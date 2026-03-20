/**
 * HYPERMASH
 * 
 * Hypermash uses Hyperdimensional Computing (HDC) to encode the topology of De Bruijn 
 * sequence graphs into binarized hypervectors. Unlike "bag-of-words" MinHash approaches, 
 * Hypermash preserves structural variation data (edges between adjacent k-mers).
 * 
 * g++ -O3 -std=c++17 -pthread -march=native hypermash.cpp -o hypermash
 */

// Explicit hardware vectorization pragmas to force the compiler to utilize 256-bit registers.
// This allows the CPU to process 8 dimensions simultaneously per clock cycle.
#pragma GCC optimize("O3,unroll-loops,fast-math")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <thread>
#include <atomic>
#include <sstream>
#include <cstdlib>
#include <memory>

// OS-Level Memory Mapping for Zero-Copy I/O (UNIX/Linux/macOS)
#ifndef _MSC_VER
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Hardware-level popcount instruction mapping.
// Popcount counts the number of '1' bits in an integer in a single CPU cycle.
#ifdef _MSC_VER
#include <intrin.h>
#define POPCOUNT64 __popcnt64
#else
#define POPCOUNT64 __builtin_popcountll
#endif

// Define software version
const std::string HYPERMASH_VERSION = "1.0";

//==============================================================================
// Global L1 Cache Lookup Table (LUT) for AVX2 Vectorization
//==============================================================================
// To avoid slow bit-shifting and conditional branching when mapping a random 64-bit 
// integer into an array of +1/-1 values, we pre-compute all 256 possible byte states.
// This 8KB table fits entirely inside the CPU's ultra-fast L1 cache.
static int BIT_TO_INT_LUT[256][8];
static bool lut_initialized = false;

//==============================================================================
// Type Definitions
//==============================================================================
// A Memory Bank is a flattened 1D array representing an M x D matrix of integers.
using MemoryBank = std::vector<int>;

// Compact memory structure for comparing sketches. 
// Uses 64-bit integers to tightly pack binary hypervectors (1 bit per dimension).
struct PackedSketch {
    size_t M;                    // Number of memory buckets
    int D;                       // Hypervector dimensions (e.g., 10,000)
    int num_blocks;              // Number of 64-bit integers needed to store D bits
    std::vector<uint64_t> bits;  // The flattened 1D array of bit-packed vectors
    std::vector<uint8_t> empty;  // Fast sentinel array: 1 if bucket is entirely empty
};

//==============================================================================
// Ultra-Fast Random Access PRNG and Hash Functions
//==============================================================================

/**
 * deterministic_hash_kmer: Hashes a 2-bit integer-encoded k-mer.
 * Uses a SIMD Within A Register (SWAR) technique to extract ASCII characters 
 * directly from a 64-bit magic constant ("TGCA") without doing slow array lookups.
 * Perfectly replicates standard FNV-1a string hash behavior at hardware speeds.
 */
static inline uint64_t deterministic_hash_kmer(uint64_t kmer_bits, int k) {
    uint64_t hash = 14695981039346656037ULL;
    static const uint64_t magic_chars = 0x54474341ULL; // "TGCA" in little-endian binary
    
    // Force the compiler to unroll the loop, removing loop-counter overhead.
    #pragma GCC unroll 32
    for (int i = 0; i < k; ++i) {
        int base_val = (kmer_bits >> (2 * (k - 1 - i))) & 3;
        uint64_t c = (magic_chars >> (base_val * 8)) & 0xFF; // SWAR extraction
        hash ^= c;
        hash *= 1099511628211ULL; // FNV-1a prime
    }
    return hash;
}

/**
 * get_block_bits: A stateless SplitMix64 pseudo-random number generator (PRNG).
 * By passing the dimension block index, we can instantly jump to any block of 
 * 64 dimensions without generating previous numbers in a loop.
 */
static inline uint64_t get_block_bits(uint64_t seed, int block_idx) {
    // Weyl Sequence: Addition prevents dimensions from clumping together mathematically.
    uint64_t z = seed + (static_cast<uint64_t>(block_idx) * 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

//==============================================================================
// HashedGraphEncoder Class
//==============================================================================
class HashedGraphEncoder {
public:
    HashedGraphEncoder(int k_size, int dimensions, size_t memory_bank_size)
        : k(k_size), D(dimensions), M(memory_bank_size) {
        if (k <= 0 || D <= 0 || M <= 0) throw std::invalid_argument("Invalid params.");
        if (D % 8 != 0) throw std::invalid_argument("Dimensions must be a multiple of 8.");
        if (k > 32) throw std::invalid_argument("Max k-mer size is 32.");
        
        // Mask to cleanly truncate overflowing bits when shifting 2-bit k-mers
        kmer_mask = (k == 32) ? ~0ULL : (1ULL << (2 * k)) - 1;

        // Initialize the global L1 AVX2 lookup table once
        if (!lut_initialized) {
            for (int i = 0; i < 256; ++i) {
                for (int j = 0; j < 8; ++j) {
                    BIT_TO_INT_LUT[i][j] = ((i >> j) & 1) ? 1 : -1;
                }
            }
            lut_initialized = true;
        }
    }

    /**
     * encode_single: The core sketching engine.
     * Takes a pre-tokenized genome and converts it into an HDC Memory Bank using 
     * a 4-Phase hyper-optimized Map-Reduce architecture.
     */
    MemoryBank encode_single(const std::vector<uint8_t>& tokenized_genome, unsigned int num_threads) {
        if (tokenized_genome.size() <= static_cast<size_t>(k)) return MemoryBank(M * D, 0);

        size_t chunk_size = (tokenized_genome.size() + num_threads - 1) / num_threads;
        if (chunk_size == 0) chunk_size = 1;

        // DATA-ORIENTED DESIGN: Structure of Arrays (SoA).
        // By separating buckets and target_hashes into distinct parallel arrays, 
        // we eliminate 4-byte struct padding, packing the L1/L2 cache perfectly tight.
        struct LocalData {
            std::unique_ptr<uint32_t[]> buckets;
            std::unique_ptr<uint64_t[]> target_hashes;
            size_t count = 0;
            std::unique_ptr<uint32_t[]> counts;

            LocalData(size_t cap, size_t m_size) {
                // Using 'new T[]' allocates raw memory WITHOUT zeroing it out, saving millions of CPU cycles.
                buckets.reset(new uint32_t[cap]);
                target_hashes.reset(new uint64_t[cap]);
                // 'new T[]()' forces zero-initialization (required for counting frequencies).
                counts.reset(new uint32_t[m_size]());
            }
        };

        std::vector<std::unique_ptr<LocalData>> thread_data(num_threads);
        std::vector<std::thread> threads;

        // ==============================================================================
        // PHASE 1: Parallel Edge Extraction & Counting
        // ==============================================================================
        for (unsigned int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t, chunk_size]() {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, tokenized_genome.size());
                if (start >= tokenized_genome.size()) {
                    thread_data[t] = std::make_unique<LocalData>(0, M);
                    return;
                }

                size_t max_edges = end - start;
                thread_data[t] = std::make_unique<LocalData>(max_edges, M);
                auto& data = *thread_data[t];

                size_t read_start = (start >= static_cast<size_t>(k)) ? start - k : 0;
                uint64_t kmer_fwd = 0, kmer_rev = 0, prev_canonical = 0;
                int valid_bases = 0;

                for (size_t j = read_start; j < end; ++j) {
                    uint8_t base = tokenized_genome[j];
                    
                    // Value > 3 indicates an 'N' or ambiguous base. Instantly resets the sequence 
                    // to break the graph at scaffold boundaries, preventing false structural variations.
                    if (base > 3) { valid_bases = 0; continue; }

                    // Rolling 2-bit hash computation
                    kmer_fwd = ((kmer_fwd << 2) | base) & kmer_mask;
                    uint64_t rev_base = (~base) & 3;
                    kmer_rev = (kmer_rev >> 2) | (rev_base << (2 * (k - 1)));
                    valid_bases++;

                    if (valid_bases >= k) {
                        uint64_t current_canonical = std::min(kmer_fwd, kmer_rev);
                        if (valid_bases > k && j >= start) {
                            // Extract graph edge: Node A (bucket) -> Node B (target_hash)
                            uint64_t prev_seed = deterministic_hash_kmer(prev_canonical, k);
                            uint32_t bucket = static_cast<uint32_t>(prev_seed % M);
                            uint64_t target_hash = deterministic_hash_kmer(current_canonical, k);
                            
                            // Save to thread-local Data-Oriented arrays
                            data.buckets[data.count] = bucket;
                            data.target_hashes[data.count] = target_hash;
                            data.count++;
                            data.counts[bucket]++; // Tally bucket frequencies for Phase 2
                        }
                        prev_canonical = current_canonical;
                    }
                }
            });
        }
        for (auto& t : threads) t.join();

        // ==============================================================================
        // PHASE 2: Flattened Prefix Sums
        // ==============================================================================
        // Calculate the exact final memory locations for every edge to prepare for a 1D sort.
        std::vector<uint32_t> global_offsets(M + 1, 0);
        std::unique_ptr<uint32_t[]> thread_write_offsets(new uint32_t[num_threads * M]);

        uint32_t current_pos = 0;
        for (size_t m = 0; m < M; ++m) {
            global_offsets[m] = current_pos;
            for (unsigned int t = 0; t < num_threads; ++t) {
                thread_write_offsets[t * M + m] = current_pos;
                current_pos += thread_data[t]->counts[m];
            }
        }
        global_offsets[M] = current_pos; 

        // ==============================================================================
        // PHASE 3: Parallel O(N) Counting Sort (Scatter)
        // ==============================================================================
        // Allocates pure, uninitialized RAM to bypass the heavy overhead of std::vector zeroing.
        auto sorted_targets = std::unique_ptr<uint64_t[]>(new uint64_t[current_pos]);
        
        threads.clear();
        for (unsigned int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                auto& data = *thread_data[t];
                uint32_t* write_offsets = &thread_write_offsets[t * M];
                
                // Scatter edges into contiguous memory blocks grouped perfectly by their target Bucket.
                for (size_t i = 0; i < data.count; ++i) {
                    uint32_t bucket = data.buckets[i];
                    uint64_t hash = data.target_hashes[i];
                    sorted_targets[write_offsets[bucket]++] = hash;
                }
            });
        }
        for (auto& t : threads) t.join();

        // ==============================================================================
        // PHASE 4: L1-Cache Locked Vector Generation
        // ==============================================================================
        MemoryBank final_bank(M * D, 0);
        threads.clear();
        size_t buckets_per_thread = (M + num_threads - 1) / num_threads;

        for (unsigned int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t, buckets_per_thread]() {
                size_t start_m = t * buckets_per_thread;
                size_t end_m = std::min(start_m + buckets_per_thread, M);
                if (start_m >= M) return;

                int full_blocks = D / 64;
                int max_leftover = D % 64;
                const uint64_t WEYL_MAGIC = 0x9e3779b97f4a7c15ULL;

                // Process generation Bucket by Bucket.
                for (size_t m = start_m; m < end_m; ++m) {
                    size_t edge_start = global_offsets[m];
                    size_t edge_end = global_offsets[m+1];
                    if (edge_start == edge_end) continue; 

                    int* row = &final_bank[m * D];

                    // Because all edges targeting this bucket are contiguous in memory (thanks to Phase 3),
                    // the CPU locks `row` into the L1 cache. We process thousands of edges instantly 
                    // without experiencing Cache Thrashing or requesting Main RAM.
                    for (size_t e = edge_start; e < edge_end; ++e) {
                        uint64_t current_z = sorted_targets[e];

                        for (int b = 0; b < full_blocks; ++b) {
                            // Fast inline SplitMix64 step
                            uint64_t z = current_z;
                            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
                            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
                            uint64_t r = z ^ (z >> 31);
                            
                            // WEYL Sequence: Simple addition avoids multiplication cycles
                            current_z += WEYL_MAGIC; 

                            int* target_row = &row[b * 64];
                            uint8_t* r_bytes = reinterpret_cast<uint8_t*>(&r);
                            
                            // The L1 LUT Vectorization Trick
                            // Compiles down to AVX2 VPADDD instructions, doing 8 dimensions simultaneously.
                            #pragma GCC unroll 8
                            for (int i = 0; i < 8; ++i) {
                                const int* lut = BIT_TO_INT_LUT[r_bytes[i]];
                                #pragma GCC unroll 8
                                for(int j = 0; j < 8; ++j) target_row[i * 8 + j] += lut[j];
                            }
                        }
                        
                        // Handle remaining dimensions if D is not a perfect multiple of 64
                        if (max_leftover > 0) {
                            uint64_t z = current_z;
                            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
                            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
                            uint64_t r = z ^ (z >> 31);
                            
                            int* target_row = &row[full_blocks * 64];
                            uint8_t* r_bytes = reinterpret_cast<uint8_t*>(&r);
                            
                            for (int i = 0; i < max_leftover / 8; ++i) {
                                const int* lut = BIT_TO_INT_LUT[r_bytes[i]];
                                #pragma GCC unroll 8
                                for(int j = 0; j < 8; ++j) target_row[i * 8 + j] += lut[j];
                            }
                            for (int i = (max_leftover / 8) * 8; i < max_leftover; ++i) {
                                target_row[i] += (((r >> i) & 1) << 1) - 1;
                            }
                        }
                    }
                }
            });
        }
        for (auto& t : threads) t.join();

        return final_bank;
    }

    /**
     * fast_compare_packed_single: 
     * Computes the mathematical Cosine Similarity between two binarized sketches.
     * Uses bitwise XOR and Hardware POPCOUNT to reduce floating-point operations 
     * to a single algebraic fraction: Sim = 1 - (2 * Hamming) / Dimensions.
     */
    static double fast_compare_packed_single(const PackedSketch& s1, const PackedSketch& s2) {
        if (s1.M != s2.M || s1.D != s2.D || s1.M == 0) return 0.0;

        uint64_t local_hamming = 0;
        uint32_t local_active = 0;
        uint32_t local_single = 0;

        int blocks = s1.num_blocks;
        const uint64_t* bits1 = s1.bits.data();
        const uint64_t* bits2 = s2.bits.data();

        for (size_t i = 0; i < s1.M; ++i) {
            // Ignore buckets that are empty in both sketches (prevents baseline inflation)
            if (s1.empty[i] && s2.empty[i]) continue;
            
            if (!s1.empty[i] && !s2.empty[i]) {
                local_active++;
                const uint64_t* row1 = &bits1[i * blocks];
                const uint64_t* row2 = &bits2[i * blocks];

                int b = 0;
                // Aggressive loop unrolling for POPCOUNT pipeline filling
                for (; b <= blocks - 4; b += 4) {
                    local_hamming += POPCOUNT64(row1[b] ^ row2[b]);
                    local_hamming += POPCOUNT64(row1[b+1] ^ row2[b+1]);
                    local_hamming += POPCOUNT64(row1[b+2] ^ row2[b+2]);
                    local_hamming += POPCOUNT64(row1[b+3] ^ row2[b+3]);
                }
                for (; b < blocks; ++b) {
                    local_hamming += POPCOUNT64(row1[b] ^ row2[b]);
                }
            } else {
                // If one bucket is empty and the other isn't, the similarity is mathematically 0.0.
                // We tally it here to increment the denominator without adding to the similarity numerator.
                local_single++; 
            }
        }

        uint32_t total_active_slots = local_active + local_single;
        if (total_active_slots == 0) return 1.0;

        // Algebraic translation: Cosine Similarity of Binarized Vectors
        double sum_similarity = (double)local_active - (2.0 * local_hamming) / s1.D;
        return sum_similarity / total_active_slots;
    }

private:
    int k, D;
    size_t M;
    uint64_t kmer_mask;
};

//==============================================================================
// File I/O
//==============================================================================
struct HypermashHeader {
    uint32_t magic_number = 0x484D5348; // ASCII "HMSH"
    char version[16];
    int k;
    int D;
    size_t M;
    char source_file[256];
};

/**
 * save_sketch: Compresses the integer MemoryBank into a binary bit-packed file.
 */
void save_sketch(const MemoryBank& bank, const std::string& filepath, const HypermashHeader& header) {
    std::ofstream outfile(filepath, std::ios::binary);
    if (!outfile) throw std::runtime_error("Cannot open file for writing: " + filepath);

    outfile.write(reinterpret_cast<const char*>(&header), sizeof(HypermashHeader));
    if (bank.empty()) return;

    size_t packed_vec_size = header.D / 8;
    std::vector<uint8_t> packed_vec(packed_vec_size);
    
    for (size_t i = 0; i < header.M; ++i) {
        std::fill(packed_vec.begin(), packed_vec.end(), 0);
        const int* row = &bank[i * header.D];
        
        // 8-WAY UNROLLED BIT PACKING
        // Bypasses extremely slow modulo math (`d % 8`) using standard binary OR gates.
        for(int d = 0; d <= header.D - 8; d += 8) {
            uint8_t byte = 0;
            if (row[d] > 0)   byte |= 1;
            if (row[d+1] > 0) byte |= 2;
            if (row[d+2] > 0) byte |= 4;
            if (row[d+3] > 0) byte |= 8;
            if (row[d+4] > 0) byte |= 16;
            if (row[d+5] > 0) byte |= 32;
            if (row[d+6] > 0) byte |= 64;
            if (row[d+7] > 0) byte |= 128;
            packed_vec[d / 8] = byte;
        }
        // Tail cleanup for remaining dimensions
        for(int d = (header.D / 8) * 8; d < header.D; ++d) {
            if (row[d] > 0) packed_vec[d / 8] |= (1 << (d % 8));
        }
        
        outfile.write(reinterpret_cast<const char*>(packed_vec.data()), packed_vec_size);
    }
}

/**
 * load_packed_sketch: Loads a bit-packed file directly into memory.
 * Uses instantaneous Hardware DMA block copies (`std::memcpy`) to bypass billions 
 * of bit-shifting loop iterations.
 */
PackedSketch load_packed_sketch(const std::string& filepath, HypermashHeader& header) {
    std::FILE* fp = std::fopen(filepath.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open sketch file " + filepath);
    
    // Quick size determination
    std::fseek(fp, 0, SEEK_END);
    size_t size = std::ftell(fp);
    std::rewind(fp);
    
    std::vector<uint8_t> buffer(size);
    if (std::fread(buffer.data(), 1, size, fp) != size) {
        std::fclose(fp);
        throw std::runtime_error("Failed to read sketch file.");
    }
    std::fclose(fp);

    std::memcpy(&header, buffer.data(), sizeof(HypermashHeader));
    if (header.magic_number != 0x484D5348) throw std::runtime_error("Not a valid Hypermash file.");

    PackedSketch sketch;
    sketch.M = header.M;
    sketch.D = header.D;
    if (sketch.M == 0) return sketch;

    sketch.num_blocks = (header.D + 63) / 64;
    sketch.bits.resize(sketch.M * sketch.num_blocks, 0);
    sketch.empty.resize(sketch.M, 0);

    size_t packed_vec_size = header.D / 8;
    size_t offset = sizeof(HypermashHeader);

    for (size_t i = 0; i < sketch.M; ++i) {
        uint8_t* row_bytes = buffer.data() + offset;
        
        // Fast SWAR emptiness check (Checking 64-bits / 8 bytes at a time)
        bool is_empty = true;
        size_t num_qwords = packed_vec_size / 8;
        uint64_t* qwords = reinterpret_cast<uint64_t*>(row_bytes);
        for (size_t q = 0; q < num_qwords; ++q) {
            if (qwords[q] != 0) { is_empty = false; break; }
        }
        if (is_empty) {
            for (size_t b = num_qwords * 8; b < packed_vec_size; ++b) {
                if (row_bytes[b] != 0) { is_empty = false; break; }
            }
        }

        sketch.empty[i] = is_empty ? 1 : 0; 

        if (!is_empty) {
            // Instantaneous direct memory copy of the pre-packed bytes
            std::memcpy(&sketch.bits[i * sketch.num_blocks], row_bytes, packed_vec_size);
        }
        offset += packed_vec_size;
    }
    return sketch;
}

/**
 * load_and_tokenize_fasta: Zero-Copy OS Memory Mapping
 * Instead of reading the file into a RAM string, it asks the Unix kernel to map the 
 * physical SSD disk blocks directly into the CPU's virtual memory space via `mmap`. 
 * Characters are parsed and converted to 0-3 integers on the fly using pointer-bumping.
 */
std::vector<uint8_t> load_and_tokenize_fasta(const std::string& filepath) {
#ifndef _MSC_VER
    // UNIX Systems: Hyper-fast mmap (Zero-Copy OS Memory)
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error("Could not open FASTA file " + filepath);

    struct stat sb;
    if (fstat(fd, &sb) == -1) throw std::runtime_error("Could not stat FASTA file.");
    size_t size = sb.st_size;

    char* buffer = static_cast<char*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (buffer == MAP_FAILED) throw std::runtime_error("mmap failed.");

    // Flat lookup array for blistering-fast translation mapping
    uint8_t char_to_base[256];
    std::fill_n(char_to_base, 256, 255); 
    char_to_base['A'] = 0; char_to_base['a'] = 0;
    char_to_base['C'] = 1; char_to_base['c'] = 1;
    char_to_base['G'] = 2; char_to_base['g'] = 2;
    char_to_base['T'] = 3; char_to_base['t'] = 3;

    // POINTER-BUMPING FASTA LOADER
    // Allocates raw array and uses ptr arithmetic to bypass std::vector bounds checks.
    std::vector<uint8_t> tokenized(size); 
    uint8_t* out_ptr = tokenized.data();
    size_t valid_count = 0;
    bool in_header = false;

    for (size_t i = 0; i < size; ++i) {
        char c = buffer[i];
        if (c == '>') {
            in_header = true;
            // Inject structural scaffold-breaker if transitioning between chromosomes
            if (valid_count > 0 && out_ptr[valid_count - 1] != 255) {
                out_ptr[valid_count++] = 255; 
            }
            continue;
        }
        if (c == '\n' || c == '\r') {
            in_header = false;
            continue;
        }
        if (!in_header && c != ' ' && c != '\t') {
            out_ptr[valid_count++] = char_to_base[static_cast<uint8_t>(c)];
        }
    }
    // Trim unused space
    tokenized.resize(valid_count);

    munmap(buffer, size);
    close(fd);
    return tokenized;
#else
    // Windows Fallback using standard block fread
    std::FILE* fp = std::fopen(filepath.c_str(), "rb");
    if (!fp) throw std::runtime_error("Could not open FASTA file " + filepath);
    
    std::fseek(fp, 0, SEEK_END);
    size_t size = std::ftell(fp);
    std::rewind(fp);
    
    std::vector<char> buffer(size);
    size_t read_size = std::fread(buffer.data(), 1, size, fp);
    std::fclose(fp);

    uint8_t char_to_base[256];
    std::fill_n(char_to_base, 256, 255); 
    char_to_base['A'] = 0; char_to_base['a'] = 0;
    char_to_base['C'] = 1; char_to_base['c'] = 1;
    char_to_base['G'] = 2; char_to_base['g'] = 2;
    char_to_base['T'] = 3; char_to_base['t'] = 3;

    std::vector<uint8_t> tokenized(read_size); 
    uint8_t* out_ptr = tokenized.data();
    size_t valid_count = 0;
    bool in_header = false;

    for (size_t i = 0; i < read_size; ++i) {
        char c = buffer[i];
        if (c == '>') {
            in_header = true;
            if (valid_count > 0 && out_ptr[valid_count - 1] != 255) {
                out_ptr[valid_count++] = 255;
            }
            continue;
        }
        if (c == '\n' || c == '\r') {
            in_header = false;
            continue;
        }
        if (!in_header && c != ' ' && c != '\t') {
            out_ptr[valid_count++] = char_to_base[static_cast<uint8_t>(c)];
        }
    }
    tokenized.resize(valid_count);

    return tokenized;
#endif
}

//==============================================================================
// Command Functions
//==============================================================================

unsigned int get_default_threads() {
    unsigned int threads = std::thread::hardware_concurrency();
    return (threads == 0) ? 1 : threads;
}

void print_help() {
    std::cout << "Hypermash (v" << HYPERMASH_VERSION << ")\n\n"
              << "Usage: hypermash <command> [options] <input_file>\n\n"
              << "Commands:\n"
              << "  sketch      Create a compact sketch from a single FASTA file.\n"
              << "  dist        Calculate similarity between two sketch files.\n"
              << "  info        Display metadata and information about a sketch file.\n"
              << "  version     Print the software version.\n"
              << "  help        Show this help message.\n\n"
              << "Options:\n"
              << "  -k <int>    K-mer size (max 32, default: 9)\n"
              << "  -d <int>    Hypervector dimensions (must be multiple of 8, default: 10000)\n"
              << "  -m <int>    Memory bank size (default: 4096)\n"
              << "  -t <int>    Number of threads (default: all available cores)\n"
              << "  -o <file>   Output file prefix (for sketch command)\n"
              << "  -l <file>   File containing a list of sketch paths, one per line (for dist command)\n";
}

void handle_sketch(const std::vector<std::string>& args) {
    std::string input_file;
    std::string output_file;
    int k = 9, D = 10000;
    size_t M = 4096;
    unsigned int num_threads = get_default_threads();

    size_t i = 2;
    while (i < args.size()) {
        if (args[i] == "-k" && i + 1 < args.size()) { k = std::stoi(args[++i]); }
        else if (args[i] == "-d" && i + 1 < args.size()) { D = std::stoi(args[++i]); }
        else if (args[i] == "-m" && i + 1 < args.size()) { M = std::stoul(args[++i]); }
        else if (args[i] == "-t" && i + 1 < args.size()) { num_threads = std::stoi(args[++i]); }
        else if (args[i] == "-o" && i + 1 < args.size()) { output_file = args[++i] + ".hms"; }
        else if (args[i][0] != '-') { 
            if (!input_file.empty()) throw std::runtime_error("Only one input FASTA file is allowed.");
            input_file = args[i]; 
        }
        i++;
    }

    if (k > 32) throw std::invalid_argument("This version uses 64-bit hardware encoding. Max k-mer size is 32.");
    if (D % 8 != 0) throw std::runtime_error("Dimensions (-d) must be a multiple of 8.");
    if (input_file.empty()) throw std::runtime_error("No input FASTA file provided.");
    if (num_threads == 0) throw std::runtime_error("Thread count must be at least 1.");

    if (output_file.empty()) {
        std::string basename = input_file;
        size_t pos = basename.find_last_of(".");
        if (pos != std::string::npos) basename = basename.substr(0, pos);
        output_file = basename + ".hms";
    }

    HashedGraphEncoder encoder(k, D, M);
    std::cout << "Processing " << input_file << "..." << std::endl;
    
    std::vector<uint8_t> tokenized_genome = load_and_tokenize_fasta(input_file);
    MemoryBank bank = encoder.encode_single(tokenized_genome, num_threads);

    HypermashHeader header;
    strncpy(header.version, HYPERMASH_VERSION.c_str(), 15); header.version[15] = '\0';
    header.k = k; header.D = D; header.M = M;
    strncpy(header.source_file, input_file.c_str(), 255); header.source_file[255] = '\0';

    save_sketch(bank, output_file, header);
}

void handle_dist(const std::vector<std::string>& args) {
    std::vector<std::string> files;
    unsigned int num_threads = get_default_threads();
    std::string list_file = "";

    // Parse command line arguments including the new -l list file feature
    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "-t" && i + 1 < args.size()) {
            num_threads = std::stoi(args[++i]);
        } else if (args[i] == "-l" && i + 1 < args.size()) {
            list_file = args[++i];
        } else if (args[i][0] != '-') {
            files.push_back(args[i]);
        }
    }

    // Parse list file if provided to bypass UNIX ARG_MAX limitations
    if (!list_file.empty()) {
        std::ifstream infile(list_file);
        if (!infile) throw std::runtime_error("Could not open list file: " + list_file);
        std::string line;
        while (std::getline(infile, line)) {
            // Trim whitespace and newlines from both ends
            line.erase(line.find_last_not_of(" \n\r\t") + 1);
            size_t start = line.find_first_not_of(" \n\r\t");
            if (start != std::string::npos) {
                files.push_back(line.substr(start));
            }
        }
    }

    if (files.size() < 2) throw std::runtime_error("dist command requires at least two sketch files (or a list file with at least two paths).");
    if (num_threads == 0) throw std::runtime_error("Thread count must be at least 1.");

    // Execution path for a simple two-file comparison
    if (files.size() == 2) {
        HypermashHeader h1, h2;
        PackedSketch b1 = load_packed_sketch(files[0], h1);
        PackedSketch b2 = load_packed_sketch(files[1], h2);

        if (h1.k != h2.k || h1.D != h2.D || h1.M != h2.M) {
            throw std::runtime_error("Sketch parameters do not match.");
        }

        double sim = HashedGraphEncoder::fast_compare_packed_single(b1, b2);
        
        // Un-distort the binarization topology using Grothendieck's Identity
        double estimated_jaccard = std::sin(sim * M_PI / 2.0);
        double mash_dist = 1.0; 
        if (estimated_jaccard > 0.0) {
            if (estimated_jaccard >= 1.0) mash_dist = 0.0;
            // Mash Mutation Probability Formula: (-1/k) * ln(Jaccard)
            else mash_dist = -1.0 / h1.k * std::log(estimated_jaccard);
        }

        std::cout << "Raw HDC Similarity: " << sim << std::endl;
        std::cout << "Estimated Jaccard:  " << estimated_jaccard << std::endl;
        std::cout << "Est. Mash Distance: " << mash_dist << std::endl;
    } 
    // Execution path for Matrix Batch Processing (All-vs-All)
    else {
        std::cerr << "Loading: " << files.size() << " sketches mapped directly to RAM..." << std::endl;
        std::vector<PackedSketch> sketches(files.size());
        std::vector<std::string> names(files.size());
        std::vector<int> k_values(files.size());
        
        // Multi-threaded load phase
        std::vector<std::thread> load_threads;
        size_t files_per_thread = (files.size() + num_threads - 1) / num_threads;
        for (unsigned int t = 0; t < num_threads; ++t) {
            load_threads.emplace_back([&, t, files_per_thread]() {
                size_t start = t * files_per_thread;
                size_t end = std::min(start + files_per_thread, files.size());
                for (size_t i = start; i < end; ++i) {
                    HypermashHeader h;
                    sketches[i] = load_packed_sketch(files[i], h);
                    k_values[i] = h.k;
                    
                    std::string basename = files[i];
                    size_t pos = basename.find_last_of("/\\");
                    if (pos != std::string::npos) basename = basename.substr(pos + 1);
                    pos = basename.find_last_of(".");
                    if (pos != std::string::npos) basename = basename.substr(0, pos);
                    names[i] = basename;
                }
            });
        }
        for (auto& t : load_threads) t.join();
        
        // MATRIX JOB GENERATION
        // Pre-calculating all valid pairs ensures zero thread-spawning overhead during processing.
        struct Job { int i; int j; };
        std::vector<Job> jobs;
        jobs.reserve((files.size() * (files.size() - 1)) / 2);
        for (size_t i = 0; i < files.size(); ++i) {
            for (size_t j = i + 1; j < files.size(); ++j) {
                if (sketches[i].M == sketches[j].M && sketches[i].D == sketches[j].D) {
                    jobs.push_back({static_cast<int>(i), static_cast<int>(j)});
                }
            }
        }

        std::cerr << "Computing " << jobs.size() << " pairwise distances..." << std::endl;
        std::vector<std::string> thread_outputs(num_threads);
        std::vector<std::thread> threads;
        
        size_t jobs_per_thread = (jobs.size() + num_threads - 1) / num_threads;

        for (unsigned int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t, jobs_per_thread]() {
                size_t start = t * jobs_per_thread;
                size_t end = std::min(start + jobs_per_thread, jobs.size());
                if (start >= jobs.size()) return;

                // Asynchronous buffering avoids console-locking delays
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(6);

                for (size_t job_idx = start; job_idx < end; ++job_idx) {
                    int i = jobs[job_idx].i;
                    int j = jobs[job_idx].j;
                    
                    double sim = HashedGraphEncoder::fast_compare_packed_single(sketches[i], sketches[j]);
                    double estimated_jaccard = std::sin(sim * M_PI / 2.0);
                    double mash_dist = 1.0; 
                    if (estimated_jaccard > 0.0) {
                        if (estimated_jaccard >= 1.0) mash_dist = 0.0;
                        else mash_dist = -1.0 / k_values[i] * std::log(estimated_jaccard);
                    }
                    
                    ss << names[i] << "\t" << names[j] << "\t" << sim << "\t" << mash_dist << "\n";
                }
                thread_outputs[t] = ss.str();
            });
        }
        for (auto& t : threads) t.join();

        // Burst dump the entire matrix to stdout instantly
        std::cout << "genome_1\tgenome_2\thdc_similarity\test_mash_dist\n";
        for (const auto& out : thread_outputs) {
            std::cout << out; 
        }
        std::cerr << "Done." << std::endl;
    }
}

void handle_info(const std::vector<std::string>& args) {
    if (args.size() != 3) throw std::runtime_error("info command requires exactly one sketch file.");
    std::string file = args[2];
    HypermashHeader header;

    load_packed_sketch(file, header); 

    std::cout << "--- Hypermash Sketch Info ---\n"
              << "File Path:          " << file << "\n"
              << "Sketch Version:     " << header.version << "\n"
              << "--- Parameters ---\n"
              << "k-mer size (k):     " << header.k << "\n"
              << "Dimensions (D):     " << header.D << "\n"
              << "Memory Bank Size (M): " << header.M << "\n"
              << "--- Source File ---\n" 
              << "- " << header.source_file << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) { print_help(); return 1; }
    std::vector<std::string> args(argv, argv + argc);
    std::string command = args[1];
    try {
        if (command == "sketch") handle_sketch(args);
        else if (command == "dist") handle_dist(args);
        else if (command == "info") handle_info(args);
        else if (command == "version") std::cout << "Hypermash version " << HYPERMASH_VERSION << std::endl;
        else if (command == "help" || command == "--help" || command == "-h") print_help();
        else { std::cerr << "Error: Unknown command '" << command << "'. Use 'hypermash help'.\n"; return 1; }
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}