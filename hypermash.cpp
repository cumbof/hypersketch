#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <iomanip>
#include <random>
#include <numeric>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <thread>
#include <atomic>
#include <mutex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Define software version
const std::string HYPERMASH_VERSION = "1.0";

//==============================================================================
// Type Definitions and Helper Functions
//==============================================================================
using Hypervector = std::vector<int>;
using MemoryBank = std::vector<Hypervector>;

// FNV-1a hash for robust, deterministic hashing.
uint64_t deterministic_hash(const std::string& str) {
    uint64_t hash = 14695981039346656037ULL;
    uint64_t fnv_prime = 1099511628211ULL;
    for (char c : str) {
        hash ^= static_cast<uint64_t>(c);
        hash *= fnv_prime;
    }
    return hash;
}

// Computes the reverse complement of a DNA sequence.
std::string reverse_complement(const std::string& dna) {
    std::string complement = "";
    complement.reserve(dna.length());
    for (char base : dna) {
        switch (base) {
            case 'A': complement += 'T'; break;
            case 'T': complement += 'A'; break;
            case 'C': complement += 'G'; break;
            case 'G': complement += 'C'; break;
            default:  complement += 'N'; break;
        }
    }
    std::reverse(complement.begin(), complement.end());
    return complement;
}

// Returns the canonical k-mer.
std::string get_canonical(const std::string& kmer) {
    std::string rev_comp = reverse_complement(kmer);
    return (kmer < rev_comp) ? kmer : rev_comp;
}


//==============================================================================
// HashedGraphEncoder Class
//==============================================================================
class HashedGraphEncoder {
public:
    HashedGraphEncoder(int k_size, int dimensions, size_t memory_bank_size)
        : k(k_size), D(dimensions), M(memory_bank_size) {
        if (k <= 0 || D <= 0 || M <= 0) {
            throw std::invalid_argument("K-mer size, dimensions, and memory bank size must be positive.");
        }
        if (D % 8 != 0) {
            throw std::invalid_argument("Dimensions must be a multiple of 8 for bit-packing.");
        }
    }

    // Parallelized encoding function (MapReduce)
    MemoryBank encode_single(const std::string& genome, unsigned int num_threads) {
        if (genome.length() < static_cast<size_t>(k + 1)) {
            return MemoryBank(M, Hypervector(D, 0));
        }

        std::vector<std::thread> threads;
        std::vector<MemoryBank> local_banks(num_threads, MemoryBank(M, Hypervector(D, 0)));

        size_t total_len = genome.length() - (k + 1);
        size_t chunk_size = (total_len + num_threads - 1) / num_threads;

        // --- MAP Phase ---
        // Each thread builds its own local MemoryBank
        for (unsigned int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, total_len);
                if (start >= total_len) return;

                MemoryBank& local_bank = local_banks[t];
                for (size_t i = start; i <= end; ++i) {
                    std::string source_kmer = genome.substr(i, k);
                    std::string target_kmer = genome.substr(i + 1, k);

                    // Skip if the kmer crosses a scaffold boundary (marked by 'N')
                    if (source_kmer.find('N') != std::string::npos || target_kmer.find('N') != std::string::npos) {
                        continue;
                    }

                    source_kmer = get_canonical(source_kmer);
                    target_kmer = get_canonical(target_kmer);

                    Hypervector target_hv = get_hypervector(target_kmer, D);
                    size_t source_index = deterministic_hash(source_kmer) % M;
                    bundle_inplace(local_bank[source_index], target_hv);
                }
            });
        }
        for (auto& t : threads) t.join();

        // --- REDUCE Phase ---
        // Bundle all local banks into a final bank, also in parallel
        MemoryBank final_bank(M, Hypervector(D, 0));
        threads.clear();
        size_t rows_chunk_size = (M + num_threads - 1) / num_threads;

        for (unsigned int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                size_t start_row = t * rows_chunk_size;
                size_t end_row = std::min(start_row + rows_chunk_size, M);

                for (size_t i = start_row; i < end_row; ++i) {
                    // Bundle all local banks for this row into the final bank
                    for (const auto& local_bank : local_banks) {
                        bundle_inplace(final_bank[i], local_bank[i]);
                    }
                }
            });
        }
        for (auto& t : threads) t.join();

        return final_bank;
    }

    // Parallelized Adaptive Retraining
    void refine_sketch(MemoryBank& bank, const std::string& genome, int max_iterations, unsigned int num_threads, double initial_error_rate) {
        if (max_iterations <= 0) return;

        std::cout << "\n[WARNING] Performing adaptive error mitigation (Retraining).\n"
                  << "          Max Iterations: " << max_iterations << "\n"
                  << "          Stopping early if error rate does not improve.\n";

        double current_best_error = initial_error_rate;

        for (int iter = 0; iter < max_iterations; ++iter) {
            // --- PRE-CALCULATION PHASE ---
            // Calculate dynamic thresholds for every bucket based on its population density
            std::vector<double> bucket_thresholds(M);
            double sqrt_D = std::sqrt(static_cast<double>(D));

            // This loop is fast enough to run serially
            for(size_t i = 0; i < M; ++i) {
                double magnitude = 0.0;
                for(int val : bank[i]) magnitude += val * val;
                magnitude = std::sqrt(magnitude);

                if (magnitude == 0) {
                    bucket_thresholds[i] = 1.0; 
                } else {
                    double expected_sim = sqrt_D / magnitude;
                    bucket_thresholds[i] = expected_sim * 0.75; 
                }
            }

            std::vector<std::thread> threads;
            std::vector<MemoryBank> local_deltas(num_threads, MemoryBank(M, Hypervector(D, 0)));

            size_t total_len = genome.length() - (k + 1);
            size_t chunk_size = (total_len + num_threads - 1) / num_threads;

            // MAP PHASE: Find weak edges and create deltas
            for (unsigned int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() { 
                    size_t start = t * chunk_size;
                    size_t end = std::min(start + chunk_size, total_len);
                    if (start >= total_len) return;

                    MemoryBank& local_delta = local_deltas[t];

                    for (size_t i = start; i <= end; ++i) {
                        std::string source_kmer = genome.substr(i, k);
                        std::string target_kmer = genome.substr(i + 1, k);

                        if (source_kmer.find('N') != std::string::npos || target_kmer.find('N') != std::string::npos) continue;

                        source_kmer = get_canonical(source_kmer);
                        target_kmer = get_canonical(target_kmer);

                        size_t source_index = deterministic_hash(source_kmer) % M;

                        const Hypervector& current_vec = bank[source_index];
                        Hypervector target_hv = get_hypervector(target_kmer, D);

                        double sim = cosine_similarity(current_vec, target_hv);

                        // Use the dynamic threshold specific to this bucket
                        if (sim < bucket_thresholds[source_index]) {
                             bundle_inplace(local_delta[source_index], target_hv);
                        }
                    }
                });
            }
            for (auto& t : threads) t.join();

            // REDUCE PHASE: Apply deltas to main bank
            threads.clear();
            size_t rows_chunk_size = (M + num_threads - 1) / num_threads;

            for (unsigned int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    size_t start_row = t * rows_chunk_size;
                    size_t end_row = std::min(start_row + rows_chunk_size, M);

                    for (size_t i = start_row; i < end_row; ++i) {
                        for (const auto& local_delta : local_deltas) {
                             bundle_inplace(bank[i], local_delta[i]);
                        }
                    }
                });
            }
            for (auto& t : threads) t.join();

            // --- EVALUATION PHASE ---
            // Calculate new fidelity to check for improvement
            double new_fidelity = calculate_sketch_fidelity(bank, genome, k, M, D, num_threads);
            double new_error = 1.0 - new_fidelity;

            std::cout << "  - Iteration " << (iter + 1) << ": Error Rate " << new_error;

            if (new_error >= current_best_error) {
                std::cout << " -> No improvement (Prev: " << current_best_error << "). Stopping early." << std::endl;
                break;
            } else {
                std::cout << " -> Improved." << std::endl;
                current_best_error = new_error;
            }
        }
    }

    // Made static so it can be shared with calculate_sketch_fidelity
    static Hypervector get_hypervector(const std::string& str, int D) {
        uint64_t seed = deterministic_hash(str);
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> dist(0, 1);
        Hypervector vec(D);
        for (int i = 0; i < D; ++i) {
            vec[i] = (dist(rng) == 0) ? -1 : 1;
        }
        return vec;
    }

    // Parallelized comparison function
    static double compare_sketches(const MemoryBank& bank1, const MemoryBank& bank2, unsigned int num_threads) {
        if (bank1.size() != bank2.size() || bank1.empty()) return 0.0;
        size_t M = bank1.size();

        // Use atomics for simple, lock-free reduction
        std::atomic<double> total_cosine_similarity{0.0};
        std::atomic<int> active_slots{0};

        std::vector<std::thread> threads;
        size_t chunk_size = (M + num_threads - 1) / num_threads;

        for (unsigned int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, M);

                double local_similarity = 0.0;
                int local_active_slots = 0;

                for (size_t i = start; i < end; ++i) {
                    bool b1_zero = std::all_of(bank1[i].begin(), bank1[i].end(), [](int v){ return v == 0; });
                    bool b2_zero = std::all_of(bank2[i].begin(), bank2[i].end(), [](int v){ return v == 0; });

                    // Only compare if at least one bucket is not empty.
                    // If one is empty and the other isn't, cosine_similarity safely returns 0.
                    if (!b1_zero || !b2_zero) {
                        local_active_slots++;
                        local_similarity += cosine_similarity(bank1[i], bank2[i]);
                    }
                }

                // Atomically add the local sums to the global totals
                double current_sim = total_cosine_similarity.load(std::memory_order_relaxed);
                while (!total_cosine_similarity.compare_exchange_weak(current_sim, current_sim + local_similarity, std::memory_order_relaxed));

                active_slots += local_active_slots;
            });
        }
        for (auto& t : threads) t.join();

        if (active_slots == 0) return 1.0;
        return total_cosine_similarity / active_slots;
    }

    // Parallelized fidelity calculation
    static double calculate_sketch_fidelity(const MemoryBank& bank, const std::string& genome, int k, size_t M, int D, unsigned int num_threads) {
        size_t len = genome.length();
        if (len < static_cast<size_t>(k + 1)) return 1.0; // No edges, perfect fidelity.

        size_t total_edges = len - k;
        std::atomic<double> total_normalized_similarity{0.0};

        std::vector<std::thread> threads;
        size_t chunk_size = (total_edges + num_threads - 1) / num_threads;

        for (unsigned int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, total_edges);
                if (start >= total_edges) return;

                double local_similarity = 0.0;

                for (size_t i = start; i <= end; ++i) {
                    std::string source_kmer = genome.substr(i, k);
                    std::string target_kmer = genome.substr(i + 1, k);
                    if (source_kmer.find('N') != std::string::npos || target_kmer.find('N') != std::string::npos) continue;

                    source_kmer = get_canonical(source_kmer);
                    target_kmer = get_canonical(target_kmer);

                    size_t source_index = deterministic_hash(source_kmer) % M;
                    const Hypervector& bundled_vec = bank[source_index];
                    Hypervector target_hv = get_hypervector(target_kmer, D);

                    double sim = cosine_similarity(bundled_vec, target_hv);
                    local_similarity += (sim + 1.0) / 2.0;
                }

                double current_sim = total_normalized_similarity.load(std::memory_order_relaxed);
                while (!total_normalized_similarity.compare_exchange_weak(current_sim, current_sim + local_similarity, std::memory_order_relaxed));
            });
        }
        for (auto& t : threads) t.join();

        if (total_edges == 0) return 1.0;

        return total_normalized_similarity / (total_edges + 1); // +1 to match original loop
    }


private:
    int k, D;
    size_t M;

    void bundle_inplace(Hypervector& v1, const Hypervector& v2) const {
        for (size_t i = 0; i < v1.size(); ++i) v1[i] += v2[i];
    }

    static double cosine_similarity(const Hypervector& v1, const Hypervector& v2) {
        double dot = 0.0, mag1_sq = 0.0, mag2_sq = 0.0;
        for(size_t i = 0; i < v1.size(); ++i) {
            dot += static_cast<double>(v1[i]) * v2[i];
            mag1_sq += static_cast<double>(v1[i]) * v1[i];
            mag2_sq += static_cast<double>(v2[i]) * v2[i];
        }
        if (mag1_sq == 0.0 && mag2_sq == 0.0) return 1.0;
        if (mag1_sq == 0.0 || mag2_sq == 0.0) return 0.0;
        return dot / (std::sqrt(mag1_sq) * std::sqrt(mag2_sq));
    }
};

//==============================================================================
// File I/O
//==============================================================================
struct HypermashHeader {
    uint32_t magic_number = 0x484D5348; // "HMSH"
    char version[16];
    int k;
    int D;
    size_t M;
    char source_file[256]; // Replaced num_genomes with single source file
};

void save_sketch(const MemoryBank& bank, const std::string& filepath, const HypermashHeader& header) {
    std::ofstream outfile(filepath, std::ios::binary);
    if (!outfile) throw std::runtime_error("Cannot open file for writing: " + filepath);

    // Write the simplified header
    outfile.write(reinterpret_cast<const char*>(&header), sizeof(HypermashHeader));

    if (bank.empty()) return;

    size_t packed_vec_size = header.D / 8;
    std::vector<uint8_t> packed_vec(packed_vec_size);
    for (const auto& vec : bank) {
        std::fill(packed_vec.begin(), packed_vec.end(), 0);
        for(int i = 0; i < header.D; ++i) {
            if (vec[i] > 0) packed_vec[i / 8] |= (1 << (i % 8));
        }
        outfile.write(reinterpret_cast<const char*>(packed_vec.data()), packed_vec_size);
    }
}

MemoryBank load_sketch(const std::string& filepath, HypermashHeader& header) {
    std::ifstream infile(filepath, std::ios::binary);
    if (!infile) throw std::runtime_error("Cannot open file for reading: " + filepath);

    infile.read(reinterpret_cast<char*>(&header), sizeof(HypermashHeader));
    if (header.magic_number != 0x484D5348) throw std::runtime_error("Not a valid Hypermash (.hms) file.");

    if (header.M == 0) return {};
    size_t packed_vec_size = header.D / 8;
    std::vector<uint8_t> packed_vec(packed_vec_size);
    MemoryBank bank(header.M, Hypervector(header.D));

    for (size_t i = 0; i < header.M; ++i) {
        infile.read(reinterpret_cast<char*>(packed_vec.data()), packed_vec_size);
        if(!infile.good()) throw std::runtime_error("Unexpected end of file while reading sketch data.");

        // A truly empty bucket is saved as entirely 0s.
        // Binarized normal vectors have virtually a 0% chance of being entirely 0s.
        // We use all-0s as "empty bucket".
        bool is_empty = true;
        for (uint8_t byte : packed_vec) {
            if (byte != 0) {
                is_empty = false;
                break;
            }
        }

        for (int j = 0; j < header.D; ++j) {
            if (is_empty) {
                bank[i][j] = 0; // Keep the bucket properly empty
            } else {
                bank[i][j] = ((packed_vec[j / 8] >> (j % 8)) & 1) ? 1 : -1;
            }
        }
    }
    return bank;
}

std::string read_genome_from_fasta(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file) throw std::runtime_error("Could not open FASTA file " + filepath);
    std::string genome, line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (line[0] == '>') {
            // Multi-scaffold Support. Add 'N' to break edges between scaffolds.
            if (!genome.empty() && genome.back() != 'N') {
                genome += "N"; 
            }
            continue;
        }

        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        for (char &c : line) { c = toupper(c); }
        genome += line;
    }
    return genome;
}

//==============================================================================
// Command Functions
//==============================================================================

// Helper to get default thread count
unsigned int get_default_threads() {
    unsigned int threads = std::thread::hardware_concurrency();
    return (threads == 0) ? 1 : threads;
}

void print_help() {
    std::cout << "Hypermash (Simple, Parallel): A tool for genome sketching (v" << HYPERMASH_VERSION << ")\n\n"
              << "Usage: hypermash <command> [options] <input_file>\n\n"
              << "Commands:\n"
              << "  sketch      Create a compact sketch from a single FASTA file.\n"
              << "  dist        Calculate similarity between two sketch files.\n"
              << "  info        Display metadata and information about a sketch file.\n"
              << "  version     Print the software version.\n"
              << "  help        Show this help message.\n\n"
              << "Options:\n"
              << "  -k <int>    K-mer size (default: 9)\n"
              << "  -d <int>    Hypervector dimensions (must be multiple of 8, default: 10000)\n"
              << "  -m <int>    Memory bank size (default: 4096)\n"
              << "  -t <int>    Number of threads (default: all available cores)\n"
              << "  -r <int>    Retraining iterations for error mitigation (default: 0)\n"
              << "  -o <file>   Output file prefix (for sketch command)\n";
}

void handle_sketch(const std::vector<std::string>& args) {
    std::string input_file;
    std::string output_file;
    int k = 9, D = 10000;
    size_t M = 4096;
    unsigned int num_threads = get_default_threads();
    int iterations = 0;

    size_t i = 2;
    while (i < args.size()) {
        if (args[i] == "-k" && i + 1 < args.size()) { k = std::stoi(args[++i]); }
        else if (args[i] == "-d" && i + 1 < args.size()) { D = std::stoi(args[++i]); }
        else if (args[i] == "-m" && i + 1 < args.size()) { M = std::stoul(args[++i]); }
        else if (args[i] == "-t" && i + 1 < args.size()) { num_threads = std::stoi(args[++i]); }
        else if (args[i] == "-r" && i + 1 < args.size()) { iterations = std::stoi(args[++i]); }
        else if (args[i] == "-o" && i + 1 < args.size()) { output_file = args[++i] + ".hms"; }
        else if (args[i][0] != '-') { 
            if (!input_file.empty()) throw std::runtime_error("Only one input FASTA file is allowed.");
            input_file = args[i]; 
        }
        i++;
    }

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

    std::cout << "Creating single sketch for " << input_file << " using " << num_threads << " thread(s)..." << std::endl;
    std::string genome = read_genome_from_fasta(input_file);
    MemoryBank bank = encoder.encode_single(genome, num_threads);

    std::cout << "Sketch complete. Calculating initial fidelity..." << std::endl;

    double fidelity_score = HashedGraphEncoder::calculate_sketch_fidelity(bank, genome, k, M, D, num_threads);
    double error_rate = 1.0 - fidelity_score;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Fidelity Score: " << fidelity_score << "\n"
              << "  Error Rate:     " << error_rate << std::endl;

    if (iterations > 0) {
        // Pass the initial error rate to allow for early stopping comparison
        encoder.refine_sketch(bank, genome, iterations, num_threads, error_rate);

        std::cout << "Retraining complete." << std::endl;
    }

    HypermashHeader header;
    strncpy(header.version, HYPERMASH_VERSION.c_str(), 15); header.version[15] = '\0';
    header.k = k; header.D = D; header.M = M;
    strncpy(header.source_file, input_file.c_str(), 255); header.source_file[255] = '\0';

    std::cout << "Saving sketch to " << output_file << "..." << std::endl;
    save_sketch(bank, output_file, header);
    std::cout << "Sketch saved successfully." << std::endl;
}

void handle_dist(const std::vector<std::string>& args) {
    if (args.size() < 4) throw std::runtime_error("dist command requires two sketch files.");

    std::string file1, file2;
    unsigned int num_threads = get_default_threads();

    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "-t" && i + 1 < args.size()) {
            num_threads = std::stoi(args[++i]);
        } else if (file1.empty()) {
            file1 = args[i];
        } else if (file2.empty()) {
            file2 = args[i];
        }
    }

    if (file1.empty() || file2.empty()) {
        throw std::runtime_error("dist command requires two sketch files.");
    }
    if (num_threads == 0) throw std::runtime_error("Thread count must be at least 1.");

    HypermashHeader h1, h2;
    MemoryBank b1 = load_sketch(file1, h1);
    MemoryBank b2 = load_sketch(file2, h2);

    if (h1.k != h2.k || h1.D != h2.D || h1.M != h2.M) {
        throw std::runtime_error("Sketch parameters do not match. Cannot compare sketches with different k, D, or M values.");
    }

    double sim = HashedGraphEncoder::compare_sketches(b1, b2, num_threads);

    // Reverse mathematical translation to approximate Mash Distance directly.
    // HDC Cosine Similarity scales as (2 / PI) * arcsin(Jaccard)
    double estimated_jaccard = std::sin(sim * M_PI / 2.0);

    double mash_dist = 1.0; 
    if (estimated_jaccard > 0.0) {
        if (estimated_jaccard >= 1.0) mash_dist = 0.0;
        else mash_dist = -1.0 / h1.k * std::log(estimated_jaccard);
    }

    std::cout << "Raw HDC Similarity: " << sim << std::endl;
    std::cout << "Estimated Jaccard:  " << estimated_jaccard << std::endl;
    std::cout << "Est. Mash Distance: " << mash_dist << std::endl;
}

void handle_info(const std::vector<std::string>& args) {
    if (args.size() != 3) throw std::runtime_error("info command requires exactly one sketch file.");
    std::string file = args[2];
    HypermashHeader header;

    load_sketch(file, header); 

    std::cout << "--- Hypermash Sketch Info ---\n"
              << "File Path:            " << file << "\n"
              << "Sketch Version:       " << header.version << "\n"
              << "--- Parameters ---\n"
              << "k-mer size (k):       " << header.k << "\n"
              << "Dimensions (D):       " << header.D << "\n"
              << "Memory Bank Size (M): " << header.M << "\n"
              << "--- Source File ---\n" 
              << "- " << header.source_file << "\n";
}

//==============================================================================
// Main Function
//==============================================================================
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