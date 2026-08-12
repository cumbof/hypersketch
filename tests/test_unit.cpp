/**
 * HyperSketch Unit Tests
 *
 * Tests the internal C++ functions directly by including the source file
 * with HYPERSKETCH_TEST_BUILD defined, which strips out main().
 *
 * Framework: Catch2 v2 (single-header, auto-downloaded by the Makefile)
 *
 * Build & run:
 *   make test-unit
 */

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

// Include the full implementation, excluding main()
#define HYPERSKETCH_TEST_BUILD
#include "../hypersketch.cpp"

#include <fstream>
#include <cstdio>
#include <algorithm>

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/** Write content to a temporary file and return its path. */
static std::string write_temp_file(const std::string& content, const std::string& suffix = ".fna") {
    std::string path = std::string("/tmp/hm_test_") +
                       std::to_string(getpid()) + "_" +
                       std::to_string(rand()) + suffix;
    std::ofstream f(path);
    f << content;
    return path;
}

static void remove_file(const std::string& path) { std::remove(path.c_str()); }

// ===========================================================================
// 1. Hash & PRNG Functions
// ===========================================================================

TEST_CASE("deterministic_hash_kmer is reproducible", "[hash]") {
    REQUIRE(deterministic_hash_kmer(0b0000ULL, 2) == deterministic_hash_kmer(0b0000ULL, 2));
    REQUIRE(deterministic_hash_kmer(0b1010ULL, 4) == deterministic_hash_kmer(0b1010ULL, 4));
    REQUIRE(deterministic_hash_kmer(0xDEADBEEFULL, 16) == deterministic_hash_kmer(0xDEADBEEFULL, 16));
}

TEST_CASE("deterministic_hash_kmer differentiates distinct k-mers", "[hash]") {
    REQUIRE(deterministic_hash_kmer(0b0000ULL, 4) != deterministic_hash_kmer(0b0001ULL, 4));
    REQUIRE(deterministic_hash_kmer(0b0000ULL, 4) != deterministic_hash_kmer(0b1111ULL, 4));
    REQUIRE(deterministic_hash_kmer(0b1010ULL, 4) != deterministic_hash_kmer(0b0101ULL, 4));
}

TEST_CASE("deterministic_hash_kmer is sensitive to k", "[hash]") {
    // Same bit pattern but different k encodes a different sequence length
    REQUIRE(deterministic_hash_kmer(0b001010ULL, 3) != deterministic_hash_kmer(0b001010ULL, 4));
}

TEST_CASE("get_block_bits is reproducible", "[prng]") {
    REQUIRE(get_block_bits(42ULL, 0) == get_block_bits(42ULL, 0));
    REQUIRE(get_block_bits(0ULL, 5)  == get_block_bits(0ULL, 5));
}

TEST_CASE("get_block_bits is seed-sensitive and block-index-sensitive", "[prng]") {
    REQUIRE(get_block_bits(42ULL, 0) != get_block_bits(43ULL, 0)); // different seed
    REQUIRE(get_block_bits(42ULL, 0) != get_block_bits(42ULL, 1)); // different block
}

// ===========================================================================
// 2. HashedGraphEncoder — constructor validation
// ===========================================================================

TEST_CASE("HashedGraphEncoder rejects invalid parameters", "[encoder][constructor]") {
    SECTION("k = 0")          { REQUIRE_THROWS_AS(HashedGraphEncoder(0,   256, 64), std::invalid_argument); }
    SECTION("k > 32")         { REQUIRE_THROWS_AS(HashedGraphEncoder(33,  256, 64), std::invalid_argument); }
    SECTION("D not mult of 8"){ REQUIRE_THROWS_AS(HashedGraphEncoder(9,   100, 64), std::invalid_argument); }
    SECTION("D = 0")          { REQUIRE_THROWS_AS(HashedGraphEncoder(9,     0, 64), std::invalid_argument); }
    SECTION("M = 0")          { REQUIRE_THROWS_AS(HashedGraphEncoder(9,   256,  0), std::invalid_argument); }
    SECTION("D = 17")         { REQUIRE_THROWS_AS(HashedGraphEncoder(9,    17, 64), std::invalid_argument); }
}

TEST_CASE("HashedGraphEncoder accepts valid boundary parameters", "[encoder][constructor]") {
    REQUIRE_NOTHROW(HashedGraphEncoder(1,  8,    1));   // minimum sensible values
    REQUIRE_NOTHROW(HashedGraphEncoder(9,  256,  64));  // common case
    REQUIRE_NOTHROW(HashedGraphEncoder(32, 10000, 4096)); // maximum k
}

// ===========================================================================
// 3. HashedGraphEncoder — encode_single
// ===========================================================================

TEST_CASE("encode_single returns a zero bank for an empty genome", "[encoder][encode]") {
    HashedGraphEncoder enc(4, 256, 64);
    MemoryBank bank = enc.encode_single({}, 1);
    REQUIRE(std::all_of(bank.begin(), bank.end(), [](int v){ return v == 0; }));
}

TEST_CASE("encode_single returns a zero bank when genome is shorter than k", "[encoder][encode]") {
    HashedGraphEncoder enc(9, 256, 64);
    std::vector<uint8_t> seq = {0, 1, 2, 3, 0}; // 5 bases < k=9
    MemoryBank bank = enc.encode_single(seq, 1);
    REQUIRE(std::all_of(bank.begin(), bank.end(), [](int v){ return v == 0; }));
}

TEST_CASE("encode_single produces a non-zero bank for a valid genome", "[encoder][encode]") {
    HashedGraphEncoder enc(4, 256, 64);
    std::vector<uint8_t> seq;
    for (int i = 0; i < 100; i++) seq.push_back(static_cast<uint8_t>(i % 4));
    MemoryBank bank = enc.encode_single(seq, 1);
    REQUIRE(std::any_of(bank.begin(), bank.end(), [](int v){ return v != 0; }));
}

TEST_CASE("encode_single is deterministic across repeated calls", "[encoder][encode]") {
    HashedGraphEncoder enc(4, 256, 64);
    std::vector<uint8_t> seq;
    for (int i = 0; i < 200; i++) seq.push_back(static_cast<uint8_t>(i % 4));
    REQUIRE(enc.encode_single(seq, 1) == enc.encode_single(seq, 1));
}

TEST_CASE("encode_single produces identical output regardless of thread count", "[encoder][encode]") {
    HashedGraphEncoder enc(4, 256, 64);
    std::vector<uint8_t> seq;
    for (int i = 0; i < 1000; i++) seq.push_back(static_cast<uint8_t>((i * 3 + 7) % 4));
    REQUIRE(enc.encode_single(seq, 1) == enc.encode_single(seq, 4));
}

TEST_CASE("encode_single does not crash on a genome containing ambiguous bases", "[encoder][encode]") {
    HashedGraphEncoder enc(4, 256, 64);
    std::vector<uint8_t> seq = {0, 1, 2, 3, 255, 0, 1, 2, 3}; // ACGT-N-ACGT
    REQUIRE_NOTHROW(enc.encode_single(seq, 1));
}

TEST_CASE("encode_single severs edges at ambiguous base boundaries", "[encoder][encode]") {
    HashedGraphEncoder enc(4, 256, 64);
    // intact:  AAAAACCCC — the 5th A creates edges AAAA→AAAC→…→CCCC (5 unique edges).
    // severed: AAAA-N-CCCC — N resets valid_bases; after N only 4 C's appear (valid==k,
    //          never valid>k), so no edges are extracted at all → zero bank.
    std::vector<uint8_t> intact  = {0,0,0,0,0, 1,1,1,1}; // 9 bases
    std::vector<uint8_t> severed = {0,0,0,0, 255, 1,1,1,1}; // N severs before any C edge
    MemoryBank b_intact  = enc.encode_single(intact,  1);
    MemoryBank b_severed = enc.encode_single(severed, 1);
    // severed produces no edges at all → all-zero bank
    REQUIRE(std::all_of(b_severed.begin(), b_severed.end(), [](int v){ return v == 0; }));
    // intact has 5 unique edges → at least some non-zero entries
    REQUIRE(std::any_of(b_intact.begin(), b_intact.end(), [](int v){ return v != 0; }));
}

// ===========================================================================
// 4. L1-Cache LUT
// ===========================================================================

TEST_CASE("BIT_TO_INT_LUT maps bits to +1/-1 correctly", "[lut]") {
    HashedGraphEncoder enc(4, 256, 64); // triggers one-time LUT initialisation
    REQUIRE(BIT_TO_INT_LUT[0b00000000][0] == -1); // bit 0 of 0x00 is 0  → -1
    REQUIRE(BIT_TO_INT_LUT[0b00000001][0] ==  1); // bit 0 of 0x01 is 1  → +1
    REQUIRE(BIT_TO_INT_LUT[0b11111111][7] ==  1); // bit 7 of 0xFF is 1  → +1
    REQUIRE(BIT_TO_INT_LUT[0b10000000][7] ==  1); // bit 7 of 0x80 is 1  → +1
    REQUIRE(BIT_TO_INT_LUT[0b10000000][0] == -1); // bit 0 of 0x80 is 0  → -1
    REQUIRE(BIT_TO_INT_LUT[0b01010101][0] ==  1); // bit 0 of 0x55 is 1  → +1
    REQUIRE(BIT_TO_INT_LUT[0b01010101][1] == -1); // bit 1 of 0x55 is 0  → -1
}

// ===========================================================================
// 5. Sketch comparison
// ===========================================================================

TEST_CASE("fast_compare_packed_single returns 0 for all-empty sketches", "[compare]") {
    PackedSketch s;
    s.M = 64; s.D = 64; s.num_blocks = 1;
    s.bits.assign(64, 0);
    s.empty.assign(64, 1); // all buckets are empty
    REQUIRE(HashedGraphEncoder::fast_compare_packed_single(s, s) == 0.0);
}

TEST_CASE("fast_compare_packed_single returns 0 for mismatched sketch dimensions", "[compare]") {
    PackedSketch s1, s2;
    s1.M = 4; s1.D = 64; s1.num_blocks = 1;
    s1.bits.assign(4, 0xFF); s1.empty.assign(4, 0);
    s2.M = 8; s2.D = 64; s2.num_blocks = 1;
    s2.bits.assign(8, 0xFF); s2.empty.assign(8, 0);
    REQUIRE(HashedGraphEncoder::fast_compare_packed_single(s1, s2) == 0.0);
}

TEST_CASE("fast_compare_packed_single returns 1.0 for identical non-empty sketches", "[compare]") {
    PackedSketch s;
    s.M = 4; s.D = 64; s.num_blocks = 1;
    s.bits.assign(4, 0xDEADBEEFCAFEBABEULL);
    s.empty.assign(4, 0); // all buckets active
    REQUIRE(HashedGraphEncoder::fast_compare_packed_single(s, s) == Approx(1.0));
}

TEST_CASE("fast_compare_packed_single is symmetric", "[compare]") {
    PackedSketch s1, s2;
    s1.M = 4; s1.D = 64; s1.num_blocks = 1;
    s1.bits = {0xAAAAAAAAAAAAAAAAULL, 0x5555555555555555ULL,
               0xF0F0F0F0F0F0F0F0ULL, 0x0F0F0F0F0F0F0F0FULL};
    s1.empty.assign(4, 0);
    s2.M = 4; s2.D = 64; s2.num_blocks = 1;
    s2.bits = {0x1234567890ABCDEFULL, 0xFEDCBA0987654321ULL,
               0xA5A5A5A5A5A5A5A5ULL, 0x5A5A5A5A5A5A5A5AULL};
    s2.empty.assign(4, 0);
    double sim_12 = HashedGraphEncoder::fast_compare_packed_single(s1, s2);
    double sim_21 = HashedGraphEncoder::fast_compare_packed_single(s2, s1);
    REQUIRE(sim_12 == Approx(sim_21));
}

TEST_CASE("fast_compare_packed_single returns a lower score for dissimilar sketches", "[compare]") {
    // Identical sketch
    PackedSketch same;
    same.M = 4; same.D = 64; same.num_blocks = 1;
    same.bits.assign(4, 0xAAAAAAAAAAAAAAAAULL);
    same.empty.assign(4, 0);
    double self_sim = HashedGraphEncoder::fast_compare_packed_single(same, same);

    // Inverted sketch — maximally different
    PackedSketch diff;
    diff.M = 4; diff.D = 64; diff.num_blocks = 1;
    diff.bits.assign(4, 0x5555555555555555ULL); // every bit flipped vs same
    diff.empty.assign(4, 0);
    double diff_sim = HashedGraphEncoder::fast_compare_packed_single(same, diff);

    REQUIRE(self_sim > diff_sim);
}

// ===========================================================================
// 6. FASTA tokenisation
// ===========================================================================

TEST_CASE("load_and_tokenize_fasta maps ACGT to 0-3", "[tokenizer]") {
    auto path = write_temp_file(">seq1\nACGT\n");
    auto tok   = load_and_tokenize_fasta(path);
    remove_file(path);
    REQUIRE(tok == std::vector<uint8_t>({0, 1, 2, 3}));
}

TEST_CASE("load_and_tokenize_fasta accepts lowercase bases", "[tokenizer]") {
    auto path = write_temp_file(">seq1\nacgt\n");
    auto tok   = load_and_tokenize_fasta(path);
    remove_file(path);
    REQUIRE(tok == std::vector<uint8_t>({0, 1, 2, 3}));
}

TEST_CASE("load_and_tokenize_fasta encodes ambiguous bases as 255", "[tokenizer]") {
    auto path = write_temp_file(">seq1\nACGNT\n");
    auto tok   = load_and_tokenize_fasta(path);
    remove_file(path);
    REQUIRE(tok.size() == 5);
    REQUIRE(tok[3] == 255); // N → 255
}

TEST_CASE("load_and_tokenize_fasta inserts a 255 separator between FASTA records", "[tokenizer]") {
    auto path = write_temp_file(">seq1\nACGT\n>seq2\nTGCA\n");
    auto tok   = load_and_tokenize_fasta(path);
    remove_file(path);
    // seq1(4) + separator(1) + seq2(4) = 9 tokens
    REQUIRE(tok.size() == 9);
    REQUIRE(tok[4] == 255);
}

TEST_CASE("load_and_tokenize_fasta throws on a missing file", "[tokenizer]") {
    REQUIRE_THROWS(load_and_tokenize_fasta("/nonexistent/path/missing.fna"));
}

// ===========================================================================
// 7. File I/O round-trip
// ===========================================================================

TEST_CASE("save_sketch / load_packed_sketch round-trip preserves header fields", "[io]") {
    const int D = 256;
    const size_t M = 16;
    MemoryBank bank(M * D, 0);
    for (size_t i = 0; i < M * D; i++) bank[i] = static_cast<int>(i % 3) - 1; // -1, 0, +1

    HyperSketchHeader hdr;
    hdr.magic_number = 0x484D5348;
    strncpy(hdr.version, "1.0", 15);
    hdr.k = 9; hdr.D = D; hdr.M = M;
    strncpy(hdr.source_file, "test.fna", 255);

    auto path = write_temp_file("", ".hms");
    save_sketch(bank, path, hdr);

    HyperSketchHeader loaded_hdr;
    PackedSketch sketch = load_packed_sketch(path, loaded_hdr);
    remove_file(path);

    REQUIRE(loaded_hdr.magic_number == hdr.magic_number);
    REQUIRE(loaded_hdr.k  == hdr.k);
    REQUIRE(loaded_hdr.D  == hdr.D);
    REQUIRE(loaded_hdr.M  == hdr.M);
    REQUIRE(sketch.M == M);
    REQUIRE(sketch.D == D);
    REQUIRE(sketch.num_blocks == (D + 63) / 64);
}

TEST_CASE("load_packed_sketch throws on an invalid magic number", "[io]") {
    auto path = write_temp_file("", ".hms");
    {
        std::ofstream f(path, std::ios::binary);
        uint8_t garbage[sizeof(HyperSketchHeader)] = {};
        f.write(reinterpret_cast<char*>(garbage), sizeof(garbage));
    }
    HyperSketchHeader hdr;
    REQUIRE_THROWS(load_packed_sketch(path, hdr));
    remove_file(path);
}

TEST_CASE("save_sketch throws when the output directory does not exist", "[io]") {
    MemoryBank bank(4 * 64, 0);
    HyperSketchHeader hdr;
    hdr.magic_number = 0x484D5348;
    strncpy(hdr.version, "1.0", 15);
    hdr.k = 4; hdr.D = 64; hdr.M = 4;
    REQUIRE_THROWS(save_sketch(bank, "/nonexistent/dir/out.hms", hdr));
}

// ===========================================================================
// 8. End-to-end encode → save → load → compare (self-similarity ≈ 1)
// ===========================================================================

TEST_CASE("encoding the same sequence twice produces identical sketches (full pipeline)", "[pipeline]") {
    const int D = 256;
    const size_t M = 64;
    HashedGraphEncoder enc(4, D, M);

    std::vector<uint8_t> seq;
    for (int i = 0; i < 500; i++) seq.push_back(static_cast<uint8_t>((i * 7 + 3) % 4));

    MemoryBank bank = enc.encode_single(seq, 1);

    HyperSketchHeader hdr;
    hdr.magic_number = 0x484D5348;
    strncpy(hdr.version, "1.0", 15);
    hdr.k = 4; hdr.D = D; hdr.M = M;
    strncpy(hdr.source_file, "test.fna", 255);

    auto path1 = write_temp_file("", ".hms");
    auto path2 = write_temp_file("", ".hms");
    save_sketch(bank, path1, hdr);
    save_sketch(bank, path2, hdr);

    HyperSketchHeader h1, h2;
    PackedSketch s1 = load_packed_sketch(path1, h1);
    PackedSketch s2 = load_packed_sketch(path2, h2);
    remove_file(path1);
    remove_file(path2);

    double sim = HashedGraphEncoder::fast_compare_packed_single(s1, s2);
    REQUIRE(sim == Approx(1.0));
}
