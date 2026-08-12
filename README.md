# hypersketch

Graph-Based Genome Sketching via Hyperdimensional Computing

## Overview

HyperSketch is a high-performance bioinformatics engine designed to compute evolutionary distances between genomes.

Unlike traditional MinHash approaches that compare unstructured sets of k-mers (a "bag-of-words"), HyperSketch sketches the topological structure of the _de Bruijn_ sequence graph. By encoding the transitions (edges) between adjacent k-mers, HyperSketch captures genomic syntax and structural variation. It utilizes Hyperdimensional Computing (HDC) to compress this vast graphical topology into a fixed-size, heavily binarized matrix representation called a Memory Bank.

The underlying C++ engine is aggressively optimized for bare-metal hardware performance, utilizing Zero-Copy OS Memory Mapping, Data-Oriented Design (SoA), SIMD AVX2 vectorization, L1-cache locking, and Statistical Adaptive Error Mitigation.

## Core Data Structures

### The Hypervector

A high-dimensional vector (default $D = 10,000$), where every element is deterministically generated as $+1$ or $-1$ using a Weyl-sequence PRNG. In high-dimensional space, any two randomly generated hypervectors are pseudo-orthogonal (their cosine similarity approaches 0).

### The Memory Bank

A densely packed matrix of size $M \times D$ (default $M = 4096$). It functions as a graphical hash table where:

- __Keys:__ The source k-mer.

- __Values:__ The bundled superposition (element-wise addition) of all target k-mers that follow the source in the genome.

Because $M$ is significantly smaller than the total k-mer vocabulary of a genome, collisions occur. The mathematics of Hyperdimensional Computing allow the system to tolerate this overlapping noise while preserving the dominant topological signals.

## The Sketching Algorithm

The encoding pipeline transforms a FASTA sequence into a Memory Bank using a 5-phase, lock-free, cache-optimized architecture:

### 1. Zero-Copy I/O & Tokenization

The genome is never loaded as a standard string. Using OS-level mmap, the physical SSD blocks are mapped directly into the CPU's virtual memory. A pointer-bumping parser tokenizes ASCII characters into 0-3 integers on the fly. Ambiguous bases (e.g., N) are flagged to instantly sever structural edges, preventing false connections across scaffold boundaries.

### 2. Edge Extraction & SWAR Hashing

The genome is chunked across available CPU threads. For every valid edge (source $\rightarrow$ target), the algorithm:

- Computes the canonical orientation to ensure strandedness independence.

- Uses SWAR (SIMD Within A Register) to hash the k-mers without slow array lookups.

- Extracts the target hash and its destination bucket.

### 3. Data-Oriented Counting Sort (Scatter)

To prevent RAM cache thrashing, HyperSketch utilizes Data-Oriented Design (Structure of Arrays). Thread-local edges are mapped using flattened Prefix Sums. An $O(N)$ parallel Counting Sort groups all edges destined for the same memory bucket into perfectly contiguous memory blocks.

### 4. L1-Cache Locked Vector Generation

With all edges perfectly grouped by bucket, threads pull a single row of the Memory Bank into the CPU's 40KB L1 Cache.
Instead of doing sequential bit-math, an 8KB Global Lookup Table (LUT) translates 64-bit random seeds into $+1/-1$ arrays instantly. The compiler fuses these lookups into AVX2 VPADDD vector instructions, generating and bundling 8 dimensions simultaneously per clock cycle before writing the finalized row back to main memory.

### 5. Statistical Adaptive Error Mitigation (Retraining)

To mitigate hash-collision interference without inflating the Memory Bank size, HyperSketch employs a dynamic, data-driven error mitigation algorithm.

- For each memory bucket, the engine calculates the true Mean ($\mu$) and Standard Deviation ($\sigma$) of the constituent edge similarities.

- Any topological edge whose similarity degrades more than one standard deviation below the local bucket's mean ($< \mu - 1\sigma$) is identified as structurally compromised and algorithmically reinforced.

- This purely statistical approach dynamically prevents runaway signal oscillation. An instant 0-cost rollback mechanism guarantees that the engine stops early if the overall error rate fails to improve, ensuring perfect model stability.

## Comparison Algorithm & Distance Estimation

To compare two genomes, HyperSketch executes a parallel, lock-free matrix boarding process:

1. __Hardware POPCOUNT:__ Because the final sketches are bit-packed, the comparison uses hardware-level __builtin_popcountll and bitwise XOR gates to compute the Hamming distance between aligned memory buckets at blazing speeds.

2. __Algebraic Translation:__ Hamming distance is algebraically converted to Binarized Cosine Similarity: Sim = 1 - (2 * Hamming) / D.

3. __Grothendieck's Identity:__ The binarized similarity is mathematically un-distorted using Grothendieck's identity to approximate the Jaccard Index: Jaccard = sin(Sim * PI / 2).

4. __Mash Distance:__ Finally, the standard MinHash mutation probability formula is applied to estimate the true evolutionary distance: d = (-1 / k) * ln(Jaccard).

## Usage

```text
HyperSketch (v1.0)

Usage: hypersketch <command> [options] <input_file>

Commands:
  sketch      Create a compact sketch from a single FASTA file.
  dist        Calculate similarity between two sketch files.
  info        Display metadata and information about a sketch file.
  version     Print the software version.
  help        Show this help message.

Options:
  -k <int>    K-mer size (max 32, default: 9)
  -d <int>    Hypervector dimensions (must be multiple of 8, default: 10000)
  -m <int>    Memory bank size (default: 4096)
  -t <int>    Number of threads (default: all available cores)
  -r <int>    Retraining iterations for error mitigation (default: 0)
  -o <file>   Output file prefix (for sketch command)
  -l <file>   File containing a list of sketch paths, one per line (for dist command)
```

## Implementation Specs

- __Language:__ C++17

- __Compiler Requirements:__ Requires modern GCC/Clang with -march=native to unlock hardware-specific AVX2, BMI2, and POPCNT instructions.

- __I/O:__ Fully asynchronous batch processing via Matrix Job Generation (bypasses ARG_MAX terminal limits using the -l list flag).

- __Storage:__ Final .hms files are aggressively bit-packed (8 dimensions per byte) using 8-way unrolled OR-gates, achieving a 32x compression ratio over raw integers.

## Credits

If you use HyperSketch in your work, please cite:

> _Manuscript in preparation_