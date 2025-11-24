# hypermash
Graph-based Genome Sketching via Hyperdimensional Computing

## Overview

Hypermash is a bioinformatics tool designed to compute the similarity between genomes. Unlike traditional methods (like Mash) that compare the set of k-mers (a "bag of words" approach), Hypermash sketches the _de Bruijn_ Graph structure of the genome.

It encodes the transitions between k-mers (syntax/structure) rather than just their presence. It utilizes Hyperdimensional Computing (HDC) to compress this vast graph structure into a fixed-size, low-memory vector representation called a Memory Bank.

## Core Data Structures

### The Hypervector

A vector of dimensions _D_ (default 10,000), where every element is randomly initialized to either +1 or -1. Any two randomly generated hypervectors are pseudo-orthogonal, meaning their cosine similarity is approximately 0.

### The Memory Bank

A matrix of size _M x D_. It acts as a hash table where the "keys" are k-mers and the "values" are the sum (superposition) of all k-mers that follow the key. Because _M_ (default 4096) is potentially much smaller than the total number of unique k-mers in a genome, multiple k-mers map to the same row (hash collisions). Hyperdimensional Computing math allows us to tolerate this noise up to a point.

### The Sketching Algorithm

The sketching process transforms a linear DNA sequence into a Memory Bank.

1. Input Processing
  - Reading: the genome is read from a FASTA file;
  - Canonical k-mers: for every window of size _k_, the tool computes the canonical k-mer. This is the lexicographically smaller sequence between the k-mer and its reverse complement. This ensures that a sequence and its reverse complement map to the same graph structure.

2. Graph Encoding (the `source > target` logic)
  - Window: `source` (k-mer at _i_) > `target` (k-mer at _i+1_);
  - Hashing: the `source` k-mer is hashed to an index _idx_ (0..._M_-1);
  - Generation: a deterministic unique Hypervector is generated for the `target` k-mer (_HV-target_);
  - Bundling: _HV-target_ is added (element-wise addition) to the row at `MemoryBank[idx]`;
  - Result: row _i_ in the bank becomes a superposition of all k-mers that ever followed any source k-mer that hashed to index _i_.

3. Parallelization (Map-Reduce Pattern)
  - To maximize speed, the encoding uses a lock-free Map-Reduce approach;
  - Map Phase: the genome is split into _N_ chunks (where _N_ is the thread count). Each thread creates its own private Memory Bank and sketches its chunk;
  - Reduce Phase: once all threads finish, a parallel reduction merges the _N_ private banks into one final global Memory Bank by summing them element-wise.

### Error Mitigation and Fidelity

Because the Memory Bank is "lossy" (due to hash collisions), the tool includes mechanisms to measure and repair signal loss.

1. Fidelity Calculation
  - This measures how well the sketch "remembers" the genome;
  - The tool re-scans the genome edges (`source > target`);
  - It compares the similarity between the Source bucket in the bank and the true Target hypervector;
  - Fidelity Score: The average normalized similarity across all edges (0.0 to 1.0);
  - Error Rate: 1.0 - Fidelity.

2. Adaptive Retraining
  - If enabled, the tool attempts to "boost" weak signals (Gradient Descent-style logic);
  - Dynamic Thresholding: the tool calculates a specific threshold for every bucket in the memory bank;
    - As a bucket accumulates vectors, its magnitude grows, and expected similarity drops;
    - Formula: `Threshold_i = (sqrt(D) / |Magnitude_i|) x 0.75`;
  - Detection: if an edge's similarity is below this dynamic threshold, it is flagged as a "weak link";
  - Correction: the target vector is added again to the source bucket. This increases the signal-to-noise ratio for that specific transition;
  - Early Stopping: after each iteration, the Error Rate is recalculated. If the error rate does not improve (or gets worse), the loop breaks immediately.

### Comparison Algorithm

To compare two genomes (Sketch A and Sketch B), the tool compares their Memory Banks row-by row.

1. Requirement: sketches must share the same _k_, _D_, and _M_ parameters;
2. Row Comparison: for every row _i_:
  - Compute cosine similarity between `BankA[i]` and `BankB[i]`;
3. Aggregation: the final similarity is the average of all row similarities.

Interpretation:
- __1.0__: the graph structures are identical;
- __0.0__: the graph structures are completely disjoint.

### Implementation Specs

- Language: C++17
- Threading: uses `std::thread`, `std::vector` for thread management, and `std::atomic` for thread-safe counters in the reduction phase;
- Storage: the final file is heavily compressed using Bit-Packing:
  - Since Hypervectors are sums of +1 and -1, final values are integers;
  - The file stores the sign bit of these integers packed into bytes (_D_ dimensions, _D_/8 bytes per row);
  - This achieves a 32x compression ratio compared to storing raw 32-bit integers.