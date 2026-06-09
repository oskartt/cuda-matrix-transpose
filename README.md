# CUDA Matrix Transposition

## Overview

The goal of this assignment was to implement and optimize matrix transposition using NVIDIA CUDA. Matrix transposition swaps rows and columns of a matrix. The project compares CPU and GPU performance and explores optimization techniques such as Unified Memory, coalesced memory access, and shared memory tiling.

Matrix size used for testing:

* Rows: 2048
* Columns: 1024

---

## Part 1 – Basic Matrix Transposition

### CPU Implementation

A sequential CPU version was implemented using nested loops. Each element at position `(row, column)` was copied to position `(column, row)` in the output matrix.

### GPU Implementation

A basic CUDA kernel was created where each thread processes one matrix element and writes it to its transposed position.

### Results

| Implementation | Time (ms) |
| -------------- | --------- |
| CPU            | 78.0      |
| Naive GPU      | 42.5      |

Verification result: PASS

The GPU implementation was faster than the CPU implementation while producing identical results.

---

## Part 2 – Optimized Matrix Transposition

### Unified Memory

Unified Memory was implemented using `cudaMallocManaged()`. This simplified memory management by allowing the CPU and GPU to access the same memory space.

### Shared Memory Tiling

A tiled transpose kernel was implemented using shared memory. Shared memory reduces global memory accesses and improves memory coalescing.

### Block Size Analysis

Different tile sizes were tested:

| Tile Size | Naive GPU (ms) | Tiled GPU (ms) |
| --------- | -------------- | -------------- |
| 8x8       | 34.112         | 0.205          |
| 16x16     | 34.044         | 0.208          |
| 32x32     | 36.846         | 0.218          |

Verification result: PASS for all tests.

### Observations

The tiled implementation significantly outperformed the naive implementation. The optimized version reduced execution time from approximately 34 ms to around 0.2 ms.

The 8x8 and 16x16 tile sizes provided the best performance, while the 32x32 configuration was slightly slower.

---

## Part 3 – Analysis

### Performance Differences

The naive GPU version performs many non-coalesced memory accesses, which limits memory bandwidth efficiency.

The tiled implementation uses shared memory and coalesced accesses, reducing memory traffic and improving performance dramatically.

### Impact of Matrix Size

As matrix size increases, GPU acceleration becomes more beneficial because more threads can execute in parallel. For very small matrices, kernel launch overhead may reduce the performance advantage.

### Possible Future Optimizations

* Use asynchronous memory transfers.
* Explore CUDA streams.
* Test larger matrices.
* Use advanced transpose kernels with loop unrolling.
* Profile the application using Nsight Compute.

---

## Conclusion

The project successfully implemented CPU and GPU matrix transposition. The optimized tiled implementation achieved a significant speedup compared to both the CPU and naive GPU versions. Shared memory and coalesced memory access were the most important factors contributing to the performance improvement.
