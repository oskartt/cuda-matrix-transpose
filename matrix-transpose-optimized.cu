// Standard input/output library, used for printf
#include <stdio.h>

// Standard library, used for rand and srand
#include <stdlib.h>

// Math library, used for fabs
#include <math.h>

// Time library, used for CPU timing
#include <time.h>

// CUDA runtime library
#include <cuda_runtime.h>

// Number of matrix rows
#define ROWS 2048

// Number of matrix columns
#define COLS 1024

// Tile size used for the optimized GPU transpose
#define TILE 16

// CPU transpose function
void cpuTranspose(float *input, float *output, int rows, int cols)
{
    // Loop through each row
    for (int r = 0; r < rows; r++)

        // Loop through each column
        for (int c = 0; c < cols; c++)

            // Write each value into its transposed position
            output[c * rows + r] = input[r * cols + c];
}

// Basic GPU transpose kernel
__global__ void naiveTranspose(float *input, float *output, int rows, int cols)
{
    // Calculate column index for this GPU thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate row index for this GPU thread
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check that the thread is inside matrix bounds
    if (x < cols && y < rows)

        // Store the value in the transposed output position
        output[x * rows + y] = input[y * cols + x];
}

// Optimized tiled GPU transpose kernel
__global__ void tiledTranspose(float *input, float *output, int rows, int cols)
{
    // Shared memory tile, with +1 to reduce memory bank conflicts
    __shared__ float tile[TILE][TILE + 1];

    // Calculate original column index
    int x = blockIdx.x * TILE + threadIdx.x;

    // Calculate original row index
    int y = blockIdx.y * TILE + threadIdx.y;

    // Load data from global memory into shared memory
    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];

    // Wait until all threads finish loading the tile
    __syncthreads();

    // Calculate transposed row position
    x = blockIdx.y * TILE + threadIdx.x;

    // Calculate transposed column position
    y = blockIdx.x * TILE + threadIdx.y;

    // Write transposed tile from shared memory to global memory
    if (x < rows && y < cols)
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}

// Function to compare two arrays
bool verify(float *a, float *b, int size)
{
    // Loop through every value
    for (int i = 0; i < size; i++)

        // If values are different by more than a tiny amount, return false
        if (fabs(a[i] - b[i]) > 1e-5)
            return false;

    // If all values match, return true
    return true;
}

// Unused placeholder function
float timeKernel(void (*dummy)())
{
    // Returns 0 because this function is not used
    return 0.0f;
}

// Main program
int main()
{
    // Total number of matrix elements
    int total = ROWS * COLS;

    // Total memory size in bytes
    size_t size = total * sizeof(float);

    // Declare pointers for input and output arrays
    float *input, *cpuOut, *naiveOut, *tiledOut;

    // Allocate unified memory for input matrix
    cudaMallocManaged(&input, size);

    // Allocate unified memory for CPU output
    cudaMallocManaged(&cpuOut, size);

    // Allocate unified memory for naive GPU output
    cudaMallocManaged(&naiveOut, size);

    // Allocate unified memory for tiled GPU output
    cudaMallocManaged(&tiledOut, size);

    // Set random seed
    srand(0);

    // Fill input matrix with random values
    for (int i = 0; i < total; i++)
        input[i] = (float)(rand() % 100);

    // Start CPU timer
    clock_t cpuStart = clock();

    // Run CPU transpose
    cpuTranspose(input, cpuOut, ROWS, COLS);

    // Stop CPU timer
    clock_t cpuEnd = clock();

    // Convert CPU time to milliseconds
    double cpuTime = 1000.0 * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;

    // Create a 16x16 thread block
    dim3 threads(TILE, TILE);

    // Calculate number of blocks needed
    dim3 blocks((COLS + TILE - 1) / TILE, (ROWS + TILE - 1) / TILE);

    // Declare CUDA timing events
    cudaEvent_t start, stop;

    // Create start event
    cudaEventCreate(&start);

    // Create stop event
    cudaEventCreate(&stop);

    // Start timing naive GPU kernel
    cudaEventRecord(start);

    // Run naive GPU transpose
    naiveTranspose<<<blocks, threads>>>(input, naiveOut, ROWS, COLS);

    // Stop timing naive GPU kernel
    cudaEventRecord(stop);

    // Wait for naive GPU kernel to finish
    cudaEventSynchronize(stop);

    // Variable for naive GPU time
    float naiveTime;

    // Calculate naive GPU time
    cudaEventElapsedTime(&naiveTime, start, stop);

    // Start timing tiled GPU kernel
    cudaEventRecord(start);

    // Run tiled GPU transpose
    tiledTranspose<<<blocks, threads>>>(input, tiledOut, ROWS, COLS);

    // Stop timing tiled GPU kernel
    cudaEventRecord(stop);

    // Wait for tiled GPU kernel to finish
    cudaEventSynchronize(stop);

    // Variable for tiled GPU time
    float tiledTime;

    // Calculate tiled GPU time
    cudaEventElapsedTime(&tiledTime, start, stop);

    // Print matrix dimensions
    printf("Matrix size: %d x %d\n", ROWS, COLS);

    // Print CPU time
    printf("CPU Time: %.3f ms\n", cpuTime);

    // Print naive GPU time
    printf("Naive GPU Time: %.3f ms\n", naiveTime);

    // Print tiled GPU time
    printf("Tiled GPU Time: %.3f ms\n", tiledTime);

    // Check and print naive GPU correctness
    printf("Naive Verification: %s\n", verify(cpuOut, naiveOut, total) ? "PASS" : "FAIL");

    // Check and print tiled GPU correctness
    printf("Tiled Verification: %s\n", verify(cpuOut, tiledOut, total) ? "PASS" : "FAIL");

    // Free input memory
    cudaFree(input);

    // Free CPU output memory
    cudaFree(cpuOut);

    // Free naive GPU output memory
    cudaFree(naiveOut);

    // Free tiled GPU output memory
    cudaFree(tiledOut);

    // End program successfully
    return 0;
}
