// Standard input/output library (for printf)
#include <stdio.h>

// Standard library (for rand, srand)
#include <stdlib.h>

// Math library (for fabs)
#include <math.h>

// Time library (for CPU timing)
#include <time.h>

// CUDA runtime library
#include <cuda_runtime.h>

// Number of rows in the matrix
#define ROWS 2048

// Number of columns in the matrix
#define COLS 1024

// CPU function to transpose a matrix
void cpuTranspose(float *input, float *output, int rows, int cols)
{
    // Loop through every row
    for (int r = 0; r < rows; r++)

        // Loop through every column
        for (int c = 0; c < cols; c++)

            // Swap row and column positions
            output[c * rows + r] = input[r * cols + c];
}

// GPU kernel function for matrix transpose
__global__ void gpuTranspose(float *input, float *output, int rows, int cols)
{
    // Calculate column index handled by this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate row index handled by this thread
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Make sure thread is inside matrix bounds
    if (x < cols && y < rows)

        // Copy value to transposed position
        output[x * rows + y] = input[y * cols + x];
}

// Function to compare CPU and GPU results
bool verify(float *a, float *b, int size)
{
    // Check every element
    for (int i = 0; i < size; i++)

        // If difference is too large, fail verification
        if (fabs(a[i] - b[i]) > 1e-5)
            return false;

    // Everything matched
    return true;
}

// Main program
int main()
{
    // Calculate size of input matrix in bytes
    size_t inputSize = ROWS * COLS * sizeof(float);

    // Calculate size of output matrix in bytes
    size_t outputSize = ROWS * COLS * sizeof(float);

    // Pointer for original matrix
    float *input;

    // Pointer for CPU result
    float *cpuOut;

    // Pointer for GPU result
    float *gpuOut;

    // Allocate unified memory for input matrix
    cudaMallocManaged(&input, inputSize);

    // Allocate unified memory for CPU output
    cudaMallocManaged(&cpuOut, outputSize);

    // Allocate unified memory for GPU output
    cudaMallocManaged(&gpuOut, outputSize);

    // Set random seed for reproducible results
    srand(0);

    // Fill input matrix with random numbers
    for (int i = 0; i < ROWS * COLS; i++)
        input[i] = (float)(rand() % 100);

    // Start CPU timer
    clock_t cpuStart = clock();

    // Run CPU transpose
    cpuTranspose(input, cpuOut, ROWS, COLS);

    // Stop CPU timer
    clock_t cpuEnd = clock();

    // Convert CPU execution time to milliseconds
    double cpuTime =
        1000.0 * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;

    // Create a block of 16x16 threads
    dim3 threads(16, 16);

    // Calculate how many blocks are needed
    dim3 blocks(
        (COLS + threads.x - 1) / threads.x,
        (ROWS + threads.y - 1) / threads.y);

    // Create CUDA timing events
    cudaEvent_t start, stop;

    // Allocate start event
    cudaEventCreate(&start);

    // Allocate stop event
    cudaEventCreate(&stop);

    // Record start time on GPU
    cudaEventRecord(start);

    // Launch transpose kernel on GPU
    gpuTranspose<<<blocks, threads>>>(input, gpuOut, ROWS, COLS);

    // Record stop time
    cudaEventRecord(stop);

    // Wait until GPU finishes
    cudaEventSynchronize(stop);

    // Variable to store GPU execution time
    float gpuTime;

    // Calculate elapsed GPU time in milliseconds
    cudaEventElapsedTime(&gpuTime, start, stop);

    // Compare CPU and GPU results
    bool correct = verify(cpuOut, gpuOut, ROWS * COLS);

    // Print CPU execution time
    printf("CPU Time: %.3f ms\n", cpuTime);

    // Print GPU execution time
    printf("GPU Time: %.3f ms\n", gpuTime);

    // Print verification result
    printf("Verification: %s\n", correct ? "PASS" : "FAIL");

    // Free input matrix memory
    cudaFree(input);

    // Free CPU output memory
    cudaFree(cpuOut);

    // Free GPU output memory
    cudaFree(gpuOut);

    // End program successfully
    return 0;
}
