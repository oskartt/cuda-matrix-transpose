#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define ROWS 2048
#define COLS 1024

void cpuTranspose(float *input, float *output, int rows, int cols)
{
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            output[c * rows + r] = input[r * cols + c];
}

__global__ void gpuTranspose(float *input, float *output, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
        output[x * rows + y] = input[y * cols + x];
}

bool verify(float *a, float *b, int size)
{
    for (int i = 0; i < size; i++)
        if (fabs(a[i] - b[i]) > 1e-5)
            return false;
    return true;
}

int main()
{
    size_t inputSize = ROWS * COLS * sizeof(float);
    size_t outputSize = ROWS * COLS * sizeof(float);

    float *input;
    float *cpuOut;
    float *gpuOut;

    cudaMallocManaged(&input, inputSize);
    cudaMallocManaged(&cpuOut, outputSize);
    cudaMallocManaged(&gpuOut, outputSize);

    srand(0);

    for (int i = 0; i < ROWS * COLS; i++)
        input[i] = (float)(rand() % 100);

    clock_t cpuStart = clock();
    cpuTranspose(input, cpuOut, ROWS, COLS);
    clock_t cpuEnd = clock();

    double cpuTime =
        1000.0 * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;

    dim3 threads(16, 16);
    dim3 blocks(
        (COLS + threads.x - 1) / threads.x,
        (ROWS + threads.y - 1) / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    gpuTranspose<<<blocks, threads>>>(input, gpuOut, ROWS, COLS);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);

    bool correct = verify(cpuOut, gpuOut, ROWS * COLS);

    printf("CPU Time: %.3f ms\n", cpuTime);
    printf("GPU Time: %.3f ms\n", gpuTime);
    printf("Verification: %s\n", correct ? "PASS" : "FAIL");

    cudaFree(input);
    cudaFree(cpuOut);
    cudaFree(gpuOut);

    return 0;
}