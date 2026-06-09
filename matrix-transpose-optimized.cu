#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define ROWS 2048
#define COLS 1024
#define TILE 16

void cpuTranspose(float *input, float *output, int rows, int cols)
{
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            output[c * rows + r] = input[r * cols + c];
}

__global__ void naiveTranspose(float *input, float *output, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
        output[x * rows + y] = input[y * cols + x];
}

__global__ void tiledTranspose(float *input, float *output, int rows, int cols)
{
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    if (x < rows && y < cols)
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}

bool verify(float *a, float *b, int size)
{
    for (int i = 0; i < size; i++)
        if (fabs(a[i] - b[i]) > 1e-5)
            return false;
    return true;
}

float timeKernel(void (*dummy)())
{
    return 0.0f;
}

int main()
{
    int total = ROWS * COLS;
    size_t size = total * sizeof(float);

    float *input, *cpuOut, *naiveOut, *tiledOut;

    cudaMallocManaged(&input, size);
    cudaMallocManaged(&cpuOut, size);
    cudaMallocManaged(&naiveOut, size);
    cudaMallocManaged(&tiledOut, size);

    srand(0);
    for (int i = 0; i < total; i++)
        input[i] = (float)(rand() % 100);

    clock_t cpuStart = clock();
    cpuTranspose(input, cpuOut, ROWS, COLS);
    clock_t cpuEnd = clock();
    double cpuTime = 1000.0 * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;

    dim3 threads(TILE, TILE);
    dim3 blocks((COLS + TILE - 1) / TILE, (ROWS + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    naiveTranspose<<<blocks, threads>>>(input, naiveOut, ROWS, COLS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float naiveTime;
    cudaEventElapsedTime(&naiveTime, start, stop);

    cudaEventRecord(start);
    tiledTranspose<<<blocks, threads>>>(input, tiledOut, ROWS, COLS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tiledTime;
    cudaEventElapsedTime(&tiledTime, start, stop);

    printf("Matrix size: %d x %d\n", ROWS, COLS);
    printf("CPU Time: %.3f ms\n", cpuTime);
    printf("Naive GPU Time: %.3f ms\n", naiveTime);
    printf("Tiled GPU Time: %.3f ms\n", tiledTime);
    printf("Naive Verification: %s\n", verify(cpuOut, naiveOut, total) ? "PASS" : "FAIL");
    printf("Tiled Verification: %s\n", verify(cpuOut, tiledOut, total) ? "PASS" : "FAIL");

    cudaFree(input);
    cudaFree(cpuOut);
    cudaFree(naiveOut);
    cudaFree(tiledOut);

    return 0;
}