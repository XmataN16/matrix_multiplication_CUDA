#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>

void init_matrix(int N, int M, int K, double* A, double* B, double* C);
void print_matrix(int N, int M, double* matrix);

#define BLOCK_SIZE 32

__global__ void blas_dgemmCUDAv1(const double* A, const double* B, double* C, int N, int M, int K)
{
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    double sum = 0.0;

    if (row < N && col < K) 
    {
        for (int i = 0; i < M; ++i) 
        {
            sum += A[row * M + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

void run_dgemmCUDAv1(const double* d_A, const double* d_B, double* d_C, int N, int M, int K)
{
    /* Set size block (BLOCK_SIZE x BLOCK_SIZE) */
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    /* Each block thread a part of the matrix with the size BLOCK_SIZE x BLOCK_SIZE */
    dim3 dimGrid((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* Run CUDA-core */
    blas_dgemmCUDAv1 << <dimGrid, dimBlock >> > (d_A, d_B, d_C, N, M, K);
}

__global__ void blas_dgemmCUDAv2(const double* A, const double* B, double* C, int N, int M, int K) {
    // Размер блоков
    __shared__ double tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double tileB[BLOCK_SIZE][BLOCK_SIZE];

    // Индексы строки и столбца для матрицы C
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    double sum = 0.0;

    // Обход всех "подблоков" A и B по горизонтали и вертикали соответственно
    for (int i = 0; i < (M + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) 
    {
        // Загрузка элементов в shared memory
        if (row < N && i * BLOCK_SIZE + threadIdx.x < M) 
        {
            tileA[threadIdx.y][threadIdx.x] = A[row * M + i * BLOCK_SIZE + threadIdx.x];
        }
        else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < K && i * BLOCK_SIZE + threadIdx.y < M) 
        {
            tileB[threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE + threadIdx.y) * K + col];
        }
        else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        // Умножение текущих подблоков A и B
        for (int j = 0; j < BLOCK_SIZE; ++j) 
        {
            sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }

        __syncthreads();
    }

    // Запись результата в глобальную память
    if (row < N && col < K) {
        C[row * K + col] = sum;
    }
}

void run_dgemmCUDAv2(const double* d_A, const double* d_B, double* d_C, int N, int M, int K)
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    blas_dgemmCUDAv2 << <dimGrid, dimBlock >> > (d_A, d_B, d_C, N, M, K);
}

void transpose_matrix(const double* B, double* B_transposed, int M, int K) 
{
    for (int i = 0; i < M; ++i) 
    {
        for (int j = 0; j < K; ++j) 
        {
            B_transposed[j * M + i] = B[i * K + j];
        }
    }
}

int main(int argc, char** argv)
{
    /* Declaration variables of matrix sizes */
    int N, M, K;

    /* Checking the number of command line arguments */
    if (argc == 2)
    {
        /* Initialization of matrix sizes */
        N = M = K = atoi(argv[0]);
    }
    else if (argc == 1)
    {
        N = M = K = 4096;
    }

    /* Allocation of memory for arrays A, B and result matrix C (host)*/
    double* A = (double*)malloc(N * M * sizeof(double));
    double* B = (double*)malloc(M * K * sizeof(double));
    double* C = (double*)malloc(N * K * sizeof(double));

    double* B_transposed = (double*)malloc(M * K * sizeof(double));
    transpose_matrix(B, B_transposed, M, K);

    if (!A || !B || !C)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    init_matrix(N, M, K, A, B, C);

    /* Allocation of memory for arrays A, B and result matrix C (device)*/
    double* d_A, * d_B, * d_C;

    cudaMalloc((void**)&d_A, N * M * sizeof(double));
    cudaMalloc((void**)&d_B, M * K * sizeof(double));
    cudaMalloc((void**)&d_C, N * K * sizeof(double));

    /* Copy arrays A, B and C out host memory in device memory*/
    cudaMemcpy(d_A, A, N * M * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_transposed, M * K * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    run_dgemmCUDAv2(d_A, d_B, d_C, N, M, K);

    cudaEventRecord(stop);

    cudaMemcpy(C, d_C, N * K * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Matrix multiplication took %f milliseconds.\n", milliseconds);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //print_matrix(N, K, C);

    /* Free allocated memory */
    free(A);
    free(B);
    free(C);

    return 0;
}

/* Function for displaying the matrix to the console */
void print_matrix(int N, int M, double* matrix)
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            printf("%f\t", matrix[i * N + j]);
        }
        printf("\n");
    }
}

/* The initialization function of matrices A, B and C */
void init_matrix(int N, int M, int K, double* A, double* B, double* C)
{
    int i, j;

    /* Initialization of matrix A (N x M) */
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            A[i * M + j] = 1.0;
        }
    }

    /* Initialization of matrix B (M x K) */
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < K; j++)
        {
            B[i * K + j] = 1.0;
        }
    }

    /* Initialization of matrix C (N x K) */
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < K; j++)
        {
            C[i * K + j] = 0.0;
        }
    }
}

