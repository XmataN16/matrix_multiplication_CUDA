#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <omp.h>
#include <stdio.h>
#include <cstdlib>

void init_matrix(int N, int M, int K, double* A, double* B, double* C);
void print_matrix(int N, int M, double* matrix);

#define BLOCK_SIZE 32

void transfer_data_to_GPU(int method, double* A, double* B, double* C, double* &d_A, double* &d_B, double* &d_C, int N, int M, int K)
{
    switch (method)
    {
    case 0: // Standard allocate
        cudaMalloc((void**)&d_A, N * M * sizeof(double));
        cudaMalloc((void**)&d_B, M * K * sizeof(double));
        cudaMalloc((void**)&d_C, N * K * sizeof(double));

        cudaMemcpy(d_A, A, N * M * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, M * K * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, C, N * K * sizeof(double), cudaMemcpyHostToDevice);
        break;

    case 1: // Pinned memory
        double* pinned_A, * pinned_B, * pinned_C;
        cudaHostAlloc((void**)&pinned_A, N * M * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void**)&pinned_B, M * K * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void**)&pinned_C, N * K * sizeof(double), cudaHostAllocDefault);

        memcpy(pinned_A, A, N * M * sizeof(double));
        memcpy(pinned_B, B, M * K * sizeof(double));
        memcpy(pinned_C, C, N * K * sizeof(double));

        cudaMalloc((void**)&d_A, N * M * sizeof(double));
        cudaMalloc((void**)&d_B, M * K * sizeof(double));
        cudaMalloc((void**)&d_C, N * K * sizeof(double));

        cudaMemcpy(d_A, pinned_A, N * M * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, pinned_B, M * K * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, pinned_C, N * K * sizeof(double), cudaMemcpyHostToDevice);
        break;

    case 2: // Unified memory
        cudaMallocManaged((void**)&d_A, N * M * sizeof(double));
        cudaMallocManaged((void**)&d_B, M * K * sizeof(double));
        cudaMallocManaged((void**)&d_C, N * K * sizeof(double));

        /* Copy data HOST to Unified Memory */
        memcpy(d_A, A, N * M * sizeof(double));
        memcpy(d_B, B, M * K * sizeof(double));
        memcpy(d_C, C, N * K * sizeof(double));

        cudaDeviceSynchronize();
        break;

    case 3: // CUDA streams
        cudaStream_t stream1, stream2;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);

        cudaMalloc((void**)&d_A, N * M * sizeof(double));
        cudaMalloc((void**)&d_B, M * K * sizeof(double));
        cudaMalloc((void**)&d_C, N * K * sizeof(double));

        cudaMemcpyAsync(d_A, A, N * M * sizeof(double), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_B, B, M * K * sizeof(double), cudaMemcpyHostToDevice, stream2);

        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        break;

    default:
        fprintf(stderr, "Unknown transfer method.\n");
        exit(1);
    }
}

template <typename Func, typename... Args>
void run_dgemmCUDA(const char* func_name, Func f, const double* d_A, const double* d_B, double* d_C, int N, int M, int K)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Start measuring time */
    cudaEventRecord(start);

    /* Set size block (BLOCK_SIZE x BLOCK_SIZE) */
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    /* Each block thread a part of the matrix with the size BLOCK_SIZE x BLOCK_SIZE */
    dim3 dimGrid((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* Run CUDA-core */
    f << <dimGrid, dimBlock >> > (d_A, d_B, d_C, N, M, K);

    /* End measuring time */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    /* Calc time running */
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%s time elapsed = %f ms\n", func_name, milliseconds);

    // Очистка ресурсов событий
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void run_cublasDgemm(const double* d_A, const double* d_B, double* d_C, int N, int M, int K) 
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Start measuring time */
    cudaEventRecord(start);

    cublasHandle_t handle;

    /* Create context cuBLAS*/
    cublasCreate(&handle);

    /*Coeffs for operation (alpha * A * B + beta * C) */ 
    const double alpha = 1.0;
    const double beta = 0.0;

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, N, M, &alpha, d_B, K, d_A, M, &beta, d_C, K);                 

    /* Free context cuBLAS */
    cublasDestroy(handle);

    /* End measuring time */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    /* Calc time running */
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("cuBLAS v2 time elapsed = %f ms\n", milliseconds);

    // Очистка ресурсов событий
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

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

__global__ void blas_dgemmCUDAv2(const double* A, const double* B, double* C, int N, int M, int K) 
{
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
        else 
        {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < K && i * BLOCK_SIZE + threadIdx.y < M) 
        {
            tileB[threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE + threadIdx.y) * K + col];
        }
        else 
        {
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

__global__ void blas_dgemmCUDAv3(const double* A, const double* B, double* C, int N, int M, int K) 
{
    // Разделяемая память с выравниванием
    __shared__ double sharedA[BLOCK_SIZE][BLOCK_SIZE + 1]; // Добавляем +1 для устранения конфликтов
    __shared__ double sharedB[BLOCK_SIZE][BLOCK_SIZE + 1]; // Аналогично для матрицы B

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    double sum = 0.0;

    for (int t = 0; t < (M + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) 
    {
        // Загрузка данных в разделяемую память
        if (row < N && t * BLOCK_SIZE + threadIdx.x < M)
            sharedA[threadIdx.y][threadIdx.x] = A[row * M + t * BLOCK_SIZE + threadIdx.x];
        else
            sharedA[threadIdx.y][threadIdx.x] = 0.0;

        if (col < K && t * BLOCK_SIZE + threadIdx.y < M)
            sharedB[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * K + col];
        else
            sharedB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        // Вычисления
        for (int i = 0; i < BLOCK_SIZE; ++i) 
        {
            sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < K) 
    {
        C[row * K + col] = sum;
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
        N = M = K = 1000;
    }

    /* Allocation of memory for arrays A, B and result matrix C (host)*/
    double* A = (double*)malloc(N * M * sizeof(double));
    double* B = (double*)malloc(M * K * sizeof(double));
    double* C = (double*)malloc(N * K * sizeof(double));

    if (!A || !B || !C)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    init_matrix(N, M, K, A, B, C);

    /* Allocation of memory for arrays A, B and result matrix C (device)*/
    double* d_A, * d_B, * d_C;

    transfer_data_to_GPU(3, A, B, C, d_A, d_B, d_C, N, M, K);

    /* Run core use global memory */
    run_dgemmCUDA("Global memory kernel (v1)", blas_dgemmCUDAv1, d_A, d_B, d_C, N, M, K);
    cudaDeviceSynchronize();

    /* Run core use shared memory */
    run_dgemmCUDA("Shared memory kernel (v2)", blas_dgemmCUDAv2, d_A, d_B, d_C, N, M, K);
    cudaDeviceSynchronize();

    /* Run core use shared-padding memory */
    run_dgemmCUDA("Shared memory with padding (v3)", blas_dgemmCUDAv3, d_A, d_B, d_C, N, M, K);
    cudaDeviceSynchronize();

    /* Run core use cublas */
    run_cublasDgemm(d_A, d_B, d_C, N, M, K);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, N * K * sizeof(double), cudaMemcpyDeviceToHost);

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

