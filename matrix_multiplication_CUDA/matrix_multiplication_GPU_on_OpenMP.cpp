#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <omp.h>

void init_matrix(int N, int M, int K, double* A, double* B, double* C);
void print_matrix(int N, int M, double* matrix);
void blas_dgemmGPU_OpenMP(int N, int M, int K, double* A, double* B, double* C);

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
        N = M = K = 2000;
    }

    /* Allocation of memory for arrays A, B and result matrix C */
    double* A = (double*)malloc(N * M * sizeof(double));
    double* B = (double*)malloc(M * K * sizeof(double));
    double* C = (double*)malloc(N * K * sizeof(double));

    if (!A || !B || !C)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    init_matrix(N, M, K, A, B, C);

    /* The start of the execution time */
    double start_time = omp_get_wtime();

    blas_dgemmGPU_OpenMP(N, M, K, A, B, C);

    /* The end of the execution time */
    double end_time = omp_get_wtime();

    //print_matrix(N, M, C);

    printf("Execution time: %f seconds\n", end_time - start_time);

    /* Free allocated memory */
    free(A);
    free(B);
    free(C);

    return 0;
}

/* Own implementation of the matrix multiplication function on GPU */
void blas_dgemmGPU_OpenMP(int N, int M, int K, double* A, double* B, double* C) 
{
    /* Declare arrays "map(to/from)" for broadcast to GPU */
    #pragma omp target data map(to: A[0:N*M], B[0:M*K]) map(from: C[0:N*K])
    {
        /* We distribute calculations into teams and streams */
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < N; i++) 
        {
            for (int j = 0; j < K; j++) 
            {
                double sum = 0.0;
                for (int k = 0; k < M; k++) 
                {
                    sum += A[i * M + k] * B[k * K + j];
                }
                C[i * K + j] = sum;
            }
        }
    }
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

