// 21140036 並列処理 期末課題
// レポート課題4 (3)
// CUDA言語による実装 (cuda_mat.c)
// (c) Takuo Yamaguchi

// compile: nvcc cuda_mat.cu -o cuda_mat

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define h2d cudaMemcpyHostToDevice
#define d2h cudaMemcpyDeviceToHost

// 要素積の計算を行う．Blockはdim3(N, N, 1), Threadはdim(1,1,1).
__global__ void adm(int *A, int *B, int *C)
{
    int idc = blockIdx.x * gridDim.x + blockIdx.y;
    C[idc] = A[idc] * B[idc];
}
// 行列積の計算を行う．Blockはdim3(N, N, 1), Threadはdim(N, 1, 1).
__global__ void cmb(int *A, int *B, int *C)
{
    // threadがA[i,k]*B[k,j]の計算を担う．
    int ida = blockIdx.x * gridDim.x + threadIdx.x;
    int idb = threadIdx.x * blockDim.x + blockIdx.y;
    int idc = blockIdx.x * gridDim.x + blockIdx.y;
    int comb = A[ida] * B[idb];
    atomicAdd(&C[idc], comb);
}
// 実行時間計測用の変数
cudaEvent_t start, end;

double t = 0; // time, msec
float t_time; // cudaEventElapsedTimeの受け皿
// 各条件の繰り返し数
const int rep = 5;
// 計算サイズN
int N[2] = {1ULL << 7, 1ULL << 10};
int main()
{
    cudaSetDevice(0);
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    int *A, *B, *C, *Ad, *Bd, *Cd;
    // 条件ごとにrep回実行し，実行時間の平均をとる．
    printf("要素積\n");
    for (int cond = 0; cond < 2; cond++)
    {
        t = 0;
        printf("\tCondition: L=%d\n", N[cond]);
        // malloc
        A = (int *)malloc(sizeof(int) * N[cond] * N[cond]);
        B = (int *)malloc(sizeof(int) * N[cond] * N[cond]);
        C = (int *)malloc(sizeof(int) * N[cond] * N[cond]);
        for (int i = 0; i < N[cond] * N[cond]; i++)
        {
            A[i] = 1;
            B[i] = 1;
            C[i] = 0;
        }
        cudaMalloc(&Ad, sizeof(int) * N[cond] * N[cond]);
        cudaMalloc(&Bd, sizeof(int) * N[cond] * N[cond]);
        cudaMalloc(&Cd, sizeof(int) * N[cond] * N[cond]);
        for (int trial = 0; trial < rep; trial++)
        {
            printf("\t\ttrial: %d, ", trial);
            //  cuda settings
            cudaMemcpy(Ad, A, sizeof(int) * N[cond] * N[cond], h2d);
            cudaMemcpy(Bd, B, sizeof(int) * N[cond] * N[cond], h2d);
            cudaMemcpy(Cd, C, sizeof(int) * N[cond] * N[cond], h2d);
            dim3 grid(N[cond], N[cond], 1);
            dim3 thread(1, 1, 1);
            // time mesurement
            cudaEventRecord(start, 0);
            adm<<<grid, thread>>>(Ad, Bd, Cd);
            cudaEventRecord(end, 0);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&t_time, start, end);
            printf("time=%lf[msec]\n", t_time);
            t += t_time / rep;
        }
        printf("\tavg time: %lf[msec]\n", t);
        // free mem
        cudaFree(Ad);
        cudaFree(Bd);
        cudaFree(Cd);
        free(A);
        free(B);
        free(C);
    }
    printf("行列積\n");
    for (int cond = 0; cond < 2; cond++)
    {
        t = 0;
        printf("\tCondition: L=%d\n", N[cond]);
        // malloc
        A = (int *)malloc(sizeof(int) * N[cond] * N[cond]);
        B = (int *)malloc(sizeof(int) * N[cond] * N[cond]);
        C = (int *)malloc(sizeof(int) * N[cond] * N[cond]);
        for (int i = 0; i < N[cond] * N[cond]; i++)
        {
            A[i] = 1;
            B[i] = 1;
            C[i] = 0;
        }
        cudaMalloc(&Ad, sizeof(int) * N[cond] * N[cond]);
        cudaMalloc(&Bd, sizeof(int) * N[cond] * N[cond]);
        cudaMalloc(&Cd, sizeof(int) * N[cond] * N[cond]);
        for (int trial = 0; trial < rep; trial++)
        {
            printf("\t\ttrial: %d, ", trial);
            // cuda settings
            cudaMemcpy(Ad, A, sizeof(int) * N[cond] * N[cond], h2d);
            cudaMemcpy(Bd, B, sizeof(int) * N[cond] * N[cond], h2d);
            cudaMemcpy(Cd, C, sizeof(int) * N[cond] * N[cond], h2d);
            dim3 grid(N[cond], N[cond], 1);
            dim3 thread(N[cond], 1, 1);
            // time mesurement
            cudaEventRecord(start, 0);
            cmb<<<grid, thread>>>(Ad, Bd, Cd);
            cudaEventRecord(end, 0);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&t_time, start, end);
            printf("time=%lf[msec]\n", t_time);
            t += t_time / rep;
        }
        printf("\tavg time: %lf[msec]\n", t);
        // free mem
        cudaFree(Ad);
        cudaFree(Bd);
        cudaFree(Cd);
        free(A);
        free(B);
        free(C);
    }
}
