// 21140036 並列処理 期末課題
// レポート課題3 (4)
// CUDA言語による実装 (cuda.cu)
// (c) Takuo Yamaguchi

// compile: nvcc cuda.cu -o cuda

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define h2d cudaMemcpyHostToDevice
#define d2h cudaMemcpyDeviceToHost
typedef unsigned long long ull;

// 計算を行う関数．
__global__ void calc_sum(ull *sum)
{
    ull i = blockIdx.x;
    ull j = blockIdx.y;
    // 計算，Atomicとし，正しく加算が行われるようにする．
    atomicAdd(sum, (i - j) * (i - j));
}

// 実行時間計測用の変数
cudaEvent_t start, end;

double t = 0; // time, msec
float t_time; // cudaEventElapsedTimeの受け皿
// 各条件の繰り返し数
const int rep = 5;
// 計算サイズL
ull L[3] = {1ULL << 7, 1ULL << 10, 1ULL << 13};
int main()
{
    cudaSetDevice(0);
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    ull result = 0;
    ull *result_d;
    cudaMalloc(&result_d, sizeof(ull));
    // 条件ごとにrep回実行し，実行時間の平均をとる．
    for (int cond = 0; cond < 3; cond++)
    {
        t = 0;
        printf("Condition: L=%llu\n", L[cond]);
        for (int trial = 0; trial < rep; trial++)
        {
            result = 0;
            printf("\ttrial: %d, ", trial);
            // cuda settings
            cudaMemcpy(result_d, &result, sizeof(ull), h2d);
            dim3 grid(L[cond], L[cond], 1);
            dim3 thread(1, 1, 1);
            // time mesurement
            cudaEventRecord(start, 0);
            calc_sum<<<grid, thread>>>(result_d);
            cudaEventRecord(end, 0);
            // fetch result
            cudaMemcpy(&result, result_d, sizeof(ull), d2h);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&t_time, start, end);
            printf("time=%lf[msec], ", t_time);
            t += t_time / rep;
            printf("result=%llu\n", result);
        }
        printf("avg time: %lf[msec]\n", t);
    }
    // free device mem
    cudaFree(result_d);
}

// Condition: L=128
//         trial: 0, time=0.033856[msec], result=44736512
//         trial: 1, time=0.034816[msec], result=44736512
//         trial: 2, time=0.036288[msec], result=44736512
//         trial: 3, time=0.034752[msec], result=44736512
//         trial: 4, time=0.034816[msec], result=44736512
// avg time: 0.034906[msec]
// Condition: L=1024
//         trial: 0, time=1.798272[msec], result=183251763200
//         trial: 1, time=1.803904[msec], result=183251763200
//         trial: 2, time=1.794112[msec], result=183251763200
//         trial: 3, time=1.802656[msec], result=183251763200
//         trial: 4, time=1.799936[msec], result=183251763200
// avg time: 1.799776[msec]
// Condition: L=8192
//         trial: 0, time=112.407234[msec], result=750599926710272
//         trial: 1, time=93.984772[msec], result=750599926710272
//         trial: 2, time=93.603134[msec], result=750599926710272
//         trial: 3, time=92.272606[msec], result=750599926710272
//         trial: 4, time=91.838303[msec], result=750599926710272
// avg time: 96.821211[msec]