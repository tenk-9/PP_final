// 21140036 並列処理 期末課題
// レポート課題4 (2)
// C言語による実装 (c_mat_omp.c)
// (c) Takuo Yamaguchi

// compile: gcc c_mat_omp.c -o c_omp -fopenmp

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

// 要素積の計算を行う関数．行列A,B,Cとそれらのサイズnを引数にとる．
void adm(int *A, int *B, int *C, int n)
{
    int id, i, j;
// 並列化
#pragma omp parallel for private(j) num_threads(8)
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            id = i * n + j;
            C[id] = A[id] * B[id];
        }
    }
}

// 行列積の計算を行う関数．
void dot(int *A, int *B, int *C, int n)
{
    int i, j, k, sum, ida, idb, idc;
// 並列化
#pragma omp parallel for private(j, k, sum) num_threads(8)
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            idc = i * n + j;
            sum = 0;
            for (k = 0; k < n; k++)
            {
                ida = i * n + k;
                idb = k * n + j;
                sum += A[ida] * B[idb];
            }
            C[idc] = sum;
        }
    }
}

// 実行時間計測用の変数
double start, end;
double time = 0; // msec
// 各条件の繰り返し数
const int rep = 5;
// 行列サイズN
int N[2] = {1ULL << 7, 1ULL << 10};
int *A, *B, *C;
int *Ad, *Bd, *Cd;
int main()
{
    // 条件ごとにrep回実行し，実行時間の平均をとる．
    // 要素積
    printf("要素積\n");
    for (int cond = 0; cond < 2; cond++)
    {
        time = 0;
        printf("\tCondition: L=%d\n", N[cond]);
        for (int trial = 0; trial < rep; trial++)
        {
            printf("\t\ttrial: %d, ", trial);
            // malloc
            A = (int *)malloc(sizeof(int) * N[cond] * N[cond]);
            B = (int *)malloc(sizeof(int) * N[cond] * N[cond]);
            C = (int *)malloc(sizeof(int) * N[cond] * N[cond]);
            for (int i = 0; i < N[cond] * N[cond]; i++)
            {
                A[i] = 1;
                B[i] = 1;
                C[i] = 1;
            }
            // time mesurement
            start = omp_get_wtime();
            adm(A, B, C, N[cond]);
            end = omp_get_wtime();
            time += 1e3 * (end - start) / rep;
            printf("time=%lf[msec]\n", 1e3 * (end - start));
            // free mem
            free(A);
            free(B);
            free(C);
        }
        printf("\tavg time: %lf[msec]\n", time);
    }
    // 行列積
    printf("行列積\n");
    for (int cond = 0; cond < 2; cond++)
    {
        time = 0;
        printf("\tCondition: L=%d\n", N[cond]);
        for (int trial = 0; trial < rep; trial++)
        {
            printf("\t\ttrial: %d, ", trial);
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
            // time mesurement
            start = omp_get_wtime();
            dot(A, B, C, N[cond]);
            end = omp_get_wtime();
            time += 1e3 * (end - start) / rep;
            printf("time=%lf[msec]\n", 1e3 * (end - start));
            // free mem
            free(A);
            free(B);
            free(C);
        }
        printf("\tavg time: %lf[msec]\n", time);
    }
}
