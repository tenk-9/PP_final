// 21140036 並列処理 期末課題
// レポート課題3 (2)
// C言語+OpenMPによる実装 (c_omp.c)
// (c) Takuo Yamaguchi

// compile: gcc c_omp.c -o c_omp -fopenmp

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
typedef unsigned long long ull;

// 計算を行う関数．i, jを引数にとり，計算結果を返す．
ull calc_sum(ull i, ull j)
{
    ull sum = 0;        // 計算結果格納用変数
    ull i_loop, j_loop; // ループ制御変数
// 並列化．並列化スレッド数は8．
#pragma omp parallel for reduction(+ : sum) private(j_loop) num_threads(8)
    for (i_loop = 0; i_loop < i; i_loop++)
    {
        for (j_loop = 0; j_loop < j; j_loop++)
        {
            sum += (i_loop - j_loop) * (i_loop - j_loop);
        }
    }
    return sum;
}

// 実行時間計測用の変数
double start, end;
double time = 0; // msec
// 各条件の繰り返し数
const int rep = 5;
// 計算サイズL
ull L[3] = {1ULL << 7, 1ULL << 10, 1ULL << 13};
int main()
{
    ull result;
    // 条件ごとにrep回実行し，実行時間の平均をとる．
    for (int cond = 0; cond < 3; cond++)
    {
        time = 0;
        printf("Condition: L=%llu\n", L[cond]);
        for (int trial = 0; trial < rep; trial++)
        {
            printf("\ttrial: %d, ", trial);
            // time mesurement
            start = omp_get_wtime();
            result = calc_sum(L[cond], L[cond]);
            end = omp_get_wtime();
            time += 1e3 * (end - start) / rep;
            printf("time=%lf[msec], ", 1e3 * (end - start));
            printf("result=%llu\n", result);
        }
        printf("avg time: %lf[msec]\n", time);
    }
}

// Condition : L = 128
//     trial : 0, time = 0.540700 [msec], result = 44736512
//     trial : 1, time = 0.008200 [msec], result = 44736512
//     trial : 2, time = 0.084300 [msec], result = 44736512
//     trial : 3, time = 0.007900 [msec], result = 44736512
//     trial : 4, time = 0.007600 [msec], result = 44736512
// avg time : 0.129740 [msec]
// Condition : L = 1024
//     trial : 0, time = 0.394200 [msec], result = 183251763200
//     trial : 1, time = 0.393100 [msec], result = 183251763200
//     trial : 2, time = 0.382800 [msec], result = 183251763200
//     trial : 3, time = 0.411600 [msec], result = 183251763200
//     trial : 4, time = 0.394900 [msec], result = 183251763200
// avg time : 0.395320 [msec]
// Condition : L = 8192
//     trial : 0, time = 22.496400 [msec], result = 750599926710272
//     trial : 1, time = 20.374899 [msec], result = 750599926710272
//     trial : 2, time = 21.820600 [msec], result = 750599926710272
//     trial : 3, time = 23.694700 [msec], result = 750599926710272
//     trial : 4, time = 21.840199 [msec], result = 750599926710272
// avg time : 22.045360 [msec]