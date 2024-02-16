// 21140036 並列処理 期末課題
// レポート課題3 (1)
// C言語による実装 (c_imp.c)
// (c) Takuo Yamaguchi

// compile: gcc c_imp.c -o c_imp -fopenmp

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
// 計算結果の保持にはunsigned long long 型を用いる．ここでは，ullとして型定義する．
// [0, 2^64 - 1]の値が格納できるため，課題3の計算では，Lのサイズは2^32まで許容される．
typedef unsigned long long ull;

// 計算を行う関数．i, jを引数にとり，計算結果を返す．
ull calc_sum(ull i, ull j)
{
    ull sum = 0;        // 計算結果格納用変数
    ull i_loop, j_loop; // ループ制御変数
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
