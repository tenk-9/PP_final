// 21140036 並列処理 期末課題
// レポート課題5 (1)
// C言語による実装 (c_img_imp.c)
// (c) Takuo Yamaguchi

// compile: gcc c_img_imp.c -o c_imp

#include <stdio.h>
#include <stdlib.h>

#define Nx 1024
#define Ny 1024

// 画像を左に90度回転させる関数．img_inは読み取るだけ，img_outを変更する．
// 左に90度回転させる変形は，転置してから上下反転させる操作と等価．
void rotate90(int *img_in, int *img_out)
{
    int id_in, id_out;
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            id_in = i * Nx + j;
            // id_out = j * Nx + i; 転置操作
            // id_out = (Ny - i - 1) * Nx + j; 上下反転操作
            id_out = (Ny - j - 1) * Nx + i; // 転置して上下反転
            img_out[id_out] = img_in[id_in];
        }
    }
}

// 画像を上下反転させる関数．img_inは読み取るだけ，img_outを変更する．
void UDswap(int *img_in, int *img_out)
{
    int id_in, id_out;
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            id_in = i * Nx + j;
            id_out = (Ny - i - 1) * Nx + j;
            img_out[id_out] = img_in[id_in];
        }
    }
}

int main()
{
    // 画像配列
    int *img_in, *img_out;
    img_in = (int *)malloc(sizeof(int) * Nx * Ny);
    img_out = (int *)malloc(sizeof(int) * Nx * Ny);
    // 画像読み込み
    FILE *fp;
    fp = fopen("./man1024.img", "r");
    for (int i = 0; i < Nx * Ny; i++)
    {
        fscanf(fp, "%d", &img_in[i]);
    }
    fclose(fp);
    // process and ouput
    // A (rotate)
    rotate90(img_in, img_out);
    fp = fopen("out/c/c_rotate.img", "w");
    for (int i = 0; i < Nx * Ny; i++)
    {
        fprintf(fp, "%d\n", img_out[i]);
    }
    fclose(fp);
    // B (UDswap)
    UDswap(img_in, img_out);
    fp = fopen("out/c/c_swap.img", "w");
    for (int i = 0; i < Nx * Ny; i++)
    {
        fprintf(fp, "%d\n", img_out[i]);
    }
    fclose(fp);
    // free mem
    free(img_in);
    free(img_out);
}