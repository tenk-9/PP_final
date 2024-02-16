// 21140036 並列処理 期末課題
// レポート課題5 (3)
// cudaによる実装 (cuda_img.cu)
// (c) Takuo Yamaguchi

// compile: nvcc cuda_img.cu -o cuda_img

#include <stdio.h>
#include <stdlib.h>

#define h2d cudaMemcpyHostToDevice
#define d2h cudaMemcpyDeviceToHost
#define Nx 1024
#define Ny 1024

// 画像を左周りに90度回転させる関数．img_inは読み取るだけ，img_outを変更する．
__global__ void rotate90(int *img_in, int *img_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int id_in = i * Nx + j;
    int id_out = (Ny - j - 1) * Nx + i;
    img_out[id_out] = img_in[id_in];
}

// 画像を上下反転させる関数．img_inは読み取るだけ，img_outを変更する．
__global__ void UDswap(int *img_in, int *img_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int id_in = i * Nx + j;
    int id_out = (Ny - i - 1) * Nx + j;
    img_out[id_out] = img_in[id_in];
}

int main()
{
    // 画像配列
    int *img_in, *img_out, *img_in_d, *img_out_d;
    img_in = (int *)malloc(sizeof(int) * Nx * Ny);
    img_out = (int *)malloc(sizeof(int) * Nx * Ny);
    cudaMalloc(&img_in_d, sizeof(int) * Nx * Ny);
    cudaMalloc(&img_out_d, sizeof(int) * Nx * Ny);
    // Block数: (4, 1024, 1), Thread数: (256, 1, 1)
    dim3 blk(4, 1024, 1);
    dim3 thd(256, 1, 1);
    // 画像読み込み
    FILE *fp;
    fp = fopen("./man1024.img", "r");
    for (int i = 0; i < Nx * Ny; i++)
    {
        fscanf(fp, "%d", &img_in[i]);
    }
    fclose(fp);
    cudaMemcpy(img_in_d, img_in, sizeof(int) * Nx * Ny, h2d);
    // process and ouput
    // A (rotate)
    rotate90<<<blk, thd>>>(img_in_d, img_out_d);
    cudaMemcpy(img_out, img_out_d, sizeof(int) * Nx * Ny, d2h);
    fp = fopen("out/cuda/cuda_rotate.img", "w");
    for (int i = 0; i < Nx * Ny; i++)
    {
        fprintf(fp, "%d\n", img_out[i]);
    }
    fclose(fp);
    // B (UDswap)
    UDswap<<<blk, thd>>>(img_in_d, img_out_d);
    cudaMemcpy(img_out, img_out_d, sizeof(int) * Nx * Ny, d2h);
    fp = fopen("out/cuda/cuda_swap.img", "w");
    for (int i = 0; i < Nx * Ny; i++)
    {
        fprintf(fp, "%d\n", img_out[i]);
    }
    fclose(fp);
    // free mem
    free(img_in);
    free(img_out);
    cudaFree(img_in_d);
    cudaFree(img_out_d);
}