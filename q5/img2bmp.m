filename = [
    "man1024.img",
    "out/c/c_rotate.img",
    "out/c/c_swap.img",
    "out/omp/omp_rotate.img",
    "out/omp/omp_swap.img",
    "out/cuda/cuda_rotate.img",
    "out/cuda/cuda_swap.img"
    ];

for i=1:size(filename)
    outfileName = erase(filename(i), ".img") + ".bmp";
    % ファイル読み込み
    mat = readmatrix(filename(i), 'FileType','text');
    % 整形
    mat = reshape(mat, [1024, 1024]); 
    % プログラムはrow-majorを前提とした記述なので，転置する(matlabはcolum-majorのため)．
    imwrite(uint8(mat'), outfileName, "bmp");
end