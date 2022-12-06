////
//// Created by user on 27.11.2022.
////
//#include <iostream>
//#include <cstdio>
//#include <cmath>
//#include <ctime>
//
//#include <cstdlib>
//#include <cstring>
//#include <ctime>
//#include <cuda_runtime.h>
//#include <cublas_v2.h>
//
//#include <cstdlib>
//#include <malloc.h>
//#include <curand.h>
//#pragma comment (lib, "cublas.lib")
//#pragma comment (lib, "curand.lib")
//#define I(i, j, rows) ((j) * (rows) + (i))
//
//void GPU_fill_rand(float* A, int nr_rows_A, int nr_cols_A);
//void gpu_blas_mmul(const float* A, const float* B, float* C, const int m, const int k, const int n);
//static void HandleError( cudaError_t err,const char *file,int line );
//void print_matrix(float* matrix, int rows, int cols);
//
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
//
//
//__device__ inline unsigned int lcg_rand(unsigned int& seed)
//{
//    seed = seed * 1664525 + 1013904223UL;
//    return seed;
//}
//
//__global__ void GenerateRandNums(unsigned int* out, unsigned int* states)
//{
//    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
//    unsigned int s = states[tid];
//    out[tid] = lcg_rand(s) + lcg_rand(s);
//    states[tid] = s;
//}
//
//int main() {
//    // Выделяем 3 массива на хосте
//    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
//
//
//    // Используются квадратные матрицы
//    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 3;
//
//    auto* h_A = (float*)malloc(nr_rows_A * nr_cols_A * sizeof(float));
//    auto* h_B = (float*)malloc(nr_rows_B * nr_cols_B * sizeof(float));
//    auto* h_C = (float*)malloc(nr_rows_C * nr_cols_C * sizeof(float));
//
//    // Выделяем 3 массива на девайсе
//    float* d_A, * d_B, * d_C;
//    HANDLE_ERROR (cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(float)));
//    HANDLE_ERROR (cudaMalloc(&d_B, nr_rows_B * nr_cols_B * sizeof(float)));
//    HANDLE_ERROR (cudaMalloc(&d_C, nr_rows_C * nr_cols_C * sizeof(float)));
//
//    // Заполняем матрицы А и В случайными числами
////    GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
////    GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
//
//
//// A
//    h_A[I(0, 0, nr_rows_A)] = 2;
//    h_A[I(0, 1, nr_rows_A)] = 1;
//    h_A[I(0, 2, nr_rows_A)] = 0;
//
//    h_A[I(1, 0, nr_rows_A)] = 3;
//    h_A[I(1, 1, nr_rows_A)] = 1;
//    h_A[I(1, 2, nr_rows_A)] = 1;
//
//    h_A[I(2, 0, nr_rows_A)] = 0;
//    h_A[I(2, 1, nr_rows_A)] = 1;
//    h_A[I(2, 2, nr_rows_A)] = 2;
//
//// B
//    h_B[I(0, 0, nr_rows_B)] = 2;
//    h_B[I(0, 1, nr_rows_B)] = 4;
//    h_B[I(0, 2, nr_rows_B)] = 1;
//
//    h_B[I(1, 0, nr_rows_B)] = 0;
//    h_B[I(1, 1, nr_rows_B)] = 1;
//    h_B[I(1, 2, nr_rows_B)] = 1;
//
//    h_B[I(2, 0, nr_rows_B)] = 1;
//    h_B[I(2, 1, nr_rows_B)] = 0;
//    h_B[I(2, 2, nr_rows_B)] = 4;
//
////    int k = 0;
////    for (int i = 0; i < nr_rows_A; i++)
////    {
////        for (int j = 0; j < nr_cols_A; ++j) {
////            h_A[i + j * nr_cols_A] = k + 1;
////            h_B[i + j * nr_cols_A] = 9 - k++;
////            printf("i = %d; A[i] = %f; B[i] = %f\n", i + j * nr_cols_A, h_A[i + j * nr_cols_A], h_B[i + j * nr_cols_A]);
////        }
////    }
//
//
//    HANDLE_ERROR (cudaMemcpy(d_A, h_A, nr_rows_A * nr_cols_A * sizeof(float), cudaMemcpyHostToDevice));
//    HANDLE_ERROR (cudaMemcpy(d_B, h_B, nr_rows_B * nr_cols_B * sizeof(float), cudaMemcpyHostToDevice));
//
//    // Можно скопировать их на хост, чтобы вывести их
//    HANDLE_ERROR (cudaMemcpy(h_A, d_A, nr_rows_A * nr_cols_A * sizeof(float), cudaMemcpyDeviceToHost));
//    HANDLE_ERROR (cudaMemcpy(h_B, d_B, nr_rows_B * nr_cols_B * sizeof(float), cudaMemcpyDeviceToHost));
//
//    printf("A =\n");
//    print_matrix(h_A, nr_rows_A, nr_cols_A);
//    printf("B =\n");
//    print_matrix(h_B, nr_rows_B, nr_cols_B);
//
//
//
//
//    cudaEvent_t start, end;
//    HANDLE_ERROR(cudaEventCreate(&start));
//    HANDLE_ERROR(cudaEventCreate(&end));
//
//    float time = 0.0f;
//    HANDLE_ERROR (cudaEventRecord(start, nullptr));
//
//    // Умножаем матрицы А и В на GPU
//    gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
//    HANDLE_ERROR (cudaEventRecord(end, nullptr));
//
//    // Копируем на хост и выводим результат умножения матриц
//    HANDLE_ERROR(cudaMemcpy(h_C, d_C, nr_rows_C * nr_cols_C * sizeof(float), cudaMemcpyDeviceToHost));
//
//    printf("A[0] = %f", h_A[0]);
//    printf("C =\n");
//    print_matrix(h_C, nr_rows_C, nr_cols_C);
//
//    HANDLE_ERROR (cudaEventElapsedTime(&time, start, end));
//    printf("Total time: %f\n", time / 1000);
//
//    // Освобождаем память на девайсе
//    HANDLE_ERROR (cudaFree(d_A));
//    HANDLE_ERROR (cudaFree(d_B));
//    HANDLE_ERROR (cudaFree(d_C));
//
//    // Освобождаем память на хосте
//    free(h_A);
//    free(h_B);
//    free(h_C);
//    return EXIT_SUCCESS;
//}
//
//
//void HandleError(cudaError_t err, const char *file, int line) {
//    if (err != cudaSuccess) {
//        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
//                file, line );
//        exit( EXIT_FAILURE );
//    }
//}
//
//
//void GPU_fill_rand(float* A, int nr_rows_A, int nr_cols_A) {
//    // Создание генератора псевдо-рандомных чисел
//    curandGenerator_t prng;
//    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
//
//    // Установка сида для генератора
//    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
//
//    // Заполнение матрицы случайными числами
//    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
//}
//
///// gpu_blas_mmul() - Функция умножения матриц
///// Перемножение матриц А и В с сохранением результата в матрице С
///// C(m,n) = A(m,k) * B(k,n)
//void gpu_blas_mmul(const float* A, const float* B, float* C, const int m, const int k, const int n) {
//    int lda = m, ldb = k, ldc = m;
//    const float alf = 1;
//    const float bet = 0;
//    const float* alpha = &alf;
//    const float* beta = &bet;
//
//
//    // Создадим хендл
//    cublasHandle_t handle;
//    cublasCreate(&handle);
//
//
//    // Умножение
//    // cublasSgemm(handle, ..., ..., m, n, k, a, A, lda, ldb, b, C, ldc)
//    // A - матрица размером m x k
//    // B - матрица размером k x n
//    // C = a * A * B + b * C
//    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
//
//    // Удалим хендл
//    cublasDestroy(handle);
//}
//
//
//void print_matrix(float* matrix, int rows, int cols) {
//    printf("%f  *   %f\n", matrix[0], matrix[cols - 1]);
//    printf("*   *   *\n");
//    printf("%f  *   %f\n", matrix[rows * (cols - 1)], matrix[cols * rows - 1]);
//}
//
//
//
//
//#include <cstdlib>
//#include <curand.h>
//#include <cublas_v2.h>
//#include <iostream>
//
////GPU_fill_rand() - Функция случайной генерации матрицы
////gpu_blas_mmul() - Функция умножения матриц
////print_matrix() - Функция вывода матрицы
//
//
////GPU_fill_rand() - Функция случайной генерации матрицы
////gpu_blas_mmul() - Функция умножения матриц
////print_matrix() - Функция вывода матрицы
//
//void print_matrix(float* matrix, int rows, int cols) {
//    printf("%f  *   %f\n", matrix[0], matrix[cols - 1]);
//    printf("*   *   *\n");
//    printf("%f  *   %f\n", matrix[rows * (cols - 1)], matrix[cols * rows - 1]);
//}
//
//
//void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n);
//void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A);
//
//int main() {
//    // Allocate 3 arrays on CPU
//    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
//    // for simplicity we are going to use square arrays
//    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 3;
//    float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
//    float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
//    float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
//    // Allocate 3 arrays on GPU
//    float *d_A, *d_B, *d_C;
//    cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
//    cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
//    cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));
//    // Fill the arrays A and B on GPU with random numbers
//    GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
//    GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
//    // Optionally we can copy the data back on CPU and print the arrays
//    cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyDeviceToHost);
//    cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyDeviceToHost);
//    std::cout << "A =" << std::endl;
//    print_matrix(h_A, nr_rows_A, nr_cols_A);
//    std::cout << "B =" << std::endl;
//    print_matrix(h_B, nr_rows_B, nr_cols_B);
//    // Multiply A and B on GPU
//    gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
//    // Copy (and print) the result on host memory
//    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
//    std::cout << "C =" << std::endl;
//    print_matrix(h_C, nr_rows_C, nr_cols_C);
//
//    //Free GPU memory
//    cudaFree(d_A);
//    cudaFree(d_B);
//    cudaFree(d_C);
//
//    // Free CPU memory
//    free(h_A);
//    free(h_B);
//    free(h_C);
//    return EXIT_SUCCESS;
//}
//
//
//
//// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
//void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
//    // Create a pseudo-random number generator
//    curandGenerator_t prng;
//    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
//    // Set the seed for the random number generator using the system clock
//    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
//
//    // Fill the array with random numbers on the device
//    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
//}
//
//
//// Multiply the arrays A and B on GPU and save the result in C
//// C(m,n) = A(m,k) * B(k,n)
//void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
//    int lda = m, ldb = k, ldc = m;
//    const float alf = 1;
//    const float bet = 0;
//    const float *alpha = &alf;
//    const float *beta = &bet;
//    // Create a handle for CUBLAS
//    cublasHandle_t handle;
//    cublasCreate(&handle);
//    // Do the actual multiplication
//    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
//    // Destroy the handle
//    cublasDestroy(handle);
//}