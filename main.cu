#include <iostream>
#include <cstdio>
#include <cmath>
#include <ctime>

using namespace std;

static void HandleError( cudaError_t err,
                         const char *file,
                         int line );
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void cudaCheckAndPrintProperties();
cudaError_t addWithCuda(double* c, const double* a, const double* b, unsigned int size, const int BLOCKS, const int THREADS_PER_BLOCK);
void printTest();

__global__ void add(int a, int b, int* c) {
    *c = a + b;
}
double fRand(double fMin, double fMax);

const extern int ARRAY_SIZE = 6000000;
const extern int NORMAL_SPREAD = 12;

int main() {
    cudaCheckAndPrintProperties();

    printf("Array size: %d", ARRAY_SIZE);
    const int BLOCKS = (!ARRAY_SIZE % 1024) ? ARRAY_SIZE / 1024 : ARRAY_SIZE / 1024 + 1;


    const int THREADS_PER_BLOCK = (!ARRAY_SIZE % BLOCKS) ? 1024 : ARRAY_SIZE / BLOCKS +
                                                                  ceil(double(ARRAY_SIZE % BLOCKS) / BLOCKS);

    printf("\nAmount of BLOCKS: %d",  BLOCKS);
    printf("\nTHREADS_PER_BLOCK: %d",  THREADS_PER_BLOCK);

    auto * a = (double *)calloc(ARRAY_SIZE, sizeof(double ));
    auto * b = (double *)calloc(ARRAY_SIZE, sizeof(double ));
    auto * c = (double *)calloc(ARRAY_SIZE, sizeof(double ));


    srand(time(nullptr));
    // инициализация

    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
        a[i] = fRand(0, 10);
        b[i] = fRand(10, 20);
    }



    // Пареллельное сложения на GPU
    if (addWithCuda(c, a, b, ARRAY_SIZE, BLOCKS, THREADS_PER_BLOCK) != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        exit( EXIT_FAILURE );
    }


    for (int i = 0; i < 3; ++i){
        printf("\n%d: %f + %f = %f", i, a[i], b[i], c[i]);
    }
    for (int i = ARRAY_SIZE - 3; i < ARRAY_SIZE; ++i){
        printf("\n%d: %f + %f = %f", i, a[i], b[i], c[i]);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    if (cudaDeviceReset() != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        exit( EXIT_FAILURE );
    }

    free(a);
    free(b);
    free(c);

    return EXIT_SUCCESS;

}


void printTest(){
    cout <<"\n\n\n";
    cout<< "TTTTTTTTTTTTTTTTTTTTTTTEEEEEEEEEEEEEEEEEEEEEE   SSSSSSSSSSSSSSS TTTTTTTTTTTTTTTTTTTTTTT\n"
           "T:::::::::::::::::::::TE::::::::::::::::::::E SS:::::::::::::::ST:::::::::::::::::::::T\n"
           "T:::::::::::::::::::::TE::::::::::::::::::::ES:::::SSSSSS::::::ST:::::::::::::::::::::T\n"
           "T:::::TT:::::::TT:::::TEE::::::EEEEEEEEE::::ES:::::S     SSSSSSST:::::TT:::::::TT:::::T\n"
           "TTTTTT  T:::::T  TTTTTT  E:::::E       EEEEEES:::::S            TTTTTT  T:::::T  TTTTTT\n"
           "        T:::::T          E:::::E             S:::::S                    T:::::T        \n"
           "        T:::::T          E::::::EEEEEEEEEE    S::::SSSS                 T:::::T        \n"
           "        T:::::T          E:::::::::::::::E     SS::::::SSSSS            T:::::T        \n"
           "        T:::::T          E:::::::::::::::E       SSS::::::::SS          T:::::T        \n"
           "        T:::::T          E::::::EEEEEEEEEE          SSSSSS::::S         T:::::T        \n"
           "        T:::::T          E:::::E                         S:::::S        T:::::T        \n"
           "        T:::::T          E:::::E       EEEEEE            S:::::S        T:::::T        \n"
           "      TT:::::::TT      EE::::::EEEEEEEE:::::ESSSSSSS     S:::::S      TT:::::::TT      \n"
           "      T:::::::::T      E::::::::::::::::::::ES::::::SSSSSS:::::S      T:::::::::T      \n"
           "      T:::::::::T      E::::::::::::::::::::ES:::::::::::::::SS       T:::::::::T      \n"
           "      TTTTTTTTTTT      EEEEEEEEEEEEEEEEEEEEEE SSSSSSSSSSSSSSS         TTTTTTTTTTT      \n" << endl;
    cout <<
            "     CCC::::::::::::CU::::::U     U::::::UD::::::::::::DDD                   A:::A                   \n"
            "   CC:::::::::::::::CU::::::U     U::::::UD:::::::::::::::DD                A:::::A                  \n"
            "  C:::::CCCCCCCC::::CUU:::::U     U:::::UUDDD:::::DDDDD:::::D              A:::::::A                 \n"
            " C:::::C       CCCCCC U:::::U     U:::::U   D:::::D    D:::::D            A:::::::::A                \n"
            "C:::::C               U:::::D     D:::::U   D:::::D     D:::::D          A:::::A:::::A               \n"
            "C:::::C               U:::::D     D:::::U   D:::::D     D:::::D         A:::::A A:::::A              \n"
            "C:::::C               U:::::D     D:::::U   D:::::D     D:::::D        A:::::A   A:::::A             \n"
            "C:::::C               U:::::D     D:::::U   D:::::D     D:::::D       A:::::A     A:::::A            \n"
            "C:::::C               U:::::D     D:::::U   D:::::D     D:::::D      A:::::AAAAAAAAA:::::A           \n"
            "C:::::C               U:::::D     D:::::U   D:::::D     D:::::D     A:::::::::::::::::::::A          \n"
            " C:::::C       CCCCCC U::::::U   U::::::U   D:::::D    D:::::D     A:::::AAAAAAAAAAAAA:::::A         \n"
            "  C:::::CCCCCCCC::::C U:::::::UUU:::::::U DDD:::::DDDDD:::::D     A:::::A             A:::::A        \n"
            "   CC:::::::::::::::C  UU:::::::::::::UU  D:::::::::::::::DD     A:::::A               A:::::A       \n"
            "     CCC::::::::::::C    UU:::::::::UU    D::::::::::::DDD      A:::::A                 A:::::A      \n"
            "        CCCCCCCCCCCCC      UUUUUUUUU      DDDDDDDDDDDDD        AAAAAAA                   AAAAAAA\n";
}

void cudaCheckAndPrintProperties(){
    printTest();
    cout << "------------------------------------------------------------------------------------------------\n";
    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));

    cout << "Found " << count << " device(s)" << endl;

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};

        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        cout << "Device name: " << prop.name << endl;
        cout << "Warp size in threads: " << prop.warpSize << endl;
        cout << "Shared memory available per block in bytes: " << prop.sharedMemPerBlock  / 1024.0 / 1024.0 << " MB" << endl;
        cout << "Total Memory: " << prop.totalGlobalMem / 1024.0 / 1024.0 << " MB" << endl;
        cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
        cout << "Maximum size of each dimension of a grid: " <<  prop.maxGridSize[0] << " | "
                                                             <<  prop.maxGridSize[1] << " | "
                                                             <<  prop.maxGridSize[2] << endl;

        cout << "Maximum size of each dimension of a block: " <<  prop.maxThreadsDim[0] << " | "
                                                              <<  prop.maxThreadsDim[1] << " | "
                                                              <<  prop.maxThreadsDim[2] << endl;
        cout << "Compute capability: " << prop.major << "." << prop.minor << endl;
    }

    int result;
    int* devResult;

    HANDLE_ERROR(cudaMalloc((void**)&devResult, sizeof(int)));

    add<<<1, 1>>>(7, 8, devResult);

    HANDLE_ERROR(cudaMemcpy(&result, devResult, sizeof(int), cudaMemcpyDeviceToHost));

    cout << "7 + 8 = " << result << endl;
    cudaFree(devResult);
    cout << "------------------------------------------------------------------------------------------------\n";
}


//__global__ — выполняется на GPU, вызывается с CPU.
__global__ void addKernel(double *c, const double *a, const double *b, const int size)
{
    // Индекс обсчитываемых компонент вектора с учетом смещения от количества блоков
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        c[i] = a[i] + b[i];
}

cudaError_t addWithCuda(double* c, const double* a, const double* b, unsigned int size, const int BLOCKS, const int THREADS_PER_BLOCK)
{
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    double allTime = 0;
    cudaError_t cudaStatus;

    // Создание обработчиков событий
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cannot create CUDA start event: %s\n",
                cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cannot create CUDA end event: %s\n",
                cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Инициализация девайса
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    // Выделения памяти на GPU
    cudaStatus = cudaMalloc(&dev_c, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&dev_a, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&dev_b, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Копирования входных векторов с хоста на девайс
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    for (int i = 0; i < 12; i++)
    {
        // Установка точки старта
        cudaStatus = cudaEventRecord(start, 0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Cannot record CUDA start event: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Запуск функции ядра на GPU
        addKernel <<< BLOCKS, THREADS_PER_BLOCK >>> (dev_c, dev_a, dev_b, size);

        // Отлов ошибок запуска ядра
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Установка точки окончания
        cudaStatus = cudaEventRecord(stop, 0);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "Cannot record CUDA end event: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Ожидание завершения обсчета функции ядра
        // Отлов ошибок работы и завершения ядра
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        // Расчет времени
        cudaStatus = cudaEventElapsedTime(&gpuTime, start, stop);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "Cannot record CUDA time event: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        printf("\nTime: %.20f", gpuTime / 1000);
        allTime += gpuTime / 1000;
    }
    printf("\nAverage time: %.20f", allTime / 12);

    // Копирования выходного вектора с девайса на хост
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Возникла ошибка/конец программы
    Error:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}



void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

double fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
