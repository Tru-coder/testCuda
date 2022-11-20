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
cudaError_t addWithCuda(double* c, const double* a, const double* b, const int BLOCKS, const int THREADS_PER_BLOCK);
void printTest();
double fRand(double fMin, double fMax);

const extern int ARRAY_SIZE = 6000000;
const extern int NORMAL_SPREAD = 12;

int main() {
    cudaCheckAndPrintProperties();

    printf("Array size: %d", ARRAY_SIZE);



    // gridDim - Количество ядер
    const int GRID_DIM[]{4096, 2048, 4096, 2048, 4096, 2048};
    // blockDim - Количество потоков в ядре
    const int BLOCK_DIM[]{1024, 1024, 256, 256, 64, 64};



    auto * a = (double *)calloc(ARRAY_SIZE, sizeof(double ));
    auto * b = (double *)calloc(ARRAY_SIZE, sizeof(double ));
    auto * c = (double *)calloc(ARRAY_SIZE, sizeof(double ));


    srand(time(nullptr));
    // инициализация
    for (int i = 0; i < ARRAY_SIZE; ++i){
        a[i] = fRand(0, 10);
        b[i] = fRand(10, 20);
    }

    // Пареллельное сложения на GPU
    for (int i = 0; i < 6; ++i){
        HANDLE_ERROR(addWithCuda(c, a, b, GRID_DIM[i], BLOCK_DIM[i]));
        for (int i = 0; i < 3; ++i){
            printf("\n%d: %f + %f = %f", i, a[i], b[i], c[i]);
        }
        for (int i = ARRAY_SIZE - 3; i < ARRAY_SIZE; ++i){
            printf("\n%d: %f + %f = %f", i, a[i], b[i], c[i]);
        }

    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    HANDLE_ERROR(cudaDeviceReset());

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
        cout << "Total Memory: " << prop.totalGlobalMem / 1024.0 / 1024.0 << " MB" << endl;
        cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
        cout << "Number of multiprocessors on device: " << prop.multiProcessorCount << endl;
        cout << "Maximum size of each dimension of a grid: " <<  prop.maxGridSize[0] << " | "
                                                             <<  prop.maxGridSize[1] << " | "
                                                             <<  prop.maxGridSize[2] << endl;

        cout << "Maximum size of each dimension of a block: " <<  prop.maxThreadsDim[0] << " | "
                                                              <<  prop.maxThreadsDim[1] << " | "
                                                              <<  prop.maxThreadsDim[2] << endl;
        cout << "Compute capability: " << prop.major << "." << prop.minor << endl;
    }

    cout << "------------------------------------------------------------------------------------------------\n";
}


//__global__ — выполняется на GPU, вызывается с CPU.
__global__ void addKernel(double *c, const double *a, const double *b)
{
  /// Каждому потоку, выполняющему addKernel,
  /// присваивается уникальный идентификатор блока
  /// и идентификатор потока, который доступен в ядре через
  /// встроенную переменную blockIdx.x и threadIdx,x.
  /// Мы используем этот индекс, чтобы определить,
  /// какие пары чисел мы хотим добавить в ядро.

    // blockIdx.x is the index of the block.
    // Each block has blockDim.x threads.
    // threadIdx.x is the index of the thead.
    // Each thread can perform 1 addition.


    // Индекс обсчитываемых компонент вектора с учетом смещения от количества блоков
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < ARRAY_SIZE)
        c[i] = a[i] + b[i];
}
// Computing
cudaError_t addWithCuda(double* c, const double* a, const double* b, const int BLOCKS, const int THREADS_PER_BLOCK)
{
    double* dev_a = nullptr;
    double* dev_b = nullptr;
    double* dev_c = nullptr;
    double allTime = 0;

    printf("\nAmount of BLOCKS: %d",  BLOCKS);
    printf("\nTHREADS_PER_BLOCK: %d",  THREADS_PER_BLOCK);

    // Создание обработчиков событий
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // Инициализация девайса
    HANDLE_ERROR(cudaSetDevice(0));

    // Выделения памяти на GPU
    HANDLE_ERROR(cudaMalloc(&dev_c, ARRAY_SIZE * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&dev_a, ARRAY_SIZE * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&dev_b, ARRAY_SIZE * sizeof(double)));

    // Копирования входных векторов с хоста на девайс
    HANDLE_ERROR(cudaMemcpy(dev_a, a, ARRAY_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, ARRAY_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    for (int i = 0; i < NORMAL_SPREAD; ++i){
        // Установка точки старта
        HANDLE_ERROR(cudaEventRecord(start, nullptr));


        // Запуск функции ядра на GPU
        addKernel <<< BLOCKS, THREADS_PER_BLOCK >>> (dev_c, dev_a, dev_b);

        // Отлов ошибок запуска ядра
        HANDLE_ERROR(cudaGetLastError());

        // Установка точки окончания
        HANDLE_ERROR(cudaEventRecord(stop, nullptr));

        /// у функций ядра при этом есть особенность – асинхронное исполнение,
        /// то есть, если после вызова ядра начал работать следующий участок кода,
        /// то это ещё не значит, что GPU выполнил расчеты.
        /// Для завершения работы заданной функции ядра необходимо использовать
        /// средства синхронизации, например event’ы. Поэтому,
        /// перед копированием результатов на хост выполняем синхронизацию нитей GPU
        /// через event


        //Хендл event'а
        cudaEvent_t syncEvent;

        HANDLE_ERROR(cudaEventCreate(&syncEvent));    //Создаем event
        HANDLE_ERROR(cudaEventRecord(syncEvent, nullptr));  //Записываем event
        HANDLE_ERROR(cudaEventSynchronize(syncEvent));  //Синхронизируем event

        // Расчет времени
        HANDLE_ERROR(cudaEventElapsedTime(&gpuTime, start, stop));
        printf("\nTime: %.20f", gpuTime / 1000);
        allTime += gpuTime / 1000;


    }
    printf("\nAverage time: %.20f", allTime / 12);

    // Копирования выходного вектора с девайса на хост
    HANDLE_ERROR(cudaMemcpy(c, dev_c, ARRAY_SIZE * sizeof(double), cudaMemcpyDeviceToHost));

    // Освобождение памяти
    HANDLE_ERROR(cudaFree(dev_c));
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));

    return cudaSuccess;
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
