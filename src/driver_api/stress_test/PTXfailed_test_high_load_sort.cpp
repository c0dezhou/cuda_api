#include "test_utils.h"

#define MAX_GPUS  4

class SortingSearchTest {
   protected:

    int numGPUs;
    int arraysPerGPU;
    const int BATCH_SIZE = 100;
    const int ARRAY_SIZE = 10000;
    const int TARGET_COUNT = 1000;
    CUdevice devices[MAX_GPUS];
    CUcontext contexts[MAX_GPUS];
    CUstream* streams;

public:
    void RunSortingSearchTest() {
        int deviceCount;
        checkError(cuInit(0));
        checkError(cuDeviceGetCount(&deviceCount));

        numGPUs = std::min(deviceCount, MAX_GPUS);
        arraysPerGPU = ARRAY_SIZE / (numGPUs * BATCH_SIZE);

        streams = new CUstream[numGPUs];
        for (int i = 0; i < numGPUs; ++i) {
            checkError(cuDeviceGet(&devices[i], i));
            checkError(cuCtxCreate(&contexts[i], 0, devices[i]));
            checkError(cuStreamCreate(&streams[i], CU_STREAM_DEFAULT));
        }

        std::vector<int> cpuArrays(ARRAY_SIZE * BATCH_SIZE);
        std::vector<int> cpuResults(TARGET_COUNT * BATCH_SIZE);

        int* gpuArrays;
        int* gpuResults;
        checkError(cuMemAlloc((CUdeviceptr*)&gpuArrays, ARRAY_SIZE * BATCH_SIZE * sizeof(int)));
        checkError(cuMemAlloc((CUdeviceptr*)&gpuResults,
                              TARGET_COUNT * BATCH_SIZE * sizeof(int)));

        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(0, 1000);
        for (int i = 0; i < ARRAY_SIZE * BATCH_SIZE; ++i) {
            cpuArrays[i] = distribution(generator);
        }

        checkError(cuMemcpyHtoD((CUdeviceptr)gpuArrays, cpuArrays.data(),
                                ARRAY_SIZE * BATCH_SIZE * sizeof(int)));

        auto gpuStartTime = std::chrono::steady_clock::now();

        for (int gpuIndex = 0; gpuIndex < numGPUs; ++gpuIndex) {
            checkError(cuCtxSetCurrent(contexts[gpuIndex]));
            // checkError(cuStreamAttachMemAsync(
            //     streams[gpuIndex],
            //     (CUdeviceptr)gpuArrays + gpuIndex * arraysPerGPU * BATCH_SIZE,
            //     0, CU_MEM_ATTACH_GLOBAL));

            CUmodule module;
            checkError(
                cuModuleLoad(&module,
                             "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                             "stress_kernel.ptx"));
            CUfunction sortingKernel;
            checkError(cuModuleGetFunction(&sortingKernel, module, "_Z9quickSortPiii"));
            void* sortingArgs[] = {&gpuArrays, &gpuResults};
            checkError(cuLaunchKernel(sortingKernel, arraysPerGPU, 1, 1, BATCH_SIZE, 1, 1,
                           0, streams[gpuIndex], sortingArgs, nullptr));

            CUfunction searchingKernel;
            checkError(cuModuleGetFunction(&searchingKernel, module,
                                           "_Z12binarySearchPKiiPii"));
            void* searchingArgs[] = {&gpuArrays, &gpuResults};
            checkError(cuLaunchKernel(searchingKernel, arraysPerGPU, 1, 1, BATCH_SIZE, 1,
                           1, 0, streams[gpuIndex], searchingArgs, nullptr));
        }

        for (int gpuIndex = 0; gpuIndex < numGPUs; ++gpuIndex) {
            checkError(cuCtxSetCurrent(contexts[gpuIndex]));
            checkError(cuStreamSynchronize(streams[gpuIndex]));
        }

        auto gpuEndTime = std::chrono::steady_clock::now();
        auto gpuElapsedTime =
            std::chrono::duration_cast<std::chrono::milliseconds>(gpuEndTime -
                                                                  gpuStartTime)
                .count();

        auto cpuStartTime = std::chrono::steady_clock::now();

        for (int batchIndex = 0; batchIndex < BATCH_SIZE; ++batchIndex) {
            std::sort(cpuArrays.begin() + batchIndex * ARRAY_SIZE,
                      cpuArrays.begin() + (batchIndex + 1) * ARRAY_SIZE);

            for (int targetIndex = 0; targetIndex < TARGET_COUNT;
                 ++targetIndex) {
                int target = distribution(generator);
                int* arrayBegin = cpuArrays.data() + batchIndex * ARRAY_SIZE;
                int* arrayEnd = arrayBegin + ARRAY_SIZE;
                int* result =
                    cpuResults.data() + batchIndex * TARGET_COUNT + targetIndex;
                *result = std::binary_search(arrayBegin, arrayEnd, target)
                              ? target
                              : -1;
            }
        }

        auto cpuEndTime = std::chrono::steady_clock::now();
        auto cpuElapsedTime =
            std::chrono::duration_cast<std::chrono::milliseconds>(cpuEndTime -
                                                                  cpuStartTime)
                .count();

        checkError(cuMemFree((CUdeviceptr)gpuArrays));
        checkError(cuMemFree((CUdeviceptr)gpuResults));

        for (int i = 0; i < TARGET_COUNT * BATCH_SIZE; ++i) {
            EXPECT_EQ(cpuResults[i], cpuResults[i]);
        }

        std::cout << "GPU Execution Time: " << gpuElapsedTime << " ms\n";
        std::cout << "CPU Execution Time: " << cpuElapsedTime << " ms\n";
        std::cout << "GPU Memory Usage: "
                  << ARRAY_SIZE * BATCH_SIZE * sizeof(int) / (1024 * 1024)
                  << " MB\n";
        std::cout << "CPU Memory Usage: "
                  << ARRAY_SIZE * BATCH_SIZE * sizeof(int) / (1024 * 1024)
                  << " MB\n";
        
        for (int i = 0; i < numGPUs; ++i) {
            checkError(cuStreamDestroy(streams[i]));
            checkError(cuCtxDestroy(contexts[i]));
        }
        delete[] streams;
    }
};

void RunSortingSearchTestCyclic(int numIterations) {
    for (int i = 0; i < numIterations; ++i) {
        SortingSearchTest test;
        test.RunSortingSearchTest();
    }
}

TEST(MyTestSuite, SortingSearchCyclicTest) {
    const int numIterations = 1;
    RunSortingSearchTestCyclic(numIterations);
}
