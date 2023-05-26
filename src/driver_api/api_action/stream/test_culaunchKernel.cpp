#include "stream_tests.h"

TEST_F(CuStreamTests, AC_BA_LaunchKernel_ValidInputParameters) {
    int gridDimX = 2, gridDimY = 2, gridDimZ = 1;
    int blockDimX = 4, blockDimY = 4, blockDimZ = 1;
    int sharedMemBytes = 64;

    int result = 0;
    int* d_result;
    cuMemAlloc((CUdeviceptr*)&d_result, sizeof(int));

    int input_a = 5;
    int input_b = 7;

    void* args[] = {&input_a, &input_b, &d_result};
    cuLaunchKernel(cuFunction, gridDimX, gridDimY, gridDimZ, blockDimX,
                   blockDimY, blockDimZ, sharedMemBytes, cuStream, args,
                   nullptr);

    EXPECT_EQ(result, CUDA_SUCCESS);
}

TEST_F(CuStreamTests, AC_EG_LaunchKernel_ExceedMaximumThreads) {
    int maxThreadsPerBlock;
    cuDeviceGetAttribute(&maxThreadsPerBlock,
                         CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cuDevice);

    int gridDimX = 2, gridDimY = 2, gridDimZ = 1;
    int blockDimX = maxThreadsPerBlock + 1, blockDimY = 4, blockDimZ = 1;
    int sharedMemBytes = 64;

    int* d_result;
    cuMemAlloc((CUdeviceptr*)&d_result, sizeof(int));

    int input_a = 5;
    int input_b = 7;

    void* args[] = {&input_a, &input_b, &d_result};
    CUresult result = cuLaunchKernel(cuFunction, gridDimX, gridDimY, gridDimZ,
                                     blockDimX, blockDimY, blockDimZ,
                                     sharedMemBytes, cuStream, args, nullptr);

    EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE);
}

TEST_F(CuStreamTests, AC_EG_LaunchKernel_ExceedMaximumSharedMemory) {
    int maxSharedMemoryPerBlock;
    cuDeviceGetAttribute(&maxSharedMemoryPerBlock,
                         CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                         cuDevice);

    int gridDimX = 2, gridDimY = 2, gridDimZ = 1;
    int blockDimX = 4, blockDimY = 4, blockDimZ = 1;
    int sharedMemBytes = maxSharedMemoryPerBlock + 1;

    int* d_result;
    cuMemAlloc((CUdeviceptr*)&d_result, sizeof(int));

    int input_a = 5;
    int input_b = 7;

    void* args[] = {&input_a, &input_b, &d_result};
    CUresult result = cuLaunchKernel(cuFunction, gridDimX, gridDimY, gridDimZ,
                                     blockDimX, blockDimY, blockDimZ,
                                     sharedMemBytes, cuStream, args, nullptr);

    EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE);
}
