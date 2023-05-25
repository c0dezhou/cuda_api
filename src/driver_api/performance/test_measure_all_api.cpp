#include "test_utils.h"

#define TIME_CUDA_API_CALL(api_call) do { \
    auto start = std::chrono::high_resolution_clock::now(); \
    api_call; \
    auto end = std::chrono::high_resolution_clock::now(); \
    std::chrono::duration<double, std::milli> diff = end - start; \
    std::cout << #api_call << " took " << diff.count() << " ms" << std::endl; \
} while (0)

TEST(PERF, measure_all_api) {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr devPtr;
    void* hostPtr;
    size_t size = 1024 * 1024;  // 1 MiB

    TIME_CUDA_API_CALL(cuInit(0));
    TIME_CUDA_API_CALL(cuDeviceGet(&dev, 0));
    TIME_CUDA_API_CALL(cuCtxCreate(&ctx, 0, dev));

    TIME_CUDA_API_CALL(cuMemAlloc(&devPtr, size));
    TIME_CUDA_API_CALL(cuMemAllocHost(&hostPtr, size));

    TIME_CUDA_API_CALL(cuMemcpyHtoD(devPtr, hostPtr, size));

    TIME_CUDA_API_CALL(cuMemcpyDtoH(hostPtr, devPtr, size));

    TIME_CUDA_API_CALL(cuMemFreeHost(hostPtr));
    TIME_CUDA_API_CALL(cuMemFree(devPtr));
    TIME_CUDA_API_CALL(cuCtxDestroy(ctx));

}
