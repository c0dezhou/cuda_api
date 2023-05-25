#include "test_utils.h"

void performCUDAOperations(int threadId,
                           CUcontext context,
                           CUdeviceptr d_data,
                           int size,
                           CUfunction kernel) {
    checkError(cuCtxSetCurrent(context));

    CUevent event1, event2;

    checkError(cuEventCreate(&event1, CU_EVENT_DEFAULT));
    checkError(cuEventCreate(&event2, CU_EVENT_DEFAULT));

    void* args[] = {&d_data, &size};
    checkError(cuEventRecord(event1, 0));
    checkError(cuLaunchKernel(kernel, (size + 255) / 256, 1, 1, 256, 1, 1, 0, 0,
                              args, nullptr));
    checkError(cuEventRecord(event2, 0));

    checkError(cuEventSynchronize(event2));

    std::cout << "Thread " << threadId << " finished" << std::endl;

    checkError(cuEventDestroy(event1));
    checkError(cuEventDestroy(event2));
}

TEST(MthsTest_, MTH_Single_Device_diffkernel_samedata_accumulative) {
    const int dataSize = 30;
    const int numThreads = 5;

    checkError(cuInit(0));
    CUdevice device;
    checkError(cuDeviceGet(&device, 0));

    CUcontext context;
    checkError(cuCtxCreate(&context, 0, device));

    CUmodule module;
    checkError(cuModuleLoad(
        &module,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx"));

    CUfunction vec_multiply_2, vec_add_3, vec_sub_1;
    checkError(
        cuModuleGetFunction(&vec_multiply_2, module, "_Z14vec_multiply_2Pfi"));
    checkError(cuModuleGetFunction(&vec_add_3, module, "_Z9vec_add_3Pfi"));
    checkError(cuModuleGetFunction(&vec_sub_1, module, "_Z9vec_sub_1Pfi"));

    CUdeviceptr d_data;
    checkError(cuMemAlloc(&d_data, dataSize * sizeof(float)));

    std::vector<float> h_data(dataSize, 1.0f);

    checkError(cuMemcpyHtoD(d_data, h_data.data(), dataSize * sizeof(float)));

    std::vector<std::thread> threads(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        CUfunction kernel;
        if (i == 0)
            kernel = vec_multiply_2;
        else if (i == 1)
            kernel = vec_add_3;
        else
            kernel = vec_sub_1;
        threads[i] = std::thread(performCUDAOperations, i, context, d_data,
                                 dataSize, kernel);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    checkError(cuMemcpyDtoH(h_data.data(), d_data, dataSize * sizeof(float)));
    // std::cout << "h_data size: " << h_data.size() << std::endl;

    bool resultIsValid = verifyResult(h_data, 2.0f);

    std::cout << "Result is " << (resultIsValid ? "valid" : "invalid")
              << std::endl;

    checkError(cuMemFree(d_data));
    checkError(cuModuleUnload(module));
    checkError(cuCtxDestroy(context));
}
