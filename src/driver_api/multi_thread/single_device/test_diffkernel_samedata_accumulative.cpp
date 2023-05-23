#include <cuda.h>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

#define CHECK_CUDA(op) __check_cuda_error((op), #op, __FILE__, __LINE__)

bool __check_cuda_error(CUresult code,
                        const char* op,
                        const char* file,
                        int line) {
    if (code != CUresult::CUDA_SUCCESS) {
        const char* err_name = nullptr;
        const char* err_message = nullptr;
        cuGetErrorName(code, &err_name);
        cuGetErrorString(code, &err_message);
        printf("%s:%d %s failed. \n code = %s, message = %s\n", file, line, op,
               err_name, err_message);
        return false;
    }
    return true;
}


void performCUDAOperations(int threadId,
                           CUcontext context,
                           CUdeviceptr d_data,
                           int size,
                           CUfunction kernel) {
    
    CHECK_CUDA(cuCtxSetCurrent(context));

    CUevent event1, event2;

    CHECK_CUDA(cuEventCreate(&event1, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&event2, CU_EVENT_DEFAULT));

    void* args[] = {&d_data, &size};
    CHECK_CUDA(cuEventRecord(event1, 0));
    CHECK_CUDA(cuLaunchKernel(kernel, (size + 255) / 256, 1, 1, 256, 1, 1, 0, 0,
                              args, nullptr));
    CHECK_CUDA(cuEventRecord(event2, 0));

    CHECK_CUDA(cuEventSynchronize(event2));

    std::cout << "Thread " << threadId << " finished" << std::endl;

    CHECK_CUDA(cuEventDestroy(event1));
    CHECK_CUDA(cuEventDestroy(event2));
}

bool verifyResult(const std::vector<float>& data, float expectedValue) {
    bool flag = true;
    for (int i = 0; i < data.size(); i++) {
        if (std::abs(data[i] - expectedValue) > 1e-5) {
            std::cout << "error index: " << i << "expect " << data[i] << std::endl;
            flag =  false;
        }
    }
    return flag;
}

int main() {
    const int dataSize = 30;
    const int numThreads = 5;

    CHECK_CUDA(cuInit(0));
    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));

    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    CUmodule module;
    CHECK_CUDA(cuModuleLoad(
        &module,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx"));

    CUfunction vec_multiply_2, vec_add_3, vec_sub_1;
    CHECK_CUDA(
        cuModuleGetFunction(&vec_multiply_2, module, "_Z14vec_multiply_2Pfi"));
    CHECK_CUDA(cuModuleGetFunction(&vec_add_3, module, "_Z9vec_add_3Pfi"));
    CHECK_CUDA(cuModuleGetFunction(&vec_sub_1, module, "_Z9vec_sub_1Pfi"));

    CUdeviceptr d_data;
    CHECK_CUDA(cuMemAlloc(&d_data, dataSize * sizeof(float)));

    std::vector<float> h_data(dataSize, 1.0f);

    CHECK_CUDA(cuMemcpyHtoD(d_data, h_data.data(), dataSize * sizeof(float)));

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

    CHECK_CUDA(cuMemcpyDtoH(h_data.data(), d_data, dataSize * sizeof(float)));
    // std::cout << "h_data size: " << h_data.size() << std::endl;

    bool resultIsValid = verifyResult(
        h_data, 2.0f);

    std::cout << "Result is " << (resultIsValid ? "valid" : "invalid")
              << std::endl;

    CHECK_CUDA(cuMemFree(d_data));
    CHECK_CUDA(cuModuleUnload(module));
    CHECK_CUDA(cuCtxDestroy(context));

    return 0;
}
