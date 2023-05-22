#include "test_utils.h"

class MthsTest : public ::testing::Test {
   protected:
    void SetUp() override {
        checkError(cuInit(0));
        checkError(cuDeviceGet(&device, 0));
        checkError(cuCtxCreate(&context, 0, device));
        checkError(
            cuModuleLoad(&module,
                         "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                         "cuda_kernel.ptx"));

        checkError(cuModuleGetFunction(&vecAdd, module, "_Z6vecAddPfS_S_"));
        checkError(cuModuleGetFunction(&vecScal, module, "_Z8vecScalePff"));
        checkError(
            cuModuleGetFunction(&add_kernel, module, "_Z10add_kernelPfS_S_i"));

        size_t size = sizeof(int) * N;
        checkError(cuMemAlloc(&d_A, size));
        checkError(cuMemAlloc(&d_B, size));
        checkError(cuMemAlloc(&d_C, size));
    }

    void TearDown() override {
        checkError(cuMemFree(d_A));
        checkError(cuMemFree(d_B));
        checkError(cuMemFree(d_C));

        checkError(cuModuleUnload(module));
        checkError(cuCtxDestroy(context));
    }

    int N = 10;
    CUdevice device = 0;
    CUmodule module;
    CUcontext context;
    CUstream stream;
    CUevent event;
    CUfunction vecAdd, vecScal, add_kernel;

    CUdeviceptr d_A;
    CUdeviceptr d_B;
    CUdeviceptr d_C;

};