#include "test_macros.h"
#include "test_utils.h"

class cuModuleTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cuInit(0);
        cuDeviceGetCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP();
        }
        cuDeviceGet(&device, 0);
        cuCtxCreate(&context, 0, device);
        cuModuleLoad(&module, fname);
        // cuModuleLoad(&module_sm80, fname_sm80);
    }

    void TearDown() override {
        cuModuleUnload(module);
        cuCtxDestroy(context);
    }

    int device_count;
    CUdevice device;
    CUcontext context;
    CUmodule module, module_sm80, module_fatbin;
    const char* fname_sm80 =
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel_sm_80.ptx";
    const char* fatbin_path =
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.fatbin";
    const char* fname =
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx";
};
