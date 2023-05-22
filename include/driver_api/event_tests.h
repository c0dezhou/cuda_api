#include "test_macros.h"
#include "test_utils.h"

class CuEventTest : public ::testing::Test {
   protected:
    virtual void SetUp() {
        cuInit(0);
        cuDeviceGet(&cuDevice, 0);
        cuCtxCreate(&cuContext, 0, cuDevice);
        cuEventCreate(&event, 0);
        cuEventCreate(&start_, 0);
        cuEventCreate(&end_, 0);

        CUresult res = cuModuleLoad(&module, fname);
        EXPECT_EQ(res, CUDA_SUCCESS);
        res = cuModuleGetFunction(&func_vector_add, module, "_Z6vecAddPfS_S_");
        EXPECT_EQ(res, CUDA_SUCCESS);
        res = cuModuleGetFunction(&func_add, module, "_Z3addiiPi");
        EXPECT_EQ(res, CUDA_SUCCESS);
        res = cuModuleGetFunction(&func_add_with_delay, module,
                                  "_Z9addKernelPiPKiS1_");
        EXPECT_EQ(res, CUDA_SUCCESS);
        res = cuModuleGetFunction(&func_delay, module, "_Z12delay_devicef");
        EXPECT_EQ(res, CUDA_SUCCESS);
    }

    virtual void TearDown() {
        cuCtxDestroy(cuContext);
        cuEventDestroy(event);
        cuEventDestroy(start_);
        cuEventDestroy(end_);
    }

    static void createAndDestroyEvent(CUevent* event) {
        cuEventCreate(event, CU_EVENT_DEFAULT);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        cuEventDestroy(*event);
    }

    static void destroyEvent(CUevent* event) {
        // std::this_thread::sleep_for(std::chrono::milliseconds(500));
        cuEventDestroy(*event);
    }

    static void waitForEvent(CUevent* event) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        CUresult result = cuEventSynchronize(*event);

        if (result != CUDA_SUCCESS) {
            const char *errorName, *errorString;
            cuGetErrorName(result, &errorName);
            cuGetErrorString(result, &errorString);
            FAIL() << "Error during event synchronization: " << errorName
                   << ": " << errorString;
        }
    }

    CUdevice cuDevice;
    CUcontext cuContext;
    CUevent event, start_, end_;
    CUresult result;
    CUmodule module;
    CUfunction func_vector_add, func_add, func_delay, func_add_with_delay;
    const char* fname =
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx";
};