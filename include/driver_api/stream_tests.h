#include "test_utils.h"

class CuStreamTests : public ::testing::Test
{
protected:
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;
    CUstream cuStream;

    void SetUp() override
    {
        cuInit(0);
        cuDeviceGet(&cuDevice, 0);
        cuCtxCreate(&cuContext, 0, cuDevice);
        cuStreamCreate(&cuStream, 0);
        cuModuleLoad(&cuModule,
                     "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                     "cuda_kernel_sm_80.ptx");
        cuModuleGetFunction(&cuFunction, cuModule, "_Z3addiiPi");
    }

    void TearDown() override
    {
        cuModuleUnload(cuModule);
        cuStreamDestroy(cuStream);
        cuCtxDestroy(cuContext);
    }
};
