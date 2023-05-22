
#include "test_macros.h"
#include "test_utils.h"

class CuMemTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cuInit(0);
        cuDeviceGet(&device, 0);
        cuCtxCreate(&context, 0, device);
    }

    void TearDown() override { cuCtxDestroy(context); }

    CUcontext context;
    CUresult res;
    CUdevice device;
};
