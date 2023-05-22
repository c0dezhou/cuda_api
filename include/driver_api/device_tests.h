#include "test_utils.h"

class CuDeviceTest : public ::testing::Test {
   protected:
    void SetUp() override {
        res = cuInit(0);
        res = cuDeviceGetCount(&device_count);
        res = cuDeviceGet(&device, 0);
        res = cuCtxCreate(&context, 0, device);
    }

    void TearDown() override { CUresult res = cuCtxDestroy(context); }

    CUdevice device;
    CUcontext context;
    CUresult res;
    int device_count;
};