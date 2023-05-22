#include "loop_common.h"

TEST(LOOPSINGLE,Stream) {
    const int loop_times = 500;
    CUdevice device;
    CUcontext context;
    LOOP(cuInit(0), 1);
    LOOP(cuDeviceGet(&device, 0), 1);
    LOOP(cuCtxCreate(&context, 0, device), 1);

    // TODO

    cuCtxDestroy(context);

}