#include "loop_common.h"

TEST(LOOPSINGLE,Device) {
    const int loop_times = 10000;
    CUdevice device;
    CUcontext context;
    LOOP(cuInit(0), 1);
    LOOP(cuDeviceGet(&device, 0), 1);
    LOOP(cuCtxCreate(&context, 0, device), 1);

    auto cuDeviceGetFunc = makeFuncPtr(cuDeviceGet);
    auto cuDeviceGetCountFunc = makeFuncPtr(cuDeviceGetCount);
    auto cuDeviceGetAttributeFunc = makeFuncPtr(cuDeviceGetAttribute);
    auto cuCtxSetCurrentFunc = makeFuncPtr(cuCtxSetCurrent);
    auto cuCtxGetCurrentFunc = makeFuncPtr(cuCtxGetCurrent);

    auto cuDeviceGetParams = [&device]() {
        int count, rand_num;
        cuDeviceGetCount(&count);
        get_random(&rand_num, 0, count-1);
        return std::make_tuple(&device, rand_num);
    };
    auto cuDeviceGetCountParams = []() {
        int count;
        return std::make_tuple(&count);
    };
    auto cuCtxSetCurrentParams = [&context]() {
        // CUcontext context;
        return std::make_tuple(context);
    };
    auto cuCtxGetCurrentParams = []() {
        CUcontext context;
        return std::make_tuple(&context);
    };
    auto cuDeviceGetAttributeParams = [&device]() {
        int pi;
        int rand_num;
        get_random(&rand_num, 1, 119);
        std::cout << rand_num <<std::endl;
        return std::make_tuple(&pi, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                               device);
        // return std::make_tuple(&pi, (CUdevice_attribute)rand_num,
        //                        device);
    };

    PRINT_FUNCNAME(loopFuncPtr(loop_times, cuDeviceGetFunc, cuDeviceGetParams()));
    PRINT_FUNCNAME(loopFuncPtr(loop_times, cuDeviceGetCountFunc,
                              cuDeviceGetCountParams()));
    // PRINT_FUNCNAME(loopFuncPtr(loop_times, cuDeviceGetAttributeFunc,
    //                            cuDeviceGetAttributeParams()));
    PRINT_FUNCNAME(
        loopFuncPtr(loop_times, cuCtxSetCurrentFunc, cuCtxSetCurrentParams()));
    PRINT_FUNCNAME(
        loopFuncPtr(loop_times, cuCtxGetCurrentFunc, cuCtxGetCurrentParams()));

    cuCtxDestroy(context);

}