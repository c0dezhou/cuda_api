#include "loop_common.h"

CUresult create_and_destroy_context(CUdevice device) {
    CUcontext context;
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));
    checkError(cuCtxDestroy(context));
    return CUDA_SUCCESS;
}

CUresult set_and_get_context(CUdevice device) {
    CUcontext context, current;
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));

    checkError(cuCtxSetCurrent(context));
    checkError(cuCtxGetCurrent(&current));
    EXPECT_EQ(context, current);

    checkError(cuCtxDestroy(context));
    return CUDA_SUCCESS;
}

TEST(LOOPSINGLE,Device) {
    const int loop_times = 1000;
    CUdevice device;
    CUcontext context;
    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    // checkError(cuCtxCreate(&context, 0, device));

    // auto cuDeviceGetFunc = makeFuncPtr(cuDeviceGet);
    auto cuDeviceGetCountFunc = makeFuncPtr(cuDeviceGetCount);
    auto cuDeviceGetAttributeFunc = makeFuncPtr(cuDeviceGetAttribute);
    auto create_and_destroy_contextFunc =
        makeFuncPtr(create_and_destroy_context);
    auto set_and_get_contextFunc = makeFuncPtr(set_and_get_context);

    // auto cuDeviceGetParams = [&device]() {
    //     int count, rand_num;
    //     cuDeviceGetCount(&count);
    //     get_random(&rand_num, 0, count-1);
    //     return std::make_tuple(&device, rand_num);
    // };
    auto cuDeviceGetCountParams = []() {
        int count;
        return std::make_tuple(&count);
    };
    auto create_and_destroy_contextParams = [&device]() {
        // CUcontext context;
        return std::make_tuple(device);
    };
    auto set_and_get_contextParams = [&device]() {
        return std::make_tuple(device);
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

    // PRINT_FUNCNAME(loopFuncPtr(loop_times, cuDeviceGetFunc, cuDeviceGetParams()));
    PRINT_FUNCNAME(loopFuncPtr(loop_times, cuDeviceGetCountFunc,
                              cuDeviceGetCountParams()));
    // PRINT_FUNCNAME(loopFuncPtr(loop_times, cuDeviceGetAttributeFunc,
    //                            cuDeviceGetAttributeParams()));
    PRINT_FUNCNAME(loopFuncPtr(loop_times, create_and_destroy_contextFunc,
                               create_and_destroy_contextParams()));
    PRINT_FUNCNAME(loopFuncPtr(loop_times, set_and_get_contextFunc,
                               set_and_get_contextParams()));

    // cuCtxDestroy(context);

}