#include "loop_common.h"


TEST(MTHSINGLE, MTH_Device) {
    const int loop_times = 10000;
    CUdevice device;
    CUcontext context;
    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));

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

    std::vector<std::thread> outer_ths;
    std::vector<std::thread> ths;

    ths.emplace_back([&cuDeviceGetFunc, &cuDeviceGetParams]() {
        loopFuncPtr(1, cuDeviceGetFunc, cuDeviceGetParams());
    });
    ths.emplace_back([&cuDeviceGetCountFunc, &cuDeviceGetCountParams]() {
        loopFuncPtr(1, cuDeviceGetCountFunc, cuDeviceGetCountParams());
    });
    ths.emplace_back([&cuCtxSetCurrentFunc, &cuCtxSetCurrentParams]() {
        loopFuncPtr(1, cuCtxSetCurrentFunc, cuCtxSetCurrentParams());
    });
    ths.emplace_back([&cuCtxGetCurrentFunc, &cuCtxGetCurrentParams]() {
        loopFuncPtr(1, cuCtxGetCurrentFunc, cuCtxGetCurrentParams());
    });

    for (int i = 0; i < 4; i++) {
        ths[i].join();
    }

    cuCtxDestroy(context);
}