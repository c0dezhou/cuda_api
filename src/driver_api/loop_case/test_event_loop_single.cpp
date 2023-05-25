#include "loop_common.h"

#define CREATE_EVENT(event, flag)  \
    get_random((int*)&flag, 0, 3); \
    cuEventCreate(&event, flag);

TEST(LOOPSINGLE,Event) {
    const int loop_times = 100;
    CUdevice device;
    CUcontext context;
    CUstream stream;
    LOOP(cuInit(0), 1);
    LOOP(cuDeviceGet(&device, 0), 1);
    LOOP(cuCtxCreate(&context, 0, device), 1);
    LOOP(cuStreamCreate(&stream, 0),1);

    auto cuEventCreateFunc = makeFuncPtr(cuEventCreate);
    auto cuEventDestroyFunc = makeFuncPtr(cuEventDestroy);
    auto cuEventElapsedTimeFunc = makeFuncPtr(cuEventElapsedTime);
    auto cuEventQueryFunc = makeFuncPtr(cuEventQuery);
    auto cuEventSynchronizeFunc = makeFuncPtr(cuEventSynchronize);

    auto cuEventCreateParams = []() {
        CUevent event;
        unsigned int flag;
        get_random((int *)&flag, 0, 3);
        return std::make_tuple(&event, flag);
    };
    auto cuEventDestroyParams = []() {
        unsigned int flag;
        CUevent event;
        CREATE_EVENT(event, flag);
        return std::make_tuple(event);
    };

    CUevent start, end;
    checkError(cuEventCreate(&start, 0));
    checkError(cuEventCreate(&end, 0));
    auto cuEventElapsedTimeParams = [&stream,&start,&end]() {
        unsigned int flag;
        
        float sec;

        cuEventRecord(start, stream);
        sleep(1);
        cuEventRecord(end, stream);
        return std::make_tuple(&sec, start, end);

    };
    checkError(cuEventDestroy(start));
    checkError(cuEventDestroy(end));

    // TODO: launch kernel
    // auto cuEventElapsedTimeLKParams = [&stream]() {
    //     return std::make_tuple(&sec, start, end);
    // };

    auto cuEventQueryParams = [&stream]() {
        unsigned int flag;
        CUevent start;
        float sec;
        CREATE_EVENT(start, flag);
        cuEventRecord(start, stream);
        sleep(0.1);
        return std::make_tuple(start);
    };

    auto cuEventSynchronizeParams = []() {
        CUstream stream1;
        checkError(cuStreamCreate(&stream1, 0));
        unsigned int flag;
        CUevent start, end;
        float sec;
        CREATE_EVENT(start,flag);
        cuEventRecord(start, stream1);
        sleep(0.1);

        return std::make_tuple(start);
    };

    // TODO: launch kernel
    // auto cuEventSynchronizeLKParams = []() {
    // };

    // PRINT_FUNCNAME(
    //     loopFuncPtr(loop_times, cuEventCreateFunc, cuEventCreateParams()));
    PRINT_FUNCNAME(loopFuncPtr(loop_times, cuEventElapsedTimeFunc,
                               cuEventElapsedTimeParams()));
    // PRINT_FUNCNAME(
    //     loopFuncPtr(loop_times, cuEventQueryFunc, cuEventQueryParams()));
    // PRINT_FUNCNAME(loopFuncPtr(loop_times, cuEventSynchronizeFunc,
    //                            cuEventSynchronizeParams()));

    // PRINT_FUNCNAME(loopFuncPtr(loop_times, cuEventDestroyFunc,
    //                            cuEventSynchronizeParams()));

    
    cuStreamDestroy(stream);
    cuCtxDestroy(context);
}