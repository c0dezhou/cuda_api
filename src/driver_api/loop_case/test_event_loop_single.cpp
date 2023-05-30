#include "loop_common.h"

#define CREATE_EVENT(event, flag)  \
    get_random((int*)&flag, 0, 3); \
    cuEventCreate(&event, flag);

CUresult create_and_destroy_event(int non_use){
    CUevent event;
    unsigned int flag;
    get_random((int*)&flag, 0, 3);
    checkError(cuEventCreate(&event, flag));
    checkError(cuEventDestroy(event));

    return CUDA_SUCCESS;
}

CUresult event_elapsedtime_and_record_and_sync(CUstream stream) {
    CUevent start, end;
    checkError(cuEventCreate(&start, 0));
    checkError(cuEventCreate(&end, 0));
    unsigned int flag;
    float sec;
    checkError(cuEventRecord(start, stream));
    sleep(0.1);
    checkError(cuEventRecord(end, stream));
    checkError(cuEventSynchronize(end));
    checkError(cuEventDestroy(start));
    checkError(cuEventDestroy(end));

    return CUDA_SUCCESS;
}

TEST(LOOPSINGLE, Event) {
    const int loop_times = 10000;
    CUdevice device;
    CUcontext context;
    CUstream stream;
    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));
    checkError(cuStreamCreate(&stream, 0));

    auto create_and_destroy_eventFunc = makeFuncPtr(create_and_destroy_event);
    auto event_elapsedtime_and_record_and_syncFunc = makeFuncPtr(event_elapsedtime_and_record_and_sync);

    auto create_and_destroy_eventParams = []() { return std::make_tuple(1); };

    auto event_elapsedtime_and_record_and_syncParams = [&stream]() {
        return std::make_tuple(stream);
    };

    // TODO: launch kernel
    // auto cuEventElapsedTimeLKParams = [&stream]() {
    //     return std::make_tuple(&sec, start, end);
    // };

    // TODO: launch kernel
    // auto cuEventSynchronizeLKParams = []() {
    // };

    PRINT_FUNCNAME(loopFuncPtr(loop_times, create_and_destroy_eventFunc,
                               create_and_destroy_eventParams()));
    PRINT_FUNCNAME(
        loopFuncPtr(loop_times, event_elapsedtime_and_record_and_syncFunc,
                    event_elapsedtime_and_record_and_syncParams()));

    
    cuStreamDestroy(stream);
    cuCtxDestroy(context);
}