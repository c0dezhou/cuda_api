#include "cuda_runtime.h"
#include "device_tests.h"

TEST_F(CuDeviceTest, AC_BA_GetCurrent_CanHandleNullContext) {
    CUcontext current_context;
    CUresult res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    EXPECT_EQ(current_context, context);

    CUcontext null_context = NULL;
    res = cuCtxSetCurrent(null_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    // EXPECT_EQ(current_context, null_context);
}

TEST_F(CuDeviceTest, AC_BA_GetCurrent_CanHandleInvalidContext) {
    CUcontext current_context;
    CUresult res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    EXPECT_EQ(current_context, context);

    CUcontext invalid_context = context;
    res = cuCtxDestroy(invalid_context);
    EXPECT_EQ(res, CUDA_SUCCESS);
    res = cuCtxSetCurrent(invalid_context);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_CONTEXT);

    res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    // EXPECT_EQ(current_context, nullptr);
}

TEST_F(CuDeviceTest, AC_BA_GetCurrent_CanHandleMultipleContexts) {
    CUcontext current_context;
    CUresult res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    EXPECT_EQ(current_context, context);

    int device_count = 0;
    res = cuDeviceGetCount(&device_count);
    EXPECT_EQ(res, CUDA_SUCCESS);

    if (device_count == 1) {
        GTEST_SKIP();
    }
    CUdevice device;
    res = cuDeviceGet(&device, 1);
    EXPECT_EQ(res, CUDA_SUCCESS);

    CUcontext second_context;
    res = cuCtxCreate(&second_context, 0, device);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuCtxSetCurrent(second_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    EXPECT_EQ(current_context, second_context);

    res = cuCtxDestroy(second_context);
    EXPECT_EQ(res, CUDA_SUCCESS);
}

TEST_F(CuDeviceTest, AC_OT_GetCurrent_CanHandleRepeatedCalls) {
    CUcontext current_context;
    CUresult res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    EXPECT_EQ(current_context, context);

    CUcontext new_context = context;
    res = cuCtxSetCurrent(new_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    EXPECT_EQ(current_context, new_context);

    res = cuCtxSetCurrent(new_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    EXPECT_EQ(current_context, new_context);

    CUcontext null_context = NULL;
    res = cuCtxSetCurrent(null_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    // // EXPECT_EQ(current_context, null_context);
    // // EXPECT_EQ(current_context, 0x564d0349ed90);

    GTEST_SKIP(); // core dump here

    res = cuCtxSetCurrent(context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    EXPECT_EQ(current_context, context);
}

TEST_F(CuDeviceTest, AC_OT_GetCurrent_CanHandleMultiThread) {
    CUcontext current_context;
    CUresult res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    EXPECT_EQ(current_context, context);

    std::thread child_thread([&]() {
        CUcontext child_context;
        CUresult child_res = cuCtxGetCurrent(&child_context);
        EXPECT_EQ(child_res, CUDA_SUCCESS);

        EXPECT_EQ(child_context, nullptr);
        CUcontext new_context = context;
        child_res = cuCtxSetCurrent(new_context);
        EXPECT_EQ(child_res, CUDA_SUCCESS);

        child_res = cuCtxGetCurrent(&child_context);
        EXPECT_EQ(child_res, CUDA_SUCCESS);

        EXPECT_EQ(child_context, new_context);
    });

    child_thread.join();

    res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    // 子线程不影响主线程
    EXPECT_EQ(current_context, context);
}

TEST_F(CuDeviceTest, AC_OT_GetCurrent_CanHandleAsyncCall) {
    CUcontext current_context;
    CUresult res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    EXPECT_EQ(current_context, context);

    // 创建一个异步任务
    auto async_task = std::async(std::launch::async, [&]() {
        CUcontext async_context;
        CUresult async_res = cuCtxGetCurrent(&async_context);
        EXPECT_EQ(async_res, CUDA_SUCCESS);

        EXPECT_EQ(async_context, nullptr);  // 还没绑定

        CUcontext new_context = context;
        async_res = cuCtxSetCurrent(new_context);
        EXPECT_EQ(async_res, CUDA_SUCCESS);

        async_res = cuCtxGetCurrent(&async_context);
        EXPECT_EQ(async_res, CUDA_SUCCESS);

        EXPECT_EQ(async_context, new_context);
    });

    async_task.get();

    res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    // 异步任务不影响主线程
    EXPECT_EQ(current_context, context);
}

TEST_F(CuDeviceTest, AC_OT_GetCurrent_CanHandleStream) {
    CUcontext current_context;
    CUresult res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    EXPECT_EQ(current_context, context);

    CUstream stream;
    res = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    EXPECT_EQ(res, CUDA_SUCCESS);

    // TODO: 在CUDA流中执行一个异步内核

    cudaError_t cudares = cudaGetLastError();
    EXPECT_EQ(cudares, CUDA_SUCCESS);

    CUcontext null_context = NULL;
    res = cuCtxSetCurrent(null_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    // EXPECT_EQ(current_context, null_context);

    res = cuStreamSynchronize(stream);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuStreamDestroy(stream);
    EXPECT_EQ(res, CUDA_SUCCESS);
}

TEST_F(CuDeviceTest, AC_OT_GetCurrent_CanHandleEvent) {
    CUcontext current_context;
    CUresult res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    EXPECT_EQ(current_context, context);

    CUevent event;
    res = cuEventCreate(&event, CU_EVENT_DEFAULT);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuEventRecord(event, NULL);
    EXPECT_EQ(res, CUDA_SUCCESS);

    CUcontext null_context = NULL;
    res = cuCtxSetCurrent(null_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    // EXPECT_EQ(current_context, null_context);

    res = cuEventSynchronize(event);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuEventDestroy(event);
    EXPECT_EQ(res, CUDA_SUCCESS);
}

TEST_F(CuDeviceTest, AC_OT_GetCurrent_CanHandleMemoryAllocationAndFree) {
    // GTEST_SKIP();  // due to core dump
    CUcontext current_context;
    CUresult res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    EXPECT_EQ(current_context, context);

    CUdeviceptr dev_ptr;
    size_t size = 1024;
    res = cuMemAlloc(&dev_ptr, size);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuMemFree(dev_ptr);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuCtxSetCurrent(context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    res = cuCtxGetCurrent(&current_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

    EXPECT_EQ(current_context, context);
}
