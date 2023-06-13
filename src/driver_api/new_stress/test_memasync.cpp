#include "test_utils.h"

void test_mem_async_case1(int d) {
    SDdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, d));

    SDcontext cuContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));

    char* h_val;
    cuMemAllocHost((void**)&h_val, sizeof(char));
    *h_val = 'a';

    char* d_val;
    cuMemAlloc((SDdeviceptr*)&d_val, sizeof(char));

    SDstream stream;
    cuStreamCreate(&stream,0);

    cuMemcpyHtoDAsync((SDdeviceptr)d_val, h_val, sizeof(char), stream);
    cuStreamSynchronize(stream);

    char* h_result;
    cuMemAllocHost((void**)&h_result, sizeof(char));
    cuMemcpyDtoHAsync(h_result, (SDdeviceptr)d_val, sizeof(char), stream);
    cuStreamSynchronize(stream);

    assert(*h_result == 'a');

    cuMemFreeHost(h_val);
    cuMemFreeHost(h_result);
    cuMemFree((SDdeviceptr)d_val);
    cuStreamDestroy(stream);
    checkError(cuCtxDestroy(cuContext));
}

void test_mem_async_case2(int d) {
    SDdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, d));

    SDcontext cuContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));

    int size = 1 << 30;  // 1GB
    char* h_vals;
    cuMemAllocHost((void**)&h_vals, size);
    memset(h_vals, 'b', size);

    char* d_vals;
    cuMemAlloc((SDdeviceptr*)&d_vals, size);

    SDstream stream;
    cuStreamCreate(&stream,0);

    cuMemcpyHtoDAsync((SDdeviceptr)d_vals, h_vals, size, stream);
    cuStreamSynchronize(stream);

    char* h_result;
    cuMemAllocHost((void**)&h_result, size);
    cuMemcpyDtoHAsync(h_result, (SDdeviceptr)d_vals, size, stream);
    cuStreamSynchronize(stream);

    for (int i = 0; i < size; ++i) {
        assert(h_result[i] == 'b');
    }

    cuMemFreeHost(h_vals);
    cuMemFreeHost(h_result);
    cuMemFree((SDdeviceptr)d_vals);
    cuStreamDestroy(stream);
    checkError(cuCtxDestroy(cuContext));
}

void test_mem_async_case3(int d) {
    SDdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, d));

    SDcontext cuContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));

    int size = 123;  // not a multiple of 8
    char* h_vals;
    cuMemAllocHost((void**)&h_vals, size);
    memset(h_vals, 'c', size);

    char* d_vals;
    cuMemAlloc((SDdeviceptr*)&d_vals, size);

    SDstream stream;
    cuStreamCreate(&stream,0);

    cuMemcpyHtoDAsync((SDdeviceptr)d_vals, h_vals, size, stream);
    cuStreamSynchronize(stream);

    char* h_result;
    cuMemAllocHost((void**)&h_result, size);
    cuMemcpyDtoHAsync(h_result, (SDdeviceptr)d_vals, size, stream);
    cuStreamSynchronize(stream);

    for (int i = 0; i < size; ++i) {
        assert(h_result[i] == 'c');
    }

    cuMemFreeHost(h_vals);
    cuMemFreeHost(h_result);
    cuMemFree((SDdeviceptr)d_vals);
    cuStreamDestroy(stream);
    checkError(cuCtxDestroy(cuContext));
}

void test_mem_async_case4(int d) {
    SDdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, d));

    SDcontext cuContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));

    int chunkSize = 100;
    int numChunks = 10000;

    char* h_vals;
    cuMemAllocHost((void**)&h_vals, chunkSize);
    memset(h_vals, 'd', chunkSize);

    char* d_vals;
    cuMemAlloc((SDdeviceptr*)&d_vals, chunkSize);

    char* h_result;
    cuMemAllocHost((void**)&h_result, chunkSize);

    SDstream stream;
    cuStreamCreate(&stream,0);

    for (int i = 0; i < numChunks; ++i) {
        cuMemcpyHtoDAsync((SDdeviceptr)d_vals, h_vals, chunkSize, stream);
        cuMemcpyDtoHAsync(h_result, (SDdeviceptr)d_vals, chunkSize, stream);
        cuStreamSynchronize(stream);

        for (int j = 0; j < chunkSize; ++j) {
            assert(h_result[j] == 'd');
        }
    }

    cuMemFreeHost(h_vals);
    cuMemFreeHost(h_result);
    cuMemFree((SDdeviceptr)d_vals);
    cuStreamDestroy(stream);
    checkError(cuCtxDestroy(cuContext));
}

TEST(STRESS_NEW, memasync) {
    int dev_count;
    checkError(cuInit(0));
    checkError(cuDeviceGetCount(&dev_count));

    for (int d = 0; d < dev_count; d++) {
        test_mem_async_case1(d);
        test_mem_async_case2(d);
        test_mem_async_case3(d);
        test_mem_async_case4(d);
    }
}