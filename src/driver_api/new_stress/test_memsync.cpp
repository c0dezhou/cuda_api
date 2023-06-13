#include "test_utils.h"

void test_mem_case1(int d) {
    CUdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, d));

    CUcontext cuContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));

    char h_val = 'a';
    char* d_val;
    cuMemAlloc((CUdeviceptr*)&d_val, sizeof(char));

    cuMemcpyHtoD((CUdeviceptr)d_val, &h_val, sizeof(char));

    char h_result;
    cuMemcpyDtoH(&h_result, (CUdeviceptr)d_val, sizeof(char));
    assert(h_result == 'a');

    cuMemFree((CUdeviceptr)d_val);
    checkError(cuCtxDestroy(cuContext));
}

void test_mem_case2(int d) {
    CUdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, d));

    CUcontext cuContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));

    int size = 1 << 30;  // 1GB
    char* h_vals = new char[size];
    memset(h_vals, 'b', size);

    char* d_vals;
    cuMemAlloc((CUdeviceptr*)&d_vals, size);

    cuMemcpyHtoD((CUdeviceptr)d_vals, h_vals, size);

    char* h_result = new char[size];
    cuMemcpyDtoH(h_result, (CUdeviceptr)d_vals, size);
    for (int i = 0; i < size; ++i) {
        assert(h_result[i] == 'b');
    }

    delete[] h_vals;
    delete[] h_result;
    cuMemFree((CUdeviceptr)d_vals);
    checkError(cuCtxDestroy(cuContext));
}

void test_mem_case3(int d) {
    CUdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, d));

    CUcontext cuContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));

    int size = 123;  // not a multiple of 8
    char* h_vals = new char[size];
    memset(h_vals, 'c', size);

    char* d_vals;
    cuMemAlloc((CUdeviceptr*)&d_vals, size);

    cuMemcpyHtoD((CUdeviceptr)d_vals, h_vals, size);

    char* h_result = new char[size];
    cuMemcpyDtoH(h_result, (CUdeviceptr)d_vals, size);
    for (int i = 0; i < size; ++i) {
        assert(h_result[i] == 'c');
    }

    delete[] h_vals;
    delete[] h_result;
    cuMemFree((CUdeviceptr)d_vals);
    checkError(cuCtxDestroy(cuContext));
}

void test_mem_case4(int d) {
    CUdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, d));

    CUcontext cuContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));

    int chunkSize = 100;
    int numChunks = 10000;
    char* h_vals = new char[chunkSize];
    memset(h_vals, 'd', chunkSize);

    char* d_vals;
    cuMemAlloc((CUdeviceptr*)&d_vals, chunkSize);

    for (int i = 0; i < numChunks; ++i) {
        cuMemcpyHtoD((CUdeviceptr)d_vals, h_vals, chunkSize);
        cuMemcpyDtoH(h_vals, (CUdeviceptr)d_vals, chunkSize);
    }

    for (int i = 0; i < chunkSize; ++i) {
        assert(h_vals[i] == 'd');
    }

    delete[] h_vals;
    cuMemFree((CUdeviceptr)d_vals);
    checkError(cuCtxDestroy(cuContext));
}

TEST(STRESS_NEW, memsync) {
    int dev_count;
    checkError(cuInit(0));
    checkError(cuDeviceGetCount(&dev_count));

    for (int d = 0; d < dev_count; d++) {
        test_mem_case1(d);
        test_mem_case2(d);
        test_mem_case3(d);
        test_mem_case4(d);
    }
}