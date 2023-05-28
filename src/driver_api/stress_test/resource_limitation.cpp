// Resource Limitation
// 旨在测试cuda driver api的资源限制，例如内存分配，线程数，流数，事件数等。我使用了一些随机的参数和循环

#include <cuda.h>
#include <gtest/gtest.h>

// A macro to check CUDA driver API functions
#define checkError(result) \
{ \
    if (result != CUDA_SUCCESS) { \
        fprintf(stderr, "CUDA driver API error %d at %s:%d\n", result, __FILE__, __LINE__); \
        FAIL(); \
    } \
}

// A stress test for cuda driver api memory allocation and deallocation
TEST(CudaDriverApiTest, MemoryAllocation) {
    CUdevice device;
    CUcontext context;
    CUdeviceptr devPtr[N]; // Array of device pointers

    // Initialize the device and the context
    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));

    // Allocate and free memory repeatedly
    for (int i = 0; i < N; i++) {
        checkError(cuMemAlloc(&devPtr[i], M)); // Allocate M bytes of device memory
        checkError(cuMemFree(devPtr[i])); // Free the device memory
    }

    // Destroy the context
    checkError(cuCtxDestroy(context));
}

// A stress test for cuda driver api thread creation and synchronization
TEST(CudaDriverApiTest, ThreadCreation) {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;

    // Initialize the device, the context, the module and the function
    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));
    checkError(cuModuleLoad(&module, "dummy_kernel.ptx")); // Load the module from a ptx file
    checkError(cuModuleGetFunction(&function, module, "dummy_kernel")); // Get the function from the module

    // Launch the kernel repeatedly with different grid and block sizes
    for (int i = 0; i < K; i++) {
        int gridDimX = (i % M) + 1; // Vary the grid size from 1 to M
        int blockDimX = (i % N) + 1; // Vary the block size from 1 to N
        void *args[] = {}; // No arguments for the kernel
        checkError(cuLaunchKernel(function, gridDimX, 1, 1, blockDimX, 1, 1, 0, NULL, args, NULL)); // Launch the kernel
        checkError(cuCtxSynchronize()); // Synchronize the context
    }

    // Unload the module and destroy the context
    checkError(cuModuleUnload(module));
    checkError(cuCtxDestroy(context));
}

// A stress test for cuda driver api stream creation and destruction
TEST(CudaDriverApiTest, StreamCreation) {
    CUdevice device;
    CUcontext context;
    CUstream stream[N]; // Array of streams

    // Initialize the device and the context
    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));

    // Create and destroy streams repeatedly
    for (int i = 0; i < N; i++) {
        checkError(cuStreamCreate(&stream[i], CU_STREAM_DEFAULT)); // Create a stream with default flags
        checkError(cuStreamDestroy(stream[i])); // Destroy the stream
    }

    // Destroy the context
    checkError(cuCtxDestroy(context));
}

// A stress test for cuda driver api event creation and destruction
TEST(CudaDriverApiTest, EventCreation) {
    CUdevice device;
    CUcontext context;
    CUevent event[N]; // Array of events

    // Initialize the device and the context
    checkError(cuInit(0));
    checkError(cuDeviceGet(&device, 0));
    checkError(cuCtxCreate(&context, 0, device));

    // Create and destroy events repeatedly
    for (int i = 0; i < N; i++) {
        checkError(cuEventCreate(&event[i], CU_EVENT_DEFAULT)); // Create an event with default flags
        checkError(cuEventDestroy(event[i])); // Destroy the event
    }

    // Destroy the context
    checkError(cuCtxDestroy(context));
}
