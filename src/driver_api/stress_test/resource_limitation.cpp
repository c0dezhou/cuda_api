// Resource Limitation
// 旨在测试cuda driver api的资源限制，例如内存分配，线程数，流数，事件数等。我使用了一些随机的参数和循环

// A stress test for cuda driver api memory allocation and deallocation
#include <stdio.h>
#include <cuda.h>

#define N 1000 // Number of memory allocations
#define M 1024 // Size of each memory allocation in bytes

int main() {
  CUdevice device;
  CUcontext context;
  CUresult result;
  CUdeviceptr devPtr[N]; // Array of device pointers

  // Initialize the device and the context
  result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    printf("cuInit failed: %d\n", result);
    return -1;
  }
  result = cuDeviceGet(&device, 0);
  if (result != CUDA_SUCCESS) {
    printf("cuDeviceGet failed: %d\n", result);
    return -1;
  }
  result = cuCtxCreate(&context, 0, device);
  if (result != CUDA_SUCCESS) {
    printf("cuCtxCreate failed: %d\n", result);
    return -1;
  }

  // Allocate and free memory repeatedly
  for (int i = 0; i < N; i++) {
    result = cuMemAlloc(&devPtr[i], M); // Allocate M bytes of device memory
    if (result != CUDA_SUCCESS) {
      printf("cuMemAlloc failed: %d\n", result);
      break;
    }
    result = cuMemFree(devPtr[i]); // Free the device memory
    if (result != CUDA_SUCCESS) {
      printf("cuMemFree failed: %d\n", result);
      break;
    }
  }

  // Destroy the context
  result = cuCtxDestroy(context);
  if (result != CUDA_SUCCESS) {
    printf("cuCtxDestroy failed: %d\n", result);
    return -1;
  }

  return 0;
}

// A stress test for cuda driver api thread creation and synchronization
#include <stdio.h>
#include <cuda.h>

#define N 1000 // Number of threads per block
#define M 1000 // Number of blocks per grid
#define K 1000 // Number of kernel launches

__global__ void dummy_kernel() {
  // Do nothing
}

int main() {
  CUdevice device;
  CUcontext context;
  CUmodule module;
  CUfunction function;
  CUresult result;

  // Initialize the device, the context, the module and the function
  result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    printf("cuInit failed: %d\n", result);
    return -1;
  }
  result = cuDeviceGet(&device, 0);
  if (result != CUDA_SUCCESS) {
    printf("cuDeviceGet failed: %d\n", result);
    return -1;
  }
  result = cuCtxCreate(&context, 0, device);
  if (result != CUDA_SUCCESS) {
    printf("cuCtxCreate failed: %d\n", result);
    return -1;
  }
  result = cuModuleLoad(&module, "dummy_kernel.ptx"); // Load the module from a ptx file
  if (result != CUDA_SUCCESS) {
    printf("cuModuleLoad failed: %d\n", result);
    return -1;
  }
  result = cuModuleGetFunction(&function, module, "dummy_kernel"); // Get the function from the module
  if (result != CUDA_SUCCESS) {
    printf("cuModuleGetFunction failed: %d\n", result);
    return -1;
  }

  // Launch the kernel repeatedly with different grid and block sizes
  for (int i = 0; i < K; i++) {
    int gridDimX = (i % M) + 1; // Vary the grid size from 1 to M
    int blockDimX = (i % N) + 1; // Vary the block size from 1 to N
    void *args[] = {}; // No arguments for the kernel
    result = cuLaunchKernel(function, gridDimX, 1, 1, blockDimX, 1, 1, 0, NULL, args, NULL); // Launch the kernel
    if (result != CUDA_SUCCESS) {
      printf("cuLaunchKernel failed: %d\n", result);
      break;
    }
    result = cuCtxSynchronize(); // Synchronize the context
    if (result != CUDA_SUCCESS) {
      printf("cuCtxSynchronize failed: %d\n", result);
      break;
    }
  }

  // Unload the module and destroy the context
  result = cuModuleUnload(module);
  if (result != CUDA_SUCCESS) {
    printf("cuModuleUnload failed: %d\n", result);
    return -1;
  }
  result = cuCtxDestroy(context);
  if (result != CUDA_SUCCESS) {
    printf("cuCtxDestroy failed: %d\n", result);
    return -1;
  }

  return 0;
}

// A stress test for cuda driver api stream creation and destruction
#include <stdio.h>
#include <cuda.h>

#define N 1000 // Number of streams to create and destroy

int main() {
  CUdevice device;
  CUcontext context;
  CUresult result;
  CUstream stream[N]; // Array of streams

  // Initialize the device and the context
  result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    printf("cuInit failed: %d\n", result);
    return -1;
  }
  result = cuDeviceGet(&device, 0);
  if (result != CUDA_SUCCESS) {
    printf("cuDeviceGet failed: %d\n", result);
    return -1;
  }
  result = cuCtxCreate(&context, 0, device);
  if (result != CUDA_SUCCESS) {
    printf("cuCtxCreate failed: %d\n", result);
    return -1;
  }

  // Create and destroy streams repeatedly
  for (int i = 0; i < N; i++) {
    result = cuStreamCreate(&stream[i], CU_STREAM_DEFAULT); // Create a stream with default flags
    if (result != CUDA_SUCCESS) {
      printf("cuStreamCreate failed: %d\n", result);
      break;
    }
    result = cuStreamDestroy(stream[i]); // Destroy the stream
    if (result != CUDA_SUCCESS) {
      printf("cuStreamDestroy failed: %d\n", result);
      break;
    }
  }

  // Destroy the context
  result = cuCtxDestroy(context);
  if (result != CUDA_SUCCESS) {
    printf("cuCtxDestroy failed: %d\n", result);
    return -1;
  }

  return 0;
}

// A stress test for cuda driver api event creation and destruction
#include <stdio.h>
#include <cuda.h>

#define N 1000 // Number of events to create and destroy

int main() {
  CUdevice device;
  CUcontext context;
  CUresult result;
  CUevent event[N]; // Array of events

  // Initialize the device and the context
  result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    printf("cuInit failed: %d\n", result);
    return -1;
  }
  result = cuDeviceGet(&device, 0);
  if (result != CUDA_SUCCESS) {
    printf("cuDeviceGet failed: %d\n", result);
    return -1;
  }
  result = cuCtxCreate(&context, 0, device);
  if (result != CUDA_SUCCESS) {
    printf("cuCtxCreate failed: %d\n", result);
    return -1;
  }

  // Create and destroy events repeatedly
  for (int i = 0; i < N; i++) {
    result = cuEventCreate(&event[i], CU_EVENT_DEFAULT); // Create an event with default flags
    if (result != CUDA_SUCCESS) {
      printf("cuEventCreate failed: %d\n", result);
      break;
    }
    result = cuEventDestroy(event[i]); // Destroy the event
    if (result != CUDA_SUCCESS) {
      printf("cuEventDestroy failed: %d\n", result);
      break;
    }
  }

  // Destroy the context
  result = cuCtxDestroy(context);
  if (result != CUDA_SUCCESS) {
    printf("cuCtxDestroy failed: %d\n", result);
    return -1;
  }

  return 0;
}
