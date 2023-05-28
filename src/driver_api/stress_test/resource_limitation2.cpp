// 
// A stress test for cuda driver api memory exhaustion
#include <stdio.h>
#include <cuda.h>

#define N 1000 // Number of memory allocations
#define M 1024 * 1024 * 1024 // Size of each memory allocation in bytes

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

  // Allocate memory until it fails
  for (int i = 0; i < N; i++) {
    result = cuMemAlloc(&devPtr[i], M); // Allocate M bytes of device memory
    if (result != CUDA_SUCCESS) {
      printf("cuMemAlloc failed: %d\n", result);
      break;
    }
    printf("Allocated %d GB of device memory\n", (i + 1));
  }

  // Free the allocated memory
  for (int i = 0; i < N; i++) {
    if (devPtr[i]) {
      result = cuMemFree(devPtr[i]); // Free the device memory
      if (result != CUDA_SUCCESS) {
        printf("cuMemFree failed: %d\n", result);
        break;
      }
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

// A stress test for cuda driver api thread timeout
#include <stdio.h>
#include <cuda.h>

#define N 1024 // Number of threads per block
#define M 1024 // Number of blocks per grid

__global__ void infinite_kernel() {
  // Loop forever
  while (true) {
    // Do nothing
  }
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
  result = cuCtxCreate(&context, CU_CTX_SCHED_BLOCKING_SYNC, device); // Create a context with blocking sync mode
  if (result != CUDA_SUCCESS) {
    printf("cuCtxCreate failed: %d\n", result);
    return -1;
  }
  result = cuModuleLoad(&module, "infinite_kernel.ptx"); // Load the module from a ptx file
  if (result != CUDA_SUCCESS) {
    printf("cuModuleLoad failed: %d\n", result);
    return -1;
  }
  result = cuModuleGetFunction(&function, module, "infinite_kernel"); // Get the function from the module
  if (result != CUDA_SUCCESS) {
    printf("cuModuleGetFunction failed: %d\n", result);
    return -1;
  }

  // Launch the kernel with a large grid and block size
  void *args[] = {}; // No arguments for the kernel
  result = cuLaunchKernel(function, M, 1, 1, N, 1, 1, 0, NULL, args, NULL); // Launch the kernel
  if (result != CUDA_SUCCESS) {
    printf("cuLaunchKernel failed: %d\n", result);
    return -1;
  }

  // Try to synchronize the context
  result = cuCtxSynchronize(); // Synchronize the context
  if (result != CUDA_SUCCESS) {
    printf("cuCtxSynchronize failed: %d\n", result);
    // Handle the error
    switch (result) {
      case CUDA_ERROR_LAUNCH_TIMEOUT:
        // The kernel exceeded the maximum execution time
        printf("The kernel exceeded the maximum execution time\n");
        break;
      case CUDA_ERROR_LAUNCH_FAILED:
        // The kernel launch failed for an unknown reason
        printf("The kernel launch failed for an unknown reason\n");
        break;
      default:
        // Some other error occurred
        printf("Some other error occurred\n");
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

// A stress test for cuda driver api stream and event synchronization and query
#include <stdio.h>
#include <cuda.h>

#define N 1000 // Number of streams and events

__global__ void dummy_kernel() {
  // Do nothing
}

int main() {
  CUdevice device;
  CUcontext context;
  CUmodule module;
  CUfunction function;
  CUresult result;
  CUstream stream[N]; // Array of streams
  CUevent event[N]; // Array of events

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

  // Create streams and events
  for (int i = 0; i < N; i++) {
    result = cuStreamCreate(&stream[i], CU_STREAM_DEFAULT); // Create a stream with default flags
    if (result != CUDA_SUCCESS) {
      printf("cuStreamCreate failed: %d\n", result);
      break;
    }
    result = cuEventCreate(&event[i], CU_EVENT_DEFAULT); // Create an event with default flags
    if (result != CUDA_SUCCESS) {
      printf("cuEventCreate failed: %d\n", result);
      break;
    }
  }

  // Launch the kernel on each stream and record an event
  for (int i = 0; i < N; i++) {
    void *args[] = {}; // No arguments for the kernel
    result = cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream[i], args, NULL); // Launch the kernel on the stream
    if (result != CUDA_SUCCESS) {
      printf("cuLaunchKernel failed: %d\n", result);
      break;
    }
    result = cuEventRecord(event[i], stream[i]); // Record an event on the stream
    if (result != CUDA_SUCCESS) {
      printf("cuEventRecord failed: %d\n", result);
      break;
    }
  }

  // Wait for all events to complete and query their status
  for (int i = 0; i < N; i++) {
    result = cuEventSynchronize(event[i]); // Wait for the event to complete
    if (result != CUDA_SUCCESS) {
      printf("cuEventSynchronize failed: %d\n", result);
        break;
      }
      // Query the event status
      result = cuEventQuery(event[i]);
      if (result == CUDA_SUCCESS) {
        printf("Event %d is completed.\n", i);
      } else if (result == CUDA_ERROR_NOT_READY) {
        printf("Event %d is not ready.\n", i);
      } else {
        printf("cuEventQuery failed: %d\n", result);
        break;
      }
    }

    // Destroy streams and events
    for (int i = 0; i < N; i++) {
      result = cuStreamDestroy(stream[i]); // Destroy the stream
      if (result != CUDA_SUCCESS) {
        printf("cuStreamDestroy failed: %d\n", result);
        break;
      }
      result = cuEventDestroy(event[i]); // Destroy the event
      if (result != CUDA_SUCCESS) {
        printf("cuEventDestroy failed: %d\n", result);
        break;
      }
    }

    // Clean up the device, the context, the module and the function
    result = cuModuleUnload(module); // Unload the module
    if (result != CUDA_SUCCESS) {
      printf("cuModuleUnload failed: %d\n", result);
      return -1;
    }
    result = cuCtxDestroy(context); // Destroy the context
    if (result != CUDA_SUCCESS) {
      printf("cuCtxDestroy failed: %d\n", result);
      return -1;
    }

    // Return success
    return 0;
  }