// A code to loop cuMemcpyDtoH, cuMemGetInfo and cuMemcpyPeerAsync 100 times and configure parameters
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// A helper function to check CUDA errors
void checkCudaError(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char* errorName;
    const char* errorString;
    cuGetErrorName(result, &errorName);
    cuGetErrorString(result, &errorString);
    fprintf(stderr, "CUDA error: %s: %s\n", errorName, errorString);
    exit(1);
  }
}

// A helper macro to loop 100 times over a function call and check the result
#define LOOP_100(func) \
  do { \
    for (int i = 0; i < 100; i++) { \
      checkCudaError(func); \
    } \
  } while (0)

// A helper macro to print a message before and after a function call
#define PRINT_MSG(func) \
  do { \
    printf("Calling %s\n", #func); \
    func; \
    printf("Done %s\n", #func); \
  } while (0)

int main() {
  // Initialize the CUDA context and device
  CUdevice device;
  CUcontext context;
  PRINT_MSG(LOOP_100(cuInit(0)));
  PRINT_MSG(LOOP_100(cuDeviceGet(&device, 0)));
  PRINT_MSG(LOOP_100(cuCtxCreate(&context, 0, device)));

  // Allocate some device memory and fill it with some data
  CUdeviceptr devicePtr;
  size_t size = 1024 * sizeof(int);
  PRINT_MSG(LOOP_100(cuMemAlloc(&devicePtr, size)));
  int data[size / sizeof(int)];
  for (int i = 0; i < size / sizeof(int); i++) {
    data[i] = i;
  }
  PRINT_MSG(LOOP_100(cuMemcpyHtoD(devicePtr, data, size)));

  // Prepare the parameters for cuMemcpyDtoH
  void* hostPtr = malloc(size); // The destination host pointer
  CUdeviceptr srcDevice = devicePtr; // The source device pointer
  size_t byteCount = size; // The number of bytes to copy

  // Loop cuMemcpyDtoH 100 times and print a message
  PRINT_MSG(LOOP_100(cuMemcpyDtoH(hostPtr, srcDevice, byteCount)));

  // Prepare the parameters for cuMemGetInfo
  size_t free; // The pointer to store the available device memory
  size_t total; // The pointer to store the total device memory

  // Loop cuMemGetInfo 100 times and print a message
  PRINT_MSG(LOOP_100(cuMemGetInfo(&free, &total)));

```c++
// The rest of the code is similar to the previous one until this point
```c

   // Prepare another CUDA context and device for peer-to-peer copy
   CUdevice peerDevice;
   CUcontext peerContext;
   PRINT_MSG(LOOP_100(cuDeviceGet(&peerDevice,1))); // Get the second device if available
   PRINT_MSG(LOOP_100(cuCtxCreate(&peerContext,0 , peerDevice))); // Create a context for the second device

   // Enable peer access between the two contexts
   PRINT_MSG(LOOP_100(cuCtxEnablePeerAccess(peerContext ,0 )));

   // Allocate some device memory on the peer device and fill it with some data
   CUdeviceptr peerDevicePtr;
   PRINT_MSG(LOOP_100(cuMemAlloc(&peerDevicePtr , size)));
   int peerData[size / sizeof(int)];
   for (int i = 0; i < size / sizeof(int); i++) {
     peerData[i] = i + size / sizeof(int);
   }
   PRINT_MSG(LOOP_100(cuMemcpyHtoD(peerDevicePtr , peerData , size)));

   // Prepare the parameters for cuMemcpyPeerAsync
   CUdeviceptr dstDevice = devicePtr; // The destination device pointer on the first device
   CUcontext dstContext = context; // The destination context on the first device
   CUdeviceptr srcDevice = peerDevicePtr; // The source device pointer on the second device
   CUcontext srcContext = peerContext; // The source context on the second device
   byteCount = size; // The number of bytes to copy

   // Create a stream for asynchronous operation
   CUstream stream;
   PRINT_MSG(LOOP_100(cuStreamCreate(&stream , CU_STREAM_DEFAULT)));

   // Loop cuMemcpyPeerAsync 100 times and print a message
   PRINT_MSG(LOOP_100(cuMemcpyPeerAsync(dstDevice , dstContext , srcDevice , srcContext , byteCount , stream)));

}
