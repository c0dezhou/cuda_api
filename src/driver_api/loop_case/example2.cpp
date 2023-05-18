// A code to loop cuMemcpyDtoH, cuMemGetInfo and cuMemcpyPeerAsync 100 times and configure parameters using function pointers, templates and handles
#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

// A helper function to check CUDA errors
void checkCudaError(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char* errorName;
    const char* errorString;
    cuGetErrorName(result, &errorName);
    cuGetErrorString(result, &errorString);
    std::cerr << "CUDA error: " << errorName << ": " << errorString << std::endl;
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
    std::cout << "Calling " << #func << std::endl; \
    func; \
    std::cout << "Done " << #func << std::endl; \
  } while (0)

// A template class to represent a function pointer with any parameters
template <typename R, typename... Args>
class FuncPtr {
 public:
  // The constructor takes a function pointer and stores it
  FuncPtr(R (*func)(Args...)) : func_(func) {}

  // The call operator takes the arguments and calls the function pointer with them
  R operator()(Args... args) {
    return func_(args...);
  }

 private:
  // The function pointer
  R (*func_)(Args...);
};

// A template function to create a FuncPtr object from a function pointer
template <typename R, typename... Args>
FuncPtr<R, Args...> makeFuncPtr(R (*func)(Args...)) {
  return FuncPtr<R, Args...>(func);
}

// A template function to loop 100 times over a FuncPtr object with given arguments
template <typename R, typename... Args>
void loop100FuncPtr(FuncPtr<R, Args...> funcPtr, Args... args) {
  LOOP_100(funcPtr(args...));
}

// A template function to print a message before and after a FuncPtr object with given arguments
template <typename R, typename... Args>
void printMsgFuncPtr(FuncPtr<R, Args...> funcPtr, Args... args) {
  PRINT_MSG(loop100FuncPtr(funcPtr, args...));
}

// A template class to represent a handle for a function pointer with any parameters
template <typename R, typename... Args>
class Handle {
 public:
  // The constructor takes a FuncPtr object and its arguments and stores them
  Handle(FuncPtr<R, Args...> funcPtr, Args... args) : funcPtr_(funcPtr), args_(args...) {}

  // The call operator calls the FuncPtr object with its arguments
  void operator()() {
    std::apply(funcPtr_, args_);
  }

 private:
  // The FuncPtr object
  FuncPtr<R, Args...> funcPtr_;
  // The arguments for the FuncPtr object
  std::tuple<Args...> args_;
};

// A template function to create a Handle object from a FuncPtr object and its arguments
template <typename R, typename... Args>
Handle<R, Args...> makeHandle(FuncPtr<R, Args...> funcPtr, Args... args) {
  return Handle<R, Args...>(funcPtr, args...);
}

int main() {
   // Initialize the CUDA context and device
   CUdevice device;
   CUcontext context;
   PRINT_MSG(LOOP_100(cuInit(0)));
   PRINT_MSG(LOOP_100(cuDeviceGet(&device ,0)));
   PRINT_MSG(LOOP_100(cuCtxCreate(&context ,0 , device)));

   // Allocate some device memory and fill it with some data
   CUdeviceptr devicePtr;
   size_t size = 1024 * sizeof(int);
   PRINT_MSG(LOOP_100(cuMemAlloc(&devicePtr , size)));
   int data[size / sizeof(int)];
   for (int i = 0; i < size / sizeof(int); i++) {
     data[i] = i;
   }
   PRINT_MSG(LOOP_100(cuMemcpyHtoD(devicePtr , data , size)));

   // Prepare the parameters for cuMemcpyDtoH using lambda expressions
   auto cuMemcpyDtoHParams = [&devicePtr , size]() {
     void* hostPtr = malloc(size); // The destination host pointer
     CUdeviceptr srcDevice = devicePtr; // The source device pointer
     size_t byteCount = size; // The number of bytes to copy
     return std::make_tuple(hostPtr , srcDevice , byteCount);
   };

   // Use C++ templates to create a FuncPtr object for cuMemcpyDtoH
   auto cuMemcpyDtoHFunc = makeFuncPtr(cuMemcpyDtoH);

   // Use C++ templates to create a Handle object for cuMemcpyDtoH and store it in an array
   Handle handles[] = {
     makeHandle(cuMemcpyDtoHFunc , cuMemcpyDtoHParams())
```c++
// The rest of the code is similar to the previous one until this point 
```c

     // Prepare another CUDA context and device for peer-to-peer copy using lambda expressions 
     auto cuDeviceGetParams = []() {
       CUdevice peerDevice;
       return std::make_tuple(&peerDevice ,1); // Get the second device if available 
     };
     auto cuCtxCreateParams = [&peerDevice]() {
       CUcontext peerContext;
       return std::make_tuple(&peerContext ,0 , peerDevice); // Create a context for the second device 
     };
     auto cuCtxEnablePeerAccessParams = [&peerContext]() {
       return std::make_tuple(peerContext ,0); // Enable peer access between the two contexts 
     };
     auto cuMemAllocParams = [&peerDevice]() {
       CUdeviceptr peerDevicePtr;
       return std::make_tuple(&peerDevicePtr , size); // Allocate some device memory on the peer device 
     };
     auto cuMemcpyHtoDParams = [&peerData , &peerDevice]() {
       CUdeviceptr peerDevicePtr;
       return std::make_tuple(peerDevicePtr , peerData , size); // Fill the device memory with some data 
     };
     auto cuMemcpyPeerAsyncParams = [&device , &context , &peerContext]() {
       CUdeviceptr dstDevice = device; // The destination device pointer on the first device 
       CUcontext dstContext = context; // The destination context on the first device 
       CUdeviceptr srcDevice = peerDevice; // The source device pointer on the second device 
       CUcontext srcContext = peerContext; // The source context on the second device 
       size_t byteCount = size; // The number of bytes to copy 

       return std::make_tuple(dstDevice , dstContext , srcDevice , srcContext , byteCount); 
     };

     // Use C++ templates to create FuncPtr objects for peer-to-peer copy functions  
     auto cuDeviceGetFunc = makeFuncPtr(cuDeviceGet);
     auto cuCtxCreateFunc = makeFuncPtr(cuCtxCreate);
     auto cuCtxEnablePeerAccessFunc = makeFuncPtr(cuCtxEnablePeerAccess);
     auto cuMemAllocFunc = makeFuncPtr(cuMemAlloc);
     auto cuMemcpyHtoDFunc = makeFuncPtr(cuMemcpyHtoD);
     auto cuMemcpyPeerAsyncFunc = makeFuncPtr(cuMemcpyPeerAsync);

     // Use C++ templates to create Handle objects for peer-to-peer copy functions and store them in an array  
     handles[] = {
       makeHandle(cuDeviceGetFunc , cuDeviceGetParams()),
       makeHandle(cuCtxCreateFunc , cuCtxCreateParams()),
       makeHandle(cuCtxEnablePeerAccessFunc , cuCtxEnablePeerAccessParams()),
       makeHandle(cuMemAllocFunc , cuMemAllocParams()),
       makeHandle(cuMemcpyHtoDFunc , cuMemcpyHtoDParams()),
       makeHandle(cuMemcpyPeerAsyncFunc , cuMemcpyPeerAsyncParams())
```c++
// The rest of the handles are omitted for brevity  
```c

      };

      // The number of handles in the array  
      int num_handles = sizeof(handles) / sizeof(Handle);

      // Initialize the random number generator with the current time as seed  
      srand(time(NULL));

      // Loop over all the handles randomly and call them  
      for (int i = 0; i < num_handles; i++) {
        // Generate a random index between 0 and num_handles -1 
        int index = rand() % num_handles;
        // Call the handle at the random index 
        handles[index]();
      }



      // Use C++ templates to create a Handle object for cuMemcpyDtoH and store it in an array
Handle handles[] = {
  makeHandle(cuMemcpyDtoHFunc , cuMemcpyDtoHParams())
};

// Use C++ templates to create FuncPtr objects for peer-to-peer copy functions  
auto cuDeviceGetFunc = makeFuncPtr(cuDeviceGet);
auto cuCtxCreateFunc = makeFuncPtr(cuCtxCreate);
auto cuCtxEnablePeerAccessFunc = makeFuncPtr(cuCtxEnablePeerAccess);
auto cuMemAllocFunc = makeFuncPtr(cuMemAlloc);
auto cuMemcpyHtoDFunc = makeFuncPtr(cuMemcpyHtoD);
auto cuMemcpyPeerAsyncFunc = makeFuncPtr(cuMemcpyPeerAsync);

// Use C++ templates to create Handle objects for peer-to-peer copy functions and add them to the array  
handles.push_back(makeHandle(cuDeviceGetFunc , cuDeviceGetParams()));
handles.push_back(makeHandle(cuCtxCreateFunc , cuCtxCreateParams()));
handles.push_back(makeHandle(cuCtxEnablePeerAccessFunc , cuCtxEnablePeerAccessParams()));
handles.push_back(makeHandle(cuMemAllocFunc , cuMemAllocParams()));
handles.push_back(makeHandle(cuMemcpyHtoDFunc , cuMemcpyHtoDParams()));
handles.push_back(makeHandle(cuMemcpyPeerAsyncFunc , cuMemcpyPeerAsyncParams()));

}}


