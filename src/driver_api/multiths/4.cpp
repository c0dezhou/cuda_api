// A function to create parameters for cuMemcpyHtoD
auto cuMemcpyHtoDParams = [&hostPtr, size]() {
  void *srcHost = hostPtr; // The source host pointer
  CUdeviceptr dstDevice = devicePtr; // The destination device pointer
  size_t byteCount = size; // The number of bytes to copy
  return std::make_tuple(dstDevice, srcHost, byteCount);
};

// Use C++ templates to create a FuncPtr object for cuMemcpyHtoD
auto cuMemcpyHtoDFunc = makeFuncPtr(cuMemcpyHtoD);

// A function to create parameters for cuMemsetD8
auto cuMemsetD8Params = [&devicePtr, size]() {
  CUdeviceptr dstDevice = devicePtr; // The destination device pointer
  unsigned char uc = 0xFF; // The value to set
  size_t n = size; // The number of bytes to set
  return std::make_tuple(dstDevice, uc, n);
};

// Use C++ templates to create a FuncPtr object for cuMemsetD8
auto cuMemsetD8Func = makeFuncPtr(cuMemsetD8);

// A function to create parameters for cuMemAllocHost
auto cuMemAllocHostParams = [size]() {
  void **pp = &hostPtr; // The pointer to the allocated host memory
  size_t bytesize = size; // The number of bytes to allocate
  return std::make_tuple(pp, bytesize);
};

// Use C++ templates to create a FuncPtr object for cuMemAllocHost
auto cuMemAllocHostFunc = makeFuncPtr(cuMemAllocHost);

// A function to create parameters for cuMemFreeHost
auto cuMemFreeHostParams = [&hostPtr]() {
  void *p = hostPtr; // The pointer to the allocated host memory
  return std::make_tuple(p);
};

// Use C++ templates to create a FuncPtr object for cuMemFreeHost
auto cuMemFreeHostFunc = makeFuncPtr(cuMemFreeHost);

// Use C++ templates to create Handle objects for the functions and store them in an array
Handle handles[] = {
  makeHandle(cuMemcpyDtoHFunc, cuMemcpyDtoHParams()),
  makeHandle(cuMemcpyHtoDFunc, cuMemcpyHtoDParams()),
  makeHandle(cuMemsetD8Func, cuMemsetD8Params()),
  makeHandle(cuMemAllocHostFunc, cuMemAllocHostParams()),
  makeHandle(cuMemFreeHostFunc, cuMemFreeHostParams())
};

// A function to run a Handle object in a thread
void runHandle(Handle handle) {
  handle(); // Call the Handle object
}

// A function to create and join threads for each Handle object in the array
void testHandles() {
  int n = sizeof(handles) / sizeof(handles[0]); // Get the number of Handle objects in the array
  std::thread threads[n]; // Create an array of threads
  for (int i = 0; i < n; i++) {
    threads[i] = std::thread(runHandle, handles[i]); // Create a thread for each Handle object and pass it as an argument
  }
  for (int i = 0; i < n; i++) {
    threads[i].join(); // Wait for each thread to finish
  }
}

/*
我不能确定这样的多线程测试会不会造成core dump或内存泄漏，因为我没有运行或调试过这段代码。你需要自己检查和测试它。一般来说，为了避免core dump或内存泄漏，你需要确保以下几点：

- 每个线程都有自己的上下文和设备指针，不要在多个线程中共享或修改它们。
- 每次分配内存后，都要在不需要时释放它，不要造成内存泄漏或重复释放。
- 每次启动内核后，都要同步上下文，以确保内核执行完成，不要造成竞争条件或死锁。
- 每次使用函数指针时，都要确保传递正确的参数类型和数量，不要造成类型错误或缓冲区溢出。
*/