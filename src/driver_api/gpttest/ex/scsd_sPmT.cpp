// # 单卡单核
// * 单进程多线程
  /*
    一个场景是使用多个线程来执行不同的CUDA操作，例如对不同的设备或模块进行操作，然后使用相同的CUDA context来管理不同线程之间的设备资源和状态12。这个场景可以测试cudadriver api的线程安全性和context共享性。

    另一个场景是使用多个线程来执行相同的CUDA操作，例如对相同的设备或模块进行操作，然后使用不同的CUDA stream来管理不同线程之间的操作顺序和并发性12。这个场景可以测试cudadriver api的stream机制和并行能力。

    还有一个场景是使用多个线程来执行交错的CUDA操作，例如对相同的设备或模块进行不同的操作，然后使用不同的CUDA event来管理不同线程之间的操作同步和依赖性12。这个场景可以测试cudadriver api的event机制和协作能力。
  */

// 1.
// 使用cuCtxPopCurrent和cuCtxPushCurrent等函数来在不同的线程之间切换CUDA上下文，以保证每个线程都能访问到正确的设备资源和状态。
// 使用cuCtxSetCurrent等函数来设置每个线程的当前CUDA上下文，以保证每个线程都能使用相同的CUDA上下文。
// 使用cuCtxSynchronize等函数来同步每个线程的CUDA操作，以保证每个线程都能正确地完成其CUDA操作。
#include <cuda.h>
#include <gtest/gtest.h>
#include <thread>

// a simple kernel that adds two vectors
__global__ void vecAdd(float *A, float *B, float *C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  C[i] = A[i] + B[i];
}

// a helper function that initializes data
void initData(float *data, int size, float value) {
  for (int i = 0; i < size; i++) {
    data[i] = value;
  }
}

// a helper function that checks data
void checkData(float *data, int size, float expected) {
  for (int i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(data[i], expected);
  }
}

// a test fixture class that sets up the CUDA context and device memory
class CudaTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    // initialize the driver API and get the device handle
    CUresult result = cuInit(0);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuDeviceGet(&device, 0);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // create a context and load the module from a PTX file
    result = cuCtxCreate(&context, 0, device);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuModuleLoad(&module, "vecAdd.ptx");
    ASSERT_EQ(result, CUDA_SUCCESS);

    // get the kernel function handle
    result = cuModuleGetFunction(&function, module, "vecAdd");
    ASSERT_EQ(result, CUDA_SUCCESS);

    // allocate device memory
    result = cuMemAlloc(&d_A, N * sizeof(float));
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuMemAlloc(&d_B, N * sizeof(float));
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuMemAlloc(&d_C, N * sizeof(float));
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  static void TearDownTestSuite() {
    // free device memory
    CUresult result = cuMemFree(d_A);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuMemFree(d_B);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuMemFree(d_C);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // unload the module and destroy the context
    result = cuModuleUnload(module);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuCtxDestroy(context);
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  void SetUp() override {
    // set the current context for each test thread
    CUresult result = cuCtxSetCurrent(context);
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  void TearDown() override {
    // synchronize the current context for each test thread
    CUresult result = cuCtxSynchronize();
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  static constexpr int N = 1000000; // size of data
  static constexpr int M = 256; // number of threads per block

  static CUdevice device; // device handle
  static CUcontext context; // context handle
  static CUmodule module; // module handle
  static CUfunction function; // kernel function handle

  static CUdeviceptr d_A; // device memory for vector A
  static CUdeviceptr d_B; // device memory for vector B
  static CUdeviceptr d_C; // device memory for vector C

};

// define the static members of the test fixture class
CUdevice CudaTest::device;
CUcontext CudaTest::context;
CUmodule CudaTest::module;
CUfunction CudaTest::function;
CUdeviceptr CudaTest::d_A;
CUdeviceptr CudaTest::d_B;
CUdeviceptr CudaTest::d_C;

// a test case that performs vector addition on one device using one thread
TEST_F(CudaTest, SingleThreadSingleDevice) {
      // allocate host memory and initialize data
  float *h_A = (float *)malloc(N * sizeof(float));
  float *h_B = (float *)malloc(N * sizeof(float));
  float *h_C = (float *)malloc(N * sizeof(float));

  initData(h_A, N, 1.0f);
  initData(h_B, N, 2.0f);
  initData(h_C, N, 0.0f);

  // copy data from host to device
  CUresult result = cuMemcpyHtoD(d_A, h_A, N * sizeof(float));
  ASSERT_EQ(result, CUDA_SUCCESS);
  result = cuMemcpyHtoD(d_B, h_B, N * sizeof(float));
  ASSERT_EQ(result, CUDA_SUCCESS);

  // set up kernel parameters and launch kernel
  void *args[] = {&d_A, &d_B, &d_C};
  int gridDim = (N + M - 1) / M;
  result = cuLaunchKernel(function, gridDim, 1, 1, M, 1, 1, 0, NULL, args, NULL);
  ASSERT_EQ(result, CUDA_SUCCESS);

  // copy data from device to host
  result = cuMemcpyDtoH(h_C, d_C, N * sizeof(float));
  ASSERT_EQ(result, CUDA_SUCCESS);

  // check data
  checkData(h_C, N, 3.0f);

  // free host memory
  free(h_A);
  free(h_B);
  free(h_C);
}

// a test case that performs vector addition on one device using multiple threads
TEST_F(CudaTest, MultiThreadSingleDevice) {
    // define the number of threads and the size of each segment
  const int numThreads = 4;
  const int segmentSize = N / numThreads;

  // allocate host memory and initialize data
  float *h_A = (float *)malloc(N * sizeof(float));
  float *h_B = (float *)malloc(N * sizeof(float));
  float *h_C = (float *)malloc(N * sizeof(float));

  initData(h_A, N, 1.0f);
  initData(h_B, N, 2.0f);
  initData(h_C, N, 0.0f);

  // create an array of threads and events
  std::thread threads[numThreads];
  CUevent events[numThreads];

  // launch each thread to perform a segment of vector addition
  for (int i = 0; i < numThreads; i++) {
    // create an event for each thread
    CUresult result = cuEventCreate(&events[i], CU_EVENT_DEFAULT);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // launch a thread with a lambda function
    threads[i] = std::thread([=]() {
      // get the current context for each thread
      CUcontext ctx;
      CUresult result = cuCtxGetCurrent(&ctx);
      ASSERT_EQ(result, CUDA_SUCCESS);
      ASSERT_EQ(ctx, context);

      // calculate the offset and size of each segment
      int offset = i * segmentSize;
      int size = (i == numThreads - 1) ? (N - offset) : segmentSize;

      // copy data from host to device for each segment
      result = cuMemcpyHtoD(d_A + offset, h_A + offset, size * sizeof(float));
      ASSERT_EQ(result, CUDA_SUCCESS);
      result = cuMemcpyHtoD(d_B + offset, h_B + offset, size * sizeof(float));
      ASSERT_EQ(result, CUDA_SUCCESS);

      // set up kernel parameters and launch kernel for each segment
      void *args[] = {&d_A, &d_B, &d_C};
      int gridDim = (size + M - 1) / M;
      result = cuLaunchKernel(function, gridDim, 1, 1, M, 1, 1, 0, NULL, args, NULL);
      ASSERT_EQ(result, CUDA_SUCCESS);

      // copy data from device to host for each segment
      result = cuMemcpyDtoH(h_C + offset, d_C + offset, size * sizeof(float));
      ASSERT_EQ(result, CUDA_SUCCESS);

      // record an event for each thread
      result = cuEventRecord(events[i], NULL);
      ASSERT_EQ(result, CUDA_SUCCESS);
    });
  }

  // wait for all threads to finish
  for (int i = 0; i < numThreads; i++) {
    threads[i].join();
    CUresult result = cuEventDestroy(events[i]);
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  // check data
  checkData(h_C, N, 3.0f);

  // free host memory
  free(h_A);
  free(h_B);
  free(h_C);
}

  
// 2.
// 使用cuStreamCreate和cuStreamDestroy等函数来为每个线程创建和销毁一个CUDA流，以保证每个线程都能使用不同的流。
// 使用cuStreamSynchronize等函数来同步每个线程的CUDA流，以保证每个线程都能正确地完成其CUDA操作。
// 使用cuStreamWaitEvent等函数来让一个流等待另一个流中的事件完成，从而实现一定的同步逻辑。
#include <cuda.h>
#include <gtest/gtest.h>
#include <thread>

// a simple kernel that adds two vectors
__global__ void vecAdd(float *A, float *B, float *C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  C[i] = A[i] + B[i];
}

// a helper function that initializes data
void initData(float *data, int size, float value) {
  for (int i = 0; i < size; i++) {
    data[i] = value;
  }
}

// a helper function that checks data
void checkData(float *data, int size, float expected) {
  for (int i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(data[i], expected);
  }
}

// a test fixture class that sets up the CUDA context and device memory
class CudaTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    // initialize the driver API and get the device handle
    CUresult result = cuInit(0);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuDeviceGet(&device, 0);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // create a context and load the module from a PTX file
    result = cuCtxCreate(&context, 0, device);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuModuleLoad(&module, "vecAdd.ptx");
    ASSERT_EQ(result, CUDA_SUCCESS);

    // get the kernel function handle
    result = cuModuleGetFunction(&function, module, "vecAdd");
    ASSERT_EQ(result, CUDA_SUCCESS);

    // allocate device memory
    result = cuMemAlloc(&d_A, N * sizeof(float));
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuMemAlloc(&d_B, N * sizeof(float));
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuMemAlloc(&d_C, N * sizeof(float));
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  static void TearDownTestSuite() {
    // free device memory
    CUresult result = cuMemFree(d_A);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuMemFree(d_B);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuMemFree(d_C);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // unload the module and destroy the context
    result = cuModuleUnload(module);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuCtxDestroy(context);
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  void SetUp() override {
    // set the current context for each test thread
    CUresult result = cuCtxSetCurrent(context);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // create a stream for each test thread
    result = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  void TearDown() override {
    // synchronize and destroy the stream for each test thread
    CUresult result = cuStreamSynchronize(stream);
    ASSERT_EQ(result, CUDA_SUCCESS);
    
    result = cuStreamDestroy(stream);
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  static constexpr int N = 1000000; // size of data
  static constexpr int M = 256; // number of threads per block

  static CUdevice device; // device handle
  static CUcontext context; // context handle
  static CUmodule module; // module handle
  static CUfunction function; // kernel function handle

  static CUdeviceptr d_A; // device memory for vector A
  static CUdeviceptr d_B; // device memory for vector B
  static CUdeviceptr d_C; // device memory for vector C

  CUstream stream; // stream handle for each test thread

};

// define the static members of the test fixture class
CUdevice CudaTest::device;
CUcontext CudaTest::context;
CUmodule CudaTest::module;
CUfunction CudaTest::function;
CUdeviceptr CudaTest::d_A;
CUdeviceptr CudaTest::d_B;
CUdeviceptr CudaTest::d_C;

// a test case that performs vector addition on one device using multiple threads and streams
TEST_F(CudaTest, MultiThreadSingleDeviceMultiStream) {
  
  // define the number of threads and the size of each segment
  const int numThreads = 4;
  const int segmentSize = N / numThreads;

  // allocate host memory and initialize data
  float *h_A = (float *)malloc(N * sizeof(float));
  float *h_B = (float *)malloc(N * sizeof(float));
  float *h_C = (float *)malloc(N * sizeof(float));

  initData(h_A, N, 1.0f);
  initData(h_B, N, 2.0f);
  initData(h_C, N, 0.0f);

  // create an array of threads and events
  std::thread threads[numThreads];
  CUevent events[numThreads];

  // launch each thread to perform a segment of vector addition using a different stream
  for (int i = 0; i < numThreads; i++) {
    // create an event for each thread
    CUresult result = cuEventCreate(&events[i], CU_EVENT_DEFAULT);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // launch a thread with a lambda function
    threads[i] = std::thread([=]() {
      // get the current context and stream for each thread
      CUcontext ctx;
      CUresult result = cuCtxGetCurrent(&ctx);
      ASSERT_EQ(result, CUDA_SUCCESS);
      ASSERT_EQ(ctx, context);

      CUstream strm;
      result = cuStreamGetCtx(stream, &strm);
      ASSERT_EQ(result, CUDA_SUCCESS);

      // calculate the offset and size of each segment
      int offset = i * segmentSize;
      int size = (i == numThreads - 1) ? (N - offset) : segmentSize;

      // copy data from host to device for each segment using the stream
      result = cuMemcpyHtoDAsync(d_A + offset, h_A + offset, size * sizeof(float), strm);
      ASSERT_EQ(result, CUDA_SUCCESS);
      result = cuMemcpyHtoDAsync(d_B + offset, h_B + offset, size * sizeof(float), strm);
      ASSERT_EQ(result, CUDA_SUCCESS);

      // set up kernel parameters and launch kernel for each segment using the stream
      void *args[] = {&d_A, &d_B, &d_C};
      int gridDim = (size + M - 1) / M;
      result = cuLaunchKernel(function, gridDim, 1, 1, M, 1, 1, 0, strm, args, NULL);
      ASSERT_EQ(result, CUDA_SUCCESS);

      // copy data from device to host for each segment using the stream
      result = cuMemcpyDtoHAsync(h_C + offset, d_C + offset, size * sizeof(float), strm);
      ASSERT_EQ(result, CUDA_SUCCESS);

      // record an event for each thread using the stream
      result = cuEventRecord(events[i], strm);
      ASSERT_EQ(result, CUDA_SUCCESS);
    });
  }

  // wait for all threads to finish
  for (int i = 0; i < numThreads; i++) {
    threads[i].join();
    CUresult result = cuEventDestroy(events[i]);
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  // check data
  checkData(h_C, N, 3.0f);

  // free host memory
  free(h_A);
  free(h_B);
  free(h_C);
}


// 3. 

#include <cuda.h>
#include <gtest/gtest.h>
#include <thread>

// a simple kernel that adds two vectors
__global__ void vecAdd(float *A, float *B, float *C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  C[i] = A[i] + B[i];
}

// a simple kernel that scales a vector by a factor
__global__ void vecScale(float *A, float factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  A[i] = A[i] * factor;
}

// a helper function that initializes data
void initData(float *data, int size, float value) {
  for (int i = 0; i < size; i++) {
    data[i] = value;
  }
}

// a helper function that checks data
void checkData(float *data, int size, float expected) {
  for (int i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(data[i], expected);
  }
}

// a test fixture class that sets up the CUDA context and device memory
class CudaTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    // initialize the driver API and get the device handle
    CUresult result = cuInit(0);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuDeviceGet(&device, 0);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // create a context and load the module from a PTX file
    result = cuCtxCreate(&context, 0, device);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuModuleLoad(&module, "vecAdd.ptx");
    ASSERT_EQ(result, CUDA_SUCCESS);

    // get the kernel function handles
    result = cuModuleGetFunction(&function1, module, "vecAdd");
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuModuleGetFunction(&function2, module, "vecScale");
    ASSERT_EQ(result, CUDA_SUCCESS);

    // allocate device memory
    result = cuMemAlloc(&d_A, N * sizeof(float));
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuMemAlloc(&d_B, N * sizeof(float));
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuMemAlloc(&d_C, N * sizeof(float));
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  static void TearDownTestSuite() {
    // free device memory
    CUresult result = cuMemFree(d_A);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuMemFree(d_B);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuMemFree(d_C);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // unload the module and destroy the context
    result = cuModuleUnload(module);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuCtxDestroy(context);
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  void SetUp() override {
    // set the current context for each test thread
    CUresult result = cuCtxSetCurrent(context);
    ASSERT_EQ(result, CUDA_SUCCESS);

    
    // create a stream for each test thread
    result = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  void TearDown() override {
    // synchronize and destroy the stream for each test thread
    CUresult result = cuStreamSynchronize(stream);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuStreamDestroy(stream);
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  static constexpr int N = 1000000; // size of data
  static constexpr int M = 256; // number of threads per block

  static CUdevice device; // device handle
  static CUcontext context; // context handle
  static CUmodule module; // module handle
  static CUfunction function1; // kernel function handle for vecAdd
  static CUfunction function2; // kernel function handle for vecScale

  static CUdeviceptr d_A; // device memory for vector A
  static CUdeviceptr d_B; // device memory for vector B
  static CUdeviceptr d_C; // device memory for vector C

  CUstream stream; // stream handle for each test thread

};

// define the static members of the test fixture class
CUdevice CudaTest::device;
CUcontext CudaTest::context;
CUmodule CudaTest::module;
CUfunction CudaTest::function1;
CUfunction CudaTest::function2;
CUdeviceptr CudaTest::d_A;
CUdeviceptr CudaTest::d_B;
CUdeviceptr CudaTest::d_C;

// a test case that performs interleaved CUDA operations on one device using multiple threads and streams
TEST_F(CudaTest, MultiThreadSingleDeviceMultiStreamInterleaved) {
  
  // define the number of threads and the size of each segment
  const int numThreads = 4;
  const int segmentSize = N / numThreads;

  // allocate host memory and initialize data
  float *h_A = (float *)malloc(N * sizeof(float));
  float *h_B = (float *)malloc(N * sizeof(float));
  float *h_C = (float *)malloc(N * sizeof(float));

  initData(h_A, N, 1.0f);
  initData(h_B, N, 2.0f);
  initData(h_C, N, 0.0f);

  // create an array of threads and events
  std::thread threads[numThreads];
  CUevent events[numThreads];

  // launch each thread to perform interleaved CUDA operations using a different stream
  for (int i = 0; i < numThreads; i++) {
    // create an event for each thread
    CUresult result = cuEventCreate(&events[i], CU_EVENT_DEFAULT);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // launch a thread with a lambda function
    threads[i] = std::thread([=]() {
      // get the current context and stream for each thread
      CUcontext ctx;
      CUresult result = cuCtxGetCurrent(&ctx);
      ASSERT_EQ(result, CUDA_SUCCESS);
      ASSERT_EQ(ctx, context);

      CUstream strm;
      result = cuStreamGetCtx(stream, &strm);
      ASSERT_EQ(result, CUDA_SUCCESS);

      // calculate the offset and size of each segment
      int offset = i * segmentSize;
      int size = (i == numThreads - 1) ? (N - offset) : segmentSize;

      // copy data from host to device for each segment using the stream
      result = cuMemcpyHtoDAsync(d_A + offset, h_A + offset, size * sizeof(float), strm);
      ASSERT_EQ(result, CUDA_SUCCESS);
      result = cuMemcpyHtoDAsync(d_B + offset, h_B + offset, size * sizeof(float), strm);
      ASSERT_EQ(result, CUDA_SUCCESS);

      // set up kernel parameters and launch kernel for vecAdd for each segment using the stream
      void *args1[] = {&d_A, &d_B, &d_C};
      int gridDim1 = (size + M - 1) / M;
      result = cuLaunchKernel(function1, gridDim1, 1, 1, M, 1, 1, 0, strm, args1, NULL);
      ASSERT_EQ(result, CUDA_SUCCESS);

      // set up kernel parameters and launch kernel for vecScale for each segment using the stream
      float factor = i + 1.0f;
      void *args2[] = {&d_C, &factor};
      int gridDim2 = (size + M - 1) / M;
      result = cuLaunchKernel(function2, gridDim2, 1, 1, M, 1, 1, 0, strm, args2, NULL);
      ASSERT_EQ(result, CUDA_SUCCESS);

      // copy data from device to host for each segment using the stream
      result = cuMemcpyDtoHAsync(h_C + offset, d_C + offset, size * sizeof(float), strm);
      ASSERT_EQ(result, CUDA_SUCCESS);

      // record an event for each thread using the stream
      result = cuEventRecord(events[i], strm);
      ASSERT_EQ(result, CUDA_SUCCESS);
    });
  }

  // wait for all threads to finish
  for (int i = 0; i < numThreads; i++) {
    threads[i].join();
    CUresult result = cuEventDestroy(events[i]);
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  // check data
  for (int i = 0; i < numThreads; i++) {
    int offset = i * segmentSize;
    int size = (i == numThreads - 1) ? (N - offset) : segmentSize;
    float expected = (i + 1.0f) * (i + 3.0f); // factor * (A[i] + B[i])
    checkData(h_C + offset, size , expected);
    
  }

  // free host memory
  free(h_A);
  free(h_B);
  free(h_C);
}
