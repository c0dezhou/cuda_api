// 以下是一些可能的多设备下的多线程CUDA操作的场景的改写思路：

// - 场景一：使用多个线程来执行不同的CUDA操作，例如对不同的设备或模块进行操作，然后使用相同的CUDA context来管理不同线程之间的设备资源和状态。这个场景可以测试CUDA驱动程序API的线程安全性和context共享性。为了改写为多设备下的场景，您可以：

//   - 在SetUpTestSuite函数中，获取多个设备的句柄，并为每个设备创建一个上下文和一个模块。
//   - 在SetUp函数中，为每个测试线程设置一个不同的上下文和模块。
//   - 在测试用例中，为每个测试线程分配一个不同的设备，并在该设备上执行相应的CUDA操作。
//   - 在TearDown函数中，恢复每个测试线程的上下文和模块。
//   - 在TearDownTestSuite函数中，销毁每个设备的上下文和模块。

// - 场景二：使用多个线程来执行相同的CUDA操作，例如对不同的设备或模块进行操作，然后使用不同的CUDA stream来管理不同线程之间的操作顺序和并发性。这个场景可以测试CUDA驱动程序API的stream机制和并行能力。为了改写为多设备下的场景，您可以：

//   - 在SetUpTestSuite函数中，获取多个设备的句柄，并为每个设备创建一个上下文和一个模块。
//   - 在SetUp函数中，为每个测试线程设置一个不同的上下文和模块，并创建一个流。
//   - 在测试用例中，为每个测试线程分配一个不同的设备，并在该设备上使用相应的流执行相应的CUDA操作。
//   - 在TearDown函数中，同步和销毁每个测试线程使用的流，并恢复每个测试线程的上下文和模块。
//   - 在TearDownTestSuite函数中，销毁每个设备的上下文和模块。

// - 场景三：使用多个线程来执行交错的CUDA操作，例如对相同的设备或模块进行不同的操作，然后使用不同的CUDA event来管理不同线程之间的操作同步和依赖性。这个场景可以测试CUDA驱动程序API的event机制和协作能力。为了改写为多设备下的场景，您可以：

//   - 在SetUpTestSuite函数中，获取多个设备的句柄，并为每个设备创建一个上下文和一个模块。
//   - 在SetUp函数中，为每个测试线程设置一个不同的上下文和模块，并创建一个流。
//   - 在测试用例中，为每个测试线程分配一个不同的设备，并在该设备上使用相应的流执行相应的CUDA操作，并记录相应的事件。
//   - 在测试用例中，使用cuStreamWaitEvent等函数来让一个流等待另一个流中的事件完成，从而实现一定的同步逻辑。
//   - 在TearDown函数中，同步和销毁每个测试线程使用的流和事件，并恢复每个测试线程的上下文和模块。
//   - 在TearDownTestSuite函数中，销毁每个设备的上下文和模块。

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
    // initialize the driver API and get the number of devices
    CUresult result = cuInit(0);
    ASSERT_EQ(result, CUDA_SUCCESS);
    result = cuDeviceGetCount(&numDevices);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // create an array of device handles and contexts
    devices = new CUdevice[numDevices];
    contexts = new CUcontext[numDevices];

    // create a context and load the module from a PTX file for each device
    for (int i = 0; i < numDevices; i++) {
      result = cuDeviceGet(&devices[i], i);
      ASSERT_EQ(result, CUDA_SUCCESS);
      result = cuCtxCreate(&contexts[i], 0, devices[i]);
      ASSERT_EQ(result, CUDA_SUCCESS);
      result = cuModuleLoad(&module, "vecAdd.ptx");
      ASSERT_EQ(result, CUDA_SUCCESS);
    }

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

    // unload the module and destroy the context for each device
    for (int i = 0; i < numDevices; i++) {
      result = cuModuleUnload(module);
      ASSERT_EQ(result, CUDA_SUCCESS);
      result = cuCtxDestroy(contexts[i]);
      ASSERT_EQ(result, CUDA_SUCCESS);
    }

    // delete the array of device handles and contexts
    delete[] devices;
    delete[] contexts;
  }

  void SetUp() override {
    
    // set the current context for each test thread
    CUresult result = cuCtxSetCurrent(contexts[deviceIndex]);
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  void TearDown() override {
    // synchronize the current context for each test thread
    CUresult result = cuCtxSynchronize();
    ASSERT_EQ(result, CUDA_SUCCESS);
  }

  static constexpr int N = 1000000; // size of data
  static constexpr int M = 256; // number of threads per block

  static int numDevices; // number of devices
  static CUdevice *devices; // array of device handles
  static CUcontext *contexts; // array of context handles
  static CUmodule module; // module handle
  static CUfunction function; // kernel function handle

  static CUdeviceptr d_A; // device memory for vector A
  static CUdeviceptr d_B; // device memory for vector B
  static CUdeviceptr d_C; // device memory for vector C

  int deviceIndex; // index of the device assigned to each test thread

};

// define the static members of the test fixture class
int CudaTest::numDevices;
CUdevice *CudaTest::devices;
CUcontext *CudaTest::contexts;
CUmodule CudaTest::module;
CUfunction CudaTest::function;
CUdeviceptr CudaTest::d_A;
CUdeviceptr CudaTest::d_B;
CUdeviceptr CudaTest::d_C;

// a test case that performs vector addition on different devices using different threads and contexts
TEST_F(CudaTest, MultiThreadMultiDevice) {
  
  // allocate host memory and initialize data
  float *h_A = (float *)malloc(N * sizeof(float));
  float *h_B = (float *)malloc(N * sizeof(float));
  float *h_C = (float *)malloc(N * sizeof(float));

  initData(h_A, N, 1.0f);
  initData(h_B, N, 2.0f);
  initData(h_C, N, 0.0f);

  // create an array of threads
  std::thread threads[numDevices];

  // launch each thread to perform vector addition on a different device
  for (int i = 0; i < numDevices; i++) {
    // assign a device index to each thread
    deviceIndex = i;

    // launch a thread with a lambda function
    threads[i] = std::thread([=]() {
      // get the current context and device for each thread
      CUcontext ctx;
      CUresult result = cuCtxGetCurrent(&ctx);
      ASSERT_EQ(result, CUDA_SUCCESS);
      ASSERT_EQ(ctx, contexts[deviceIndex]);

      CUdevice dev;
      result = cuCtxGetDevice(&dev);
      ASSERT_EQ(result, CUDA_SUCCESS);
      ASSERT_EQ(dev, devices[deviceIndex]);

      // copy data from host to device
      result = cuMemcpyHtoD(d_A, h_A, N * sizeof(float));
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
      checkData(h_C, N , 3.0f);
    });
  }

  // wait for all threads to finish
  for (int i = 0; i < numDevices; i++) {
    threads[i].join();
  }

  // free host memory
  free(h_A);
  free(h_B);
  free(h_C);
}



// 2.
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

// Test code using gtest
#include <gtest/gtest.h>
#include <cuda.h>

// Define the number of threads and devices
#define N 256
#define NUM_DEVICES 2

// Define a helper function to check CUDA errors
void checkCudaError(CUresult result, const char* msg)
{
    if (result != CUDA_SUCCESS) {
        const char* error;
        cuGetErrorName(result, &error);
        std::cerr << "CUDA error: " << msg << ": " << error << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Define a helper function to load a PTX module
CUmodule loadModule(const char* filename)
{
    CUmodule module;
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* buffer = new char[size];
    if (!file.read(buffer, size)) {
        std::cerr << "Cannot read file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    checkCudaError(cuModuleLoadData(&module, buffer), "cuModuleLoadData");
    delete[] buffer;
    return module;
}

// Define a helper function to get a kernel function from a module
CUfunction getFunction(CUmodule module, const char* name)
{
    CUfunction function;
    checkCudaError(cuModuleGetFunction(&function, module, name), "cuModuleGetFunction");
    return function;
}

// Define a test fixture class
class VecAddTest : public ::testing::Test {
protected:
    // Initialize the CUDA driver API and devices
    void SetUp() override {
        checkCudaError(cuInit(0), "cuInit");
        checkCudaError(cuDeviceGetCount(&deviceCount), "cuDeviceGetCount");
        ASSERT_GE(deviceCount, NUM_DEVICES) << "Not enough devices";
        for (int i = 0; i < NUM_DEVICES; i++) {
            checkCudaError(cuDeviceGet(&devices[i], i), "cuDeviceGet");
            checkCudaError(cuCtxCreate(&contexts[i], 0, devices[i]), "cuCtxCreate");
            modules[i] = loadModule("VecAdd.ptx");
            functions[i] = getFunction(modules[i], "VecAdd");
            checkCudaError(cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING), "cuStreamCreate");
            checkCudaError(cuMemAlloc(&d_A[i], N * sizeof(float)), "cuMemAlloc");
            checkCudaError(cuMemAlloc(&d_B[i], N * sizeof(float)), "cuMemAlloc");
            checkCudaError(cuMemAlloc(&d_C[i], N * sizeof(float)), "cuMemAlloc");
        }
    }

    // Clean up the CUDA driver API and devices
    void TearDown() override {
        for (int i = 0; i < NUM_DEVICES; i++) {
            checkCudaError(cuMemFree(d_A[i]), "cuMemFree");
            checkCudaError(cuMemFree(d_B[i]), "cuMemFree");
            checkCudaError(cuMemFree(d_C[i]), "cuMemFree");
            checkCudaError(cuStreamDestroy(streams[i]), "cuStreamDestroy");
            checkCudaError(cuModuleUnload(modules[i]), "cuModuleUnload");
            checkCudaError(cuCtxDestroy(contexts[i]), "cuCtxDestroy");
        }
    }

    // Launch the kernel on each device with different streams
    void launchKernel(float* h_A, float* h_B, float* h_C) {
        for (int i = 0; i < NUM_DEVICES; i++) {
            // Set the current context
            checkCudaError(cuCtxSetCurrent(contexts[i]), "cuCtxSetCurrent");

            // Copy the host input vectors to the device
            checkCudaError(cuMemcpyHtoDAsync(d_A[i], h_A, N * sizeof(float), streams[i]), "cuMemcpyHtoDAsync");
            checkCudaError(cuMemcpyHtoDAsync(d_B[i], h_B, N * sizeof(float), streams[i]), "cuMemcpyHtoDAsync");

            // Set up the kernel parameters
            void* args[] = {&d_A[i], &d_B[i], &d_C[i]};
            
            // Launch the kernel
            checkCudaError(cuLaunchKernel(functions[i], 1, 1, 1, N, 1, 1, 0, streams[i], args, nullptr), "cuLaunchKernel");

            // Copy the device output vector to the host
            checkCudaError(cuMemcpyDtoHAsync(h_C + i * N, d_C[i], N * sizeof(float), streams[i]), "cuMemcpyDtoHAsync");

            // Synchronize the stream
            checkCudaError(cuStreamSynchronize(streams[i]), "cuStreamSynchronize");
        }
    }

    int deviceCount;
    CUdevice devices[NUM_DEVICES];
    CUcontext contexts[NUM_DEVICES];
    CUmodule modules[NUM_DEVICES];
    CUfunction functions[NUM_DEVICES];
    CUstream streams[NUM_DEVICES];
    CUdeviceptr d_A[NUM_DEVICES];
    CUdeviceptr d_B[NUM_DEVICES];
    CUdeviceptr d_C[NUM_DEVICES];
};

// Define a test case using the fixture class
TEST_F(VecAddTest, Result) {
    // Allocate and initialize host input and output vectors
    float* h_A = new float[N * NUM_DEVICES];
    float* h_B = new float[N * NUM_DEVICES];
    float* h_C = new float[N * NUM_DEVICES];

    for (int i = 0; i < N * NUM_DEVICES; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = 0.0f;
    }

    // Launch the kernel on each device with different streams
    launchKernel(h_A, h_B, h_C);
    // Verify the results
    for (int i = 0; i < N * NUM_DEVICES; i++) {
        EXPECT_NEAR(h_A[i] + h_B[i], h_C[i], 1e-5) << "at index " << i;
    }

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}


// 3.
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

// Test code using gtest
#include <gtest/gtest.h>
#include <cuda.h>

// Define the number of threads and devices
#define N 256
#define NUM_DEVICES 2

// Define a helper function to check CUDA errors
void checkCudaError(CUresult result, const char* msg)
{
    if (result != CUDA_SUCCESS) {
        const char* error;
        cuGetErrorName(result, &error);
        std::cerr << "CUDA error: " << msg << ": " << error << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Define a helper function to load a PTX module
CUmodule loadModule(const char* filename)
{
    CUmodule module;
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* buffer = new char[size];
    if (!file.read(buffer, size)) {
        std::cerr << "Cannot read file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    checkCudaError(cuModuleLoadData(&module, buffer), "cuModuleLoadData");
    delete[] buffer;
    return module;
}

// Define a helper function to get a kernel function from a module
CUfunction getFunction(CUmodule module, const char* name)
{
    CUfunction function;
    checkCudaError(cuModuleGetFunction(&function, module, name), "cuModuleGetFunction");
    return function;
}

// Define a test fixture class
class VecAddTest : public ::testing::Test {
protected:
    // Initialize the CUDA driver API and devices
    void SetUp() override {
        checkCudaError(cuInit(0), "cuInit");
        checkCudaError(cuDeviceGetCount(&deviceCount), "cuDeviceGetCount");
        ASSERT_GE(deviceCount, NUM_DEVICES) << "Not enough devices";
        for (int i = 0; i < NUM_DEVICES; i++) {
            checkCudaError(cuDeviceGet(&devices[i], i), "cuDeviceGet");
            checkCudaError(cuCtxCreate(&contexts[i], 0, devices[i]), "cuCtxCreate");
            modules[i] = loadModule("VecAdd.ptx");
            functions[i] = getFunction(modules[i], "VecAdd");
            checkCudaError(cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING), "cuStreamCreate");
            checkCudaError(cuEventCreate(&events[i], CU_EVENT_DEFAULT), "cuEventCreate");
            checkCudaError(cuMemAlloc(&d_A[i], N * sizeof(float)), "cuMemAlloc");
            checkCudaError(cuMemAlloc(&d_B[i], N * sizeof(float)), "cuMemAlloc");
            checkCudaError(cuMemAlloc(&d_C[i], N * sizeof(float)), "cuMemAlloc");
        }
    }

    // Clean up the CUDA driver API and devices
    void TearDown() override {
        for (int i = 0; i < NUM_DEVICES; i++) {
            checkCudaError(cuMemFree(d_A[i]), "cuMemFree");
            checkCudaError(cuMemFree(d_B[i]), "cuMemFree");
            checkCudaError(cuMemFree(d_C[i]), "cuMemFree");
            checkCudaError(cuEventDestroy(events[i]), "cuEventDestroy");
            checkCudaError(cuStreamDestroy(streams[i]), "cuStreamDestroy");
            checkCudaError(cuModuleUnload(modules[i]), "cuModuleUnload");
            checkCudaError(cuCtxDestroy(contexts[i]), "cuCtxDestroy");
        }
    }

    // Launch the kernel on each device with different streams and events
    void launchKernel(float* h_A, float* h_B, float* h_C) {
        for (int i = 0; i < NUM_DEVICES; i++) {
            // Set the current context
            checkCudaError(cuCtxSetCurrent(contexts[i]), "cuCtxSetCurrent");

            // Copy the host input vectors to the device
            checkCudaError(cuMemcpyHtoDAsync(d_A[i], h_A + i * N / 2, N / 2 * sizeof(float), streams[i]), "cuMemcpyHtoDAsync");
            checkCudaError(cuMemcpyHtoDAsync(d_B[i], h_B + i * N / 2, N / 2 * sizeof(float), streams[i]), "cuMemcpyHtoDAsync");

            // Set up the kernel parameters
            void* args[] = {&d_A[i], &d_B[i], &d_C[i]};
            
            // Launch the kernel
            checkCudaError(cuLaunchKernel(functions[i], 1, 1, 1, N / 2, 1, 1, 0, streams[i], args, nullptr), "cuLaunchKernel");

            // Record an event after the kernel launch
            checkCudaError(cuEventRecord(events[i], streams[i]), "cuEventRecord");

            // Wait for the event from the other device before copying the device output vector to the host
            int j = (i + 1) % NUM_DEVICES;
            checkCudaError(cuStreamWaitEvent(streams[j], events[i], 0), "cuStreamWaitEvent");

            // Copy the device output vector to the host
            checkCudaError(cuMemcpyDtoHAsync(h_C + j * N / 2, d_C[j], N / 2 * sizeof(float), streams[j]), "cuMemcpyDtoHAsync");

            // Synchronize the stream
            checkCudaError(cuStreamSynchronize(streams[j]), "cuStreamSynchronize");
        }
    }

    int deviceCount;
    CUdevice devices[NUM_DEVICES];
    CUcontext contexts[NUM_DEVICES];
    CUmodule modules[NUM_DEVICES];
    CUfunction functions[NUM_DEVICES];
    CUstream streams[NUM_DEVICES];
    CUevent events[NUM_DEVICES];
    CUdeviceptr d_A[NUM_DEVICES];
    CUdeviceptr d_B[NUM_DEVICES];
    CUdeviceptr d_C[NUM_DEVICES];
};

// Define a test case using the fixture class
TEST_F(VecAddTest, Result) {
    // Allocate and initialize host input and output vectors
    float* h_A = new float[N * NUM_DEVICES];
    float* h_B = new float[N * NUM_DEVICES];
    float* h_C = new float[N * NUM_DEVICES];

    for (int i = 0; i < N * NUM_DEVICES; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = 0.0f;
    }

    // Launch the kernel on each device with different streams and events
    launchKernel(h_A, h_B, h_C);

    // Verify the results
    for (int i = 0; i < N * NUM_DEVICES; i++) {
        EXPECT_NEAR(h_A[i] + h_B[i], h_C[i], 1e-5) << "at index " << i;
    }

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}
