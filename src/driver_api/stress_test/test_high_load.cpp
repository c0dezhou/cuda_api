// Stress test1：你想要在一个GPU上执行一个大规模的矩阵乘法任务，该任务需要多次重复执行，并且每次执行都需要随机生成不同的输入矩阵。你可以使用cuda driver api来创建一个stream，并在该stream上循环执行以下步骤：分配主机和设备内存，生成随机矩阵，拷贝矩阵到设备内存，调用矩阵乘法的kernel函数，拷贝结果回主机内存，释放主机和设备内存。你可以使用gtest来计时每次执行的时间，并检查结果的正确性。
// Stress test2：你想要在一个GPU上执行一个深度学习任务，该任务需要对多个批次的数据进行前向传播和反向传播，并且每次传播都需要随机生成不同的权重和梯度。你可以使用cuda driver api来创建两个stream，一个负责前向传播，一个负责反向传播，并在每个stream上循环执行以下步骤：分配主机和设备内存，生成随机权重和梯度，拷贝数据到设备内存，调用前向传播或反向传播的kernel函数，拷贝结果回主机内存，释放主机和设备内存。你可以使用gtest来计时每次执行的时间，并检查结果的正确性。

// Include the necessary headers
#include <cuda.h>
#include <gtest/gtest.h>
#include <cstdlib>

// Define a macro to check cuda errors
#define CUDA_CHECK(call) \
  do { \
    CUresult error = call; \
    ASSERT_EQ(error, CUDA_SUCCESS) << "CUDA error: " << error; \
  } while (0)

// Define a test fixture class for cuda driver api
class CudaDriverTest : public ::testing::Test {
 protected:
  // Set up the test environment
  void SetUp() override {
    // Initialize the cuda driver api
    CUDA_CHECK(cuInit(0));
    // Get the first device handle
    CUDA_CHECK(cuDeviceGet(&device_, 0));
    // Create a context for the device
    CUDA_CHECK(cuCtxCreate(&context_, 0, device_));
    // Load the module containing the kernel functions
    CUDA_CHECK(cuModuleLoad(&module_, "kernel.ptx"));
  }

  // Tear down the test environment
  void TearDown() override {
    // Unload the module
    CUDA_CHECK(cuModuleUnload(module_));
    // Destroy the context
    CUDA_CHECK(cuCtxDestroy(context_));
  }

  // Declare some common variables
  CUdevice device_;
  CUcontext context_;
  CUmodule module_;
};

// Define a test case for stress test one
TEST_F(CudaDriverTest, StressTestOne) {
  // Create a stream
  CUstream stream;
  CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  // Get the kernel function handle
  CUfunction kernel;
  CUDA_CHECK(cuModuleGetFunction(&kernel, module_, "matmul"));
  // Allocate host and device memory for input and output matrices
  const int N = 1000;
  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  CUDA_CHECK(cuMemAllocHost((void**)&h_A, N * N * sizeof(float)));
  CUDA_CHECK(cuMemAllocHost((void**)&h_B, N * N * sizeof(float)));
  CUDA_CHECK(cuMemAllocHost((void**)&h_C, N * N * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_A, N * N * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_B, N * N * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_C, N * N * sizeof(float)));
  // Repeat the matrix multiplication for 100 times
  for (int i = 0; i < 100; i++) {
    // Generate random matrices
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        h_A[j * N + k] = rand() % 10;
        h_B[j * N + k] = rand() % 10;
      }
    }
    // Copy input matrices to device memory on stream
    CUDA_CHECK(cuMemcpyHtoDAsync(d_A, h_A, N * N * sizeof(float), stream));
    CUDA_CHECK(cuMemcpyHtoDAsync(d_B, h_B, N * N * sizeof(float), stream));
    // Set up kernel parameters
    void *args[] = {&d_A, &d_B, &d_C, &N};
    // Launch kernel function on stream
    CUDA_CHECK(cuLaunchKernel(kernel, N / 256, 1, 1, 256, 1, 1, 0, stream, args, NULL));
    // Copy output matrix back to host memory on stream
    CUDA_CHECK(cuMemcpyDtoHAsync(h_C, d_C, N * N * sizeof(float), stream));
    // Synchronize stream
    CUDA_CHECK(cuStreamSynchronize(stream));
    // Verify the results
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        float expected = 0;
        for (int l = 0; l < N; l++) {
          expected += h_A[j * N + l] * h_B[l * N + k];
        }
        ASSERT_FLOAT_EQ(h_C[j * N + k], expected);
      }
    }
  }
  // Free host and device memory
  CUDA_CHECK(cuMemFreeHost(h_A));
  CUDA_CHECK(cuMemFreeHost(h_B));
  CUDA_CHECK(cuMemFreeHost(h_C));
  CUDA_CHECK(cuMemFree(d_A));
  CUDA_CHECK(cuMemFree(d_B));
  CUDA_CHECK(cuMemFree(d_C));
  // Destroy stream
  CUDA_CHECK(cuStreamDestroy(stream));
}

// Define a test case for stress test three
TEST_F(CudaDriverTest, StressTestThree) {
  // Create two streams
  CUstream stream1, stream2;
  CUDA_CHECK(cuStreamCreate(&stream1, CU_STREAM_DEFAULT));
  CUDA_CHECK(cuStreamCreate(&stream2, CU_STREAM_DEFAULT));
  // Get the kernel function handles
  CUfunction kernel1, kernel2;
  CUDA_CHECK(cuModuleGetFunction(&kernel1, module_, "forward"));
  CUDA_CHECK(cuModuleGetFunction(&kernel2, module_, "backward"));
  // Allocate host and device memory for input and output data
  const int N = 1000;
  float *h_input, *h_output, *d_input, *d_output;
  CUDA_CHECK(cuMemAllocHost((void**)&h_input, N * sizeof(float)));
  CUDA_CHECK(cuMemAllocHost((void**)&h_output, N * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_input, N * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_output, N * sizeof(float)));
  // Repeat the deep learning task for 100 times
  for (int i = 0; i < 100; i++) {
    // Generate random weights and gradients
    for (int j = 0; j < N; j++) {
      h_input[j] = rand() % 10;
      h_output[j] = rand() % 10;
    }
    // Copy input data to device memory on stream one
    CUDA_CHECK(cuMemcpyHtoDAsync(d_input, h_input, N * sizeof(float), stream1));
    // Set up kernel parameters
    void *args1[] = {&d_input, &d_output, &N};
    void *args2[] = {&d_output, &d_input, &N};
    // Launch kernel function for forward propagation on stream one
    CUDA_CHECK(cuLaunchKernel(kernel1, N / 256, 1, 1, 256, 1, 1, 0, stream1, args1, NULL));
    // Create an event to record the completion of forward propagation
    CUevent event;
    CUDA_CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventRecord(event, stream1));
    // Launch kernel function for backward propagation on stream two
    // Wait for the event before launching the kernel
    CUDA_CHECK(cuStreamWaitEvent(stream2, event, 0));
    CUDA_CHECK(cuLaunchKernel(kernel2, N / 256, 1, 1, 256, 1, 1, 0, stream2, args2, NULL));
    // Copy output data back to host memory on stream two
    CUDA_CHECK(cuMemcpyDtoHAsync(h_output, d_input, N * sizeof(float), stream2));
    // Synchronize stream two
    CUDA_CHECK(cuStreamSynchronize(stream2));
    // Verify the results
    for (int j = 0; j < N; j++) {
      ASSERT_FLOAT_EQ(h_output[j], h_input[j] * h_input[j] * h_input[j]);
    }
  }
  // Free host and device memory
  CUDA_CHECK(cuMemFreeHost(h_input));
  CUDA_CHECK(cuMemFreeHost(h_output));
  CUDA_CHECK(cuMemFree(d_input));
  CUDA_CHECK(cuMemFree(d_output));
  // Destroy streams and event
  CUDA_CHECK(cuStreamDestroy(stream1));
  CUDA_CHECK(cuStreamDestroy(stream2));
  CUDA_CHECK(cuEventDestroy(event));
}

// Run all the tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


// kernel
// cbrtf是一个数学函数，它计算一个浮点数的立方根。它是cbrt函数的单精度版本，cbrt函数计算一个双精度数的立方根。cbrtf函数在<math.h>头文件中定义1。你可以使用cbrtf函数来计算一些数学表达式，例如：


// Define a kernel function for matrix multiplication
__global__ void matmul(float *A, float *B, float *C, int N) {
  // Get the thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Check the thread index is within the matrix range
  if (i < N) {
    // Compute one row of the output matrix
    for (int j = 0; j < N; j++) {
      float sum = 0;
      for (int k = 0; k < N; k++) {
        sum += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// Define a kernel function for forward propagation
__global__ void forward(float *input, float *output, int N) {
  // Get the thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Check the thread index is within the data range
  if (i < N) {
    // Compute the output as the square of the input
    output[i] = input[i] * input[i];
  }
}

// Define a kernel function for backward propagation
__global__ void backward(float *output, float *input, int N) {
  // Get the thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Check the thread index is within the data range
  if (i < N) {
    // Compute the input as the cube root of the output
    input[i] = cbrtf(output[i]);
  }
}
