// 场景一：你想要在GPU上执行两个不同的kernel函数，分别处理两个不同的数据集。你可以创建两个stream，分别将数据集拷贝到GPU上，并在对应的stream上调用kernel函数。这样，两个kernel函数可以在GPU上同时运行，而不需要等待对方完成。
// 场景二：你想要在GPU上执行一个kernel函数，但是该函数需要多次访问主机内存中的数据。你可以创建一个stream，并将数据分成多个块，每次拷贝一个块到GPU上，并在该stream上调用kernel函数。这样，你可以利用stream Overlapping来隐藏数据传输的延迟，让kernel函数在等待数据的同时继续执行其他部分。
// 场景三：你想要在GPU上执行一个复杂的计算任务，该任务可以分解为多个子任务，每个子任务都需要一定的输入和输出数据。你可以创建多个stream，并将每个子任务分配给一个stream。每个stream负责将输入数据拷贝到GPU上，并调用相应的kernel函数，并将输出数据拷贝回主机内存。这样，你可以利用stream Overlapping来并行执行多个子任务，提高计算效率。


// Include the necessary headers
#include <cuda.h>
#include <gtest/gtest.h>

// Define a macro to check cuda errors
#define CUDA_CHECK(call) \
  do { \
    CUresult error = call; \
    ASSERT_EQ(error, CUDA_SUCCESS) << "CUDA error: " << error; \
  } while (0)

// Define a test fixture class for cuda driver api and stream Overlapping
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

// Define a test case for scenario one
TEST_F(CudaDriverTest, ScenarioOne) {
  // Create two streams
  CUstream stream1, stream2;
  CUDA_CHECK(cuStreamCreate(&stream1, CU_STREAM_DEFAULT));
  CUDA_CHECK(cuStreamCreate(&stream2, CU_STREAM_DEFAULT));
  // Get the kernel function handles
  CUfunction kernel1, kernel2;
  CUDA_CHECK(cuModuleGetFunction(&kernel1, module_, "kernel1"));
  CUDA_CHECK(cuModuleGetFunction(&kernel2, module_, "kernel2"));
  // Allocate host and device memory for data sets
  const int N = 1000;
  float *h_data1, *h_data2, *d_data1, *d_data2;
  CUDA_CHECK(cuMemAllocHost((void**)&h_data1, N * sizeof(float)));
  CUDA_CHECK(cuMemAllocHost((void**)&h_data2, N * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_data1, N * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_data2, N * sizeof(float)));
  // Initialize host data sets
  for (int i = 0; i < N; i++) {
    h_data1[i] = i;
    h_data2[i] = i + 1;
  }
  // Copy data sets to device memory asynchronously
  CUDA_CHECK(cuMemcpyHtoDAsync(d_data1, h_data1, N * sizeof(float), stream1));
  CUDA_CHECK(cuMemcpyHtoDAsync(d_data2, h_data2, N * sizeof(float), stream2));
  // Set up kernel parameters
  void *args1[] = {&d_data1, &N};
  void *args2[] = {&d_data2, &N};
  // Launch kernel functions on streams
  CUDA_CHECK(cuLaunchKernel(kernel1, N / 256, 1, 1, 256, 1, 1, 0, stream1, args1, NULL));
  CUDA_CHECK(cuLaunchKernel(kernel2, N / 256, 1, 1, 256, 1, 1, 0, stream2, args2, NULL));
  // Copy data sets back to host memory asynchronously
  CUDA_CHECK(cuMemcpyDtoHAsync(h_data1, d_data1, N * sizeof(float), stream1));
  CUDA_CHECK(cuMemcpyDtoHAsync(h_data2, d_data2, N * sizeof(float), stream2));
  // Synchronize streams
  CUDA_CHECK(cuStreamSynchronize(stream1));
  CUDA_CHECK(cuStreamSynchronize(stream2));
  // Verify the results
  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(h_data1[i], i * i);
    ASSERT_FLOAT_EQ(h_data2[i], (i + 1) * (i + 1));
  }
  // Free host and device memory
  CUDA_CHECK(cuMemFreeHost(h_data1));
  CUDA_CHECK(cuMemFreeHost(h_data2));
  CUDA_CHECK(cuMemFree(d_data1));
  CUDA_CHECK(cuMemFree(d_data2));
  // Destroy streams
  CUDA_CHECK(cuStreamDestroy(stream1));
  CUDA_CHECK(cuStreamDestroy(stream2));
}

// Define a test case for scenario two
TEST_F(CudaDriverTest, ScenarioTwo) {
  // Create a stream
  CUstream stream;
  CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  // Get the kernel function handle
  CUfunction kernel;
  CUDA_CHECK(cuModuleGetFunction(&kernel, module_, "kernel"));
  // Allocate host and device memory for data
  const int N = 1000;
  const int M = 10;
  float *h_data, *d_data;
  CUDA_CHECK(cuMemAllocHost((void**)&h_data, N * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_data, N * sizeof(float)));
  // Initialize host data
  for (int i = 0; i < N; i++) {
    h_data[i] = i;
  }
  // Copy data to device memory in chunks and launch kernel function on stream
  for (int i = 0; i < N; i += M) {
    CUDA_CHECK(cuMemcpyHtoDAsync(d_data + i, h_data + i, M * sizeof(float), stream));
    void *args[] = {&d_data, &i, &M};
    CUDA_CHECK(cuLaunchKernel(kernel, M / 256, 1, 1, 256, 1, 1, 0, stream, args, NULL));
  }
  // Copy data back to host memory on stream
  CUDA_CHECK(cuMemcpyDtoHAsync(h_data, d_data, N * sizeof(float), stream));
  // Synchronize stream
  CUDA_CHECK(cuStreamSynchronize(stream));
  // Verify the results
  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(h_data[i], i * i);
  }
  // Free host and device memory
  CUDA_CHECK(cuMemFreeHost(h_data));
  CUDA_CHECK(cuMemFree(d_data));
  // Destroy stream
  CUDA_CHECK(cuStreamDestroy(stream));
}

// Define a test case for scenario three
TEST_F(CudaDriverTest, ScenarioThree) {
  // Create multiple streams
  const int K = 4;
  CUstream streams[K];
  for (int i = 0; i < K; i++) {
    CUDA_CHECK(cuStreamCreate(&streams[i], CU_STREAM_DEFAULT));
  }
  // Get the kernel function handles
  CUfunction kernels[K];
  for (int i = 0; i < K; i++) {
    char name[10];
    sprintf(name, "kernel%d", i + 1);
    CUDA_CHECK(cuModuleGetFunction(&kernels[i], module_, name));
  }
  // Allocate host and device memory for input and output data
  const int N = 1000;
  float *h_input, *h_output, *d_input, *d_output;
  CUDA_CHECK(cuMemAllocHost((void**)&h_input, N * sizeof(float)));
  CUDA_CHECK(cuMemAllocHost((void**)&h_output, N * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_input, N * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_output, N * sizeof(float)));
  // Initialize host input data
  for (int i = 0; i < N; i++) {
    h_input[i] = i;
  }
  // Copy input data to device memory on stream zero
  CUDA_CHECK(cuMemcpyHtoDAsync(d_input, h_input, N * sizeof(float), streams[0]));
  // Launch kernel functions on different streams
  for (int i = 0; i < K; i++) {
    void *args[] = {&d_input, &d_output, &N};
    CUDA_CHECK(cuLaunchKernel(kernels[i], N / 256, 1, 1, 256, 1, 1, 0, streams[i], args, NULL));
  }
  // Copy output data back to host memory on stream zero
  CUDA_CHECK(cuMemcpyDtoHAsync(h_output, d_output, N * sizeof(float), streams[0]));
  // Synchronize stream zero
  CUDA_CHECK(cuStreamSynchronize(streams[0]));
  // Verify the results
  for (int i = 0; i < N; i++) {
    float expected = h_input[i];
    for (int j = 0; j < K; j++) {
      expected = expected * expected;
    }
    ASSERT_FLOAT_EQ(h_output[i], expected);
  }
  // Free host and device memory
  CUDA_CHECK(cuMemFreeHost(h_input));
  CUDA_CHECK(cuMemFreeHost(h_output));
  CUDA_CHECK(cuMemFree(d_input));
  CUDA_CHECK(cuMemFree(d_output));
  // Destroy streams
  for (int i = 0; i < K; i++) {
    CUDA_CHECK(cuStreamDestroy(streams[i]));
  }
}

// Run all the tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
