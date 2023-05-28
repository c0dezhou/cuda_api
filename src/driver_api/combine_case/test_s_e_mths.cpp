// Case一：你想要在多个GPU上执行一个大规模的矩阵乘法任务，该任务可以分解为多个子任务，每个子任务都需要一定的输入和输出数据。你可以创建多个进程，每个进程负责一个GPU，并在每个进程中创建多个stream，每个stream负责一个子任务。你可以使用Event来同步不同stream之间的数据传输和计算，以及不同进程之间的数据交换。
// Case二：你想要在一个GPU上执行一个图像处理任务，该任务包括多个步骤，每个步骤都需要一定的输入和输出数据。你可以创建多个线程，每个线程负责一个步骤，并在每个线程中创建一个stream，每个stream负责一个图像。你可以使用Event来同步不同线程之间的数据传输和计算，以及主机和设备之间的数据拷贝。
// Case三：你想要在一个GPU上执行一个深度学习任务，该任务包括前向传播和反向传播两个阶段，每个阶段都需要一定的输入和输出数据。你可以创建两个stream，一个负责前向传播，一个负责反向传播，并在每个stream中调用相应的kernel函数。你可以使用Event来同步两个stream之间的数据传输和计算，以及主机和设备之间的数据拷贝。
// Include the necessary headers
#include <cuda.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <pthread.h>

// Define a macro to check cuda errors
#define CUDA_CHECK(call) \
  do { \
    CUresult error = call; \
    ASSERT_EQ(error, CUDA_SUCCESS) << "CUDA error: " << error; \
  } while (0)

// Define a test fixture class for Event, multi-threading, multi-processing, and multi-streaming
class CudaEventTest : public ::testing::Test {
 protected:
  // Set up the test environment
  void SetUp() override {
    // Initialize MPI
    MPI_Init(NULL, NULL);
    // Get the rank and size of MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    // Initialize the cuda driver api
    CUDA_CHECK(cuInit(0));
    // Get the device handle according to the rank
    CUDA_CHECK(cuDeviceGet(&device_, rank_));
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
    // Finalize MPI
    MPI_Finalize();
  }

  // Declare some common variables
  int rank_;
  int size_;
  CUdevice device_;
  CUcontext context_;
  CUmodule module_;
};

// Define a test case for case one
TEST_F(CudaEventTest, CaseOne) {
  // Create multiple streams
  const int K = 4;
  CUstream streams[K];
  for (int i = 0; i < K; i++) {
    CUDA_CHECK(cuStreamCreate(&streams[i], CU_STREAM_DEFAULT));
  }
  // Get the kernel function handle
  CUfunction kernel;
  CUDA_CHECK(cuModuleGetFunction(&kernel, module_, "matmul"));
  // Allocate host and device memory for input and output matrices
  const int N = 1000;
  const int M = N / size_;
  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  CUDA_CHECK(cuMemAllocHost((void**)&h_A, N * M * sizeof(float)));
  CUDA_CHECK(cuMemAllocHost((void**)&h_B, N * N * sizeof(float)));
  CUDA_CHECK(cuMemAllocHost((void**)&h_C, N * M * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_A, N * M * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_B, N * N * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_C, N * M * sizeof(float)));
  // Initialize host input matrices
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      h_A[i * M + j] = rank_ + i + j;
    }
    for (int j = 0; j < N; j++) {
      h_B[i * N + j] = rank_ + i + j;
    }
  }
  // Copy input matrices to device memory on stream zero
  CUDA_CHECK(cuMemcpyHtoDAsync(d_A, h_A, N * M * sizeof(float), streams[0]));
  CUDA_CHECK(cuMemcpyHtoDAsync(d_B, h_B, N * N * sizeof(float), streams[0]));
  // Create an event to record the completion of data transfer
  CUevent event;
  CUDA_CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
  CUDA_CHECK(cuEventRecord(event, streams[0]));
  // Launch kernel functions on different streams
  for (int i = 0; i < K; i++) {
    // Wait for the event before launching the kernel
    CUDA_CHECK(cuStreamWaitEvent(streams[i], event, 0));
    // Set up kernel parameters
    void *args[] = {&d_A, &d_B, &d_C, &N, &M, &i, &K};
    // Launch the kernel
    CUDA_CHECK(cuLaunchKernel(kernel, M / 256, 1, 1, 256, 1, 1, 0, streams[i], args, NULL));
  }
  // Copy output matrix back to host memory on stream zero
  CUDA_CHECK(cuMemcpyDtoHAsync(h_C, d_C, N * M * sizeof(float), streams[0]));
  // Synchronize stream zero
  CUDA_CHECK(cuStreamSynchronize(streams[0]));
  // Exchange output matrices among different processes using MPI
  float *h_C_all;
  CUDA_CHECK(cuMemAllocHost((void**)&h_C_all, N * N * sizeof(float)));
  MPI_Allgather(h_C, N * M, MPI_FLOAT, h_C_all, N * M, MPI_FLOAT, MPI_COMM_WORLD);
  // Verify the results
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float expected = 0;
      for (int k = 0; k < N; k++) {
        expected += (rank_ + i + k) * (rank_ + k + j);
      }
      ASSERT_FLOAT_EQ(h_C_all[i * N + j], expected);
    }
  }
  // Free host and device memory
  CUDA_CHECK(cuMemFreeHost(h_A));
  CUDA_CHECK(cuMemFreeHost(h_B));
  CUDA_CHECK(cuMemFreeHost(h_C));
  CUDA_CHECK(cuMemFreeHost(h_C_all));
  CUDA_CHECK(cuMemFree(d_A));
  CUDA_CHECK(cuMemFree(d_B));
  CUDA_CHECK(cuMemFree(d_C));
  // Destroy streams and event
  for (int i = 0; i < K; i++) {
    CUDA_CHECK(cuStreamDestroy(streams[i]));
  }
  CUDA_CHECK(cuEventDestroy(event));
}

// Define a test case for case two
TEST_F(CudaEventTest, CaseTwo) {
  // Create multiple threads
  const int K = 4;
  pthread_t threads[K];
  // Define a thread function
  void* thread_func(void* arg) {
    // Get the thread id
    int tid = (int)(long)arg;
    // Create a stream
    CUstream stream;
    CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    // Get the kernel function handle
    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module_, "image_process"));
    // Allocate host and device memory for input and output images
    const int N = 1000;
    const int M = 1000;
    unsigned char *h_input, *h_output, *d_input, *d_output;
    CUDA_CHECK(cuMemAllocHost((void**)&h_input, N * M * sizeof(unsigned char)));
    CUDA_CHECK(cuMemAllocHost((void**)&h_output, N * M * sizeof(unsigned char)));
    CUDA_CHECK(cuMemAlloc(&d_input, N * M * sizeof(unsigned char)));
    CUDA_CHECK(cuMemAlloc(&d_output, N * M * sizeof(unsigned char)));
    // Initialize host input image
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        h_input[i * M + j] = tid + i + j;
      }
    }
    // Copy input image to device memory on stream
    CUDA_CHECK(cuMemcpyHtoDAsync(d_input, h_input, N * M * sizeof(unsigned char), stream));
    // Set up kernel parameters
    void *args[] = {&d_input, &d_output, &N, &M};
    // Launch kernel function on stream
    CUDA_CHECK(cuLaunchKernel(kernel, N / 256, 1, 1, 256, 1, 1, 0, stream, args, NULL));
    // Copy output image back to host memory on stream
    CUDA_CHECK(cuMemcpyDtoHAsync(h_output, d_output, N * M * sizeof(unsigned char), stream));
    // Synchronize stream
    CUDA_CHECK(cuStreamSynchronize(stream));
    // Verify the results
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        ASSERT_EQ(h_output[i * M + j], h_input[i * M + j] + tid);
      }
    }
    // Free host and device memory
    CUDA_CHECK(cuMemFreeHost(h_input));
    CUDA_CHECK(cuMemFreeHost(h_output));
    CUDA_CHECK(cuMemFree(d_input));
    CUDA_CHECK(cuMemFree(d_output));
    // Destroy stream
    CUDA_CHECK(cuStreamDestroy(stream));
    // Return from the thread function
    return NULL;
  }
  // Create and join threads
  for (int i = 0; i < K; i++) {
    pthread_create(&threads[i], NULL, thread_func, (void*)(long)i);
  }
  for (int i = 0; i < K; i++) {
    pthread_join(threads[i], NULL);
  }
}

// Define a test case for case three
TEST_F(CudaEventTest, CaseThree) {
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
  // Initialize host input data
  for (int i = 0; i < N; i++) {
    h_input[i] = i;
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
  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(h_output[i], h_input[i] * h_input[i] * h_input[i]);
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
