// 场景一：想要在GPU上执行两个不同的kernel函数，分别处理两个不同的数据集。你可以创建两个stream，分别将数据集拷贝到GPU上，并在对应的stream上调用kernel函数。这样，两个kernel函数可以在GPU上同时运行，而不需要等待对方完成。
// 场景二：想要在GPU上执行一个kernel函数，但是该函数需要多次访问主机内存中的数据。你可以创建一个stream，并将数据分成多个块，每次拷贝一个块到GPU上，并在该stream上调用kernel函数。这样，你可以利用stream Overlapping来隐藏数据传输的延迟，让kernel函数在等待数据的同时继续执行其他部分。
// 场景三：想要在GPU上执行一个复杂的计算任务，该任务可以分解为多个子任务，每个子任务都需要一定的输入和输出数据。你可以创建多个stream，并将每个子任务分配给一个stream。每个stream负责将输入数据拷贝到GPU上，并调用相应的kernel函数，并将输出数据拷贝回主机内存。这样，你可以利用stream Overlapping来并行执行多个子任务，提高计算效率。

#include "test_utils.h"


#include <cuda.h>
#include <gtest/gtest.h>

// // The kernel function you provided
// __global__ void vec_multiply_2_withidx(float* data, int startIndex, int endIndex) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= startIndex && tid < endIndex) {
//         data[tid] *= 2.0f;
//     }
// }

// A test case using gtest
TEST(CudaDriverApiTest, TwoStreams) {
    // Initialize the CUDA driver API
    checkError(cuInit(0));

    // Get the first device
    CUdevice device;
    checkError(cuDeviceGet(&device, 0));

    // Create a context
    CUcontext context;
    checkError(cuCtxCreate(&context, 0, device));
    // Allocate two arrays on host memory
    const int N = 1024;
    float* h_data1 = new float[N];
    float* h_data2 = new float[N];

    // Initialize the host arrays with some values
    for (int i = 0; i < N; i++) {
        h_data1[i] = i * 0.1f;
        h_data2[i] = i * 0.2f;
    }

    // Allocate two arrays on device memory
    CUdeviceptr d_data1, d_data2;
    checkError(cuMemAlloc(&d_data1, N * sizeof(float)));
    checkError(cuMemAlloc(&d_data2, N * sizeof(float)));

    // Create two streams
    CUstream stream1, stream2;
    checkError(cuStreamCreate(&stream1, 0));
    checkError(cuStreamCreate(&stream2, 0));

    // Copy the host arrays to the device arrays asynchronously on the streams
    // checkError(cuMemcpyHtoDAsync(d_data1, h_data1, N * sizeof(float), stream1));
    // checkError(cuMemcpyHtoDAsync(d_data2, h_data2, N * sizeof(float), stream2));

    // Set up the kernel parameters
    void* args1[3] = {&d_data1, &N/4, &N/2};
    void* args2[3] = {&d_data2, &N/2, &N};
    
    // Create two events for each stream
CUevent start1, stop1, start2, stop2;
checkError(cuEventCreate(&start1, 0));
checkError(cuEventCreate(&stop1, 0));
checkError(cuEventCreate(&start2, 0));
checkError(cuEventCreate(&stop2, 0));
// Create two events for the total time
CUevent start_total, stop_total;
checkError(cuEventCreate(&start_total, 0));
checkError(cuEventCreate(&stop_total, 0));

// Record the start event before copying data on stream1
checkError(cuEventRecord(start_total, 0));

// Record the start event on stream1 before copying data
checkError(cuEventRecord(start1, stream1));

// Copy the host array to the device array asynchronously on stream1
checkError(cuMemcpyHtoDAsync(d_data1, h_data1, N * sizeof(float), stream1));

// Record the stop event on stream1 after launching the kernel
checkError(cuEventRecord(stop1, stream1));

// Launch the kernel on stream1
checkError(cuLaunchKernel(vec_multiply_2_withidx, N/256, 1, 1, 256, 1, 1, 0, stream1, args1, NULL));

// Record the start event on stream2 before copying data
checkError(cuEventRecord(start2, stream2));

// Copy the host array to the device array asynchronously on stream2
checkError(cuMemcpyHtoDAsync(d_data2, h_data2, N * sizeof(float), stream2));

// Record the stop event on stream2 after launching the kernel
checkError(cuEventRecord(stop2, stream2));

// Launch the kernel on stream2
checkError(cuLaunchKernel(vec_multiply_2_withidx, N/256, 1, 1, 256, 1, 1, 0, stream2, args2, NULL));

checkError(cuEventRecord(stop_total, 0));

// Synchronize the events
checkError(cuEventSynchronize(stop1));
checkError(cuEventSynchronize(stop2));

// Calculate the elapsed time for each stream
float time1, time2;
checkError(cuEventElapsedTime(&time1, start1, stop1));
checkError(cuEventElapsedTime(&time2, start2, stop2));

// Print the results
printf("Time for stream1: %f ms\n", time1);
printf("Time for stream2: %f ms\n", time2);


// Synchronize the event
checkError(cuEventSynchronize(stop_total));

// Calculate the elapsed time for the total time
float time_total;
checkError(cuEventElapsedTime(&time_total, start_total, stop_total));

// Print the result
printf("Time for total: %f ms\n", time_total);

// Destroy the events
checkError(cuEventDestroy(start_total));
checkError(cuEventDestroy(stop_total));

// Destroy the events
checkError(cuEventDestroy(start1));
checkError(cuEventDestroy(stop1));
checkError(cuEventDestroy(start2));
checkError(cuEventDestroy(stop2));

    // // Launch the kernel on the streams
    // checkError(cuLaunchKernel(vec_multiply_2_withidx, N/256, 1, 1, 256, 1, 1, 0, stream1, args1, NULL));
    // checkError(cuLaunchKernel(vec_multiply_2_withidx, N/256, 1, 1, 256, 1, 1, 0, stream2, args2, NULL));

    // // Synchronize the streams
    // checkError(cuStreamSynchronize(stream1));
    // checkError(cuStreamSynchronize(stream2));

    // Copy the device arrays back to the host arrays asynchronously on the streams
    checkError(cuMemcpyDtoHAsync(h_data1, d_data1, N * sizeof(float), stream1));
    checkError(cuMemcpyDtoHAsync(h_data2, d_data2, N * sizeof(float), stream2));

    // Synchronize the streams
    checkError(cuStreamSynchronize(stream1));
    checkError(cuStreamSynchronize(stream2));

    // Check the results
    for (int i = 0; i < N; i++) {
        if (i >= N/4 && i < N/2) {
            EXPECT_FLOAT_EQ(h_data1[i], i * 0.1f * 2.0f);
        } else {
            EXPECT_FLOAT_EQ(h_data1[i], i * 0.1f);
        }
        if (i >= N/2 && i < N) {
            EXPECT_FLOAT_EQ(h_data2[i], i * 0.2f * 2.0f);
        } else {
            EXPECT_FLOAT_EQ(h_data2[i], i * 0.2f);
        }
    }

    // Free the memory and destroy the streams
    delete[] h_data1;
    delete[] h_data2;
    checkError(cuMemFree(d_data1));
    checkError(cuMemFree(d_data2));
    checkError(cuStreamDestroy(stream1));
    checkError(cuStreamDestroy(stream2));

    // Destroy the context
    checkError(cuCtxDestroy(context));
}



class StreamOverlapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    checkError(cuInit(0));
    checkError(cuDeviceGet(&device_, 0));
    checkError(cuCtxCreate(&context_, 0, device_));
    checkError(cuModuleLoad(
        &module_,
        "cuda.ptx"));
  }

  void TearDown() override {
    checkError(cuModuleUnload(module_));
    checkError(cuCtxDestroy(context_));
  }

  CUdevice device_;
  CUcontext context_;
  CUmodule module_;
};

TEST_F(StreamOverlapTest, ScenarioOne) {
  CUstream stream1, stream2;
  checkError(cuStreamCreate(&stream1, CU_STREAM_DEFAULT));
  checkError(cuStreamCreate(&stream2, CU_STREAM_DEFAULT));
  CUfunction kernel1, kernel2;
  checkError(cuModuleGetFunction(&kernel1, module_, "kernel1"));
  checkError(cuModuleGetFunction(&kernel2, module_, "kernel2"));
  int N = 1000;
  float *h_data1, *h_data2, *d_data1, *d_data2;
  checkError(cuMemAllocHost((void**)&h_data1, N * sizeof(float)));
  checkError(cuMemAllocHost((void**)&h_data2, N * sizeof(float)));
  checkError(cuMemAlloc((CUdeviceptr*)&d_data1, N * sizeof(float)));
  checkError(cuMemAlloc((CUdeviceptr*)&d_data2, N * sizeof(float)));

  for (int i = 0; i < N; i++) {
    h_data1[i] = i;
    h_data2[i] = i + 1;
  }
  checkError(cuMemcpyHtoDAsync((CUdeviceptr)d_data1, h_data1,
                               N * sizeof(float), stream1));
  checkError(cuMemcpyHtoDAsync((CUdeviceptr)d_data2, h_data2,
                               N * sizeof(float), stream2));

  void *args1[] = {&d_data1, &N};
  void *args2[] = {&d_data2, &N};
  checkError(cuLaunchKernel(kernel1, N / 256, 1, 1, 256, 1, 1, 0, stream1, args1, NULL));
  checkError(cuLaunchKernel(kernel2, N / 256, 1, 1, 256, 1, 1, 0, stream2, args2, NULL));

  checkError(cuMemcpyDtoHAsync(h_data1, (CUdeviceptr)d_data1, N * sizeof(float),
                               stream1));
  checkError(cuMemcpyDtoHAsync(h_data2, (CUdeviceptr)d_data2, N * sizeof(float),
                               stream2));

  checkError(cuStreamSynchronize(stream1));
  checkError(cuStreamSynchronize(stream2));

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(h_data1[i], i * i);
    ASSERT_FLOAT_EQ(h_data2[i], (i + 1) * (i + 1));
  }

  checkError(cuMemFreeHost(h_data1));
  checkError(cuMemFreeHost(h_data2));
  checkError(cuMemFree((CUdeviceptr)d_data1));
  checkError(cuMemFree((CUdeviceptr)d_data2));

  checkError(cuStreamDestroy(stream1));
  checkError(cuStreamDestroy(stream2));
}


TEST_F(StreamOverlapTest, ScenarioTwo) {

  CUstream stream;
  checkError(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  CUfunction kernel;
  checkError(cuModuleGetFunction(&kernel, module_, "kernel"));

  int N = 1000;
  int M = 10;
  float *h_data, *d_data;
  checkError(cuMemAllocHost((void**)&h_data, N * sizeof(float)));
  checkError(cuMemAlloc((CUdeviceptr*)&d_data, N * sizeof(float)));

  for (int i = 0; i < N; i++) {
    h_data[i] = i;
  }

  for (int i = 0; i < N; i += M) {
    checkError(cuMemcpyHtoDAsync((CUdeviceptr)d_data + i, h_data + i,
                                 M * sizeof(float), stream));
    void *args[] = {&d_data, &i, &M};
    checkError(cuLaunchKernel(kernel, M / 256, 1, 1, 256, 1, 1, 0, stream, args, NULL));
  }
  checkError(cuMemcpyDtoHAsync(h_data, (CUdeviceptr)d_data, N * sizeof(float),
                               stream));

  checkError(cuStreamSynchronize(stream));

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(h_data[i], i * i);
  }

  checkError(cuMemFreeHost(h_data));
  checkError(cuMemFree((CUdeviceptr)d_data));

  checkError(cuStreamDestroy(stream));
}

TEST_F(StreamOverlapTest, ScenarioThree) {
  const int K = 4;
  CUstream streams[K];
  for (int i = 0; i < K; i++) {
    checkError(cuStreamCreate(&streams[i], CU_STREAM_DEFAULT));
  }
  CUfunction kernels[K];
  for (int i = 0; i < K; i++) {
    char name[10];
    sprintf(name, "kernel%d", i + 1);
    checkError(cuModuleGetFunction(&kernels[i], module_, name));
  }
  int N = 1000;
  float *h_input, *h_output, *d_input, *d_output;
  checkError(cuMemAllocHost((void**)&h_input, N * sizeof(float)));
  checkError(cuMemAllocHost((void**)&h_output, N * sizeof(float)));
  checkError(cuMemAlloc((CUdeviceptr*)&d_input, N * sizeof(float)));
  checkError(cuMemAlloc((CUdeviceptr*)&d_output, N * sizeof(float)));

  for (int i = 0; i < N; i++) {
    h_input[i] = i;
  }

  checkError(cuMemcpyHtoDAsync((CUdeviceptr)d_input, h_input, N * sizeof(float),
                               streams[0]));

  for (int i = 0; i < K; i++) {
    void *args[] = {&d_input, &d_output, &N};
    checkError(cuLaunchKernel(kernels[i], N / 256, 1, 1, 256, 1, 1, 0, streams[i], args, NULL));
  }

  checkError(cuMemcpyDtoHAsync(h_output, (CUdeviceptr)d_output,
                               N * sizeof(float), streams[0]));

  checkError(cuStreamSynchronize(streams[0]));

  for (int i = 0; i < N; i++) {
    float expected = h_input[i];
    for (int j = 0; j < K; j++) {
      expected = expected * expected;
    }
    ASSERT_FLOAT_EQ(h_output[i], expected);
  }

  checkError(cuMemFreeHost(h_input));
  checkError(cuMemFreeHost(h_output));
  checkError(cuMemFree((CUdeviceptr)d_input));
  checkError(cuMemFree((CUdeviceptr)d_output));

  for (int i = 0; i < K; i++) {
    checkError(cuStreamDestroy(streams[i]));
  }
}