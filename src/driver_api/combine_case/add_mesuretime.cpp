// 场景一：想要在GPU上执行两个不同的kernel函数，分别处理两个不同的数据集。你可以创建两个stream，分别将数据集拷贝到GPU上，并在对应的stream上调用kernel函数。这样，两个kernel函数可以在GPU上同时运行，而不需要等待对方完成。
// 场景二：想要在GPU上执行一个kernel函数，但是该函数需要多次访问主机内存中的数据。你可以创建一个stream，并将数据分成多个块，每次拷贝一个块到GPU上，并在该stream上调用kernel函数。这样，你可以利用stream Overlapping来隐藏数据传输的延迟，让kernel函数在等待数据的同时继续执行其他部分。
// 场景三：想要在GPU上执行一个复杂的计算任务，该任务可以分解为多个子任务，每个子任务都需要一定的输入和输出数据。你可以创建多个stream，并将每个子任务分配给一个stream。每个stream负责将输入数据拷贝到GPU上，并调用相应的kernel函数，并将输出数据拷贝回主机内存。这样，你可以利用stream Overlapping来并行执行多个子任务，提高计算效率。

#include "test_utils.h"


class StreamOverlapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    checkError(cuInit(0));
    checkError(cuDeviceGet(&device_, 0));
    checkError(cuCtxCreate(&context_, 0, device_));
    checkError(cuModuleLoad(
        &module_,
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx"));
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