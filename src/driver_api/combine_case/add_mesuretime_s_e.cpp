// Case一：你想要在多个GPU上执行一个大规模的矩阵乘法任务，该任务可以分解为多个子任务，每个子任务都需要一定的输入和输出数据。你可以创建多个进程，每个进程负责一个GPU，并在每个进程中创建多个stream，每个stream负责一个子任务。你可以使用Event来同步不同stream之间的数据传输和计算，以及不同进程之间的数据交换。
// Case三：你想要在一个GPU上执行一个深度学习任务，该任务包括前向传播和反向传播两个阶段，每个阶段都需要一定的输入和输出数据。你可以创建两个stream，一个负责前向传播，一个负责反向传播，并在每个stream中调用相应的kernel函数。你可以使用Event来同步两个stream之间的数据传输和计算，以及主机和设备之间的数据拷贝。
// Include the necessary headers

#include <mpi.h>
#include "test_utils.h"

class CudaEventTest : public ::testing::Test {
 protected:
  void SetUp() override {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    checkError(cuInit(0));
    checkError(cuDeviceGet(&device_, rank_));
    checkError(cuCtxCreate(&context_, 0, device_));
    checkError(cuModuleLoad(&module_, "kernel.ptx"));
  }

  void TearDown() override {
    checkError(cuModuleUnload(module_));
    checkError(cuCtxDestroy(context_));
    MPI_Finalize();
  }

  int rank_;
  int size_;
  CUdevice device_;
  CUcontext context_;
  CUmodule module_;
};

TEST_F(CudaEventTest, CaseOne) {
  const int K = 4;
  CUstream streams[K];
  for (int i = 0; i < K; i++) {
    checkError(cuStreamCreate(&streams[i], CU_STREAM_DEFAULT));
  }
  CUfunction kernel;
  checkError(cuModuleGetFunction(&kernel, module_, "matmul"));
  const int N = 1000;
  const int M = N / size_;
  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  checkError(cuMemAllocHost((void**)&h_A, N * M * sizeof(float)));
  checkError(cuMemAllocHost((void**)&h_B, N * N * sizeof(float)));
  checkError(cuMemAllocHost((void**)&h_C, N * M * sizeof(float)));
  checkError(cuMemAlloc(&d_A, N * M * sizeof(float)));
  checkError(cuMemAlloc(&d_B, N * N * sizeof(float)));
  checkError(cuMemAlloc(&d_C, N * M * sizeof(float)));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      h_A[i * M + j] = rank_ + i + j;
    }
    for (int j = 0; j < N; j++) {
      h_B[i * N + j] = rank_ + i + j;
    }
  }
  checkError(cuMemcpyHtoDAsync(d_A, h_A, N * M * sizeof(float), streams[0]));
  checkError(cuMemcpyHtoDAsync(d_B, h_B, N * N * sizeof(float), streams[0]));
  CUevent event;
  checkError(cuEventCreate(&event, CU_EVENT_DEFAULT));
  checkError(cuEventRecord(event, streams[0]));
  for (int i = 0; i < K; i++) {
    checkError(cuStreamWaitEvent(streams[i], event, 0));
    void *args[] = {&d_A, &d_B, &d_C, &N, &M, &i, &K};
    checkError(cuLaunchKernel(kernel, M / 256, 1, 1, 256, 1, 1, 0, streams[i], args, NULL));
  }
  checkError(cuMemcpyDtoHAsync(h_C, d_C, N * M * sizeof(float), streams[0]));
  checkError(cuStreamSynchronize(streams[0]));
  // Exchange output matrices among different processes using MPI
  float *h_C_all;
  checkError(cuMemAllocHost((void**)&h_C_all, N * N * sizeof(float)));
  MPI_Allgather(h_C, N * M, MPI_FLOAT, h_C_all, N * M, MPI_FLOAT, MPI_COMM_WORLD);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float expected = 0;
      for (int k = 0; k < N; k++) {
        expected += (rank_ + i + k) * (rank_ + k + j);
      }
      ASSERT_FLOAT_EQ(h_C_all[i * N + j], expected);
    }
  }
  checkError(cuMemFreeHost(h_A));
  checkError(cuMemFreeHost(h_B));
  checkError(cuMemFreeHost(h_C));
  checkError(cuMemFreeHost(h_C_all));
  checkError(cuMemFree(d_A));
  checkError(cuMemFree(d_B));
  checkError(cuMemFree(d_C));
  for (int i = 0; i < K; i++) {
    checkError(cuStreamDestroy(streams[i]));
  }
  checkError(cuEventDestroy(event));
}

TEST_F(CudaEventTest, CaseTwo) {
  const int K = 4;
  pthread_t threads[K];
  void* thread_func(void* arg) {
    int tid = (int)(long)arg;
    CUstream stream;
    checkError(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    CUfunction kernel;
    checkError(cuModuleGetFunction(&kernel, module_, "image_process"));
    const int N = 1000;
    const int M = 1000;
    unsigned char *h_input, *h_output, *d_input, *d_output;
    checkError(cuMemAllocHost((void**)&h_input, N * M * sizeof(unsigned char)));
    checkError(cuMemAllocHost((void**)&h_output, N * M * sizeof(unsigned char)));
    checkError(cuMemAlloc(&d_input, N * M * sizeof(unsigned char)));
    checkError(cuMemAlloc(&d_output, N * M * sizeof(unsigned char)));

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        h_input[i * M + j] = tid + i + j;
      }
    }

    checkError(cuMemcpyHtoDAsync(d_input, h_input, N * M * sizeof(unsigned char), stream));

    void *args[] = {&d_input, &d_output, &N, &M};

    checkError(cuLaunchKernel(kernel, N / 256, 1, 1, 256, 1, 1, 0, stream, args, NULL));

    checkError(cuMemcpyDtoHAsync(h_output, d_output, N * M * sizeof(unsigned char), stream));

    checkError(cuStreamSynchronize(stream));

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        ASSERT_EQ(h_output[i * M + j], h_input[i * M + j] + tid);
      }
    }

    checkError(cuMemFreeHost(h_input));
    checkError(cuMemFreeHost(h_output));
    checkError(cuMemFree(d_input));
    checkError(cuMemFree(d_output));

    checkError(cuStreamDestroy(stream));
    return NULL;
  }

  for (int i = 0; i < K; i++) {
    pthread_create(&threads[i], NULL, thread_func, (void*)(long)i);
  }
  for (int i = 0; i < K; i++) {
    pthread_join(threads[i], NULL);
  }
}


TEST_F(CudaEventTest, CaseThree) {

  CUstream stream1, stream2;
  checkError(cuStreamCreate(&stream1, CU_STREAM_DEFAULT));
  checkError(cuStreamCreate(&stream2, CU_STREAM_DEFAULT));

  CUfunction kernel1, kernel2;
  checkError(cuModuleGetFunction(&kernel1, module_, "forward"));
  checkError(cuModuleGetFunction(&kernel2, module_, "backward"));

  const int N = 1000;
  float *h_input, *h_output, *d_input, *d_output;
  checkError(cuMemAllocHost((void**)&h_input, N * sizeof(float)));
  checkError(cuMemAllocHost((void**)&h_output, N * sizeof(float)));
  checkError(cuMemAlloc(&d_input, N * sizeof(float)));
  checkError(cuMemAlloc(&d_output, N * sizeof(float)));

  for (int i = 0; i < N; i++) {
    h_input[i] = i;
  }

  checkError(cuMemcpyHtoDAsync(d_input, h_input, N * sizeof(float), stream1));

  void *args1[] = {&d_input, &d_output, &N};
  void *args2[] = {&d_output, &d_input, &N};

  checkError(cuLaunchKernel(kernel1, N / 256, 1, 1, 256, 1, 1, 0, stream1, args1, NULL));

  CUevent event;
  checkError(cuEventCreate(&event, CU_EVENT_DEFAULT));
  checkError(cuEventRecord(event, stream1));

  checkError(cuStreamWaitEvent(stream2, event, 0));
  checkError(cuLaunchKernel(kernel2, N / 256, 1, 1, 256, 1, 1, 0, stream2, args2, NULL));

  checkError(cuMemcpyDtoHAsync(h_output, d_input, N * sizeof(float), stream2));

  checkError(cuStreamSynchronize(stream2));

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(h_output[i], h_input[i] * h_input[i] * h_input[i]);
  }

  checkError(cuMemFreeHost(h_input));
  checkError(cuMemFreeHost(h_output));
  checkError(cuMemFree(d_input));
  checkError(cuMemFree(d_output));

  checkError(cuStreamDestroy(stream1));
  checkError(cuStreamDestroy(stream2));
  checkError(cuEventDestroy(event));
}

