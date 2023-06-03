#include <mpi.h>
#include "test_utils.h"

// 每个 MPI 进程都有一个 CUDA 上下文和一个对应于它自己的设备的模块
class CudaEventTest : public ::testing::Test {
 protected:
  void SetUp() override {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);

    devices_ = new CUdevice[size_];
    contexts_ = new CUcontext[size_];
    modules_ = new CUmodule[size_];

    for (int i = 0; i < size_; ++i) {
      checkError(cuInit(0));
      checkError(cuDeviceGet(&devices_[i], i));
      checkError(cuCtxCreate(&contexts_[i], 0, devices_[i]));
      checkError(cuModuleLoad(&modules_[i], "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx"));
    }
  }

  void TearDown() override {
    for (int i = 0; i < size_; ++i) {
      checkError(cuModuleUnload(modules_[i]));
      checkError(cuCtxDestroy(contexts_[i]));
    }

    delete[] devices_;
    delete[] contexts_;
    delete[] modules_;

    MPI_Finalize();
  }

  int rank_;
  int size_;
  CUdevice* devices_;
  CUcontext* contexts_;
  CUmodule* modules_;
};


extern "C" __global__ void matmul(float* A, float* B, float* C, int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < N && idy < M) {
        float sum = 0;
        for (int i = 0; i < N; i++) {
            sum += A[idx * N + i] * B[i * N + idy];
        }
        C[idx * N + idy] = sum;
    }
}

TEST_F(CudaEventTest, CaseOne) {
  const int K = 4;
  CUstream streams[K];
  for (int i = 0; i < K; i++) {
    checkError(cuStreamCreate(&streams[i], CU_STREAM_DEFAULT));
  }

  checkError(cuCtxPushCurrent(contexts_[rank_]));

  CUfunction kernel;
  checkError(cuModuleGetFunction(&kernel, modules_[rank_], "matmul"));

  const int N = 1000;
  const int M = N / size_; // each process computes a portion of the result

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  checkError(cuMemAllocHost((void**)&h_A, N * M * sizeof(float)));
  checkError(cuMemAllocHost((void**)&h_B, N * N * sizeof(float))); // all processes need the full B matrix
  checkError(cuMemAllocHost((void**)&h_C, N * M * sizeof(float)));
  checkError(cuMemAlloc(&d_A, N * M * sizeof(float)));
  checkError(cuMemAlloc(&d_B, N * N * sizeof(float)));
  checkError(cuMemAlloc(&d_C, N * M * sizeof(float)));

  // Initialize h_A and h_B with some values
  // ... omitted for brevity
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

  // Make sure all streams wait for the data transfer to complete before executing the kernel
  for (int i = 0; i < K; i++) {
    checkError(cuStreamWaitEvent(streams[i], event, 0));
    void *args[] = {&d_A, &d_B, &d_C, &N, &M, &i, &K};
    checkError(cuLaunchKernel(kernel, M / 256, 1, 1, 256, 1, 1, 0, streams[i], args, NULL));
  }

  checkError(cuMemcpyDtoHAsync(h_C, d_C, N * M * sizeof(float), streams[0]));
  checkError(cuStreamSynchronize(streams[0]));

  // Switch back to the original context
  checkError(cuCtxPopCurrent(NULL));

  // Exchange output matrices among different processes using MPI
  float *h_C_all;
  checkError(cuMemAllocHost((void**)&h_C_all, N * N * sizeof(float)));
  MPI_Allgather(h_C, N * M, MPI_FLOAT, h_C_all, N * M, MPI_FLOAT, MPI_COMM_WORLD);

  // Verify results
  // ... omitted for brevity
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
}

extern "C" {

__global__ void image_process(unsigned char* input, unsigned char* output, int width, int height) {
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = idx_y * width + idx_x;

    if(idx_x < width && idx_y < height) {
        unsigned char pixel = input[idx];
        // A simple operation: increase pixel value by a constant factor
        output[idx] = pixel + 10;
    }
}

}


TEST_F(CudaEventTest, CaseTwo) {
  const int K = 4;
  pthread_t threads[K];
  
  void* thread_func(void* arg) {
    int tid = (int)(long)arg;

    // Set the context for this thread
    checkError(cuCtxSetCurrent(contexts_[tid]));

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
