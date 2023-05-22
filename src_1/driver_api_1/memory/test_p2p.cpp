#include <gtest/gtest.h>
#include <cuda_runtime.h>

// 定义一个测试类
class cuMemcpyPeertoPeerAsyncTest : public ::testing::Test {
protected:
  // 在每个测试之前执行
  void SetUp() override {
    // 获取GPU数量
    cudaGetDeviceCount(&device_count);
    // 如果只有一个GPU，跳过所有测试
    if (device_count < 2) {
      GTEST_SKIP();
    }
    // 分配两个GPU上的内存
    cudaSetDevice(0);
    cudaMalloc(&dptr1, size);
    cudaSetDevice(1);
    cudaMalloc(&dptr2, size);
    // 创建一个流
    cudaStreamCreate(&stream);
  }

  // 在每个测试之后执行
  void TearDown() override {
    // 释放内存和流
    cudaSetDevice(0);
    cudaFree(dptr1);
    cudaSetDevice(1);
    cudaFree(dptr2);
    cudaStreamDestroy(stream);
  }

  // 定义一些常量和变量
  const size_t size = 1024; // 内存大小
  int device_count; // GPU数量
  void *dptr1; // GPU0上的内存指针
  void *dptr2; // GPU1上的内存指针
  cudaStream_t stream; // CUDA流
};

// 基本行为测试：从GPU0复制到GPU1，然后同步流，检查结果是否正确
TEST_F(cuMemcpyPeertoPeerAsyncTest, BasicBehavior) {
  // 初始化GPU0上的内存为0~1023的整数
  int *hptr = new int[size / sizeof(int)];
  for (int i = 0; i < size / sizeof(int); i++) {
    hptr[i] = i;
  }
  cudaSetDevice(0);
  cudaMemcpy(dptr1, hptr, size, cudaMemcpyHostToDevice);

  // 调用cuMemcpyPeertoPeerAsync从GPU0复制到GPU1
  cuMemcpyPeertoPeerAsync(dptr2, 1, dptr1, 0, size, stream);

  // 同步流，等待复制完成
  cudaStreamSynchronize(stream);

  // 检查GPU1上的内存是否与GPU0上的相同
  cudaMemcpy(hptr, dptr2, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size / sizeof(int); i++) {
    EXPECT_EQ(hptr[i], i);
  }

  delete[] hptr;
}

// 异常测试：传入无效的参数，检查是否返回错误码
TEST_F(cuMemcpyPeertoPeerAsyncTest, InvalidArguments) {
  // dstDevice或srcDevice超出范围，应返回CUDA_ERROR_INVALID_DEVICE
  EXPECT_EQ(cuMemcpyPeertoPeerAsync(dptr2, device_count, dptr1, 0, size, stream), CUDA_ERROR_INVALID_DEVICE);
  EXPECT_EQ(cuMemcpyPeertoPeerAsync(dptr2, -1, dptr1, 0, size, stream), CUDA_ERROR_INVALID_DEVICE);
  
  // dstPtr或srcPtr为空指针，应返回CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(cuMemcpyPeertoPeerAsync(nullptr, 1, dptr1, 0, size, stream), CUDA_ERROR_INVALID_VALUE);
  EXPECT_EQ(cuMemcpyPeertoPeerAsync(dptr2, 1, nullptr, 0, size, stream), CUDA_ERROR_INVALID_VALUE);

  // ByteCount为0，应返回CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(cuMemcpyPeertoPeerAsync(dptr2, device_count -1 , dptr1 , device_count -2 ,0 , stream), CUDA_ERROR_INVALID_VALUE);

}


// 边界值测试：传入边界值，检查是否正确处理
TEST_F(cuMemcpyPeertoPeerAsyncTest, BoundaryValues) {
  // 初始化GPU0上的内存为0~1023的整数
  int *hptr = new int[size / sizeof(int)];
  for (int i = 0; i < size / sizeof(int); i++) {
    hptr[i] = i;
  }
  cudaSetDevice(0);
  cudaMemcpy(dptr1, hptr, size, cudaMemcpyHostToDevice);

  // 调用cuMemcpyPeertoPeerAsync从GPU0复制到GPU1，ByteCount为1，只复制第一个字节
  cuMemcpyPeertoPeerAsync(dptr2, 1, dptr1, 0, 1, stream);

  // 同步流，等待复制完成
  cudaStreamSynchronize(stream);

  // 检查GPU1上的内存的第一个字节是否与GPU0上的相同，其他字节是否为0
  cudaMemcpy(hptr, dptr2, size, cudaMemcpyDeviceToHost);
  EXPECT_EQ(hptr[0] & 0xFF, 0);
  for (int i = 1; i < size / sizeof(int); i++) {
    EXPECT_EQ(hptr[i], 0);
  }

  // 调用cuMemcpyPeertoPeerAsync从GPU0复制到GPU1，ByteCount为size，复制整个内存
  cuMemcpyPeertoPeerAsync(dptr2, 1, dptr1, 0, size, stream);

  // 同步流，等待复制完成
  cudaStreamSynchronize(stream);

  // 检查GPU1上的内存是否与GPU0上的相同
  cudaMemcpy(hptr, dptr2, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size / sizeof(int); i++) {
    EXPECT_EQ(hptr[i], i);
  }

  delete[] hptr;
}

// 同步或异步行为测试：检查cuMemcpyPeertoPeerAsync是否是异步的，是否可以与其他操作重叠
TEST_F(cuMemcpyPeertoPeerAsyncTest, SyncOrAsyncBehavior) {
  // 初始化GPU0上的内存为0~1023的整数
  int *hptr = new int[size / sizeof(int)];
  for (int i = 0; i < size / sizeof(int); i++) {
    hptr[i] = i;
  }
  cudaSetDevice(0);
  cudaMemcpy(dptr1, hptr, size, cudaMemcpyHostToDevice);

  // 定义一个简单的核函数，将输入数组的每个元素加1
  auto kernel = [] __device__ (int *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
      arr[i]++;
    }
  };

  // 调用cuMemcpyPeertoPeerAsync从GPU0复制到GPU1，并在GPU0上启动核函数
  cuMemcpyPeertoPeerAsync(dptr2, 1, dptr1, 0, size, stream);
  kernel<<<(size + 255) / 256, 256>>>(static_cast<int*>(dptr1), size / sizeof(int));

  // 同步流，等待复制和核函数完成
  cudaStreamSynchronize(stream);

  // 检查GPU1上的内存是否与GPU0上的原始值相同，GPU0上的内存是否被核函数修改
  cudaMemcpy(hptr, dptr2, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size / sizeof(int); i++) {
    EXPECT_EQ(hptr[i], i);
  }
  
  cudaMemcpy(hptr, dptr1, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size / sizeof(int); i++) {
    EXPECT_EQ(hptr[i], i + 1);
  }

  delete[] hptr;
}

// 重复调用测试：检查多次调用cuMemcpyPeertoPeerAsync是否会产生正确的结果
TEST_F(cuMemcpyPeertoPeerAsyncTest, RepeatedCalls) {
  // 初始化GPU0上的内存为0~1023的整数
  int *hptr = new int[size / sizeof(int)];
  for (int i = 0; i < size / sizeof(int); i++) {
    hptr[i] = i;
  }
  cudaSetDevice(0);
  cudaMemcpy(dptr1, hptr, size, cudaMemcpyHostToDevice);

  // 调用cuMemcpyPeertoPeerAsync从GPU0复制到GPU1，然后从GPU1复制回GPU0，共10次
  for (int i = 0; i < 10; i++) {
    cuMemcpyPeertoPeerAsync(dptr2, 1, dptr1, 0, size, stream);
    cuMemcpyPeertoPeerAsync(dptr1, 0, dptr2, 1, size, stream);
  }

  // 同步流，等待复制完成
  cudaStreamSynchronize(stream);

  // 检查GPU0上的内存是否与原始值相同
  cudaMemcpy(hptr, dptr1, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size / sizeof(int); i++) {
    EXPECT_EQ(hptr[i], i);
  }

  delete[] hptr;
}

// 其他测试：检查cuMemcpyPeertoPeerAsync是否支持不同大小和对齐的内存
TEST_F(cuMemcpyPeertoPeerAsyncTest, DifferentSizesAndAlignments) {
  // 初始化GPU0上的内存为0~1023的整数
  int *hptr = new int[size / sizeof(int)];
  for (int i = 0; i < size / sizeof(int); i++) {
    hptr[i] = i;
  }
  cudaSetDevice(0);
  cudaMemcpy(dptr1, hptr, size, cudaMemcpyHostToDevice);

  // 调用cuMemcpyPeertoPeerAsync从GPU0复制到GPU1，ByteCount为size - 1，复制除了最后一个字节的所有内存
  cuMemcpyPeertoPeerAsync(dptr2, 1, dptr1, 0, size - 1, stream);

  // 同步流，等待复制完成
  cudaStreamSynchronize(stream);

  // 检查GPU1上的内存是否与GPU0上的除了最后一个字节的所有内存相同，最后一个字节是否为0
  cudaMemcpy(hptr, dptr2, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size / sizeof(int) - 1; i++) {
    EXPECT_EQ(hptr[i], i);
  }
  EXPECT_EQ(hptr[size / sizeof(int) - 1] & 0xFF, 0);

  // 调用cuMemcpyPeertoPeerAsync从GPU0复制到GPU1，dstPtr和srcPtr都加上3，ByteCount为size - 6，复制不对齐的内存
  cuMemcpyPeertoPeerAsync(static_cast<char*>(dptr2) + 3, 1, static_cast<char*>(dptr1) + 3, 0, size - 6, stream);

  // 同步流，等待复制完成
  cudaStreamSynchronize(stream);

  // 检查GPU1上的内存是否与GPU0上的不对齐的内存相同，前后三个字节是否为0
  cudaMemcpy(hptr, dptr2, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size / sizeof(int); i++) {
    if (i == 0 || i == size / sizeof(int) - 1) {
      EXPECT_EQ(hptr[i], 0);
    } else {
      EXPECT_EQ(hptr[i], (i << 8) + (i - 1));
    }
    
    delete[] hptr;
}
