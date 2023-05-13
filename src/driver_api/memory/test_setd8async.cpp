#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 定义测试夹具类
class cuMemsetD8AsyncTest : public ::testing::Test {
 protected:
  // 在每个测试开始之前执行
  void SetUp() override {
    // 分配设备内存
    size_t N = 100;
    cuMemAlloc(&d_ptr_, N);

    // 分配主机内存
    h_ptr_ = new unsigned char[N];
  }

  // 在每个测试结束之后执行
  void TearDown() override {
    // 释放内存
    delete[] h_ptr_;
    cuMemFree(d_ptr_);
  }

  // 成员变量
  CUdeviceptr d_ptr_; // 设备内存指针
  unsigned char* h_ptr_; // 主机内存指针
};

// 基本行为测试
TEST_F(cuMemsetD8AsyncTest, BasicBehavior) {
  // 设置设备内存为0xFF
  unsigned char uc = 0xFF;
  cuMemsetD8Async(d_ptr_, uc, N, NULL);

  // 同步设备
  cuDeviceSynchronize();

  // 拷贝设备内存到主机内存
  cuMemcpyDtoH(h_ptr_, d_ptr_, N);

  // 检查主机内存中的值是否都为0xFF
  for (size_t i = 0; i < N; i++) {
    EXPECT_EQ(h_ptr_[i], uc);
  }
}

// 异常测试
TEST_F(cuMemsetD8AsyncTest, ExceptionHandling) {
  // 设置一个无效的目标设备指针
  CUdeviceptr d_ptr = NULL;
  
  // 设置一个无效的值
  unsigned char uc = -1;

  // 调用cuMemsetD8Async，期望返回CUDA_ERROR_INVALID_VALUE
  CUresult res = cuMemsetD8Async(d_ptr, uc, N, NULL);
  EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
}

// 边界值测试
TEST_F(cuMemsetD8AsyncTest, BoundaryValue) {
   // 设置设备内存为0x00
   unsigned char uc = 0x00;
   cuMemsetD8Async(d_ptr_, uc, N, NULL);

   // 同步设备
   cuDeviceSynchronize();

   // 拷贝设备内存到主机内存
   cuMemcpyDtoH(h_ptr_, d_ptr_, N);

   // 检查主机内存中的值是否都为0x00
   for (size_t i = 0; i < N; i++) {
     EXPECT_EQ(h_ptr_[i], uc);
   }

   // 设置设备内存为0xFF
   uc = 0xFF;
   cuMemsetD8Async(d_ptr_, uc, N, NULL);

   // 同步设备
   cuDeviceSynchronize();

   // 拷贝设备内存到主机内存
   cuMemcpyDtoH(h_ptr_, d_ptr_, N);

   // 检查主机内存中的值是否都为0xFF
   for (size_t i = 0; i < N; i++) {
     EXPECT_EQ(h_ptr_[i], uc);
   }

   // 设置设备内存为0x80，这是一个边界值，因为它是最高位为1的最小值
   uc = 0x80;
   cuMemsetD8Async(d_ptr_, uc, N, NULL);

   // 同步设备
   cuDeviceSynchronize();

   // 拷贝设备内存到主机内存
   cuMemcpyDtoH(h_ptr_, d_ptr_, N);

   // 检查主机内存中的值是否都为0x80
   for (size_t i = 0; i < N; i++) {
     EXPECT_EQ(h_ptr_[i], uc);
   }

   // 设置设备内存为0x7F，这是一个边界值，因为它是最高位为0的最大值
   uc = 0x7F;
   cuMemsetD8Async(d_ptr_, uc, N, NULL);

   // 同步设备
   cuDeviceSynchronize();

   // 拷贝设备内存到主机内存
   cuMemcpyDtoH(h_ptr_, d_ptr_, N);

   // 检查主机内存中的值是否都为0x7F
   for (size_t i = 0; i < N; i++) {
     EXPECT_EQ(h_ptr_[i], uc);
   }
}


  // 同步或异步行为测试
TEST_F(cuMemsetD8AsyncTest, SyncOrAsyncBehavior) {
  // 设置设备内存为0xFF
  unsigned char uc = 0xFF;
  cuMemsetD8Async(d_ptr_, uc, N, NULL);

  // 不同步设备，直接拷贝设备内存到主机内存
  cuMemcpyDtoH(h_ptr_, d_ptr_, N);

  // 检查主机内存中的值是否都为0xFF，期望失败，因为cuMemsetD8Async是异步的
  for (size_t i = 0; i < N; i++) {
    EXPECT_NE(h_ptr_[i], uc);
  }

  // 同步设备
  cuDeviceSynchronize();

  // 再次拷贝设备内存到主机内存
  cuMemcpyDtoH(h_ptr_, d_ptr_, N);

  // 检查主机内存中的值是否都为0xFF，期望成功，因为cuMemsetD8Async已经完成
  for (size_t i = 0; i < N; i++) {
    EXPECT_EQ(h_ptr_[i], uc);
  }
}

// 重复调用测试
TEST_F(cuMemsetD8AsyncTest, RepeatedCall) {
   // 设置设备内存为0xFF
   unsigned char uc = 0xFF;
   cuMemsetD8Async(d_ptr_, uc, N, NULL);

   // 同步设备
   cuDeviceSynchronize();

   // 拷贝设备内存到主机内存
   cuMemcpyDtoH(h_ptr_, d_ptr_, N);

   // 检查主机内存中的值是否都为0xFF
   for (size_t i = 0; i < N; i++) {
     EXPECT_EQ(h_ptr_[i], uc);
   }

   // 再次设置设备内存为0x00
   uc = 0x00;
   cuMemsetD8Async(d_ptr_, uc, N, NULL);

   // 同步设备
   cuDeviceSynchronize();

   // 再次拷贝设备内存到主机内存
   cuMemcpyDtoH(h_ptr_, d_ptr_, N);

   // 检查主机内存中的值是否都为0x00
   for (size_t i = 0; i < N; i++) {
     EXPECT_EQ(h_ptr_[i], uc);
   }
}

// 其他你能想到的测试
TEST_F(cuMemsetD8AsyncTest, OtherTest) {
   // 设置一个CUDA流
   CUstream hStream;
   cuStreamCreate(&hStream, CU_STREAM_DEFAULT);

   // 设置设备内存为0xFF，使用CUDA流
   unsigned char uc = 0xFF;
   cuMemsetD8Async(d_ptr_, uc, N, hStream);

   // 在CUDA流上执行一个简单的核函数，将设备内存中的每个值加1
   dim3 gridDim((N + 255) / 256);
   dim3 blockDim(256);
   addOne<<<gridDim, blockDim, 0, hStream>>>(d_ptr_, N);

   // 同步CUDA流
   cuStreamSynchronize(hStream);

   // 拷贝设备内存到主机内存
   cuMemcpyDtoH(h_ptr_, d_ptr_, N);

   // 检查主机内存中的值是否都为0x00，因为0xFF + 1 = 0x00（溢出）
   for (size_t i = 0; i < N; i++) {
     EXPECT_EQ(h_ptr_[i], 0x00);
   }

   // 销毁CUDA流
   cuStreamDestroy(hStream);
}
