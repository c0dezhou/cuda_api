// 定义一个测试夹具类
class cuMemcpyDtoDTest : public ::testing::Test {
 protected:
  // 在每个测试开始之前执行
  void SetUp() override {
    // 分配两块设备内存
    size_t size = N * sizeof(int);
    cuMemAlloc(&d_src_, size);
    cuMemAlloc(&d_dst_, size);

    // 分配一块主机内存
    h_ptr_ = new int[N];

    // 初始化主机内存中的数据
    for (size_t i = 0; i < N; i++) {
      h_ptr_[i] = i;
    }

    // 拷贝主机内存中的数据到源设备内存
    cuMemcpyHtoD(d_src_, h_ptr_, size);
  }

  // 在每个测试结束之后执行
  void TearDown() override {
    // 释放内存
    delete[] h_ptr_;
    cuMemFree(d_src_);
    cuMemFree(d_dst_);
  }

  // 成员变量
  CUdeviceptr d_src_; // 源设备内存指针
  CUdeviceptr d_dst_; // 目标设备内存指针
  int* h_ptr_; // 主机内存指针
  static const size_t N = 100; // 元素个数
};

// 定义一个基本行为测试用例
TEST_F(cuMemcpyDtoDTest, BasicBehavior) {
  // 拷贝源设备内存中的数据到目标设备内存
  size_t size = N * sizeof(int);
  cuMemcpyDtoD(d_dst_, d_src_, size);

  // 同步设备
  cuDeviceSynchronize();

  // 拷贝目标设备内存中的数据到主机内存
  cuMemcpyDtoH(h_ptr_, d_dst_, size);

  // 检查主机内存中的数据是否和源设备内存中的数据一致
  for (size_t i = 0; i < N; i++) {
    EXPECT_EQ(h_ptr_[i], i);
  }
}

// 定义一个异常测试用例
TEST_F(cuMemcpyDtoDTest, InvalidArguments) {
  // 设置一个无效的源设备指针
  CUdeviceptr d_src_invalid = NULL;

  // 设置一个无效的目标设备指针
  CUdeviceptr d_dst_invalid = NULL;

  // 设置一个无效的大小
  size_t size_invalid = (N + 1) * sizeof(int);

  // 调用cuMemcpyDtoD，期望返回CUDA_ERROR_INVALID_VALUE
  CUresult res = cuMemcpyDtoD(d_dst_invalid, d_src_, N * sizeof(int));
  EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);

  res = cuMemcpyDtoD(d_dst_, d_src_invalid, N)
}

// 定义一个边界值测试用例
TEST_F(cuMemcpyDtoDTest, BoundaryValues) {
  // 拷贝0字节，期望返回CUDA_SUCCESS
  CUresult res = cuMemcpyDtoD(d_dst_, d_src_, 0);
  EXPECT_EQ(res, CUDA_SUCCESS);

  // 拷贝整个设备内存，期望返回CUDA_ERROR_INVALID_VALUE
  size_t size = 0;
  cuMemGetInfo(NULL, &size);
  res = cuMemcpyDtoD(d_dst_, d_src_, size);
  EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);

  // 拷贝重叠的区域，期望返回CUDA_SUCCESS
  size = N * sizeof(int);
  res = cuMemcpyDtoD(d_dst_ + size / 2, d_src_, size / 2);
  EXPECT_EQ(res, CUDA_SUCCESS);

  // 同步设备
  cuDeviceSynchronize();

  // 拷贝目标设备内存中的数据到主机内存
  cuMemcpyDtoH(h_ptr_, d_dst_, size);

  // 检查主机内存中的数据是否正确
  for (size_t i = 0; i < N / 2; i++) {
    EXPECT_EQ(h_ptr_[i], i);
  }
  for (size_t i = N / 2; i < N; i++) {
    EXPECT_EQ(h_ptr_[i], i - N / 2);
  }
}

// 定义一个同步或异步行为测试用例
TEST_F(cuMemcpyDtoDTest, SyncOrAsyncBehavior) {
  // 设置一个CUDA流
  CUstream hStream;
  cuStreamCreate(&hStream, CU_STREAM_DEFAULT);

  // 拷贝源设备内存中的数据到目标设备内存，使用CUDA流
  size_t size = N * sizeof(int);
  cuMemcpyDtoDAsync(d_dst_, d_src_, size, hStream);

  // 不同步设备，直接拷贝目标设备内存到主机内存
  cuMemcpyDtoH(h_ptr_, d_dst_, size);

  // 检查主机内存中的数据是否都为0，期望成功，因为cuMemcpyDtoDAsync是异步的
  for (size_t i = 0; i < N; i++) {
    EXPECT_EQ(h_ptr_[i], 0);
  }

  // 同步CUDA流
  cuStreamSynchronize(hStream);

  // 再次拷贝目标设备内存到主机内存
  cuMemcpyDtoH(h_ptr_, d_dst_, size);

  // 检查主机内存中的数据是否和源设备内存中的数据一致，期望成功，因为cuMemcpyDtoDAsync已经完成
  for (size_t i = 0; i < N; i++) {
    EXPECT_EQ(h_ptr_[i], i);
  }

  // 销毁CUDA流
  cuStreamDestroy(hStream);
}

// 定义一个重复调用测试用例
TEST_F(cuMemcpyDtoDTest, RepeatedCalls) {
   // 拷贝源设备内存中的数据到目标设备内存
   size_t size = N * sizeof(int);
   cuMemcpyDtoD(d_dst_, d_src_, size);

   // 同步设备
   cuDeviceSynchronize();

   // 拷贝目标设备内存中的数据到主机内存
   cuMemcpyDtoH(h_ptr_, d_dst_, size);

   // 检查主机内存中的数据是否和源设备内存中的数据一致
   for (size_t i = 0; i < N; i++) {
     EXPECT_EQ(h_ptr_[i], i);
   }

   // 再次拷贝源设备内存中的数据到目标设备内存，但是反转顺序
   for (size_t i = N -1; i >=0; --i) {
     cuMemcpyDtoD(d_dst_ + (N -1 -i) * sizeof(int), d_src_ + i * sizeof(int), sizeof(int));
   }

   //// 同步设备
   cuDeviceSynchronize();

   // 拷贝目标设备内存中的数据到主机内存
   cuMemcpyDtoH(h_ptr_, d_dst_, size);

   // 检查主机内存中的数据是否和源设备内存中的数据反转一致
   for (size_t i = 0; i < N; i++) {
     EXPECT_EQ(h_ptr_[i], N - 1 - i);
   }
}
