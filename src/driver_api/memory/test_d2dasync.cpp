// 定义一个测试类
class cuMemcpyDtoDAsyncTest : public ::testing::Test {
protected:
  // 在每个测试之前执行
  void SetUp() override {
    // 初始化CUDA环境
    cuInit(0);
    // 获取第一个可用的CUDA设备
    cuDeviceGet(&device, 0);
    // 创建一个CUDA上下文
    cuCtxCreate(&context, 0, device);
    // 分配两块设备内存
    cuMemAlloc(&d_src, size);
    cuMemAlloc(&d_dst, size);
    // 创建一个CUDA流
    cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
  }

  // 在每个测试之后执行
  void TearDown() override {
    // 释放设备内存
    cuMemFree(d_src);
    cuMemFree(d_dst);
    // 销毁CUDA流
    cuStreamDestroy(stream);
    // 销毁CUDA上下文
    cuCtxDestroy(context);
  }

  // 定义一些成员变量
  CUdevice device; // 设备句柄
  CUcontext context; // 上下文句柄
  CUdeviceptr d_src; // 源设备指针
  CUdeviceptr d_dst; // 目标设备指针
  CUstream stream; // 流句柄
  size_t size = 1024; // 复制的字节数
};

// 定义一个基本行为测试，检查函数是否能正确复制数据
TEST_F(cuMemcpyDtoDAsyncTest, BasicBehavior) {
  // 定义一些变量
  int h_src[size]; // 源主机数组
  int h_dst[size]; // 目标主机数组
  int h_ref[size]; // 参考主机数组

  // 初始化源主机数组和参考主机数组为随机值
  for (int i = 0; i < size; i++) {
    h_src[i] = rand();
    h_ref[i] = h_src[i];
  }

  // 将源主机数组复制到源设备内存
  cuMemcpyHtoD(d_src, h_src, size * sizeof(int));

  // 调用cuMemcpyDtoDAsync函数，将源设备内存复制到目标设备内存，使用默认流（0）
  cuMemcpyDtoDAsync(d_dst, d_src, size * sizeof(int), 0);

  // 等待默认流完成所有操作
  cuStreamSynchronize(0);

  // 将目标设备内存复制到目标主机数组
  cuMemcpyDtoH(h_dst, d_dst, size * sizeof(int));

  // 检查目标主机数组和参考主机数组是否相等
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(h_dst[i], h_ref[i]);
  }
}

// 定义一个异常测试，检查函数是否能正确处理无效的参数
TEST_F(cuMemcpyDtoDAsyncTest, InvalidArguments) {
  // 定义一些变量
  CUresult result; // 函数返回值

  // 调用cuMemcpyDtoDAsync函数，传入无效的目标设备指针
  result = cuMemcpyDtoDAsync(0, d_src, size * sizeof(int), 0);
  // 检查函数是否返回CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE);

  // 调用cuMemcpyDtoDAsync函数，传入无效的源设备指针
  result = cuMemcpyDtoDAsync(d_dst, 0, size * sizeof(int), 0);
  // 检查函数是否返回CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE);

  // 调用cuMemcpyDtoDAsync函数，传入无效的字节数
  result = cuMemcpyDtoDAsync(d_dst, d_src, 0, 0);
  // 检查函数是否返回CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE);

  // 调用cuMemcpyDtoDAsync函数，传入无效的流标识符
  result = cuMemcpyDtoDAsync(d_dst, d_src, size * sizeof(int), (CUstream)-1);
  // 检查函数是否返回CUDA_ERROR_INVALID_HANDLE
  EXPECT_EQ(result, CUDA_ERROR_INVALID_HANDLE);
}

// 定义一个边界值测试，检查函数是否能正确处理极端的参数
TEST_F(cuMemcpyDtoDAsyncTest, BoundaryValues) {
  // 定义一些变量
  int h_src[size]; // 源主机数组
  int h_dst[size]; // 目标主机数组
  int h_ref[size]; // 参考主机数组

  // 初始化源主机数组和参考主机数组为随机值
  for (int i = 0; i < size; i++) {
    h_src[i] = rand();
    h_ref[i] = h_src[i];
  }

  // 将源主机数组复制到源设备内存
  cuMemcpyHtoD(d_src, h_src, size * sizeof(int));

  // 调用cuMemcpyDtoDAsync函数，将源设备内存复制到目标设备内存，使用最大的字节数（size * sizeof(int)）
  cuMemcpyDtoDAsync(d_dst, d_src, size * sizeof(int), stream);

  // 等待流完成所有操作
  cuStreamSynchronize(stream);

  // 将目标设备内存复制到目标主机数组
  cuMemcpyDtoH(h_dst, d_dst, size * sizeof(int));

  // 检查目标主机数组和参考主机数组是否相等
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(h_dst[i], h_ref[i]);
  }

  // 调用cuMemcpyDtoDAsync函数，将源设备内存复制到目标设备内存，使用最小的字节数（1）
  cuMemcpyDtoDAsync(d_dst, d_src, 1, stream);

  // 等待流完成所有操作
  cuStreamSynchronize(stream);

  // 将目标设备内存复制到目标主机数组
  cuMemcpyDtoH(h_dst, d_dst, size * sizeof(int));

  // 检查目标主机数组的第一个元素和参考主机数组的第一个元素是否相等
  EXPECT_EQ(h_dst[0], h_ref[0]);
}

// 定义一个同步或异步行为测试，检查函数是否能正确处理不同的流
TEST_F(cuMemcpyDtoDAsyncTest, SyncOrAsyncBehavior) {
  // 定义一些变量
  int h_src[size]; // 源主机数组
  int h_dst[size]; // 目标主机数组
  int h_ref[size]; // 参考主机数组

  // 初始化源主机数组和参考主机数组为随机值
  for (int i = 0; i < size; i++) {
    h_src[i] = rand();
    h_ref[i] = h_src[i];
  }

  // 将源主机数组复制到源设备内存
  cuMemcpyHtoD(d_src, h_src, size * sizeof(int));

  // 调用cuMemcpyDtoDAsync函数，将源设备内存复制到目标设备内存，使用默认流（0）
  cuMemcpyDtoDAsync(d_dst, d_src, size * sizeof(int), 0);

  // 尝试将目标设备内存复制到目标主机数组，不等待默认流完成操作
  cuMemcpyDtoH(h_dst, d_dst, size * sizeof(int));

  // 检查目标主机数组和参考主机数组是否不相等（因为复制操作还没有完成）
  for (int i = 0; i < size; i++) {
    EXPECT_NE(h_dst[i], h_ref[i]);
  }

  // 等待默认流完成所有操作
  cuStreamSynchronize(0);

  // 再次将目标设备内存复制到目标主机数组// 再次将目标设备内存复制到目标主机数组
  cuMemcpyDtoH(h_dst, d_dst, size * sizeof(int));

  // 检查目标主机数组和参考主机数组是否相等（因为复制操作已经完成）
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(h_dst[i], h_ref[i]);
  }

  // 调用cuMemcpyDtoDAsync函数，将源设备内存复制到目标设备内存，使用非默认流（stream）
  cuMemcpyDtoDAsync(d_dst, d_src, size * sizeof(int), stream);

  // 尝试将目标设备内存复制到目标主机数组，不等待非默认流完成操作
  cuMemcpyDtoH(h_dst, d_dst, size * sizeof(int));

  // 检查目标主机数组和参考主机数组是否不相等（因为复制操作还没有完成）
  for (int i = 0; i < size; i++) {
    EXPECT_NE(h_dst[i], h_ref[i]);
  }

  // 等待非默认流完成所有操作
  cuStreamSynchronize(stream);

  // 再次将目标设备内存复制到目标主机数组
  // 再次将目标设备内存复制到目标主机数组
  cuMemcpyDtoH(h_dst, d_dst, size * sizeof(int));

  // 检查目标主机数组和参考主机数组是否相等（因为复制操作已经完成）
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(h_dst[i], h_ref[i]);
  }
}

// 定义一个重复调用测试，检查函数是否能正确处理多次复制
TEST_F(cuMemcpyDtoDAsyncTest, RepeatedCalls) {
  // 定义一些变量
  int h_src[size]; // 源主机数组
  int h_dst[size]; // 目标主机数组
  int h_ref[size]; // 参考主机数组

  // 初始化源主机数组和参考主机数组为随机值
  for (int i = 0; i < size; i++) {
    h_src[i] = rand();
    h_ref[i] = h_src[i];
  }

  // 将源主机数组复制到源设备内存
  cuMemcpyHtoD(d_src, h_src, size * sizeof(int));

  // 调用cuMemcpyDtoDAsync函数，将源设备内存复制到目标设备内存，使用流（stream）
  cuMemcpyDtoDAsync(d_dst, d_src, size * sizeof(int), stream);

  // 再次调用cuMemcpyDtoDAsync函数，将源设备内存的前半部分复制到目标设备内存的后半部分，使用同一个流（stream）
  cuMemcpyDtoDAsync(d_dst + size / 2, d_src, size / 2 * sizeof(int), stream);

  // 修改参考主机数组的后半部分为源主机数组的前半部分
  for (int i = size / 2; i < size; i++) {
    h_ref[i] = h_src[i - size / 2];
  }

  // 等待流完成所有操作
  // 等待流完成所有操作
  cuStreamSynchronize(stream);

  // 将目标设备内存复制到目标主机数组
  cuMemcpyDtoH(h_dst, d_dst, size * sizeof(int));

  // 检查目标主机数组和参考主机数组是否相等
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(h_dst[i], h_ref[i]);
  }
}

// 定义一个其他任何你能想到的测试，检查函数是否能正确处理不同的内存对齐方式
TEST_F(cuMemcpyDtoDAsyncTest, MemoryAlignment) {
  // 定义一些变量
  int h_src[size]; // 源主机数组
  int h_dst[size]; // 目标主机数组
  int h_ref[size]; // 参考主机数组

  // 初始化源主机数组和参考主机数组为随机值
  for (int i = 0; i < size; i++) {
    h_src[i] = rand();
    h_ref[i] = h_src[i];
  }

  // 将源主机数组复制到源设备内存
  cuMemcpyHtoD(d_src, h_src, size * sizeof(int));

  // 调用cuMemcpyDtoDAsync函数，将源设备内存的第一个元素复制到目标设备内存的第二个元素，使用流（stream）
  cuMemcpyDtoDAsync(d_dst + sizeof(int), d_src, sizeof(int), stream);

  // 修改参考主机数组的第二个元素为源主机数组的第一个元素
  h_ref[1] = h_src[0];

  // 等待流完成所有操作
  // 等待流完成所有操作
  cuStreamSynchronize(stream);

  // 将目标设备内存复制到目标主机数组
  cuMemcpyDtoH(h_dst, d_dst, size * sizeof(int));

  // 检查目标主机数组的第二个元素和参考主机数组的第二个元素是否相等
  EXPECT_EQ(h_dst[1], h_ref[1]);
}
