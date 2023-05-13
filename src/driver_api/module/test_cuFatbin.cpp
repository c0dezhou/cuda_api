// 定义一个测试类
class cuModuleLoadFatBinaryTest : public ::testing::Test {
protected:
  // 在每个测试用例之前执行
  virtual void SetUp() {
    // 初始化CUDA环境
    CUresult res = cuInit(0);
    ASSERT_EQ(CUDA_SUCCESS, res);
    // 获取第一个CUDA设备
    res = cuDeviceGet(&device, 0);
    ASSERT_EQ(CUDA_SUCCESS, res);
    // 创建一个CUDA上下文
    res = cuCtxCreate(&context, 0, device);
    ASSERT_EQ(CUDA_SUCCESS, res);
  }

  // 在每个测试用例之后执行
  virtual void TearDown() {
    // 销毁CUDA上下文
    CUresult res = cuCtxDestroy(context);
    ASSERT_EQ(CUDA_SUCCESS, res);
  }

  // 定义一些成员变量
  CUdevice device; // CUDA设备
  CUcontext context; // CUDA上下文
  CUmodule module; // CUDA模块
};

// 测试正常加载一个fat binary对象
TEST_F(cuModuleLoadFatBinaryTest, LoadValidFatBinary) {
  // 准备一个有效的fat binary对象
  const void* fatCubin = ...; // 省略细节
  // 调用cuModuleLoadFatBinary函数
  CUresult res = cuModuleLoadFatBinary(&module, fatCubin);
  // 检查返回值是否为CUDA_SUCCESS
  EXPECT_EQ(CUDA_SUCCESS, res);
  // 检查模块句柄是否非空
  EXPECT_NE(nullptr, module);
}

// 测试加载一个空指针作为fat binary对象
TEST_F(cuModuleLoadFatBinaryTest, LoadNullFatBinary) {
  // 准备一个空指针作为fat binary对象
  const void* fatCubin = nullptr;
  // 调用cuModuleLoadFatBinary函数
  CUresult res = cuModuleLoadFatBinary(&module, fatCubin);
  // 检查返回值是否为CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, res);
}

// 测试加载一个无效的fat binary对象
TEST_F(cuModuleLoadFatBinaryTest, LoadInvalidFatBinary) {
  // 准备一个无效的fat binary对象
  const void* fatCubin = ...; // 省略细节，可以是随机数据或者损坏的数据等
  // 调用cuModuleLoadFatBinary函数
  CUresult res = cuModuleLoadFatBinary(&module, fatCubin);
  // 检查返回值是否为CUDA_ERROR_INVALID_SOURCE
  EXPECT_EQ(CUDA_ERROR_INVALID_SOURCE, res);
}

// 测试使用一个空指针作为模块句柄
TEST_F(cuModuleLoadFatBinaryTest, LoadNullModule) {
  // 准备一个有效的fat binary对象
  const void* fatCubin = ...; // 省略细节
  // 调用cuModuleLoadFatBinary函数，使用一个空指针作为模块句柄
  CUresult res = cuModuleLoadFatBinary(nullptr, fatCubin);
  // 检查返回值是否为CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, res);
}

// 测试加载一个不匹配的fat binary对象
TEST_F(cuModuleLoadFatBinaryTest, LoadMismatchedFatBinary) {
  // 准备一个不匹配的fat binary对象，即它不包含当前设备的目标架构
  const void* fatCubin = ...; // 省略细节
  // 调用cuModuleLoadFatBinary函数
  CUresult res = cuModuleLoadFatBinary(&module, fatCubin);
  // 检查返回值是否为CUDA_ERROR_NO_BINARY_FOR_GPU
  EXPECT_EQ(CUDA_ERROR_NO_BINARY_FOR_GPU, res);
}

// 测试加载一个已经加载过的fat binary对象
TEST_F(cuModuleLoadFatBinaryTest, LoadRepeatedFatBinary) {
  // 准备一个有效的fat binary对象
  const void* fatCubin = ...; // 省略细节
  // 调用cuModuleLoadFatBinary函数，第一次加载成功
  CUresult res = cuModuleLoadFatBinary(&module, fatCubin);
  EXPECT_EQ(CUDA_SUCCESS, res);
  EXPECT_NE(nullptr, module);
  // 调用cuModuleLoadFatBinary函数，第二次加载失败
  CUmodule module2;
  res = cuModuleLoadFatBinary(&module2, fatCubin);
  EXPECT_EQ(CUDA_ERROR_ALREADY_MAPPED, res);
}

// 测试异步加载一个fat binary对象
TEST_F(cuModuleLoadFatBinaryTest, LoadAsyncFatBinary) {
  // 准备一个有效的fat binary对象
  const void* fatCubin = ...; // 省略细节
  // 创建一个CUDA流
  CUstream stream;
  CUresult res = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
  EXPECT_EQ(CUDA_SUCCESS, res);
  // 在流上调用cuModuleLoadFatBinary函数，异步加载模块
  res = cuModuleLoadFatBinary(&module, fatCubin);
  // 检查返回值是否为CUDA_SUCCESS
  EXPECT_EQ(CUDA_SUCCESS, res);
  // 检查模块句柄是否非空
  EXPECT_NE(nullptr, module);
  // 等待流上的操作完成
  res = cuStreamSynchronize(stream);
  EXPECT_EQ(CUDA_SUCCESS, res);
  // 销毁流
  res = cuStreamDestroy(stream);
  EXPECT_EQ(CUDA_SUCCESS, res);
}
