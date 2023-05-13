// 定义一个测试类
class cuModuleUnloadTest : public ::testing::Test {
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

// 测试正常卸载一个CUDA模块
TEST_F(cuModuleUnloadTest, UnloadValidModule) {
  // 准备一个有效的fat binary对象
  const void* fatCubin = ...; // 省略细节
  // 调用cuModuleLoadFatBinary函数，加载模块
  CUresult res = cuModuleLoadFatBinary(&module, fatCubin);
  ASSERT_EQ(CUDA_SUCCESS, res);
  ASSERT_NE(nullptr, module);
  // 调用cuModuleUnload函数，卸载模块
  res = cuModuleUnload(module);
  // 检查返回值是否为CUDA_SUCCESS
  EXPECT_EQ(CUDA_SUCCESS, res);
}

// 测试卸载一个空指针作为模块句柄
TEST_F(cuModuleUnloadTest, UnloadNullModule) {
  // 准备一个空指针作为模块句柄
  CUmodule module = nullptr;
  // 调用cuModuleUnload函数，卸载模块
  CUresult res = cuModuleUnload(module);
  // 检查返回值是否为CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, res);
}

// 测试卸载一个无效的模块句柄
TEST_F(cuModuleUnloadTest, UnloadInvalidModule) {
  // 准备一个无效的模块句柄，可以是随机数据或者已经卸载过的模块句柄等
  CUmodule module = ...; // 省略细节
  // 调用cuModuleUnload函数，卸载模块
  CUresult res = cuModuleUnload(module);
  // 检查返回值是否为CUDA_ERROR_INVALID_HANDLE
  EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, res);
}

// 测试卸载一个已经被其他上下文使用的模块句柄
TEST_F(cuModuleUnloadTest, UnloadSharedModule) {
  // 准备一个有效的fat binary对象
  const void* fatCubin = ...; // 省略细节
  // 调用cuModuleLoadFatBinary函数，加载模块
  CUresult res = cuModuleLoadFatBinary(&module, fatCubin);
  ASSERT_EQ(CUDA_SUCCESS, res);
  ASSERT_NE(nullptr, module);
  // 创建一个新的CUDA上下文
  CUcontext context2;
  res = cuCtxCreate(&context2, 0, device);
  ASSERT_EQ(CUDA_SUCCESS, res);
  // 在新的上下文中使用模块句柄
  CUfunction function;
  res = cuModuleGetFunction(&function, module, "kernel");
  ASSERT_EQ(CUDA_SUCCESS, res);
  // 切换回原来的上下文
  res = cuCtxSetCurrent(context);
  ASSERT_EQ(CUDA_SUCCESS, res);
  // 调用cuModuleUnload函数，卸载模块
  res = cuModuleUnload(module);
  // 检查返回值是否为CUDA_ERROR_CONTEXT_IS_DESTROYED
  EXPECT_EQ(CUDA_ERROR_CONTEXT_IS_DESTROYED, res);
  // 销毁新的上下文
  res = cuCtxDestroy(context2);
  ASSERT_EQ(CUDA_SUCCESS, res);
}

// 测试异步卸载一个CUDA模块
TEST_F(cuModuleUnloadTest, UnloadAsyncModule) {
  // 准备一个有效的fat binary对象
  const void* fatCubin = ...; // 省略细节
  // 调用cuModuleLoadFatBinary函数，加载模块
  CUresult res = cuModuleLoadFatBinary(&module, fatCubin);
  ASSERT_EQ(CUDA_SUCCESS, res);
  ASSERT_NE(nullptr, module);
  // 创建一个CUDA流
  CUstream stream;
  res = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
  EXPECT_EQ(CUDA_SUCCESS, res);
  // 在流上调用cuModuleUnload函数，异步卸载模块
  res = cuModuleUnload(module);
  // 检查返回值是否为CUDA_SUCCESS
  EXPECT_EQ(CUDA_SUCCESS, res);
  // 等待流上的操作完成
  res = cuStreamSynchronize(stream);
  EXPECT_EQ(CUDA_SUCCESS, res);
  // 销毁流
  res = cuStreamDestroy(stream);
  EXPECT_EQ(CUDA_SUCCESS, res);
}

// 测试反复调用cuModuleUnload函数
TEST_F(cuModuleUnloadTest, UnloadRepeatedModule) {
  // 准备一个有效的fat binary对象
  const void* fatCubin = ...; // 省略细节
  // 调用cuModuleLoadFatBinary函数，加载模块
  CUresult res = cuModuleLoadFatBinary(&module, fatCubin);
  ASSERT_EQ(CUDA_SUCCESS, res);
  ASSERT_NE(nullptr, module);
  // 调用cuModuleUnload函数，第一次卸载成功
  res = cuModuleUnload(module);
  EXPECT_EQ(CUDA_SUCCESS, res);
  // 调用cuModuleUnload函数，第二次卸载失败
  res = cuModuleUnload(module);
  EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, res);
}
