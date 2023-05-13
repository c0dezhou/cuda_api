
// 定义一个测试类
class cuModuleLoadDataTest : public ::testing::Test {
protected:
  // 在每个测试之前执行
  void SetUp() override {
    // 初始化CUDA驱动API
    CUresult res = cuInit(0);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 获取第一个可用的设备
    int deviceCount = 0;
    res = cuDeviceGetCount(&deviceCount);
    ASSERT_EQ(res, CUDA_SUCCESS);
    ASSERT_GT(deviceCount, 0);

    CUdevice device;
    res = cuDeviceGet(&device, 0);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 创建一个上下文
    CUcontext context;
    res = cuCtxCreate(&context, 0, device);
    ASSERT_EQ(res, CUDA_SUCCESS);
  }


  // 在每个测试之后执行
  void TearDown() override {
    // 销毁当前上下文
    CUresult res = cuCtxDestroy(cuCtxGetCurrent());
    ASSERT_EQ(res, CUDA_SUCCESS);
  }
};

// 基本行为测试
TEST_F(cuModuleLoadDataTest, BasicBehavior) {
  // 使用一个有效的cubin文件指针作为参数
  const char* fname = "vectorAdd.cubin";
  // 映射文件到内存中
  FILE* fp = fopen(fname, "rb");
  ASSERT_NE(fp, nullptr);
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  void* image = malloc(size);
  ASSERT_NE(image, nullptr);
  fread(image, size, 1, fp);
  fclose(fp);
  
  CUmodule module;
  // 调用cuModuleLoadData，并检查返回值是否为CUDA_SUCCESS
  CUresult res = cuModuleLoadData(&module, image);
  EXPECT_EQ(res, CUDA_SUCCESS);
  // 如果成功，检查模块是否正确加载到当前上下文中
  if (res == CUDA_SUCCESS) {
    // 获取模块中的一个函数
    CUfunction function;
    res = cuModuleGetFunction(&function, module, "vectorAdd");
    EXPECT_EQ(res, CUDA_SUCCESS);
    // 如果成功，执行该函数
    if (res == CUDA_SUCCESS) {
      // 设置函数参数和网格大小
      int N = 1024;
      float *h_A = new float[N];
      float *h_B = new float[N];
      float *h_C = new float[N];
      for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
      }
      CUdeviceptr d_A, d_B, d_C;
      res = cuMemAlloc(&d_A, N * sizeof(float));
      ASSERT_EQ(res, CUDA_SUCCESS);
      res = cuMemAlloc(&d_B, N * sizeof(float));
      ASSERT_EQ(res, CUDA_SUCCESS);
      res = cuMemAlloc(&d_C, N * sizeof(float));
      ASSERT_EQ(res, CUDA_SUCCESS);
      res = cuMemcpyHtoD(d_A, h_A, N * sizeof(float));
      ASSERT_EQ(res, CUDA_SUCCESS);
      res = cuMemcpyHtoD(d_B, h_B, N * sizeof(float));
      ASSERT_EQ(res, CUDA_SUCCESS);
      void* args[] = {&d_A, &d_B, &d_C};
      int blockSize = 256;
      int gridSize = (N + blockSize - 1) / blockSize;
      // 调用cuLaunchKernel，并检查返回值是否为CUDA_SUCCESS
      res = cuLaunchKernel(function,
                           gridSize, 1, 1,
                           blockSize, 1, 1,
                           0,
                           NULL,
                           args,
                           NULL);

  EXPECT_EQ(res, CUDA_SUCCESS);
  // 如果成功，检查结果是否正确
  if (res == CUDA_SUCCESS) {
    res = cuMemcpyDtoH(h_C, d_C, N * sizeof(float));
    ASSERT_EQ(res, CUDA_SUCCESS);
    for (int i = 0; i < N; i++) {
      EXPECT_FLOAT_EQ(h_C[i], h_A[i] + h_B[i]);
    }
  }
  // 释放内存
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  res = cuMemFree(d_A);
  ASSERT_EQ(res, CUDA_SUCCESS);
  res = cuMemFree(d_B);
  ASSERT_EQ(res, CUDA_SUCCESS);
  res = cuMemFree(d_C);
  ASSERT_EQ(res, CUDA_SUCCESS);
    }
    // 卸载模块
    res = cuModuleUnload(module);
    EXPECT_EQ(res, CUDA_SUCCESS);
    // 释放文件指针
    free(image);
}

// 异常测试
TEST_F(cuModuleLoadDataTest, Exception) {
  // 使用一个无效的文件指针（例如NULL指针）作为参数
  CUmodule module;
  // 调用cuModuleLoadData，并检查返回值是否为CUDA_ERROR_INVALID_VALUE
  CUresult res = cuModuleLoadData(&module, NULL);
  EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
  // 如果失败，检查模块是否没有加载到当前上下文中
  if (res != CUDA_SUCCESS) {
    // 尝试获取模块中的一个函数
    CUfunction function;
    res = cuModuleGetFunction(&function, module, "vectorAdd");
    EXPECT_NE(res, CUDA_SUCCESS);
    // 如果失败，不执行该函数
  }
}


// 边界值测试
TEST_F(cuModuleLoadDataTest, BoundaryValue) {
  // 使用一个最大的文件大小作为参数
  const char* fname = "large_file.cubin";
  // 映射文件到内存中
  FILE* fp = fopen(fname, "rb");
  ASSERT_NE(fp, nullptr);
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  void* image = malloc(size);
  ASSERT_NE(image, nullptr);
  fread(image, size, 1, fp);
  fclose(fp);

  CUmodule module;
  // 调用cuModuleLoadData，并检查返回值是否为CUDA_SUCCESS或错误代码
  CUresult res = cuModuleLoadData(&module, image);
  // 如果成功，检查模块是否正确加载到当前上下文中，并卸载模块
  if (res == CUDA_SUCCESS) {
    // 获取模块中的一个函数
    CUfunction function;
    res = cuModuleGetFunction(&function, module, "large_function");
    EXPECT_EQ(res, CUDA_SUCCESS);
    // 卸载模块
    res = cuModuleUnload(module);
    EXPECT_EQ(res, CUDA_SUCCESS);
  }
  // 如果失败，检查模块是否没有加载到当前上下文中
  else {
    // 尝试获取模块中的一个函数
    CUfunction function;
    res = cuModuleGetFunction(&function, module, "large_function");
    EXPECT_NE(res, CUDA_SUCCESS);
    // 如果失败，不执行该函数
  }

  free(image);
}

// 同步或异步行为测试
TEST_F(cuModuleLoadDataTest, SyncOrAsyncBehavior) {
  // 使用一个有效的cubin文件指针作为参数
  const char* fname = "vectorAdd.cubin";
  // 映射文件到内存中
  FILE* fp = fopen(fname, "rb");
  ASSERT_NE(fp, nullptr);
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  void* image = malloc(size);
  ASSERT_NE(image, nullptr);
  fread(image, size, 1, fp);
  fclose(fp);

  CUmodule module;
  // 调用cuModuleLoadData，并检查返回值是否为CUDA_SUCCESS
  CUresult res = cuModuleLoadData(&module, image);
  EXPECT_EQ(res, CUDA_SUCCESS);
  // 如果成功，检查模块是否正确加载到当前上下文中
  if (res == CUDA_SUCCESS) {
    // 获取模块中的一个函数
    CUfunction function;
    res = cuModuleGetFunction(&function, module, "vectorAdd");
    EXPECT_EQ(res, CUDA_SUCCESS);
    // 如果成功，执行该函数
    if (res == CUDA_SUCCESS) {
      // 设置函数参数和网格大小
      int N = 1024;
      float *h_A = new float[N];
      float *h_B = new float[N];
      float *h_C = new float[N];
      for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
      }
      CUdeviceptr d_A, d_B, d_C;
      res = cuMemAlloc(&d_A, N * sizeof(float));
      ASSERT_EQ(res, CUDA_SUCCESS);
      res = cuMemAlloc(&d_B, N * sizeof(float));
      ASSERT_EQ(res, CUDA_SUCCESS);
      res = cuMemAlloc(&d_C, N * sizeof(float));
      ASSERT_EQ(res, CUDA_SUCCESS);
      res = cuMemcpyHtoD(d_A, h_A, N * sizeof(float));
      ASSERT_EQ(res, CUDA_SUCCESS);
      res = cuMemcpyHtoD(d_B, h_B, N * sizeof(float));
      ASSERT_EQ(res, CUDA_SUCCESS);
      void* args[] = {&d_A, &d_B


  // ...
  // 释放内存
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_D;
  res = cuMemFree(d_A);
  ASSERT_EQ(res, CUDA_SUCCESS);
  res = cuMemFree(d_B);
  ASSERT_EQ(res, CUDA_SUCCESS);
  res = cuMemFree(d_C);
  ASSERT_EQ(res, CUDA_SUCCESS);
  res = cuMemFree(d_D);
  ASSERT_EQ(res, CUDA_SUCCESS);
    }
    // 卸载模块
    res = cuModuleUnload(module);
    EXPECT_EQ(res, CUDA_SUCCESS);
    // 释放文件指针
    free(image);
}
