// 定义一个测试类
class cuModuleLoadTest : public ::testing::Test {
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
TEST_F(cuModuleLoadTest, BasicBehavior) {
  // 使用一个有效的cubin文件名作为参数
  const char* fname = "vectorAdd.cubin";
  CUmodule module;
  // 调用cuModuleLoad，并检查返回值是否为CUDA_SUCCESS
  CUresult res = cuModuleLoad(&module, fname);
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
  }
}

// 异常测试
TEST_F(cuModuleLoadTest, Exception) {
  // 使用一个无效的文件名作为参数
  const char* fname = "invalid_file";
  CUmodule module;
  // 调用cuModuleLoad，并检查返回值是否为CUDA_ERROR_FILE_NOT_FOUND
  CUresult res = cuModuleLoad(&module, fname);
  EXPECT_EQ(res, CUDA_ERROR_FILE_NOT_FOUND);
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
TEST_F(cuModuleLoadTest, BoundaryValue) {
  // 使用一个最长的文件名作为参数
  const char* fname = "a_very_very_very_very_very_very_very_very_very_very_long_file_name.cubin";
  CUmodule module;
  // 调用cuModuleLoad，并检查返回值是否为CUDA_SUCCESS或错误代码
  CUresult res = cuModuleLoad(&module, fname);
  // 如果成功，检查模块是否正确加载到当前上下文中，并卸载模块
  if (res == CUDA_SUCCESS) {
    // 获取模块中的一个函数
    CUfunction function;
    res = cuModuleGetFunction(&function, module, "vectorAdd");
    EXPECT_EQ(res, CUDA_SUCCESS);
    // 卸载模块
    res = cuModuleUnload(module);
    EXPECT_EQ(res, CUDA_SUCCESS);
  }
  // 如果失败，检查模块是否没有加载到当前上下文中
  else {
    // 尝试获取模块中的一个函数
    CUfunction function;
    res = cuModuleGetFunction(&function, module, "vectorAdd");
    EXPECT_NE(res, CUDA_SUCCESS);
  }
}

// 同步或异步行为测试
TEST_F(cuModuleLoadTest, SyncOrAsyncBehavior) {
  // 使用一个有效的cubin文件名作为参数
  const char* fname = "vectorAdd.cubin";
  CUmodule module;
  // 调用cuModuleLoad，并检查返回值是否为CUDA_SUCCESS
  CUresult res = cuModuleLoad(&module, fname);
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
      // 在调用cuLaunchKernel之前，执行一些其他的CUDA操作，并检查返回值是否为CUDA_SUCCESS
      CUdeviceptr d_D;
      res = cuMemAlloc(&d_D, N * sizeof(float));
      EXPECT_EQ(res, CUDA_SUCCESS);

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
  // 在调用cuLaunchKernel之后，执行一些其他的CUDA操作，并检查返回值是否为CUDA_SUCCESS
  res = cuMemsetD8(d_D, 0, N * sizeof(float));
  EXPECT_EQ(res, CUDA_SUCCESS);
  // 检查这些操作是否正确执行，以及是否有任何互相影响或干扰
  float *h_D = new float[N];
  res = cuMemcpyDtoH(h_D, d_D, N * sizeof(float));
  ASSERT_EQ(res, CUDA_SUCCESS);
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(h_D[i], 0);
  }

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
  }
}

// 重复调用测试
TEST_F(cuModuleLoadTest, RepeatedCall) {
  // 使用同一个有效的cubin文件名作为参数，多次调用cuModuleLoad，并检查返回值是否为CUDA_SUCCESS
  const char* fname = "vectorAdd.cubin";
  CUmodule module1, module2;
  CUresult res = cuModuleLoad(&module1, fname);
  EXPECT_EQ(res, CUDA_SUCCESS);
  res = cuModuleLoad(&module2, fname);
  EXPECT_EQ(res, CUDA_SUCCESS);
  // 如果成功，检查模块是否正确加载到当前上下文中，并卸载模块
  if (res == CUDA_SUCCESS) {
    // 获取模块中的一个函数
    CUfunction function1, function2;
    res = cuModuleGetFunction(&function1, module1, "vectorAdd");
    EXPECT_EQ(res, CUDA_SUCCESS);
    res = cuModuleGetFunction(&function2, module2, "vectorAdd");
    EXPECT_EQ(res, CUDA_SUCCESS);
    // 如果成功，检查函数是否相同或不同
    EXPECT_NE(function1, function2); // 不同的模块应该返回不同的函数指针
    // 卸载模块
    res = cuModuleUnload(module1);
    EXPECT_EQ(res, CUDA_SUCCESS);
    res = cuModuleUnload(module2);
    EXPECT_EQ(res, CUDA_SUCCESS);
  // 使用不同的有效的cubin文件名作为参数，多次调用cuModuleLoad，并检查返回值是否为CUDA_SUCCESS
  const char* fname1 = "vectorAdd.cubin";
  const char* fname2 = "matrixMul.cubin";
  CUmodule module1, module2;
  res = cuModuleLoad(&module1, fname1);
  EXPECT_EQ(res, CUDA_SUCCESS);
  res = cuModuleLoad(&module2, fname2);
  EXPECT_EQ(res, CUDA_SUCCESS);
  // 如果成功，检查模块是否正确加载到当前上下文中，并卸载模块
  if (res == CUDA_SUCCESS) {
    // 获取模块中的一个函数
    CUfunction function1, function2;
    res = cuModuleGetFunction(&function1, module1, "vectorAdd");
    EXPECT_EQ(res, CUDA_SUCCESS);
    res = cuModuleGetFunction(&function2, module2, "matrixMul");
    EXPECT_EQ(res, CUDA_SUCCESS);
    // 如果成功，检查函数是否相同或不同
    EXPECT_NE(function1, function2); // 不同的模块应该返回不同的函数指针
    // 卸载模块
    res = cuModuleUnload(module1);
    EXPECT_EQ(res, CUDA_SUCCESS);
    res = cuModuleUnload(module2);
    EXPECT_EQ(res, CUDA_SUCCESS);
  }
}}

// 测试不同的cubin或PTX文件内容
TEST_F(cuModuleLoadTest, DifferentFileContent) {
  // 使用一个具有不同计算能力的cubin文件名作为参数
  const char* fname = "vectorAdd_sm_35.cubin";
  CUmodule module;
  // 调用cuModuleLoad，并检查返回值是否为CUDA_SUCCESS或错误代码
  CUresult res = cuModuleLoad(&module, fname);
  // 如果成功，检查模块是否正确加载到当前上下文中，并卸载模块
  if (res == CUDA_SUCCESS) {
    // 获取模块中的一个函数
    CUfunction function;
    res = cuModuleGetFunction(&function, module, "vectorAdd");
    EXPECT_EQ(res, CUDA_SUCCESS);
    // 卸载模块
    res = cuModuleUnload(module);
    EXPECT_EQ(res, CUDA_SUCCESS);
  }
  // 如果失败，检查模块是否没有加载到当前上下文中
  else {
    // 尝试获取模块中的一个函数
    CUfunction function;
    res = cuModuleGetFunction(&function, module, "vectorAdd");
    EXPECT_NE(res, CUDA_SUCCESS);
  }
}
