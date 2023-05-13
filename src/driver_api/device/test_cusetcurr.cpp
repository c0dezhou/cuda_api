// 定义一个测试夹具类
class CuCtxSetCurrentTest : public ::testing::Test {
 protected:
  // 在每个测试之前执行
  void SetUp() override {
    // 初始化CUDA驱动API
    CUresult res = cuInit(0);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 获取设备数量
    int device_count = 0;
    res = cuDeviceGetCount(&device_count);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 如果没有设备，跳过测试
    if (device_count == 0) {
      GTEST_SKIP();
    }

    // 获取第一个设备
    CUdevice device;
    res = cuDeviceGet(&device, 0);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 创建一个CUDA上下文
    CUcontext context;
    res = cuCtxCreate(&context, 0, device);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 将上下文保存为成员变量
    context_ = context;
  }

  // 在每个测试之后执行
  void TearDown() override {
    // 销毁CUDA上下文
    CUresult res = cuCtxDestroy(context_);
    ASSERT_EQ(res, CUDA_SUCCESS);
  }

  // 一个成员变量，保存创建的CUDA上下文
  CUcontext context_;
};

// 测试cuCtxSetCurrent能否成功处理一个空的CUDA上下文
TEST_F(CuCtxSetCurrentTest, CanHandleNullContext) {
  // 获取当前绑定的CUDA上下文
  CUcontext current_context;
  CUresult res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否与创建的CUDA上下文相同
  EXPECT_EQ(current_context, context_);

  // 调用cuCtxSetCurrent，传入一个空的CUDA上下文
  CUcontext null_context = NULL;
  res = cuCtxSetCurrent(null_context);
  EXPECT_EQ(res, CUDA_SUCCESS);

  // 再次获取当前绑定的CUDA上下文
  res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否为空
  EXPECT_EQ(current_context, null_context);
}

// 测试cuCtxSetCurrent能否正确处理无效的CUDA上下文
TEST_F(CuCtxSetCurrentTest, CanHandleInvalidContext) {
  // 获取当前绑定的CUDA上下文
  CUcontext current_context;
  CUresult res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否与创建的CUDA上下文相同
  EXPECT_EQ(current_context, context_);

  // 调用cuCtxSetCurrent，传入一个无效的CUDA上下文（例如一个已经销毁的上下文）
  CUcontext invalid_context = context_; // 这里假设使用创建的上下文
  res = cuCtxDestroy(invalid_context); // 销毁这个上下文
  ASSERT_EQ(res, CUDA_SUCCESS);
  res = cuCtxSetCurrent(invalid_context); // 尝试将这个已经销毁的上下文绑定到线程
  EXPECT_EQ(res, CUDA_ERROR_INVALID_CONTEXT); // 应该返回错误码

  // 再次获取当前绑定的CUDA上下文
  res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否为空（因为之前传入了一个空的上下文）
  EXPECT_EQ(current_context, NULL);
}

// 测试cuCtxSetCurrent能否正确处理未初始化的CUDA驱动API
TEST_F(CuCtxSetCurrentTest, CanHandleUninitializedDriver) {
  // 获取当前绑定的CUDA上下文
  CUcontext current_context;
  CUresult res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否与创建的CUDA上下文相同
  EXPECT_EQ(current_context, context_);

  // 调用cuDriverExit，退出CUDA驱动API
  res = cuDriverExit();
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 调用cuCtxSetCurrent，传入一个非空的CUDA上下文（可以是同一个或者不同的）
  CUcontext new_context = context_; // 这里假设使用同一个上下文
  res = cuCtxSetCurrent(new_context);
  EXPECT_EQ(res, CUDA_ERROR_NOT_INITIALIZED); // 应该返回错误码

}

// 测试cuCtxSetCurrent能否正确处理多个CUDA上下文
TEST_F(CuCtxSetCurrentTest, CanHandleMultipleContexts) {
  // 获取当前绑定的CUDA上下文
  CUcontext current_context;
  CUresult res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否与创建的CUDA上下文相同
  EXPECT_EQ(current_context, context_);

  // 获取第二个设备（如果存在的话）
  int device_count = 0;
  res = cuDeviceGetCount(&device_count);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 如果只有一个设备，跳过测试
  if (device_count == 1) {
    GTEST_SKIP();
  }

  // 获取第二个设备
  CUdevice device;
  res = cuDeviceGet(&device, 1);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 创建一个第二个设备的CUDA上下文
  CUcontext second_context;
  res = cuCtxCreate(&second_context, 0, device);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 调用cuCtxSetCurrent，传入第二个设备的CUDA上下文
  res = cuCtxSetCurrent(second_context);
  EXPECT_EQ(res, CUDA_SUCCESS);

  // 再次获取当前绑定的CUDA上下文
  res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否与第二个设备的CUDA上下文相同
  EXPECT_EQ(current_context, second_context);

  // 销毁第二个设备的CUDA上下文
  res = cuCtxDestroy(second_context);
  ASSERT_EQ(res, CUDA_SUCCESS);
}

// 测试cuCtxSetCurrent能否正确处理重复调用
TEST_F(CuCtxSetCurrentTest, CanHandleRepeatedCalls) {
  // 获取当前绑定的CUDA上下文
  CUcontext current_context;
  CUresult res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否与创建的CUDA上下文相同
  EXPECT_EQ(current_context, context_);

  // 调用cuCtxSetCurrent，传入一个非空的CUDA上下文（可以是同一个或者不同的）
  CUcontext new_context = context_; // 这里假设使用同一个上下文
  res = cuCtxSetCurrent(new_context);
  EXPECT_EQ(res, CUDA_SUCCESS);

  // 再次获取当前绑定的CUDA上下文
  res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否与传入的CUDA上下文相同
  EXPECT_EQ(current_context, new_context);

  // 再次调用cuCtxSetCurrent，传入同一个非空的CUDA上下文
  res = cuCtxSetCurrent(new_context);
  EXPECT_EQ(res, CUDA_SUCCESS);

   // 再次获取当前绑定的CUDA上下文
   res = cuCtxGetCurrent(&current_context);
   ASSERT_EQ(res, CUDA_SUCCESS);

   // 检查当前绑定的CUDA上下文是否与传入的CUDA上下文相同
   EXPECT_EQ(current_context, new_context);

   // 调用cuCtxSetCurrent，传入一个空的CUDA上下文
   CUcontext null_context = NULL;
   res = cuCtxSetCurrent(null_context);
   EXPECT_EQ(res, CUDA_SUCCESS);

    // 再次获取当前绑定的CUDA上下文
    res = cuCtxGetCurrent(&current_context);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 检查当前绑定的CUDA上下文是否为空
    EXPECT_EQ(current_context, null_context);

    // 再次调用cuCtxSetCurrent，传入一个空的CUDA上下文
    res = cuCtxSetCurrent(null_context);
    EXPECT_EQ(res, CUDA_SUCCESS);

     // 再次获取当前绑定的CUDA上下文
     res = cuCtxGetCurrent(&current_context);
     ASSERT_EQ(res, CUDA_SUCCESS);

     // 检查当前绑定的CUDA上下文是否为空
     EXPECT_EQ(current_context, null_context);

     // 调用cuCtxSetCurrent，传入之前创建的非空的CUDA上下文
     res = cuCtxSetCurrent(context_);
     EXPECT_EQ(res, CUDA_SUCCESS);

      // 再次获取当前绑定的CUDA上下文
      res = cuCtxGetCurrent(&current_context);
      ASSERT_EQ(res, CUDA_SUCCESS);

      // 检查当前绑定的CUDA上下文是否与创建的CUDA上下文相同
      EXPECT_EQ(current_context, context_);
}

// 测试cuCtxSetCurrent能否正确处理多线程的情况
TEST_F(CuCtxSetCurrentTest, CanHandleMultiThread) {
  // 获取当前绑定的CUDA上下文
  CUcontext current_context;
  CUresult res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否与创建的CUDA上下文相同
  EXPECT_EQ(current_context, context_);

  // 创建一个子线程
  std::thread child_thread([&]() {
    // 在子线程中获取当前绑定的CUDA上下文
    CUcontext child_context;
    CUresult child_res = cuCtxGetCurrent(&child_context);
    ASSERT_EQ(child_res, CUDA_SUCCESS);

    // 检查子线程中的CUDA上下文是否为空（因为没有绑定过）
    EXPECT_EQ(child_context, NULL);

    // 在子线程中调用cuCtxSetCurrent，传入一个非空的CUDA上下文（可以是同一个或者不同的）
    CUcontext new_context = context_; // 这里假设使用同一个上下文
    child_res = cuCtxSetCurrent(new_context);
    EXPECT_EQ(child_res, CUDA_SUCCESS);

    // 在子线程中再次获取当前绑定的CUDA上下文
    child_res = cuCtxGetCurrent(&child_context);
    ASSERT_EQ(child_res, CUDA_SUCCESS);

    // 检查子线程中的CUDA上下文是否与传入的CUDA上下文相同
    EXPECT_EQ(child_context, new_context);
  });

  // 等待子线程结束
  child_thread.join();

  // 在主线程中再次获取当前绑定的CUDA上下文
  res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查主线程中的CUDA上下文是否与创建的CUDA上下文相同（因为子线程不影响主线程）
  EXPECT_EQ(current_context, context_);
}

// 测试cuCtxSetCurrent能否正确处理异步调用
TEST_F(CuCtxSetCurrentTest, CanHandleAsyncCall) {
  // 获取当前绑定的CUDA上下文
  CUcontext current_context;
  CUresult res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否与创建的CUDA上下文相同
  EXPECT_EQ(current_context, context_);

  // 创建一个异步任务，使用std::async
  auto async_task = std::async(std::launch::async, [&]() {
    // 在异步任务中获取当前绑定的CUDA上下文
    CUcontext async_context;
    CUresult async_res = cuCtxGetCurrent(&async_context);
    ASSERT_EQ(async_res, CUDA_SUCCESS);

    // 检查异步任务中的CUDA上下文是否为空（因为没有绑定过）
    EXPECT_EQ(async_context, NULL);

    // 在异步任务中调用cuCtxSetCurrent，传入一个非空的CUDA上下文（可以是同一个或者不同的）
    CUcontext new_context = context_; // 这里假设使用同一个上下文
    async_res = cuCtxSetCurrent(new_context);
    EXPECT_EQ(async_res, CUDA_SUCCESS);

    // 在异步任务中再次获取当前绑定的CUDA上下文
    async_res = cuCtxGetCurrent(&async_context);
    ASSERT_EQ(async_res, CUDA_SUCCESS);

    // 检查异步任务中的CUDA上下文是否与传入的CUDA上下文相同
    EXPECT_EQ(async_context, new_context);
  });

  // 等待异步任务结束，并获取返回值（如果有的话）
  auto async_result = async_task.get();

  // 在主线程中再次获取当前绑定的CUDA上下文
  res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查主线程中的CUDA上下文是否与创建的CUDA上下文相同（因为异步任务不影响主线程）
  EXPECT_EQ(current_context, context_);
}


// 测试cuCtxSetCurrent能否正确处理CUDA流
TEST_F(CuCtxSetCurrentTest, CanHandleStream) {
  // 获取当前绑定的CUDA上下文
  CUcontext current_context;
  CUresult res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否与创建的CUDA上下文相同
  EXPECT_EQ(current_context, context_);

  // 创建一个CUDA流
  CUstream stream;
  res = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 在CUDA流中执行一个异步内核（这里假设有一个名为kernel的内核函数）
  kernel<<<1, 1, 0, stream>>>();
  res = cudaGetLastError();
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 调用cuCtxSetCurrent，传入一个空的CUDA上下文
  CUcontext null_context = NULL;
  res = cuCtxSetCurrent(null_context);
  EXPECT_EQ(res, CUDA_SUCCESS);

  // 再次获取当前绑定的CUDA上下文
  res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否为空
  EXPECT_EQ(current_context, null_context);

  // 等待CUDA流完成
  res = cuStreamSynchronize(stream);
  EXPECT_EQ(res, CUDA_SUCCESS);

  // 销毁CUDA流
  res = cuStreamDestroy(stream);
  ASSERT_EQ(res, CUDA_SUCCESS);
}

// 测试cuCtxSetCurrent能否正确处理CUDA事件
TEST_F(CuCtxSetCurrentTest, CanHandleEvent) {
  // 获取当前绑定的CUDA上下文
  CUcontext current_context;
  CUresult res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否与创建的CUDA上下文相同
  EXPECT_EQ(current_context, context_);

  // 创建一个CUDA事件
  CUevent event;
  res = cuEventCreate(&event, CU_EVENT_DEFAULT);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 在当前上下文中记录一个CUDA事件
  res = cuEventRecord(event, NULL);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 调用cuCtxSetCurrent，传入一个空的CUDA上下文
  CUcontext null_context = NULL;
  res = cuCtxSetCurrent(null_context);
  EXPECT_EQ(res, CUDA_SUCCESS);

  // 再次获取当前绑定的CUDA上下文
  res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否为空
  EXPECT_EQ(current_context, null_context);

  // 等待CUDA事件完成
  res = cuEventSynchronize(event);
  EXPECT_EQ(res, CUDA_SUCCESS);

  // 销毁CUDA事件
  res = cuEventDestroy(event);
  ASSERT_EQ(res, CUDA_SUCCESS);
}

// 测试cuCtxSetCurrent能否正确处理CUDA内存分配和释放
TEST_F(CuCtxSetCurrentTest, CanHandleMemoryAllocationAndFree) {
  // 获取当前绑定的CUDA上下文
  CUcontext current_context;
  CUresult res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否与创建的CUDA上下文相同
  EXPECT_EQ(current_context, context_);

  // 在当前上下文中分配一块CUDA设备内存
  CUdeviceptr dev_ptr;
  size_t size = 1024;
  res = cuMemAlloc(&dev_ptr, size);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 调用cuCtxSetCurrent，传入一个空的CUDA上下文
  CUcontext null_context = NULL;
  res = cuCtxSetCurrent(null_context);
  EXPECT_EQ(res, CUDA_SUCCESS);

  // 再次获取当前绑定的CUDA上下文
  res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否为空
  EXPECT_EQ(current_context, null_context);

  // 在空的上下文中尝试释放之前分配的CUDA设备内存
  res = cuMemFree(dev_ptr);
  EXPECT_EQ(res, CUDA_ERROR_INVALID_CONTEXT); // 应该返回错误码

  // 调用cuCtxSetCurrent，传入之前创建的非空的CUDA上下文
  res = cuCtxSetCurrent(context_);
  EXPECT_EQ(res, CUDA_SUCCESS);

  // 再次获取当前绑定的CUDA上下文
  res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否与创建的CUDA上下文相同
  EXPECT_EQ(current_context, context_);

   // 在非空的上下文中释放之前分配的CUDA设备内存
   res = cuMemFree(dev_ptr);
   EXPECT_EQ(res, CUDA_SUCCESS);
}

// 测试cuCtxSetCurrent能否正确处理CUDA模块加载和卸载
TEST_F(CuCtxSetCurrentTest, CanHandleModuleLoadAndUnload) {
  // 获取当前绑定的CUDA上下文
  CUcontext current_context;
  CUresult res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否与创建的CUDA上下文相同
  EXPECT_EQ(current_context, context_);

  // 在当前上下文中加载一个CUDA模块（这里假设有一个名为module.ptx的文件）
  CUmodule module;
  res = cuModuleLoad(&module, "module.ptx");
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 调用cuCtxSetCurrent，传入一个空的CUDA上下文
  CUcontext null_context = NULL;
  res = cuCtxSetCurrent(null_context);
  EXPECT_EQ(res, CUDA_SUCCESS);

  // 再次获取当前绑定的CUDA上下文
  res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否为空
  EXPECT_EQ(current_context, null_context);

  // 在空的上下文中尝试卸载之前加载的CUDA模块
  res = cuModuleUnload(module);
  EXPECT_EQ(res, CUDA_ERROR_INVALID_CONTEXT); // 应该返回错误码

  // 调用cuCtxSetCurrent，传入之前创建的非空的CUDA上下文
  res = cuCtxSetCurrent(context_);
  EXPECT_EQ(res, CUDA_SUCCESS);

  // 再次获取当前绑定的CUDA上下文
  res = cuCtxGetCurrent(&current_context);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 检查当前绑定的CUDA上下文是否与创建的CUDA上下文相同
  EXPECT_EQ(current_context, context_);

   // 在非空的上下文中卸载之前加载的CUDA模块
   res = cuModuleUnload(module);
   EXPECT_EQ(res, CUDA_SUCCESS);
}
