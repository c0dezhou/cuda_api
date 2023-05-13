// 定义一个测试类
class cuMemGetInfoTest : public ::testing::Test {
protected:
  // 在每个测试之前执行
  void SetUp() override {
    // 初始化CUDA环境
    cuInit(0);
    // 获取第一个可用的CUDA设备
    cuDeviceGet(&device, 0);
    // 创建一个CUDA上下文
    cuCtxCreate(&context, 0, device);
  }

  // 在每个测试之后执行
  void TearDown() override {
    // 销毁CUDA上下文
    cuCtxDestroy(context);
  }

  // 定义一些成员变量
  CUdevice device; // 设备句柄
  CUcontext context; // 上下文句柄
};

// 定义一个基本行为测试，检查函数是否能正确获取内存信息
TEST_F(cuMemGetInfoTest, BasicBehavior) {
  // 定义一些变量
  size_t free; // 空闲内存
  size_t total; // 总内存

  // 调用cuMemGetInfo函数，获取空闲和总内存
  cuMemGetInfo(&free, &total);

  // 检查空闲和总内存是否大于0
  EXPECT_GT(free, 0);
  EXPECT_GT(total, 0);

  // 检查空闲内存是否小于等于总内存
    // 检查空闲内存是否小于等于总内存
  EXPECT_LE(free, total);
}

// 定义一个异常测试，检查函数是否能正确处理无效的参数
TEST_F(cuMemGetInfoTest, InvalidArguments) {
  // 定义一些变量
  size_t free; // 空闲内存
  size_t total; // 总内存
  CUresult result; // 函数返回值

  // 调用cuMemGetInfo函数，传入无效的空闲内存指针
  result = cuMemGetInfo(NULL, &total);
  // 检查函数是否返回CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE);

  // 调用cuMemGetInfo函数，传入无效的总内存指针
  result = cuMemGetInfo(&free, NULL);
  // 检查函数是否返回CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE);

  // 调用cuMemGetInfo函数，传入无效的空闲和总内存指针
  result = cuMemGetInfo(NULL, NULL);
  // 检查函数是否返回CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE);
}

// 定义一个边界值测试，检查函数是否能正确处理极端的内存分配和释放
TEST_F(cuMemGetInfoTest, BoundaryValues) {
  // 定义一些变量
  size_t free; // 空闲内存
  size_t total; // 总内存
  size_t free_before; // 分配前的空闲内存
  size_t free_after; // 分配后的空闲内存
  CUdeviceptr d_ptr; // 设备指针

  // 调用cuMemGetInfo函数，获取分配前的空闲和总内存
  cuMemGetInfo(&free_before, &total);

  // 调用cuMemAlloc函数，分配最大的设备内存（free_before）
  cuMemAlloc(&d_ptr, free_before);

  // 调用cuMemGetInfo函数，获取分配后的空闲和总内存
  cuMemGetInfo(&free_after, &total);

  // 检查分配后的空闲内存是否接近0（允许一定的误差）
  EXPECT_NEAR(free_after, 0, 1024);

  // 调用cuMemFree函数，释放设备内存

    // 创建一个第二个CUDA上下文
  cuCtxCreate(&context2, 0, device);

  // 切换到第二个CUDA上下文
  cuCtxSetCurrent(context2);

  // 调用cuMemGetInfo函数，获取第二个上下文的空闲和总内存
  cuMemGetInfo(&free2, &total2);

  // 检查第二个上下文的空闲和总内存是否和第一个上下文的相同
  EXPECT_EQ(free2, free1);
  EXPECT_EQ(total2, total1);

  // 销毁第二个CUDA上下文
  cuCtxDestroy(context2);
}
