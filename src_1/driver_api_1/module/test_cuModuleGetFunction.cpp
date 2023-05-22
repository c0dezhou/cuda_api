#include <gtest/gtest.h>
#include <cuda.h>

// 定义一个测试类
class cuModuleGetFunctionTest : public ::testing::Test {
protected:
  // 在每个测试之前执行
  void SetUp() override {
    // 获取GPU数量
    cuDeviceGetCount(&device_count);
    // 如果没有GPU，跳过所有测试
    if (device_count == 0) {
      GTEST_SKIP();
    }
    // 创建一个上下文
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);
    // 加载一个模块
    cuModuleLoad(&module, "test.ptx");
  }

  // 在每个测试之后执行
  void TearDown() override {
    // 卸载模块和销毁上下文
    cuModuleUnload(module);
    cuCtxDestroy(context);
  }

  // 定义一些常量和变量
  int device_count; // GPU数量
  CUdevice device; // GPU设备
  CUcontext context; // CUDA上下文
  CUmodule module; // CUDA模块
};

// 基本行为测试：传入有效的参数，检查是否能正确获取函数句柄
TEST_F(cuModuleGetFunctionTest, BasicBehavior) {
  // 定义一个函数句柄
  CUfunction function;
  
  // 调用cuModuleGetFunction从模块中获取一个名为kernel的函数
  CUresult result = cuModuleGetFunction(&function, module, "kernel");

  // 检查返回值是否为CUDA_SUCCESS，函数句柄是否不为空
  EXPECT_EQ(result, CUDA_SUCCESS);
  EXPECT_NE(function, nullptr);
}

// 异常测试：传入无效的参数，检查是否返回错误码
TEST_F(cuModuleGetFunctionTest, InvalidArguments) {
  // 定义一个函数句柄
  CUfunction function;

  // hfunc为空指针，应返回CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(cuModuleGetFunction(nullptr, module, "kernel"), CUDA_ERROR_INVALID_VALUE);

  // hmod为空指针，应返回CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(cuModuleGetFunction(&function, nullptr, "kernel"), CUDA_ERROR_INVALID_VALUE);

  // name为空指针，应返回CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(cuModuleGetFunction(&function, module, nullptr), CUDA_ERROR_INVALID_VALUE);

  // name指向一个不存在的函数名，应返回CUDA_ERROR_NOT_FOUND
  EXPECT_EQ(cuModuleGetFunction(&function, module, "not_exist"), CUDA_ERROR_NOT_FOUND);
}

// 边界值测试：传入边界值，检查是否正确处理
TEST_F(cuModuleGetFunctionTest, BoundaryValues) {
  // 定义一个函数句柄
  CUfunction function;

  // name指向一个空字符串，应返回CUDA_ERROR_NOT_FOUND
  EXPECT_EQ(cuModuleGetFunction(&function, module, ""), CUDA_ERROR_NOT_FOUND);

  // name指向一个只有一个字符的函数名，假设模块中有这个函数，应返回CUDA_SUCCESS
  EXPECT_EQ(cuModuleGetFunction(&function, module, "f"), CUDA_SUCCESS);
  EXPECT_NE(function, nullptr);
}

// 其他测试：检查cuModuleGetFunction是否支持不同的模块和函数
TEST_F(cuModuleGetFunctionTest, DifferentModulesAndFunctions) {
  // 定义两个函数句柄
  CUfunction function1, function2;

  // 调用cuModuleGetFunction从模块中获取两个不同的函数
  CUresult result1 = cuModuleGetFunction(&function1, module, "kernel1");
  CUresult result2 = cuModuleGetFunction(&function2, module, "kernel2");

  // 检查返回值是否为CUDA_SUCCESS，函数句柄是否不为空，且不相同
  EXPECT_EQ(result1, CUDA_SUCCESS);
  EXPECT_EQ(result2, CUDA_SUCCESS);
  EXPECT_NE(function1, nullptr);
  EXPECT_NE(function2, nullptr);
  EXPECT_NE(function1, function2);

  // 加载另一个模块
  CUmodule module2;
  cuModuleLoad(&module2, "test2.ptx");

  // 调用cuModuleGetFunction从另一个模块中获取一个函数
  CUresult result3 = cuModuleGetFunction(&function3, module2, "kernel3");

  // 检查返回值是否为CUDA_SUCCESS，函数句柄是否不为空，且与前两个不相同
  EXPECT_EQ(result3, CUDA_SUCCESS);
  EXPECT_NE(function3, nullptr);
  EXPECT_NE(function3, function1);
  EXPECT_NE(function3, function2);

  // 卸载另一个模块
  cuModuleUnload(module2);
}

// 重复调用测试：检查多次调用cuModuleGetFunction是否会产生正确的结果
TEST_F(cuModuleGetFunctionTest, RepeatedCalls) {
  // 定义三个函数句柄
  CUfunction function1, function2, function3;

  // 调用cuModuleGetFunction从模块中获取一个名为kernel1的函数
  CUresult result1 = cuModuleGetFunction(&function1, module, "kernel1");

  // 检查返回值是否为CUDA_SUCCESS，函数句柄是否不为空
  EXPECT_EQ(result1, CUDA_SUCCESS);
  EXPECT_NE(function1, nullptr);

  // 再次调用cuModuleGetFunction从模块中获取同一个函数
  CUresult result2 = cuModuleGetFunction(&function2, module, "kernel1");

  // 检查返回值是否为CUDA_SUCCESS，函数句柄是否不为空，且与前一个相同
  EXPECT_EQ(result2, CUDA_SUCCESS);
  EXPECT_NE(function2, nullptr);
  EXPECT_EQ(function2, function1);

  // 调用cuModuleGetFunction从模块中获取一个名为kernel2的函数
  CUresult result3 = cuModuleGetFunction(&function3, module, "kernel2");

  // 检查返回值是否为CUDA_SUCCESS，函数句柄是否不为空，且与前两个不相同
  EXPECT_EQ(result3, CUDA_SUCCESS);
  EXPECT_NE(function3, nullptr);
  EXPECT_NE(function3, function1);
  EXPECT_NE(function3, function2);
}

