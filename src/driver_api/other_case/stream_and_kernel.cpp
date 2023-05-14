#include <gtest/gtest.h>
#include <cuda.h>
#include <thread>

// 测试类：KernelTest
class KernelTest : public ::testing::Test {
protected:
  // 初始化CUDA驱动API
  void SetUp() override {
    CUresult res = cuInit(0);
    ASSERT_EQ(res, CUDA_SUCCESS);
  }
};

// 线程类：KernelThread
class KernelThread : public std::thread {
public:
  // 构造函数：接收设备ID、上下文指针、内核函数指针和参数数组，并启动线程
  KernelThread(int device_id, CUcontext context, CUfunction kernel, void** params) : device_id_(device_id), context_(context), kernel_(kernel), params_(params) {
    start();
  }

  // 启动线程
  void start() {
    std::thread::operator=(std::thread(&KernelThread::run, this));
  }

  // 等待线程结束
  void join() {
    std::thread::join();
  }

private:
  // 线程运行的方法
  void run() {
    // 设置当前线程的上下文
    CUresult res = cuCtxSetCurrent(context_);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 启动内核函数
    res = cuLaunchKernel(kernel_, 1, 1, 1, 1, 1, 1, 0, NULL, params_, NULL);
    ASSERT_EQ(res, CUDA_SUCCESS);

    // 等待当前设备完成所有操作
    res = cuCtxSynchronize();
    ASSERT_EQ(res, CUDA_SUCCESS);
  }

  // 设备ID、上下文指针、内核函数指针和参数数组
  int device_id_;
  CUcontext context_;
  CUfunction kernel_;
  void** params_;
};

// 测试方法：单流单kernel的执行
TEST_F(KernelTest, SingleStreamSingleKernel) {
  // 获取第一个设备的ID
  int device_id = 0;

  // 创建第一个设备的上下文
  CUcontext context = nullptr;
  CUresult res = cuCtxCreate(&context, 0, device_id);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 创建一个流对象，并将其设置为当前流
  CUstream stream = nullptr;
  res = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
  ASSERT_EQ(res, CUDA_SUCCESS);
  res = cuCtxSetCurrent(stream);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 加载内核函数模块
  CUmodule module = nullptr;
  res = cuModuleLoad(&module, "kernel.cubin");
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 获取内核函数指针
  CUfunction kernel = nullptr;
  res = cuModuleGetFunction(&kernel, module, "kernel");
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 准备内核函数参数
  int n = 10;
  float* arr_on_host = new float[n];
  for (int i = 0; i < n; i++) {
    arr_on_host[i] = i + 1;
  }
  // 分配设备内存，并将主机内存复制到设备内存
  float* arr_on_device = nullptr;
  res = cuMemAlloc(&arr_on_device, n * sizeof(float));
  ASSERT_EQ(res, CUDA_SUCCESS);
  res = cuMemcpyHtoD(arr_on_device, arr_on_host, n * sizeof(float));
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 设置内核函数参数
  void* params[] = {&arr_on_device, &n};

  // 创建一个线程对象，传入设备ID、上下文指针、内核函数指针和参数数组，并启动它
  KernelThread thread(device_id, context, kernel, params);
  
  // 等待线程执行完毕
  thread.join();

  // 检查是否有错误发生
  ASSERT_NO_FATAL_FAILURE();

  // 将设备内存复制回主机内存，并检查结果
  res = cuMemcpyDtoH(arr_on_host, arr_on_device, n * sizeof(float));
  ASSERT_EQ(res, CUDA_SUCCESS);
  for (int i = 0; i < n; i++) {
    ASSERT_EQ(arr_on_host[i], (i + 1) * (i + 1));
  }

  // 释放主机内存和设备内存
  delete[] arr_on_host;
  res = cuMemFree(arr_on_device);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 销毁流对象
  res = cuStreamDestroy(stream);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 销毁内核函数模块
  res = cuModuleUnload(module);
  ASSERT_EQ(res, CUDA_SUCCESS);

  // 销毁设备上下文
  res = cuCtxDestroy(context);
  ASSERT_EQ(res, CUDA_SUCCESS);
}
