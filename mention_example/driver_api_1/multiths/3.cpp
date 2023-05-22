// 假设你已经安装了cuda toolkit，并且有一个cuda device可用
#include <iostream>
#include <thread>
#include <vector>
#include <cuda.h>

// 定义一个函数指针的类，用于存储不同的cuda driver api函数
class DriverFunc {
public:
    // 构造函数，接受一个指向cuda driver api函数的指针
    DriverFunc(CUresult (*func)(CUdeviceptr, CUdeviceptr, size_t)) {
        this->func = func;
    }

    // 调用函数指针，传入参数
    void operator()(CUdeviceptr a, CUdeviceptr b, size_t n) {
        func(a, b, n);
    }

private:
    // 函数指针，指向一个接受两个CUdeviceptr和一个size_t大小的cuda driver api函数
    CUresult (*func)(CUdeviceptr, CUdeviceptr, size_t);
};

// 定义一个简单的cuda driver api函数，用于计算两个数组的和
CUresult add(CUdeviceptr a, CUdeviceptr b, size_t n) {
    // 获取当前上下文
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);

    // 创建并加载模块
    CUmodule mod;
    cuModuleLoad(&mod, "add.ptx");

    // 获取内核函数句柄
    CUfunction func;
    cuModuleGetFunction(&func, mod, "add");

    // 设置内核函数参数
    void* args[] = {&a, &b, &n};

    // 启动内核函数
    cuLaunchKernel(func, 1, 1, 1, n, 1, 1, 0, NULL, args, NULL);

    // 同步上下文
    cuCtxSynchronize();

    // 卸载模块
    cuModuleUnload(mod);

    // 返回结果
    return CUDA_SUCCESS;
}

// 定义一个简单的cuda driver api函数，用于计算两个数组的差
CUresult sub(CUdeviceptr a, CUdeviceptr b, size_t n) {
    // 获取当前上下文
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);

    // 创建并加载模块
    CUmodule mod;
    cuModuleLoad(&mod, "sub.ptx");

    // 获取内核函数句柄
    CUfunction func;
    cuModuleGetFunction(&func, mod, "sub");

    // 设置内核函数参数
    void* args[] = {&a, &b, &n};

    // 启动内核函数
cuLaunchKernel(func, 1, 1, 1, n, 1, 1, 0, NULL, args, NULL);

// 同步上下文
cuCtxSynchronize();

// 卸载模块
cuModuleUnload(mod);

// 返回结果
return CUDA_SUCCESS;
}

// 定义一个函数，用于创建并执行多个线程，每个线程调用不同的cuda driver api函数
void run_threads(std::vector<DriverFunc> funcs, CUdeviceptr a,
                 CUdeviceptr b, size_t n) {
  // 获取可用的cuda device数量
  int device_count;
  cuDeviceGetCount(&device_count);
  std::cout << "Available cuda devices: " << device_count << std::endl;

  // 创建一个线程数组，大小为函数数组的大小
  std::thread threads[funcs.size()];

  // 遍历每个函数
  for (int i = 0; i < funcs.size(); i++) {
      // 设置当前线程使用的cuda device，假设每个线程使用不同的device
      cuDeviceSet(i % device_count);
      std::cout << "Thread " << i << " using device " << i % device_count << std::endl;

      // 创建并推送一个新的上下文到当前线程和设备上
      CUcontext ctx;
      cuCtxCreate(&ctx,CU_CTX_SCHED_AUTO,i % device_count);
      cuCtxPushCurrent(ctx);

      // 创建一个线程，调用当前函数，并传入参数
      threads[i] = std::thread(funcs[i], a,b,n);
  }

  // 等待所有线程完成
  for (int i = 0; i < funcs.size(); i++) {
      threads[i].join();
      // 弹出并销毁当前线程和设备上的上下文
      CUcontext ctx;
      cuCtxPopCurrent(&ctx);
      cuCtxDestroy(ctx);
  }
}

// 主函数，测试多线程并发调用不同的cuda driver api函数
int main() {
  // 定义数组大小
  const size_t N = 10;

  // 分配主机内存，并初始化数据
  int* h_a = new int[N];
  int* h_b = new int[N];
  for (int i = 0; i < N; i++) {
      h_a[i] = i;
      h_b[i] = i * 2;
}

// 分配设备内存，并拷贝数据
CUdeviceptr d_a;
CUdeviceptr d_b;
cuMemAlloc(&d_a,N * sizeof(int));
cuMemAlloc(&d_b,N * sizeof(int));
cuMemcpyHtoD(d_a,h_a,N * sizeof(int));
cuMemcpyHtoD(d_b,h_b,N * sizeof(int));

// 定义两个函数指针的类对象，分别指向add和sub函数
DriverFunc add_func(add);
DriverFunc sub_func(sub);

// 创建一个函数指针的类对象数组，存储两个对象
std::vector<DriverFunc> funcs;
funcs.push_back(add_func);
funcs.push_back(sub_func);

// 调用run_threads函数，传入函数对象数组和参数
run_threads(funcs,d_a,d_b,N);

// 拷贝结果回主机内存，并打印

cuMemcpyDtoH(h_a,d_a,N * sizeof(int));
for (int i = 0; i < N; i++) {
std::cout << h_a[i] << std::endl;
}

//释放内存，并返回

delete[] h_a;
delete[] h_b;
cuMemFree(d_a);
cuMemFree(d_b);
return 0;
}
