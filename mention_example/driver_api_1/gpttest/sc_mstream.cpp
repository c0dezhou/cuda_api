// CUDA驱动头文件cuda.h
#include <cuda.h>
#include <stdio.h>
#include <string.h>
// C++多线程头文件thread
#include <thread>
// C++互斥锁头文件mutex
#include <mutex>

// 宏定义checkDriver (op)，在宏体内部调用__check_cuda_driver ( (op), #op, FILE, LINE )，其中#op表示将op参数转化为字符串。
#define checkDriver(op) __check_cuda_driver((op), #op, __FILE__, __LINE__)

bool __check_cuda_driver(CUresult code, const char* op, const char* file, int line){
  if(code != CUresult::CUDA_SUCCESS){
    const char* err_name = nullptr;
    const char* err_message = nullptr;
    cuGetErrorName(code, &err_name);
    cuGetErrorString(code, &err_message);
    printf("%s:%d %s failed. \n code = %s, message = %s\n", file, line, op, err_name, err_message);
    return false;
  }
  return true;
}

// 定义一个全局变量N，表示数组的大小
const int N = 100;

// 定义一个全局变量device，表示设备句柄
CUdevice device = 0;

// 定义一个全局变量module，表示模块句柄
CUmodule module = nullptr;

// 定义一个全局变量kernel，表示内核函数句柄
CUfunction kernel = nullptr;

// 定义一个全局变量context，表示上下文句柄
CUcontext context = nullptr;

// 定义一个全局变量stream，表示流句柄
CUstream stream = nullptr;

// 定义一个全局变量mutex，表示互斥锁对象
std::mutex mutex;

// 定义一个函数run_kernel，表示在一个线程中执行内核函数的逻辑
void run_kernel(int thread_id){
  // 获取当前线程的ID
  printf("Thread %d started.\n", thread_id);

  // 设置当前上下文
  checkDriver(cuCtxSetCurrent(context));

  // 分配设备内存
  CUdeviceptr d_a = nullptr;
  CUdeviceptr d_b = nullptr;
  CUdeviceptr d_c = nullptr;
  size_t size = sizeof(int) * N;
  checkDriver(cuMemAlloc(&d_a, size));
  checkDriver(cuMemAlloc(&d_b, size));
  checkDriver(cuMemAlloc(&d_c, size));

  // 复制主机数据到设备
  int h_a[N];
  int h_b[N];
  for(int i=0;i<N;i++){
    h_a[i] = i + thread_id; // 每个线程的数组元素不同
    h_b[i] = i + thread_id;
  }
  checkDriver(cuMemcpyHtoD(d_a, h_a, size));
  checkDriver(cuMemcpyHtoD(d_b, h_b, size));

  // 启动内核函数
  int block_size = 256;
  int grid_size = (N + block_size -1) / block_size;
  void* args[] = {&d_a,&d_b,&d_c,&N};
  // 使用互斥锁保证只有一个线程可以启动内核函数
  mutex.lock();
  checkDriver(cuLaunchKernel(kernel,
                             grid_size,
                             block_size,
                             args,
                             NULL,
                             stream)); // 使用共享的流
  mutex.unlock();

   // 等待内核执行完成
   checkDriver(cuStreamSynchronize(stream));

   // 复制设备数据到主机
   int h_c[N];
   checkDriver(cuMemcpyDtoH(h_c,d_c,size));

   // 打印结果
   for(int i=0;i<N;i++){
     printf("%d + %d = %d\n",h_a[i],h_b[i],h_c[i]);
   }

   //释放设备内存
   checkDriver(cuMemFree(d_a));
   checkDriver(cuMemFree(d_b));
   checkDriver(cuMemFree(d_c));

   printf("Thread %d finished.\n", thread_id);
}

int main(){
  // 检查cuda driver的初始化
  checkDriver(cuInit(0));

  // 获取设备句柄
  checkDriver(cuDeviceGet(&device, 0));

  // 加载PTX或CUBIN文件
  checkDriver(cuModuleLoad(&module, "kernel.ptx"));

  // 获取内核函数句柄
  checkDriver(cuModuleGetFunction(&kernel, module, "add"));

  // 创建上下文
  checkDriver(cuCtxCreate(&context, 0, device));

  // 创建流
  checkDriver(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  // 定义一个常量M，表示线程的数量
  const int M = 4;

  // 创建M个线程
  std::thread threads[M];
  for(int i=0;i<M;i++){
    threads[i] = std::thread(run_kernel, i); // 线程ID为i
  }

  // 等待所有线程结束
  for(int i=0;i<M;i++){
    threads[i].join();
    printf("Thread %d joined.\n", i);
  }

  // 销毁流
  checkDriver(cuStreamDestroy(stream));

  // 销毁上下文
  checkDriver(cuCtxDestroy(context));

  // 卸载模块
  checkDriver(cuModuleUnload(module));

  return 0;
}


// /////////////////////////////////////////////////////////////////////////////////////////

// CUDA驱动头文件cuda.h
#include <cuda.h>
#include <stdio.h>
#include <string.h>
// C++多线程头文件thread
#include <thread>

// 宏定义checkDriver (op)，在宏体内部调用__check_cuda_driver ( (op), #op, FILE, LINE )，其中#op表示将op参数转化为字符串。
#define checkDriver(op) __check_cuda_driver((op), #op, __FILE__, __LINE__)

bool __check_cuda_driver(CUresult code, const char* op, const char* file, int line){
  if(code != CUresult::CUDA_SUCCESS){
    const char* err_name = nullptr;
    const char* err_message = nullptr;
    cuGetErrorName(code, &err_name);
    cuGetErrorString(code, &err_message);
    printf("%s:%d %s failed. \n code = %s, message = %s\n", file, line, op, err_name, err_message);
    return false;
  }
  return true;
}

// 定义一个全局变量N，表示数组的大小
const int N = 100;

// 定义一个全局变量device，表示设备句柄
CUdevice device = 0;

// 定义一个全局变量module，表示模块句柄
CUmodule module = nullptr;

// 定义一个全局变量kernel，表示内核函数句柄
CUfunction kernel = nullptr;

// 定义一个全局变量context，表示上下文句柄
CUcontext context = nullptr;

// 定义一个全局变量stream，表示流句柄
CUstream stream = nullptr;

// 定义一个全局变量event，表示事件句柄
CUevent event = nullptr;

// 定义一个函数run_kernel，表示在一个线程中执行内核函数的逻辑
void run_kernel(int thread_id){
  // 获取当前线程的ID
  printf("Thread %d started.\n", thread_id);

  // 设置当前上下文
  checkDriver(cuCtxSetCurrent(context));

  // 分配设备内存
  CUdeviceptr d_a = nullptr;
  CUdeviceptr d_b = nullptr;
  CUdeviceptr d_c = nullptr;
  size_t size = sizeof(int) * N;
  checkDriver(cuMemAlloc(&d_a, size));
  checkDriver(cuMemAlloc(&d_b, size));
  checkDriver(cuMemAlloc(&d_c, size));

  // 复制主机数据到设备
  int h_a[N];
  int h_b[N];
  for(int i=0;i<N;i++){
    h_a[i] = i + thread_id; // 每个线程的数组元素不同
    h_b[i] = i + thread_id;
  }
  checkDriver(cuMemcpyHtoD(d_a, h_a, size));
  checkDriver(cuMemcpyHtoD(d_b, h_b, size));

  // 启动内核函数
  int block_size = 256;
  int grid_size = (N + block_size -1) / block_size;
  void* args[] = {&d_a,&d_b,&d_c,&N};
  // 使用事件来保证只有一个线程可以启动内核函数
  checkDriver(cuStreamWaitEvent(stream, event, 0)); // 等待事件完成
  checkDriver(cuLaunchKernel(kernel,
                             grid_size,
                             block_size,
                             args,
                             NULL,
                             stream)); // 使用共享的流
  checkDriver(cuEventRecord(event, stream)); // 记录事件

   // 等待内核执行完成
   checkDriver(cuStreamSynchronize(stream));

   // 复制设备数据到主机
   int h_c[N];
   checkDriver(cuMemcpyDtoH(h_c,d_c,size));

   // 打印结果
   for(int i=0;i<N;i++){
     printf("%d + %d = %d\n",h_a[i],h_b[i],h_c[i]);
   }

   //释放设备内存
   checkDriver(cuMemFree(d_a));
   checkDriver(cuMemFree(d_b));
   checkDriver(cuMemFree(d_c));

   printf("Thread %d finished.\n", thread_id);
}

int main(){
  // 检查cuda driver的初始化
  checkDriver(cuInit(0));

  // 获取设备句柄
  checkDriver(cuDeviceGet(&device, 0));

  // 加载PTX或CUBIN文件
  checkDriver(cuModuleLoad(&module, "kernel.ptx"));

  // 获取内核函数句柄
  checkDriver(cuModuleGetFunction(&kernel, module, "add"));

  // 创建上下文
  checkDriver(cuCtxCreate(&context, 0, device));

  // 创建流
  checkDriver(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  // 创建事件
    checkDriver(cuEventCreate(&event, CU_EVENT_DEFAULT));

  // 定义一个常量M，表示线程的数量
  const int M = 4;

  // 创建M个线程
  std::thread threads[M];
  for(int i=0;i<M;i++){
    threads[i] = std::thread(run_kernel, i); // 线程ID为i
  }

  // 等待所有线程结束
  for(int i=0;i<M;i++){
    threads[i].join();
    printf("Thread %d joined.\n", i);
  }

  // 销毁事件
  checkDriver(cuEventDestroy(event));

  // 销毁流
  checkDriver(cuStreamDestroy(stream));

  // 销毁上下文
  checkDriver(cuCtxDestroy(context));

  // 卸载模块
  checkDriver(cuModuleUnload(module));

  return 0;
}

