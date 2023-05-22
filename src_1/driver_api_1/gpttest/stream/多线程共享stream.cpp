#include <cuda.h>
#include <gtest/gtest.h>
#include <iostream>
#include <thread>

// 定义一些常量
const int NUM_THREADS = 4; // 线程数
const int NUM_TASKS = 3; // 每个线程的任务数
const int NUM_ELEMENTS = 1000000; // 每个任务处理的元素数
const int BLOCK_SIZE = 256; // CUDA内核的块大小

// 定义一个共享的stream
CUstream shared_stream;

// 定义一个CUDA内核函数，用于对数组进行加法操作
__global__ void add_kernel(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// 定义一个函数，用于检查CUDA调用是否成功
void check_cuda(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char *error;
    cuGetErrorName(result, &error);
    std::cerr << "CUDA error: " << error << std::endl;
    exit(1);
  }
}

// 定义一个函数，用于初始化CUDA环境和共享stream
void init_cuda() {
  // 初始化CUDA驱动API
  check_cuda(cuInit(0));

  // 获取第一个可用的CUDA设备
  CUdevice device;
  check_cuda(cuDeviceGet(&device, 0));

  // 创建一个CUDA上下文
  CUcontext context;
  check_cuda(cuCtxCreate(&context, 0, device));

  // 创建一个共享的stream
  check_cuda(cuStreamCreate(&shared_stream, CU_STREAM_NON_BLOCKING));
}

// 定义一个函数，用于释放CUDA资源
void cleanup_cuda() {
  // 销毁共享的stream
  check_cuda(cuStreamDestroy(shared_stream));

  // 销毁CUDA上下文
  check_cuda(cuCtxDestroy(cuCtxGetCurrent()));
}

// 定义一个函数，用于分配和初始化主机内存
void init_host_memory(float **a, float **b, float **c, int n) {
  // 分配主机内存
  check_cuda(cuMemHostAlloc((void **)a, n * sizeof(float), CU_MEMHOSTALLOC_PORTABLE));
  check_cuda(cuMemHostAlloc((void **)b, n * sizeof(float), CU_MEMHOSTALLOC_PORTABLE));
  check_cuda(cuMemHostAlloc((void **)c, n * sizeof(float), CU_MEMHOSTALLOC_PORTABLE));

  // 初始化主机内存
  for (int i = 0; i < n; i++) {
    (*a)[i] = i;
    (*b)[i] = i + 1;
    (*c)[i] = 0;
  }
}

// 定义一个函数，用于释放主机内存
void free_host_memory(float *a, float *b, float *c) {
  // 释放主机内存
  check_cuda(cuMemFreeHost(a));
  check_cuda(cuMemFreeHost(b));
  check_cuda(cuMemFreeHost(c));
}

// 定义一个函数，用于分配和初始化设备内存
void init_device_memory(float **d_a, float **d_b, float **d_c, int n) {
  // 分配设备内存
  check_cuda(cuMemAlloc((CUdeviceptr *)d_a, n * sizeof(float)));
  check_cuda(cuMemAlloc((CUdeviceptr *)d_b, n * sizeof(float)));
  check_cuda(cuMemAlloc((CUdeviceptr *)d_c, n * sizeof(float)));
}

// 定义一个函数，用于释放设备内存
void free_device_memory(float *d_a, float *d_b, float *d_c) {
  // 释放设备内存
  check_cuda(cuMemFree((CUdeviceptr)d_a));
  check_cuda(cuMemFree((CUdeviceptr)d_b));
  check_cuda(cuMemFree((CUdeviceptr)d_c));
}

// 定义一个函数，用于执行单个任务
void execute_task(int task_id) {
  // 计算任务处理的数据范围
  int offset = task_id * NUM_ELEMENTS;
  int size = NUM_ELEMENTS;

  // 分配和初始化主机内存
  float *h_a, *h_b, *h_c;
  init_host_memory(&h_a, &h_b, &h_c, size);

  // 分配和初始化设备内存
  float *d_a, *d_b, *d_c;
  init_device_memory(&d_a, &d_b, &d_c, size);

  // 将主机内存中的数据拷贝到设备内存中，使用共享的stream和异步调用
  check_cuda(cuMemcpyHtoDAsync((CUdeviceptr)(d_a), h_a, size * sizeof(float), shared_stream));
  check_cuda(cuMemcpyHtoDAsync((CUdeviceptr)(d_b), h_b, size * sizeof(float), shared_stream));

  // 调用CUDA内核函数，使用共享的stream和动态并行
  int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  void *kernel_args[] = {&d_a, &d_b, &d_c, &size};
  check_cuda(cuLaunchKernel(add_kernel, grid_size, 1, 1, BLOCK_SIZE, 1, 1, 0, shared_stream, kernel_args, NULL));

  // 将设备内存中的结果拷贝回主机内存中，使用共享的stream和异步调用
  check_cuda(cuMemcpyDtoHAsync(h_c, (CUdeviceptr)(d_c), size * sizeof(float), shared_stream));

  // 同步共享的stream，等待所有操作完成
  check_cuda(cuStreamSynchronize(shared_stream));

  // 检查结果是否正确，并打印相关信息
  bool passed = true;
  for (int i = 0; i < size; i++) {
    if (h_c[i] != h_a[i] + h_b[i]) {
      passed = false;
      break;
    }
  }
  std::cout << "Task " << task_id << " result: " << (passed ? "PASSED" : "FAILED") << std::endl;

  // 释放主机内存和设备内存
  free_host_memory(h_a, h_b, h_c);
  free_device_memory(d_a, d_b, d_c);
}

// 定义一个函数，用于获取当前时间
double get_time() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

// 定义main函数
int main() {
  // 初始化CUDA环境和共享stream
  init_cuda();

  // 创建一个数组，用于存储线程对象
  std::thread threads[NUM_THREADS];

  // 获取开始时间
  double start_time = get_time();

  // 创建多个线程，并为每个线程分配不同的任务
  for (int i = 0; i < NUM_THREADS; i++) {
    threads[i] = std::thread(execute_task, i * NUM_TASKS);
  }

  // 同步所有线程，等待它们完成
  for (int i = 0; i < NUM_THREADS; i++) {
    threads[i].join();
  }

  // 获取结束时间
  double end_time = get_time();

  // 计算并打印执行时间和性能
  double elapsed_time = end_time - start_time;
  double bandwidth = NUM_THREADS * NUM_TASKS * NUM_ELEMENTS * sizeof(float) * 2 / elapsed_time / (1 << 30);
  std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;
  std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

  // 清理CUDA资源
  cleanup_cuda();

  return 0;
}


#include <cuda.h>
#include <gtest/gtest.h>
#include <iostream>
#include <thread>

// 定义一些常量
const int NUM_THREADS = 4; // 线程数
const int NUM_ELEMENTS = 1000000; // 处理的元素数
const int BLOCK_SIZE = 256; // CUDA内核的块大小

// 定义一个共享的stream
CUstream shared_stream;

// 定义一个CUDA内核函数，用于对数组进行加法操作
__global__ void add_kernel(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// 定义一个CUDA内核函数，用于对数组进行乘法操作
__global__ void mul_kernel(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] * b[i];
  }
}

// 定义一个CUDA内核函数，用于对数组进行减法操作
__global__ void sub_kernel(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] - b[i];
  }
}

// 定义一个CUDA内核函数，用于对数组进行除法操作
__global__ void div_kernel(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && b[i] != 0) {
    c[i] = a[i] / b[i];
  }
}

// 定义一个函数，用于检查CUDA调用是否成功
void check_cuda(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char *error;
    cuGetErrorName(result, &error);
    std::cerr << "CUDA error: " << error << std::endl;
    exit(1);
  }
}

// 定义一个函数，用于初始化CUDA环境和共享stream
void init_cuda() {
  // 初始化CUDA驱动API
  check_cuda(cuInit(0));

  // 获取第一个可用的CUDA设备
  CUdevice device;
  check_cuda(cuDeviceGet(&device, 0));

  // 创建一个CUDA上下文
  CUcontext context;
  check_cuda(cuCtxCreate(&context, 0, device));

  // 创建一个共享的stream
  check_cuda(cuStreamCreate(&shared_stream, CU_STREAM_NON_BLOCKING));
}

// 定义一个函数，用于释放CUDA资源
void cleanup_cuda() {
  // 销毁共享的stream
  check_cuda(cuStreamDestroy(shared_stream));

  // 销毁CUDA上下文
  check_cuda(cuCtxDestroy(cuCtxGetCurrent()));
}

// 定义一个函数，用于分配和初始化主机内存
void init_host_memory(float **a, float **b, float **c, int n) {
  // 分配主机内存
  check_cuda(cuMemHostAlloc((void **)a, n * sizeof(float), CU_MEMHOSTALLOC_PORTABLE));
  check_cuda(cuMemHostAlloc((void **)b, n * sizeof(float), CU_MEMHOSTALLOC_PORTABLE));
  check_cuda(cuMemHostAlloc((void **)c, n * sizeof(float), CU_MEMHOSTALLOC_PORTABLE));

  // 初始化主机内存
  for (int i = 0; i < n; i++) {
    (*a)[i] = i;
    (*b)[i] = i + 1;
    (*c)[i] = 0;
  }
}

// 定义一个函数，用于释放主机内存
void free_host_memory(float *a, float *b, float *c) {
  // 释放主机内存
  check_cuda(cuMemFreeHost(a));
  check_cuda(cuMemFreeHost(b));
  check_cuda(cuMemFreeHost(c));
}

// 定义一个函数，用于分配和初始化设备内存
void init_device_memory(float **d_a, float **d_b, float **d_c, int n) {
  // 分配设备内存
  check_cuda(cuMemAlloc((CUdeviceptr *)d_a, n * sizeof(float)));
  check_cuda(cuMemAlloc((CUdeviceptr *)d_b, n * sizeof(float)));
  check_cuda(cuMemAlloc((CUdeviceptr *)d_c, n * sizeof(float)));
}

// 定义一个函数，用于释放设备内存
void free_device_memory(float *d_a, float *d_b, float *d_c) {
  // 释放设备内存
}

// 定义一个函数，用于执行单个任务
void execute_task(int thread_id) {
  // 分配和初始化主机内存
  float *h_a, *h_b, *h_c;
  init_host_memory(&h_a, &h_b, &h_c, NUM_ELEMENTS);

  // 分配和初始化设备内存
  float *d_a, *d_b, *d_c;
  init_device_memory(&d_a, &d_b, &d_c, NUM_ELEMENTS);

  // 将主机内存中的数据拷贝到设备内存中，使用共享的stream和异步调用
  check_cuda(cuMemcpyHtoDAsync((CUdeviceptr)(d_a), h_a, NUM_ELEMENTS * sizeof(float), shared_stream));
  check_cuda(cuMemcpyHtoDAsync((CUdeviceptr)(d_b), h_b, NUM_ELEMENTS * sizeof(float), shared_stream));

  // 根据线程的编号选择不同的CUDA内核函数来处理数据
  CUfunction kernel;
  const char *kernel_name;
  switch (thread_id) {
    case 0:
      kernel = add_kernel;
      kernel_name = "add_kernel";
      break;
    case 1:
      kernel = mul_kernel;
      kernel_name = "mul_kernel";
      break;
    case 2:
      kernel = sub_kernel;
      kernel_name = "sub_kernel";
      break;
    case 3:
      kernel = div_kernel;
      kernel_name = "div_kernel";
      break;
    default:
      std::cerr << "Invalid thread id: " << thread_id << std::endl;
      exit(1);
  }

  // 调用CUDA内核函数，使用共享的stream和动态并行
  int grid_size = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
  void *kernel_args[] = {&d_a, &d_b, &d_c, &NUM_ELEMENTS};
  check_cuda(cuLaunchKernel(kernel, grid_size, 1, 1, BLOCK_SIZE, 1, 1, 0, shared_stream, kernel_args, NULL));

  // 将设备内存中的结果拷贝回主机内存中，使用共享的stream和异步调用
  check_cuda(cuMemcpyDtoHAsync(h_c, (CUdeviceptr)(d_c), NUM_ELEMENTS * sizeof(float), shared_stream));

  // 同步共享的stream，等待所有操作完成
  check_cuda(cuStreamSynchronize(shared_stream));

  // 打印线程的编号和所选的CUDA内核函数
  std::cout << "Thread " << thread_id << " executed " << kernel_name << std::endl;

  // TODO: 检查结果是否正确，并打印相关信息

  // 释放主机内存和设备内存
}

#include <cuda.h>
#include <gtest/gtest.h>
#include <iostream>
#include <thread>

// 定义一些常量
const int NUM_THREADS = 4; // 线程数
const int NUM_ELEMENTS = 1000000; // 处理的元素数
const int BLOCK_SIZE = 256; // CUDA内核的块大小

// 定义一个共享的stream
CUstream shared_stream;

// 定义一个CUDA内核函数，用于对数组进行加法操作
__global__ void add_kernel(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// 定义一个CUDA内核函数，用于对数组进行乘法操作
__global__ void mul_kernel(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] * b[i];
  }
}

// 定义一个CUDA内核函数，用于对数组进行减法操作
__global__ void sub_kernel(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] - b[i];
  }
}

// 定义一个CUDA内核函数，用于对数组进行除法操作
__global__ void div_kernel(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && b[i] != 0) {
    c[i] = a[i] / b[i];
  }
}

// 定义一个函数，用于检查CUDA调用是否成功
void check_cuda(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char *error;
    cuGetErrorName(result, &error);
    std::cerr << "CUDA error: " << error << std::endl;
    exit(1);
  }
}

// 定义一个函数，用于初始化CUDA环境和共享stream
void init_cuda() {
  // 初始化CUDA驱动API
  check_cuda(cuInit(0));

  // 获取第一个可用的CUDA设备
  CUdevice device;
  check_cuda(cuDeviceGet(&device, 0));

  // 创建一个CUDA上下文
  CUcontext context;
  check_cuda(cuCtxCreate(&context, 0, device));

  // 创建一个共享的stream
  check_cuda(cuStreamCreate(&shared_stream, CU_STREAM_NON_BLOCKING));
}

// 定义一个函数，用于释放CUDA资源
void cleanup_cuda() {
  // 销毁共享的stream
  check_cuda(cuStreamDestroy(shared_stream));

  // 销毁CUDA上下文
  check_cuda(cuCtxDestroy(cuCtxGetCurrent()));
}

// 定义一个函数，用于分配和初始化主机内存
void init_host_memory(float **a, float **b, float **c, int n) {
  // 分配主机内存
  check_cuda(cuMemHostAlloc((void **)a, n * sizeof(float), CU_MEMHOSTALLOC_PORTABLE));
  check_cuda(cuMemHostAlloc((void **)b, n * sizeof(float), CU_MEMHOSTALLOC_PORTABLE));
  check_cuda(cuMemHostAlloc((void **)c, n * sizeof(float), CU_MEMHOSTALLOC_PORTABLE));

  // 初始化主机内存
  for (int i = 0; i < n; i++) {
    (*a)[i] = i;
    (*b)[i] = i + 1;
    (*c)[i] = 0;
  }
}

// 定义一个函数，用于执行单个任务
void execute_task(int thread_id) {
  // 分配和初始化主机内存
  float *h_a, *h_b, *h_c;
  init_host_memory(&h_a, &h_b, &h_c, NUM_ELEMENTS);

  // 分配和初始化设备内存
  float *d_a, *d_b, *d_c;
  init_device_memory(&d_a, &d_b, &d_c, NUM_ELEMENTS);

  // 将主机内存中的数据拷贝到设备内存中，使用共享的stream和异步调用
  check_cuda(cuMemcpyHtoDAsync((CUdeviceptr)(d_a), h_a, NUM_ELEMENTS * sizeof(float), shared_stream));
  check_cuda(cuMemcpyHtoDAsync((CUdeviceptr)(d_b), h_b, NUM_ELEMENTS * sizeof(float), shared_stream));

  // 根据线程的编号选择不同的CUDA内核函数来处理数据
  CUfunction kernel;
  const char *kernel_name;
  switch (thread_id) {
    case 0:
      kernel = add_kernel;
      kernel_name = "add_kernel";
      break;
    case 1:
      kernel = mul_kernel;
      kernel_name = "mul_kernel";
      break;
    case 2:
      kernel = sub_kernel;
      kernel_name = "sub_kernel";
      break;
    case 3:
      kernel = div_kernel;
      kernel_name = "div_kernel";
      break;
    default:
      std::cerr << "Invalid thread id: " << thread_id << std::endl;
      exit(1);
  }

  // 使用cuStreamWaitEvent函数来实现线程之间的同步，使得一个线程需要等待另一个线程完成后才能开始
  // 假设线程0需要等待线程1，线程2需要等待线程3
  if (thread_id == 0 || thread_id == 2) {
    // 创建一个事件对象
    CUevent event;
    check_cuda(cuEventCreate(&event, CU_EVENT_DEFAULT));

    // 在共享的stream上记录事件
    check_cuda(cuEventRecord(event, shared_stream));

    // 获取另一个线程的编号
    int other_thread_id = thread_id == 0 ? 1 : 3;

    // 在另一个线程的共享stream上等待事件完成
    check_cuda(cuStreamWaitEvent(shared_stream[other_thread_id], event, 0));

    // 销毁事件对象
    check_cuda(cuEventDestroy(event));
  }

  // 调用CUDA内核函数，使用共享的stream和动态并行
  int grid_size = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
  void *kernel_args[] = {&d_a, &d_b, &d_c, &NUM_ELEMENTS};
  check_cuda(cuLaunchKernel(kernel, grid_size, 1, 1, BLOCK_SIZE, 1, 1, 0, shared_stream[thread_id], kernel_args, NULL));

  // 将设备内存中的结果拷贝回主机内存中，使用共享的stream和异步调用
  check_cuda(cuMemcpyDtoHAsync(h_c, (CUdeviceptr)(d_c), NUM_ELEMENTS * sizeof(float), shared_stream[thread_id]));

  // 同步共享的stream，等待所有操作完成
  check_cuda(cuStreamSynchronize(shared_stream[thread_id]));

//   检查结果是否正确，并打印相关信息。
// 使用std::cout打印线程的编号和所选的CUDA内核函数。
// 释放主机内存和设备内存
}
