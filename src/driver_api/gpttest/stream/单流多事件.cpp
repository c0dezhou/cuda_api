#include <cuda.h>
#include <gtest/gtest.h>
#include <iostream>

// 定义一些常量
const int NUM_TASKS = 4; // 任务数
const int NUM_ELEMENTS = 1000000; // 每个任务处理的元素数
const int BLOCK_SIZE = 256; // CUDA内核的块大小

// 定义一个单stream
CUstream single_stream;

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

// 定义一个函数，用于初始化CUDA环境和单stream
void init_cuda() {
  // 初始化CUDA驱动API
  check_cuda(cuInit(0));

  // 获取第一个可用的CUDA设备
  CUdevice device;
  check_cuda(cuDeviceGet(&device, 0));

  // 创建一个CUDA上下文
  CUcontext context;
  check_cuda(cuCtxCreate(&context, 0, device));

  // 创建一个单stream
  check_cuda(cuStreamCreate(&single_stream, CU_STREAM_NON_BLOCKING));
}

// 定义一个函数，用于释放CUDA资源
void cleanup_cuda() {
  // 销毁单stream
  check_cuda(cuStreamDestroy(single_stream));

  // 销毁CUDA上下文
  check_cuda(cuCtxDestroy(cuCtxGetCurrent()));
}

// 定义一个函数，用于分配和初始化设备内存
void init_device_memory(float **d_a, float **d_b, float **d_c, int n) {
  // 分配设备内存，并使用cuMemAlloc函数。
  check_cuda(cuMemAlloc((CUdeviceptr *)d_a, n * sizeof(float)));
  check_cuda(cuMemAlloc((CUdeviceptr *)d_b, n * sizeof(float)));
  check_cuda(cuMemAlloc((CUdeviceptr *)d_c, n * sizeof(float)));

  // 初始化设备内存，这一步可以省略，因为后面会将主机内存中的数据拷贝到设备内存中。
}

// 定义一个函数，用于检查结果是否正确，并打印相关信息
void check_result(float *a, float *b, float *c, int n, const char *kernel_name) {
  // 定义一个容差值
  const float epsilon = 1e-6f;

  // 遍历数组，比较每个元素的值
  for (int i = 0; i < n; i++) {
    // 计算期望的结果
    float expected;
    if (strcmp(kernel_name, "add_kernel") == 0) {
      expected = a[i] + b[i];
    } else if (strcmp(kernel_name, "mul_kernel") == 0) {
      expected = a[i] * b[i];
    } else if (strcmp(kernel_name, "sub_kernel") == 0) {
      expected = a[i] - b[i];
    } else if (strcmp(kernel_name, "div_kernel") == 0) {
      expected = a[i] / b[i];
    } else {
      std::cerr << "Unknown kernel name: " << kernel_name << std::endl;
      exit(1);
    }

    // 比较实际的结果和期望的结果，如果相差超过容差值，则打印错误信息并退出
    if (fabs(c[i] - expected) > epsilon) {
      std::cerr << "Error: result mismatch at index " << i << std::endl;
      std::cerr << "Actual: " << c[i] << ", Expected: " << expected << std::endl;
      exit(1);
    }
  }

  // 如果没有错误，则打印成功信息
  std::cout << "Success: result matched for " << kernel_name << std::endl;
}

// 定义一个主函数，用于执行多个任务，并在每个任务之间插入一个事件，并且让主机等待每个事件完成后再继续执行。
int main() {
  // 初始化CUDA环境和单stream。
  init_cuda();

  // 分配和初始化主机内存和设备内存。
  float *h_a[NUM_TASKS], *h_b[NUM_TASKS], *h_c[NUM_TASKS]; // 主机内存指针数组
  float *d_a[NUM_TASKS], *d_b[NUM_TASKS], *d_c[NUM_TASKS]; // 设备内存指针数组
  for (int i = 0; i < NUM_TASKS; i++) {
    init_host_memory(&h_a[i], &h_b[i], &h_c[i], NUM_ELEMENTS);
    init_device_memory(&d_a[i], &d_b[i], &d_c[i], NUM_ELEMENTS);
  }

  // 使用单stream来执行多个任务，并在每个任务之间插入一个事件，并且让主机等待每个事件完成后再继续执行。
  for (int i = 0; i < NUM_TASKS; i++) {
    // 将主机内存中的数据拷贝到设备内存中，使用单stream和异步调用
    check_cuda(cuMemcpyHtoDAsync((CUdeviceptr)(d_a[i]), h_a[i], NUM_ELEMENTS * sizeof(float), single_stream));
    check_cuda(cuMemcpyHtoDAsync((CUdeviceptr)(d_b[i]), h_b[i], NUM_ELEMENTS * sizeof(float), single_stream));

    // 根据任务的编号选择不同的CUDA内核函数来处理数据
    CUfunction kernel;
    const char *kernel_name;
    switch (i) {
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
        std::cerr << "Invalid task id: " << i << std::endl;
        exit(1);
    }

    // 调用CUDA内核函数，使用单stream和动态并行
    int grid_size = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void *kernel_args[] = {&d_a[i], &d_b[i], &d_c[i], &NUM_ELEMENTS};
    check_cuda(cuLaunchKernel(kernel, grid_size, 1, 1, BLOCK_SIZE, 1, 1, 0, single_stream, kernel_args, NULL));

    // 将设备内存中的结果拷贝回主机内存中，使用单stream和异步调用
    check_cuda(cuMemcpyDtoHAsync(h_c[i], (CUdeviceptr)(d_c[i]), NUM_ELEMENTS * sizeof(float), single_stream));

    // 在单stream上记录一个事件
    CUevent event;
    check_cuda(cuEventCreate(&event, CU_EVENT_DEFAULT));
    check_cuda(cuEventRecord(event, single_stream));

    // 让主机等待事件完成后再继续执行
    check_cuda(cuEventSynchronize(event));

    // 销毁事件对象
    check_cuda(cuEventDestroy(event));
  }

  // 检查结果是否正确，并打印相关信息。
}
