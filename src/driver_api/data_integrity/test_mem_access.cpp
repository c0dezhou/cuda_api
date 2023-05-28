// 如果一个函数试图访问一个已经被释放或者没有被分配的内存地址，或者如果一个函数试图访问一个不属于当前上下文或者不在统一地址空间的内存地址，那么可能会发生非法内存访问错误2。为了避免这种错误，我们需要设计一些测试用例来检查CUDA驱动程序API是否能够正确地检测和处理这些非法内存访问的情况。以下是一些可能的测试用例：

// 写测例时加上memcpy

// 使用cuMemAlloc分配一个设备内存地址，并使用cuMemFree释放它。然后使用cuMemsetD8或者其他cuMemset类的函数试图设置该地址上的一段字节为一个特定的值。期望结果是CUDA驱动程序API返回一个错误码，表示非法内存访问。
#include "test_utils.h"

class CudaMemsetTest1 : public testing::Test {
protected:
  CUdeviceptr d_ptr;
  unsigned char value;
  size_t size;

  void SetUp() override {
    cuInit(0);
    size = 1024;
    CUresult res = cuMemAlloc(&d_ptr, size);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuMemFree(d_ptr);
    ASSERT_EQ(res, CUDA_SUCCESS);
    value = 42;
  }

  void TearDown() override {
  }
};


TEST_F(CudaMemsetTest1, IllegalMemoryAccess1) {
  CUresult res = cuMemsetD8(d_ptr, value, size);
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
  
  res = cuMemsetD16(d_ptr, value, size / 2);

  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
  res = cuMemsetD32(d_ptr, value, size / 4);

  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
}




// 使用cuMemAlloc分配一个设备内存地址，并将其传递给一个在另一个上下文中运行的cuMemset类的函数。期望结果是CUDA驱动程序API返回一个错误码，表示非法内存访问。


// Define a test case class that inherits from testing::Test
class CudaMemsetTest2 : public testing::Test {
protected:

  CUdeviceptr d_ptr;
  unsigned char value;
  size_t size;
  CUcontext ctx1, ctx2;
  CUstream stream1, stream2;


  void SetUp() override {
    cuInit(0);
    size = 1024;
    value = 42;
    CUdevice dev;
    CUresult res = cuDeviceGet(&dev, 0);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxCreate(&ctx1, 0, dev);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuMemAlloc(&d_ptr, size);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxCreate(&ctx2, 0, dev);
    ASSERT_EQ(res, CUDA_SUCCESS);
    
    res = cuStreamCreate(&stream1, CU_STREAM_DEFAULT);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuStreamCreate(&stream2, CU_STREAM_DEFAULT);
    ASSERT_EQ(res, CUDA_SUCCESS);
  }

  void TearDown() override {
    CUresult res = cuMemFree(d_ptr);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuStreamDestroy(stream1);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuStreamDestroy(stream2);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxDestroy(ctx1);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxDestroy(ctx2);
    ASSERT_EQ(res, CUDA_SUCCESS);
  }
};


TEST_F(CudaMemsetTest2, IllegalMemoryAccess) {
  
  CUresult res = cuCtxSetCurrent(ctx1);
  ASSERT_EQ(res, CUDA_SUCCESS);

  res = cuMemsetD8Async(d_ptr, value, size, stream1);
  ASSERT_EQ(res, CUDA_SUCCESS);
  
  res = cuCtxSetCurrent(ctx2);
  ASSERT_EQ(res, CUDA_SUCCESS);
  
  res = cuMemsetD8Async(d_ptr, value, size, stream2);
  
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
}


// 使用cuMemAlloc分配一个设备内存地址，并使用cuMemsetD8或者其他cuMemset类的函数试图设置该地址上的一段字节为一个特定的值，但是要设置的字节数不是8位，16位或者32位的倍数。期望结果是CUDA驱动程序API返回一个错误码，表示非法内存访问。

TEST_F(CudaMemsetTest1, IllegalMemoryAccess111) {
  CUresult res = cuMemsetD8(d_ptr, value, size);
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
  res = cuMemsetD16(d_ptr, value, size / 2);
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
  res = cuMemsetD32(d_ptr, value, size / 4);
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
}



// 实现使用cuMemAlloc分配一个设备内存地址，并将其传递给一个在另一个上下文中运行的CUDA核函数。期望结果是CUDA驱动程序API返回一个错误码，表示非法内存访问。


// __global__ void Kernel(CUdeviceptr d_ptr) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   float *d_data = (float *)d_ptr;
//   d_data[idx] += 1.0f;
// }

class CudaKernelTest : public testing::Test {
protected:
  CUdeviceptr d_ptr;
  size_t size;
  CUcontext ctx1, ctx2;
  CUstream stream1, stream2;

  void SetUp() override {
    cuInit(0);
    size = 1024 * sizeof(float);
    CUresult res = cuMemAlloc(&d_ptr, size);
    ASSERT_EQ(res, CUDA_SUCCESS);
    CUdevice dev;
    res = cuDeviceGet(&dev, 0);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxCreate(&ctx1, 0, dev);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxCreate(&ctx2, 0, dev);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuStreamCreate(&stream1, CU_STREAM_DEFAULT);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuStreamCreate(&stream2, CU_STREAM_DEFAULT);
    ASSERT_EQ(res, CUDA_SUCCESS);
  }


  void TearDown() override {
    CUresult res = cuMemFree(d_ptr);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuStreamDestroy(stream1);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuStreamDestroy(stream2);
    ASSERT_EQ(res, CUDA_SUCCESS);
    
    res = cuCtxDestroy(ctx1);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxDestroy(ctx2);
    ASSERT_EQ(res, CUDA_SUCCESS);
  }
};


TEST_F(CudaKernelTest, IllegalMemoryAccess) {
  CUresult res = cuCtxSetCurrent(ctx1);
  CUmodule module_;
  CUfunction kernel;
  ASSERT_EQ(res, CUDA_SUCCESS);
  int N = 1024;
  void *args[] = {&d_ptr, &N};
  int blockSize = 256;
  int gridSize = (size / sizeof(float) + blockSize - 1) / blockSize; // Grid size
  checkError(cuModuleLoad(
      &module_,
      "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx"));
  checkError(cuModuleGetFunction(&kernel, module_, "_Z9vec_sub_1Pfi"));

  res = cuLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, 0, stream1,
                       args, NULL);

  ASSERT_EQ(res, CUDA_SUCCESS);

  res = cuCtxSetCurrent(ctx2);

  ASSERT_EQ(res, CUDA_SUCCESS);

  res = cuLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, 0, stream2, args, NULL);

  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
}



// 使用cuMemHostAlloc分配一个主机内存地址，并使用cuMemHostGetDevicePointer获取其对应的设备内存地址。然后在另一个设备上使用该设备内存地址，或者在没有启用统一地址空间的情况下使用该设备内存地址。期望结果是CUDA驱动程序API返回一个错误码，表示非法内存访问。

class CudaMemHostTest : public testing::Test {
protected:
  void *h_ptr;
  CUdeviceptr d_ptr1, d_ptr2;
  size_t size;
  CUdevice dev1, dev2;
  CUcontext ctx1, ctx2;

  void SetUp() override {
    cuInit(0);
    size = 1024 * sizeof(float);
    CUresult res = cuMemHostAlloc(&h_ptr, size, CU_MEMHOSTALLOC_PORTABLE);
    ASSERT_EQ(res, CUDA_SUCCESS);
    int deviceCount;
    res = cuDeviceGetCount(&deviceCount);
    ASSERT_EQ(res, CUDA_SUCCESS);
    ASSERT_GE(deviceCount, 2);
    res = cuDeviceGet(&dev1, 0);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuDeviceGet(&dev2, 1);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxCreate(&ctx1, 0, dev1);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxCreate(&ctx2, 0, dev2);
    ASSERT_EQ(res, CUDA_SUCCESS);
  }

  void TearDown() override {
    CUresult res = cuMemFreeHost(h_ptr);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxDestroy(ctx1);
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxDestroy(ctx2);
    ASSERT_EQ(res, CUDA_SUCCESS);
  }
};


TEST_F(CudaMemHostTest, IllegalMemoryAccess) {
  CUresult res = cuCtxSetCurrent(ctx1);
  ASSERT_EQ(res, CUDA_SUCCESS);
  res = cuMemHostGetDevicePointer(&d_ptr1, h_ptr, 0);
  ASSERT_EQ(res, CUDA_SUCCESS);
  res = cuCtxSetCurrent(ctx2);
  ASSERT_EQ(res, CUDA_SUCCESS);
  res = cuMemHostGetDevicePointer(&d_ptr2, h_ptr, 0);
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
  res = cuMemsetD8(d_ptr1, 0, size);
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
}



// 实现使用cuMemAlloc分配一个设备内存地址，并使用cuMemFree释放它。然后再次使用cuMemFree释放同一个地址，或者使用其他函数（如cuMemcpy，cumemcpyhtod, dtoh）访问该地址。期望结果是CUDA驱动程序API返回一个错误码，表示非法内存访问。

class CudaMemFreeTest : public testing::Test {
protected:
  CUdeviceptr d_ptr;
  void *h_ptr;
  size_t size;

  void SetUp() override {
    cuInit(0);
    size = 1024 * sizeof(float);
    CUresult res = cuMemAlloc(&d_ptr, size);
    ASSERT_EQ(res, CUDA_SUCCESS);
    h_ptr = malloc(size);
    ASSERT_NE(h_ptr, nullptr);
  }

  void TearDown() override {
    free(h_ptr);
  }
};

TEST_F(CudaMemFreeTest, IllegalMemoryAccess) {
  CUresult res = cuMemFree(d_ptr);
  ASSERT_EQ(res, CUDA_SUCCESS);

  res = cuMemFree(d_ptr);
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);

  res = cuMemcpyDtoH(h_ptr, d_ptr, size);
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);

  res = cuMemcpyHtoD(d_ptr, h_ptr, size);
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
}


