// 如果一个函数试图访问一个已经被释放或者没有被分配的内存地址，或者如果一个函数试图访问一个不属于当前上下文或者不在统一地址空间的内存地址，那么可能会发生非法内存访问错误2。为了避免这种错误，我们需要设计一些测试用例来检查CUDA驱动程序API是否能够正确地检测和处理这些非法内存访问的情况。以下是一些可能的测试用例：


// 写测例时加上memcpy

// 使用cuMemAlloc分配一个设备内存地址，并使用cuMemFree释放它。然后使用cuMemsetD8或者其他cuMemset类的函数试图设置该地址上的一段字节为一个特定的值。期望结果是CUDA驱动程序API返回一个错误码，表示非法内存访问。
// Include gtest header file
#include <gtest/gtest.h>
// Include CUDA driver API header file
#include <cuda.h>

// Define a test case class that inherits from testing::Test
class CudaMemsetTest : public testing::Test {
protected:
  // Define some member variables
  CUdeviceptr d_ptr; // Device memory address
  unsigned char value; // Value to set
  size_t size; // Number of bytes to set

  // Define a SetUp function that runs before each test case
  void SetUp() override {
    // Initialize CUDA driver API
    cuInit(0);
    // Allocate device memory
    size = 1024; // Arbitrary size
    CUresult res = cuMemAlloc(&d_ptr, size);
    // Check allocation result
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Free device memory
    res = cuMemFree(d_ptr);
    // Check free result
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Set value to set
    value = 42; // Arbitrary value
  }

  // Define a TearDown function that runs after each test case
  void TearDown() override {
    // Nothing to do here
  }
};

// Define a test case function that uses the test case class
TEST_F(CudaMemsetTest, IllegalMemoryAccess) {
  // Try to set device memory with cuMemsetD8
  CUresult res = cuMemsetD8(d_ptr, value, size);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
  // Try to set device memory with cuMemsetD16
  res = cuMemsetD16(d_ptr, value, size / 2);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
  // Try to set device memory with cuMemsetD32
  res = cuMemsetD32(d_ptr, value, size / 4);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
}

// Define a main function that runs all tests
int main(int argc, char **argv) {
  // Initialize gtest
  testing::InitGoogleTest(&argc, argv);
  // Run all tests and return the result
  return RUN_ALL_TESTS();
}



// 使用cuMemAlloc分配一个设备内存地址，并将其传递给一个在另一个上下文中运行的cuMemset类的函数。期望结果是CUDA驱动程序API返回一个错误码，表示非法内存访问。
// Include gtest header file
#include <gtest/gtest.h>
// Include CUDA driver API header file
#include <cuda.h>

// Define a test case class that inherits from testing::Test
class CudaMemsetTest : public testing::Test {
protected:
  // Define some member variables
  CUdeviceptr d_ptr; // Device memory address
  unsigned char value; // Value to set
  size_t size; // Number of bytes to set
  CUcontext ctx1, ctx2; // Two different CUDA contexts
  CUstream stream1, stream2; // Two different CUDA streams

  // Define a SetUp function that runs before each test case
  void SetUp() override {
    // Initialize CUDA driver API
    cuInit(0);
    // Allocate device memory
    size = 1024; // Arbitrary size
    CUresult res = cuMemAlloc(&d_ptr, size);
    // Check allocation result
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Set value to set
    value = 42; // Arbitrary value
    // Create two different CUDA contexts
    CUdevice dev;
    res = cuDeviceGet(&dev, 0); // Get the first device
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxCreate(&ctx1, 0, dev); // Create the first context
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxCreate(&ctx2, 0, dev); // Create the second context
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Create two different CUDA streams
    res = cuStreamCreate(&stream1, CU_STREAM_DEFAULT); // Create the first stream
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuStreamCreate(&stream2, CU_STREAM_DEFAULT); // Create the second stream
    ASSERT_EQ(res, CUDA_SUCCESS);
  }

  // Define a TearDown function that runs after each test case
  void TearDown() override {
    // Free device memory
    CUresult res = cuMemFree(d_ptr);
    // Check free result
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Destroy two different CUDA streams
    res = cuStreamDestroy(stream1); // Destroy the first stream
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuStreamDestroy(stream2); // Destroy the second stream
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Destroy two different CUDA contexts
    res = cuCtxDestroy(ctx1); // Destroy the first context
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxDestroy(ctx2); // Destroy the second context
    ASSERT_EQ(res, CUDA_SUCCESS);
  }
};

// Define a test case function that uses the test case class
TEST_F(CudaMemsetTest, IllegalMemoryAccess) {
  // Set the current context to ctx1
  CUresult res = cuCtxSetCurrent(ctx1);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Try to set device memory with cuMemsetD8Async in stream1
  res = cuMemsetD8Async(d_ptr, value, size, stream1);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Set the current context to ctx2
  res = cuCtxSetCurrent(ctx2);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Try to set device memory with cuMemsetD8Async in stream2
  res = cuMemsetD8Async(d_ptr, value, size, stream2);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
}

// Define a main function that runs all tests
int main(int argc, char **argv) {
  // Initialize gtest
  testing::InitGoogleTest(&argc, argv);
  // Run all tests and return the result
  return RUN_ALL_TESTS();
}


// 使用cuMemAlloc分配一个设备内存地址，并使用cuMemsetD8或者其他cuMemset类的函数试图设置该地址上的一段字节为一个特定的值，但是要设置的字节数不是8位，16位或者32位的倍数。期望结果是CUDA驱动程序API返回一个错误码，表示非法内存访问。
// Include gtest header file
#include <gtest/gtest.h>
// Include CUDA driver API header file
#include <cuda.h>

// Define a test case class that inherits from testing::Test
class CudaMemsetTest : public testing::Test {
protected:
  // Define some member variables
  CUdeviceptr d_ptr; // Device memory address
  unsigned char value; // Value to set
  size_t size; // Number of bytes to set

  // Define a SetUp function that runs before each test case
  void SetUp() override {
    // Initialize CUDA driver API
    cuInit(0);
    // Allocate device memory
    size = 1023; // Arbitrary size that is not a multiple of 8, 16 or 32 bits
    CUresult res = cuMemAlloc(&d_ptr, size);
    // Check allocation result
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Set value to set
    value = 42; // Arbitrary value
  }

  // Define a TearDown function that runs after each test case
  void TearDown() override {
    // Free device memory
    CUresult res = cuMemFree(d_ptr);
    // Check free result
    ASSERT_EQ(res, CUDA_SUCCESS);
  }
};

// Define a test case function that uses the test case class
TEST_F(CudaMemsetTest, IllegalMemoryAccess) {
  // Try to set device memory with cuMemsetD8
  CUresult res = cuMemsetD8(d_ptr, value, size);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
  // Try to set device memory with cuMemsetD16
  res = cuMemsetD16(d_ptr, value, size / 2);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
  // Try to set device memory with cuMemsetD32
  res = cuMemsetD32(d_ptr, value, size / 4);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
}

// Define a main function that runs all tests
int main(int argc, char **argv) {
  // Initialize gtest
  testing::InitGoogleTest(&argc, argv);
  // Run all tests and return the result
  return RUN_ALL_TESTS();
}


// 实现使用cuMemAlloc分配一个设备内存地址，并将其传递给一个在另一个上下文中运行的CUDA核函数。期望结果是CUDA驱动程序API返回一个错误码，表示非法内存访问。
// Include gtest header file
#include <gtest/gtest.h>
// Include CUDA driver API header file
#include <cuda.h>

// Define a CUDA kernel that takes a device memory address as argument
__global__ void Kernel(CUdeviceptr d_ptr) {
  // Get the thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Cast the device pointer to a float pointer
  float *d_data = (float *)d_ptr;
  // Do some simple operation on the device memory
  d_data[idx] += 1.0f;
}

// Define a test case class that inherits from testing::Test
class CudaKernelTest : public testing::Test {
protected:
  // Define some member variables
  CUdeviceptr d_ptr; // Device memory address
  size_t size; // Number of bytes to allocate
  CUcontext ctx1, ctx2; // Two different CUDA contexts
  CUstream stream1, stream2; // Two different CUDA streams

  // Define a SetUp function that runs before each test case
  void SetUp() override {
    // Initialize CUDA driver API
    cuInit(0);
    // Allocate device memory
    size = 1024 * sizeof(float); // Arbitrary size
    CUresult res = cuMemAlloc(&d_ptr, size);
    // Check allocation result
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Create two different CUDA contexts
    CUdevice dev;
    res = cuDeviceGet(&dev, 0); // Get the first device
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxCreate(&ctx1, 0, dev); // Create the first context
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxCreate(&ctx2, 0, dev); // Create the second context
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Create two different CUDA streams
    res = cuStreamCreate(&stream1, CU_STREAM_DEFAULT); // Create the first stream
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuStreamCreate(&stream2, CU_STREAM_DEFAULT); // Create the second stream
    ASSERT_EQ(res, CUDA_SUCCESS);
  }

  // Define a TearDown function that runs after each test case
  void TearDown() override {
    // Free device memory
    CUresult res = cuMemFree(d_ptr);
    // Check free result
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Destroy two different CUDA streams
    res = cuStreamDestroy(stream1); // Destroy the first stream
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuStreamDestroy(stream2); // Destroy the second stream
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Destroy two different CUDA contexts
    res = cuCtxDestroy(ctx1); // Destroy the first context
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxDestroy(ctx2); // Destroy the second context
    ASSERT_EQ(res, CUDA_SUCCESS);
  }
};

// Define a test case function that uses the test case class
TEST_F(CudaKernelTest, IllegalMemoryAccess) {
  // Set the current context to ctx1
  CUresult res = cuCtxSetCurrent(ctx1);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Launch the kernel in stream1
  void *args[] = {&d_ptr}; // Kernel arguments
  int blockSize = 256; // Arbitrary block size
  int gridSize = (size / sizeof(float) + blockSize - 1) / blockSize; // Grid size
  res = cuLaunchKernel(Kernel, gridSize, 1, 1, blockSize, 1, 1, 0, stream1, args, NULL);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Set the current context to ctx2
  res = cuCtxSetCurrent(ctx2);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Launch the kernel in stream2
  res = cuLaunchKernel(Kernel, gridSize, 1, 1, blockSize, 1, 1, 0, stream2, args, NULL);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
}

// Define a main function that runs all tests
int main(int argc, char **argv) {
  // Initialize gtest
  testing::InitGoogleTest(&argc, argv);
  // Run all tests and return the result
  return RUN_ALL_TESTS();
}


// 使用cuMemHostAlloc分配一个主机内存地址，并使用cuMemHostGetDevicePointer获取其对应的设备内存地址。然后在另一个设备上使用该设备内存地址，或者在没有启用统一地址空间的情况下使用该设备内存地址。期望结果是CUDA驱动程序API返回一个错误码，表示非法内存访问。
// Include gtest header file
#include <gtest/gtest.h>
// Include CUDA driver API header file
#include <cuda.h>

// Define a test case class that inherits from testing::Test
class CudaMemHostTest : public testing::Test {
protected:
  // Define some member variables
  void *h_ptr; // Host memory address
  CUdeviceptr d_ptr1, d_ptr2; // Device memory addresses
  size_t size; // Number of bytes to allocate
  CUdevice dev1, dev2; // Two different CUDA devices
  CUcontext ctx1, ctx2; // Two different CUDA contexts

  // Define a SetUp function that runs before each test case
  void SetUp() override {
    // Initialize CUDA driver API
    cuInit(0);
    // Allocate host memory with portable flag
    size = 1024 * sizeof(float); // Arbitrary size
    CUresult res = cuMemHostAlloc(&h_ptr, size, CU_MEMHOSTALLOC_PORTABLE);
    // Check allocation result
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Get two different CUDA devices
    int deviceCount;
    res = cuDeviceGetCount(&deviceCount); // Get the number of devices
    ASSERT_EQ(res, CUDA_SUCCESS);
    ASSERT_GE(deviceCount, 2); // Check there are at least two devices
    res = cuDeviceGet(&dev1, 0); // Get the first device
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuDeviceGet(&dev2, 1); // Get the second device
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Create two different CUDA contexts
    res = cuCtxCreate(&ctx1, 0, dev1); // Create the first context
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxCreate(&ctx2, 0, dev2); // Create the second context
    ASSERT_EQ(res, CUDA_SUCCESS);
  }

  // Define a TearDown function that runs after each test case
  void TearDown() override {
    // Free host memory
    CUresult res = cuMemFreeHost(h_ptr);
    // Check free result
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Destroy two different CUDA contexts
    res = cuCtxDestroy(ctx1); // Destroy the first context
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxDestroy(ctx2); // Destroy the second context
    ASSERT_EQ(res, CUDA_SUCCESS);
  }
};

// Define a test case function that uses the test case class
TEST_F(CudaMemHostTest, IllegalMemoryAccess) {
  // Set the current context to ctx1
  CUresult res = cuCtxSetCurrent(ctx1);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Get the device pointer corresponding to the host pointer in ctx1
  res = cuMemHostGetDevicePointer(&d_ptr1, h_ptr, 0);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Set the current context to ctx2
  res = cuCtxSetCurrent(ctx2);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Try to get the device pointer corresponding to the host pointer in ctx2 without unified address space enabled
  res = cuMemHostGetDevicePointer(&d_ptr2, h_ptr, 0);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
  // Try to use the device pointer from ctx1 in ctx2
  res = cuMemsetD8(d_ptr1, 0, size);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
}

// Define a main function that runs all tests
int main(int argc, char **argv) {
  // Initialize gtest
  testing::InitGoogleTest(&argc, argv);
  // Run all tests and return the result
  return RUN_ALL_TESTS();
}


// 实现使用cuMemAlloc分配一个设备内存地址，并使用cuMemFree释放它。然后再次使用cuMemFree释放同一个地址，或者使用其他函数（如cuMemcpy，cumemcpyhtod, dtoh）访问该地址。期望结果是CUDA驱动程序API返回一个错误码，表示非法内存访问。
// Include gtest header file
#include <gtest/gtest.h>
// Include CUDA driver API header file
#include <cuda.h>

// Define a test case class that inherits from testing::Test
class CudaMemFreeTest : public testing::Test {
protected:
  // Define some member variables
  CUdeviceptr d_ptr; // Device memory address
  void *h_ptr; // Host memory address
  size_t size; // Number of bytes to allocate and copy

  // Define a SetUp function that runs before each test case
  void SetUp() override {
    // Initialize CUDA driver API
    cuInit(0);
    // Allocate device memory
    size = 1024 * sizeof(float); // Arbitrary size
    CUresult res = cuMemAlloc(&d_ptr, size);
    // Check allocation result
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Allocate host memory
    h_ptr = malloc(size);
    // Check allocation result
    ASSERT_NE(h_ptr, nullptr);
  }

  // Define a TearDown function that runs after each test case
  void TearDown() override {
    // Free host memory
    free(h_ptr);
  }
};

// Define a test case function that uses the test case class
TEST_F(CudaMemFreeTest, IllegalMemoryAccess) {
  // Free device memory
  CUresult res = cuMemFree(d_ptr);
  // Check free result
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Try to free device memory again
  res = cuMemFree(d_ptr);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
  // Try to copy from device memory to host memory
  res = cuMemcpyDtoH(h_ptr, d_ptr, size);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
  // Try to copy from host memory to device memory
  res = cuMemcpyHtoD(d_ptr, h_ptr, size);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
}

// Define a main function that runs all tests
int main(int argc, char **argv) {
  // Initialize gtest
  testing::InitGoogleTest(&argc, argv);
  // Run all tests and return the result
  return RUN_ALL_TESTS();
}

