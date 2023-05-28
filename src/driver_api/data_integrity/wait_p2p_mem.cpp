
// 使用cuMemAlloc在两个不同的设备上分别分配一个设备内存地址，并使用cuMemcpyPeer或者cuMemcpyPeerAsync在两个设备之间进行内存拷贝。期望结果是CUDA驱动程序API能够成功地完成内存拷贝。
// •	使用cuMemAlloc在两个不同的设备上分别分配一个设备内存地址，并使用cuMemcpyPeer或者cuMemcpyPeerAsync在两个设备之间进行内存拷贝，但是其中一个设备的内存地址没有被分配或者超出了范围。期望结果是CUDA驱动程序API返回一个错误码，表示非法内存访问。
// •	使用cuMemAlloc在两个不同的设备上分别分配一个设备内存地址，并使用cuMemcpyPeer或者cuMemcpyPeerAsync在两个设备之间进行内存拷贝，但是两个设备之间没有启用对等访问。期望结果是CUDA驱动程序API返回一个错误码，表示非法内存访问。
// •	使用cuMemAllocManaged分配一个托管内存地址，并将其传递给两个不同的设备上运行的CUDA核函数。期望结果是CUDA驱动程序API能够自动地在两个设备之间迁移托管内存，并正确地执行CUDA核函数。

#include "test_utils.h"

// Define a CUDA kernel that takes a managed memory address as argument
__global__ void Kernel(CUdeviceptr d_ptr) {
  // Get the thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Cast the device pointer to a float pointer
  float *d_data = (float *)d_ptr;
  // Do some simple operation on the managed memory
  d_data[idx] += 1.0f;
}

// Define a test case class that inherits from testing::Test
class CudaMemcpyPeerTest : public testing::Test {
protected:
  // Define some member variables
  CUdeviceptr d_ptr1, d_ptr2; // Device memory addresses
  CUdeviceptr m_ptr; // Managed memory address
  size_t size; // Number of bytes to allocate and copy
  CUdevice dev1, dev2; // Two different CUDA devices
  CUcontext ctx1, ctx2; // Two different CUDA contexts

  // Define a SetUp function that runs before each test case
  void SetUp() override {
    // Initialize CUDA driver API
    cuInit(0);
    // Get two different CUDA devices
    int deviceCount;
    CUresult res = cuDeviceGetCount(&deviceCount); // Get the number of devices
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
    // Allocate device memory on both devices
    size = 1024 * sizeof(float); // Arbitrary size
    res = cuMemAlloc(&d_ptr1, size); // Allocate on the first device
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuMemAlloc(&d_ptr2, size); // Allocate on the second device
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Allocate managed memory with portable flag
    res = cuMemAllocManaged(&m_ptr, size, CU_MEMALLOC_PORTABLE);
    ASSERT_EQ(res, CUDA_SUCCESS);
  }

  // Define a TearDown function that runs after each test case
  void TearDown() override {
    // Free device memory on both devices
    CUresult res = cuMemFree(d_ptr1); // Free on the first device
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuMemFree(d_ptr2); // Free on the second device
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Free managed memory
    res = cuMemFree(m_ptr);
    ASSERT_EQ(res, CUDA_SUCCESS);
    // Destroy two different CUDA contexts
    res = cuCtxDestroy(ctx1); // Destroy the first context
    ASSERT_EQ(res, CUDA_SUCCESS);
    res = cuCtxDestroy(ctx2); // Destroy the second context
    ASSERT_EQ(res, CUDA_SUCCESS);
  }
};

// Define a test case function that tests successful memory copy between two devices
TEST_F(CudaMemcpyPeerTest, Success) {
  // Set the current context to ctx1
  CUresult res = cuCtxSetCurrent(ctx1);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Enable peer access to ctx2
  res = cuCtxEnablePeerAccess(ctx2, 0);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Copy from device memory on dev1 to device memory on dev2 with cuMemcpyPeer
  res = cuMemcpyPeer(d_ptr2, ctx2, d_ptr1, ctx1, size);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Copy from device memory on dev2 to device memory on dev1 with cuMemcpyPeerAsync
  CUstream stream;
  res = cuStreamCreate(&stream, CU_STREAM_DEFAULT); // Create a stream
  ASSERT_EQ(res, CUDA_SUCCESS);
  res = cuMemcpyPeerAsync(d_ptr1, ctx1, d_ptr2, ctx2, size, stream); // Copy asynchronously
  ASSERT_EQ(res, CUDA_SUCCESS);
  res = cuStreamSynchronize(stream); // Wait for the copy to finish
  ASSERT_EQ(res, CUDA_SUCCESS);
  res = cuStreamDestroy(stream); // Destroy the stream
  ASSERT_EQ(res, CUDA_SUCCESS);
}

// Define a test case function that tests illegal memory access error when copying between two devices with invalid address or size
TEST_F(CudaMemcpyPeerTest, InvalidAddressOrSize) {
  // Set the current context to ctx1
  CUresult res = cuCtxSetCurrent(ctx1);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Enable peer access to ctx2
  res = cuCtxEnablePeerAccess(ctx2, 0);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Try to copy from device memory on dev1 to device memory on dev2 with cuMemcpyPeer with an invalid source address
  res = cuMemcpyPeer(d_ptr2, ctx2, d_ptr1 + size, ctx1, size);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
  // Try to copy from device memory on dev1 to device memory on dev2 with cuMemcpyPeer with an invalid destination address
  res = cuMemcpyPeer(d_ptr2 + size, ctx2, d_ptr1, ctx1, size);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
  // Try to copy from device memory on dev1 to device memory on dev2 with cuMemcpyPeer with an invalid size
  res = cuMemcpyPeer(d_ptr2, ctx2, d_ptr1, ctx1, size + 1);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
}

// Define a test case function that tests illegal memory access error when copying between two devices without peer access enabled
TEST_F(CudaMemcpyPeerTest, NoPeerAccess) {
  // Set the current context to ctx1
  CUresult res = cuCtxSetCurrent(ctx1);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Try to copy from device memory on dev1 to device memory on dev2 with cuMemcpyPeer without enabling peer access
  res = cuMemcpyPeer(d_ptr2, ctx2, d_ptr1, ctx1, size);
  // Check the result is illegal memory access error
  ASSERT_EQ(res, CUDA_ERROR_ILLEGAL_ADDRESS);
}

// Define a test case function that tests successful kernel execution on two devices with managed memory
TEST_F(CudaMemcpyPeerTest, ManagedMemory) {
  // Set the current context to ctx1
  CUresult res = cuCtxSetCurrent(ctx1);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Launch the kernel on dev1 with managed memory as argument
  void *args[] = {&m_ptr}; // Kernel arguments
  int blockSize = 256; // Arbitrary block size
  int gridSize = (size / sizeof(float) + blockSize - 1) / blockSize; // Grid size
  res = cuLaunchKernel(Kernel, gridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, NULL);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Set the current context to ctx2
  res = cuCtxSetCurrent(ctx2);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
  // Launch the kernel on dev2 with managed memory as argument
  res = cuLaunchKernel(Kernel, gridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, NULL);
  // Check the result is success
  ASSERT_EQ(res, CUDA_SUCCESS);
}

// Define a main function that runs all tests
int main(int argc, char **argv) {
  // Initialize gtest
  testing::InitGoogleTest(&argc, argv);
  // Run all tests and return the result
  return RUN_ALL_TESTS();
}
