#include <gtest/gtest.h>
#include <cuda.h>

// A helper function to check cuda driver api status
void CheckCudaStatus(CUresult status) {
  ASSERT_EQ(status, CUDA_SUCCESS) << "CUDA driver api error: " << status;
}

// A test fixture class to manage cuda memory and host memory
class CudaMemoryTest : public ::testing::Test {
 protected:
  // The size of the memory to allocate
  static constexpr size_t kMemorySize = 1024;
  // The number of times to repeat the test
  static constexpr int kRepeatTimes = 10;

  // Set up the test fixture
  void SetUp() override {
    // Allocate cuda memory and host memory
    CheckCudaStatus(cuMemAlloc(&cuda_memory_, kMemorySize));
    CheckCudaStatus(cuMemHostAlloc(&host_memory_, kMemorySize, 0));
  }

  // Tear down the test fixture
  void TearDown() override {
    // Free cuda memory and host memory
    CheckCudaStatus(cuMemFree(cuda_memory_));
    CheckCudaStatus(cuMemFreeHost(host_memory_));
  }

  // A helper function to compare cuda memory and host memory data
  void CompareMemoryData() {
    // Copy cuda memory data to host memory
    CheckCudaStatus(cuMemcpyDtoH(host_memory_data_, cuda_memory_, kMemorySize));
    // Compare the data byte by byte
    for (size_t i = 0; i < kMemorySize; ++i) {
      EXPECT_EQ(host_memory_data_[i], cuda_memory_data_[i]) << "Data mismatch at index " << i;
    }
  }

  // A helper function to perform the test logic
  void TestMemoryConsistency() {
    // Initialize cuda memory with some random data
    for (size_t i = 0; i < kMemorySize; ++i) {
      cuda_memory_data_[i] = rand() % 256;
    }
    CheckCudaStatus(cuMemcpyHtoD(cuda_memory_, cuda_memory_data_, kMemorySize));
    // Copy cuda memory to host memory and compare the data
    CheckCudaStatus(cuMemcpyDtoH(host_memory_, cuda_memory_, kMemorySize));
    CompareMemoryData();
    // Copy host memory back to cuda memory and compare the data
    CheckCudaStatus(cuMemcpyHtoD(cuda_memory_, host_memory_, kMemorySize));
    CompareMemoryData();
  }

  // The cuda memory handle
  CUdeviceptr cuda_memory_;
  // The host memory pointer
  void* host_memory_;
  // The cuda memory data buffer
  unsigned char cuda_memory_data_[kMemorySize];
  // The host memory data buffer
  unsigned char host_memory_data_[kMemorySize];
};

// A test case to verify cuda memory and host memory consistency
TEST_F(CudaMemoryTest, MemoryConsistency) {
  // Repeat the test for a number of times
  for (int i = 0; i < kRepeatTimes; ++i) {
    TestMemoryConsistency();
  }
}
