#include "test_utils.h"

class CudaMemoryTest : public ::testing::Test {
 protected:
  static constexpr size_t kMemorySize = 1024;
  static constexpr int kRepeatTimes = 10;

  void SetUp() override {
      checkError(cuInit(0));
      checkError(cuDeviceGetCount(&device_count));
      ASSERT_GT(device_count, 0) << "No cuda device available";

      checkError(cuDeviceGet(&device, 0));

      checkError(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
      checkError(cuMemAlloc(&cuda_memory_, kMemorySize));
      checkError(cuMemHostAlloc(&host_memory_, kMemorySize, 0));
  }

  void TearDown() override {
    checkError(cuMemFree(cuda_memory_));
    checkError(cuMemFreeHost(host_memory_));
    checkError(cuCtxDestroy(context));
  }

  void CompareMemoryData() {
    checkError(cuMemcpyDtoH(host_memory_data_, cuda_memory_, kMemorySize));
    for (size_t i = 0; i < kMemorySize; ++i) {
      EXPECT_EQ(host_memory_data_[i], cuda_memory_data_[i]) << "Data mismatch at index " << i;
    }
  }

  void TestMemoryConsistency() {
    for (size_t i = 0; i < kMemorySize; ++i) {
      cuda_memory_data_[i] = rand() % 256;
    }
    checkError(cuMemcpyHtoD(cuda_memory_, cuda_memory_data_, kMemorySize));
    checkError(cuMemcpyDtoH(host_memory_, cuda_memory_, kMemorySize));
    CompareMemoryData();
    checkError(cuMemcpyHtoD(cuda_memory_, host_memory_, kMemorySize));
    CompareMemoryData();
  }

  CUcontext context;
  CUdevice device;
  int device_count;
  CUdeviceptr cuda_memory_;
  void* host_memory_;
  unsigned char cuda_memory_data_[kMemorySize];
  unsigned char host_memory_data_[kMemorySize];
};


TEST_F(CudaMemoryTest, MemoryConsistency) {
  for (int i = 0; i < kRepeatTimes; ++i) {
    TestMemoryConsistency();
  }
}
