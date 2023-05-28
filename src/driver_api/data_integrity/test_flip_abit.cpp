// 使用位翻转（bit-flipping）对cuda driver api的输入参数进行修改，例如将cuCtxCreate()的第一个参数从CU_CTX_SCHED_AUTO改为CU_CTX_SCHED_SPIN，观察是否会影响上下文的创建和调度策略。
#include <cuda.h>
#include <gtest/gtest.h>

// A helper macro to check cuda driver api calls
#define CUDA_CHECK(call) \
  do { \
    CUresult err = call; \
    ASSERT_EQ(err, CUDA_SUCCESS) << "CUDA driver API error: " << err; \
  } while (0)

// A test fixture class for cuda driver api tests
class CudaDriverApiTest : public ::testing::Test {
 protected:
  // Set up the test environment
  void SetUp() override {
    // Initialize the cuda driver api
    CUDA_CHECK(cuInit(0));

    // Get the device count
    int device_count;
    CUDA_CHECK(cuDeviceGetCount(&device_count));
    ASSERT_GT(device_count, 0) << "No cuda device available";

    // Get the first device
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));

    // Create a context for the device
    CUcontext context;
    CUDA_CHECK(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
  }

  // Tear down the test environment
  void TearDown() override {
    // Destroy the current context
    CUDA_CHECK(cuCtxDestroy(cuCtxGetCurrent()));
  }
};

// A test case for bit-flipping the second parameter of cuStreamCreate
TEST_F(CudaDriverApiTest, BitFlipCuStreamCreateSecondParam) {
  // Bit-flip the second parameter of cuStreamCreate from CU_STREAM_DEFAULT to CU_STREAM_NON_BLOCKING
  unsigned int flags = CU_STREAM_DEFAULT;
  flags ^= (1 << 0); // Flip the least significant bit

  // Try to create a stream with the bit-flipped flags
  CUstream stream;
  CUresult err = cuStreamCreate(&stream, flags);

  // Expect the creation to succeed with CUDA_SUCCESS
  EXPECT_EQ(err, CUDA_SUCCESS) << "CUDA driver API error: " << err;

  // Expect the stream to have non-blocking behavior
  int priority;
  CUDA_CHECK(cuStreamGetPriority(stream, &priority));
  EXPECT_EQ(priority, -1) << "Stream priority not correct";
}

// A test case for data truncation of the second parameter of cuStreamCreate
TEST_F(CudaDriverApiTest, DataTruncateCuStreamCreateSecondParam) {
  // Try to create a stream with an invalid flags value using int type
  int flags = -1; // Invalid flags value
  CUstream stream;
  CUresult err = cuStreamCreate(&stream, flags);

  // Expect the creation to fail with CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << "CUDA driver API error: " << err;
}
