// 在加载和卸载CUDA模块后，CUDA函数的地址和参数是否正确
#include <gtest/gtest.h>
#include <cuda.h>

// A helper function to check cuda driver api status
void CheckCudaStatus(CUresult status) {
  ASSERT_EQ(status, CUDA_SUCCESS) << "CUDA driver api error: " << status;
}

// A test fixture class to manage cuda stream and event
class CudaStreamTest : public ::testing::Test {
 protected:
  // The number of times to repeat the test
  static constexpr int kRepeatTimes = 10;

  // Set up the test fixture
  void SetUp() override {
    // Create a cuda stream and an event
    CheckCudaStatus(cuStreamCreate(&cuda_stream_, CU_STREAM_DEFAULT));
    CheckCudaStatus(cuEventCreate(&cuda_event_, CU_EVENT_DEFAULT));
  }

  // Tear down the test fixture
  void TearDown() override {
    // Destroy the cuda stream and the event
    CheckCudaStatus(cuStreamDestroy(cuda_stream_));
    CheckCudaStatus(cuEventDestroy(cuda_event_));
  }

  // A helper function to compare cuda stream and event data
  void CompareStreamData() {
    // Get the cuda stream attributes
    CUstreamAttrValue attributes[] = {
      CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
      CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_TYPE,
      CU_STREAM_ATTRIBUTE_PRIORITY,
      CU_STREAM_ATTRIBUTE_EXEC_AFFINITY,
      CU_STREAM_ATTRIBUTE_EXEC_AFFINITY_SM_COUNT
    };
    int num_attributes = sizeof(attributes) / sizeof(attributes[0]);
    int values[num_attributes];
    for (int i = 0; i < num_attributes; ++i) {
      CheckCudaStatus(cuStreamGetAttribute(&values[i], attributes[i], cuda_stream_));
    }
    // Compare the stream attributes with the expected values
    // Note: the expected values are hard-coded here for simplicity, but they may vary depending on the device
    int expected_values[] = {0, 0, 0, 0, 0};
    for (int i = 0; i < num_attributes; ++i) {
      EXPECT_EQ(values[i], expected_values[i]) << "Stream attribute mismatch at index " << i;
    }
    // Get the cuda event attributes
    CUevent_attribute attributes2[] = {
      CU_EVENT_ATTRIBUTE_SYNCHRONIZATION_TYPE,
      CU_EVENT_ATTRIBUTE_GPU_TIMESTAMP
    };
    int num_attributes2 = sizeof(attributes2) / sizeof(attributes2[0]);
    int values2[num_attributes2];
    for (int i = 0; i < num_attributes2; ++i) {
      CheckCudaStatus(cuEventGetAttribute(&values2[i], attributes2[i], cuda_event_));
    }
    // Compare the event attributes with the expected values
    // Note: the expected values are hard-coded here for simplicity, but they may vary depending on the device
    int expected_values2[] = {0, 0};
    for (int i = 0; i < num_attributes2; ++i) {
      EXPECT_EQ(values2[i], expected_values2[i]) << "Event attribute mismatch at index " << i;
    }
  }

  // A helper function to perform the test logic
  void TestStreamConsistency() {
    // Get the cuda stream and event from the fixture and compare the data
    CompareStreamData();
    // Record an event on the stream and synchronize it
    CheckCudaStatus(cuEventRecord(cuda_event_, cuda_stream_));
    CheckCudaStatus(cuEventSynchronize(cuda_event_));
    // Destroy and recreate the cuda stream and event and compare the data
    CheckCudaStatus(cuStreamDestroy(cuda_stream_));
    CheckCudaStatus(cuEventDestroy(cuda_event_));
    CheckCudaStatus(cuStreamCreate(&cuda_stream_, CU_STREAM_DEFAULT));
    CheckCudaStatus(cuEventCreate(&cuda_event_, CU_EVENT_DEFAULT));
    CompareStreamData();
  }

  // The cuda stream handle
  CUstream cuda_stream_;
  // The cuda event handle
  CUevent cuda_event_;
};

// A test case to verify cuda stream and event consistency
TEST_F(CudaStreamTest, StreamConsistency) {
  // Repeat the test for a number of times
  for (int i = 0; i < kRepeatTimes; ++i) {
    TestStreamConsistency();
  }
}

// ..................'
#include <gtest/gtest.h>
#include <cuda.h>

// A helper function to check cuda driver api status
void CheckCudaStatus(CUresult status) {
  ASSERT_EQ(status, CUDA_SUCCESS) << "CUDA driver api error: " << status;
}

// A test fixture class to manage cuda module and function
class CudaModuleTest : public ::testing::Test {
 protected:
  // The name of the cuda module file
  static constexpr const char* kModuleFile = "test.ptx";
  // The name of the cuda function in the module
  static constexpr const char* kFunctionName = "test_kernel";

  // Set up the test fixture
  void SetUp() override {
    // Load the cuda module from the file
    CheckCudaStatus(cuModuleLoad(&cuda_module_, kModuleFile));
    // Get the cuda function from the module
    CheckCudaStatus(cuModuleGetFunction(&cuda_function_, cuda_module_, kFunctionName));
  }

  // Tear down the test fixture
  void TearDown() override {
    // Unload the cuda module
    CheckCudaStatus(cuModuleUnload(cuda_module_));
  }

  // A helper function to compare cuda function and module data
  void CompareFunctionData() {
    // Get the cuda function address and parameter size
    CUdeviceptr address;
    size_t size;
    CheckCudaStatus(cuFuncGetAttribute(&address, CU_FUNC_ATTRIBUTE_ENTRY_POINT_ADDRESS, cuda_function_));
    CheckCudaStatus(cuFuncGetAttribute(&size, CU_FUNC_ATTRIBUTE_PARAM_SIZE_BYTES, cuda_function_));
    // Compare the function address and parameter size with the expected values
    // Note: the expected values are hard-coded here for simplicity, but they may vary depending on the module file and the device
    CUdeviceptr expected_address = 0x12345678;
    size_t expected_size = 16;
    EXPECT_EQ(address, expected_address) << "Function address mismatch";
    EXPECT_EQ(size, expected_size) << "Function parameter size mismatch";
  }

  // The cuda module handle
  CUmodule cuda_module_;
  // The cuda function handle
  CUfunction cuda_function_;
};

// A test case to verify cuda module and function consistency
TEST_F(CudaModuleTest, ModuleConsistency) {
  // Get the cuda function from the module and compare the data
  CheckCudaStatus(cuModuleGetFunction(&cuda_function_, cuda_module_, kFunctionName));
  CompareFunctionData();
  // Unload and reload the cuda module and compare the data
  CheckCudaStatus(cuModuleUnload(cuda_module_));
  CheckCudaStatus(cuModuleLoad(&cuda_module_, kModuleFile));
  CheckCudaStatus(cuModuleGetFunction(&cuda_function_, cuda_module_, kFunctionName));
  CompareFunctionData();
}
