// 使用数据截断（data truncation）对cuda driver api的输入参数进行修改，例如将cuMemAlloc()的第二个参数从size_t类型改为int类型，观察是否会导致内存分配失败或错误
// A test case for data truncation of the second parameter of cuMemAlloc
TEST_F(CudaDriverApiTest, DataTruncateCuMemAllocSecondParam) {
  // Get the device memory size
  size_t total_mem, free_mem;
  CUDA_CHECK(cuMemGetInfo(&free_mem, &total_mem));

  // Try to allocate more than the device memory size using int type
  int size = total_mem + 1; // May cause integer overflow
  CUdeviceptr dptr;
  CUresult err = cuMemAlloc(&dptr, size);

  // Expect the allocation to fail with CUDA_ERROR_OUT_OF_MEMORY
  EXPECT_EQ(err, CUDA_ERROR_OUT_OF_MEMORY) << "CUDA driver API error: " << err;
}

// A test case for data truncation of the third parameter of cuMemcpyHtoD
TEST_F(CudaDriverApiTest, DataTruncateCuMemcpyHtoDThirdParam) {
  // Allocate a host memory buffer of 8 bytes
  char* hbuf = (char*)malloc(8);
  ASSERT_NE(hbuf, nullptr) << "Host memory allocation failed";

  // Allocate a device memory buffer of 8 bytes
  CUdeviceptr dbuf;
  CUDA_CHECK(cuMemAlloc(&dbuf, 8));

  // Try to copy more than 8 bytes from host to device using short type
  short size = 9; // Truncate the size to 16 bits
  CUresult err = cuMemcpyHtoD(dbuf, hbuf, size);

  // Expect the copy to fail with CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << "CUDA driver API error: " << err;

  // Free the host and device memory buffers
  free(hbuf);
  CUDA_CHECK(cuMemFree(dbuf));
}

// A test case for data truncation of the second parameter of cuStreamWaitEvent
TEST_F(CudaDriverApiTest, DataTruncateCuStreamWaitEventSecondParam) {
  // Create a stream
  CUstream stream;
  CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  // Create an event
  CUevent event;
  CUDA_CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));

  // Record the event on the stream
  CUDA_CHECK(cuEventRecord(event, stream));

  // Try to make the stream wait for an invalid event using char type
  char invalid_event = -1; // Truncate the event to 8 bits
  CUresult err = cuStreamWaitEvent(stream, (CUevent)invalid_event, 0);

  // Expect the wait to fail with CUDA_ERROR_INVALID_HANDLE
  EXPECT_EQ(err, CUDA_ERROR_INVALID_HANDLE) << "CUDA driver API error: " << err;

  // Destroy the stream and the event
  CUDA_CHECK(cuStreamDestroy(stream));
  CUDA_CHECK(cuEventDestroy(event));
}

// A test case for data truncation of the first parameter of cuMemAlloc
TEST_F(CudaDriverApiTest, DataTruncateCuMemAllocFirstParam) {
  // Try to allocate a device memory buffer using float type
  float size = 8.5; // Truncate the size to 32 bits
  CUdeviceptr dptr;
  CUresult err = cuMemAlloc(&dptr, (size_t)size);

  // Expect the allocation to succeed with CUDA_SUCCESS
  EXPECT_EQ(err, CUDA_SUCCESS) << "CUDA driver API error: " << err;

  // Free the device memory buffer
  CUDA_CHECK(cuMemFree(dptr));
}

// A test case for data truncation of the first parameter of cuMemFree
TEST_F(CudaDriverApiTest, DataTruncateCuMemFreeFirstParam) {
  // Allocate a device memory buffer of 8 bytes
  CUdeviceptr dbuf;
  CUDA_CHECK(cuMemAlloc(&dbuf, 8));

  // Try to free an invalid device pointer using int type
  int invalid_ptr = -1; // Truncate the pointer to 32 bits
  CUresult err = cuMemFree((CUdeviceptr)invalid_ptr);

  // Expect the free to fail with CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << "CUDA driver API error: " << err;

  // Free the original device memory buffer
  CUDA_CHECK(cuMemFree(dbuf));
}

// A test case for data truncation of the third parameter of cuMemcpyDtoH
TEST_F(CudaDriverApiTest, DataTruncateCuMemcpyDtoHThirdParam) {
  // Allocate a device memory buffer of 8 bytes
  CUdeviceptr dbuf;
  CUDA_CHECK(cuMemAlloc(&dbuf, 8));

  // Allocate a host memory buffer of 8 bytes
  char* hbuf = (char*)malloc(8);
  ASSERT_NE(hbuf, nullptr) << "Host memory allocation failed";

  // Try to copy more than 8 bytes from device to host using short type
  short size = 9; // Truncate the size to 16 bits
  CUresult err = cuMemcpyDtoH(hbuf, dbuf, size);

  // Expect the copy to fail with CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << "CUDA driver API error: " << err;

  // Free the device and host memory buffers
  CUDA_CHECK(cuMemFree(dbuf));
  free(hbuf);
}

// A test case for data truncation of the second parameter of cuStreamCreate
TEST_F(CudaDriverApiTest, DataTruncateCuStreamCreateSecondParam) {
  // Try to create a stream with an invalid flags value using uint64_t type
  uint64_t flags = 0xFFFFFFFFFFFFFFFF; // Truncate the flags to 64 bits
  CUstream stream;
  CUresult err = cuStreamCreate(&stream, (unsigned int)flags);

  // Expect the creation to fail with CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << "CUDA driver API error: " << err;
}

// A test case for data truncation of the first parameter of cuMemAlloc
TEST_F(CudaDriverApiTest, DataTruncateCuMemAllocFirstParam) {
  // Try to allocate a device memory buffer using float type
  float size = 8.5; // Truncate the size to 32 bits
  CUdeviceptr dptr;
  CUresult err = cuMemAlloc(&dptr, (size_t)size);

  // Expect the allocation to succeed with CUDA_SUCCESS
  EXPECT_EQ(err, CUDA_SUCCESS) << "CUDA driver API error: " << err;

  // Free the device memory buffer
  CUDA_CHECK(cuMemFree(dptr));
}

// A test case for data truncation of the first parameter of cuMemsetD8
TEST_F(CudaDriverApiTest, DataTruncateCuMemsetD8FirstParam) {
  // Allocate a device memory buffer of 8 bytes
  CUdeviceptr dbuf;
  CUDA_CHECK(cuMemAlloc(&dbuf, 8));

  // Try to set the device memory buffer to a value using float type
  float value = 1.5; // Truncate the value to 32 bits
  CUresult err = cuMemsetD8(dbuf, (unsigned char)value, 8);

  // Expect the set to succeed with CUDA_SUCCESS
  EXPECT_EQ(err, CUDA_SUCCESS) << "CUDA driver API error: " << err;

  // Free the device memory buffer
  CUDA_CHECK(cuMemFree(dbuf));
}

// A test case for data truncation of the second parameter of cuMemsetD8
TEST_F(CudaDriverApiTest, DataTruncateCuMemsetD8SecondParam) {
  // Allocate a device memory buffer of 8 bytes
  CUdeviceptr dbuf;
  CUDA_CHECK(cuMemAlloc(&dbuf, 8));

  // Try to set the device memory buffer to an invalid value using int type
  int value = -1; // Truncate the value to 32 bits
  CUresult err = cuMemsetD8(dbuf, (unsigned char)value, 8);

  // Expect the set to succeed with CUDA_SUCCESS
  EXPECT_EQ(err, CUDA_SUCCESS) << "CUDA driver API error: " << err;

  // Free the device memory buffer
  CUDA_CHECK(cuMemFree(dbuf));
}

// A test case for data truncation of the third parameter of cuMemsetD8
TEST_F(CudaDriverApiTest, DataTruncateCuMemsetD8ThirdParam) {
  // Allocate a device memory buffer of 8 bytes
  CUdeviceptr dbuf;
  CUDA_CHECK(cuMemAlloc(&dbuf, 8));

  // Try to set more than 8 bytes of the device memory buffer using short type
  short size = 9; // Truncate the size to 16 bits
  CUresult err = cuMemsetD8(dbuf, 0, size);

  // Expect the set to fail with CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << "CUDA driver API error: " << err;

  // Free the device memory buffer
  CUDA_CHECK(cuMemFree(dbuf));
}

// A template function to test data truncation of the first parameter of cuMemsetD*
template <typename T>
void TestTruncateCuMemsetDFirstParam(T value) {
  // Allocate a device memory buffer of 8 bytes
  CUdeviceptr dbuf;
  CUDA_CHECK(cuMemAlloc(&dbuf, 8));

  // Try to set the device memory buffer to a value using T type
  CUresult err = cuMemsetD(dbuf, value, sizeof(T));

  // Expect the set to succeed with CUDA_SUCCESS
  EXPECT_EQ(err, CUDA_SUCCESS) << "CUDA driver API error: " << err;

  // Free the device memory buffer
  CUDA_CHECK(cuMemFree(dbuf));
}

// A macro to instantiate the template function for different types and values
#define TEST_TRUNCATE_CUMEMSETD_FIRST_PARAM(type, value) \
  TEST_F(CudaDriverApiTest, DataTruncateCuMemsetD##type##FirstParam) { \
    TestTruncateCuMemsetDFirstParam<type>(value); \
  }

// Instantiate the template function for different types and values
TEST_TRUNCATE_CUMEMSETD_FIRST_PARAM(float, 1.5) // Truncate float to 32 bits
TEST_TRUNCATE_CUMEMSETD_FIRST_PARAM(double, 2.5) // Truncate double to 64 bits
TEST_TRUNCATE_CUMEMSETD_FIRST_PARAM(uint64_t, 0xFFFFFFFFFFFFFFFF) // Truncate uint64_t to 64 bits
// Instantiate the template function for different types and values
TEST_TRUNCATE_CUMEMSETD_FIRST_PARAM(int, -1) // Truncate int to 32 bits
TEST_TRUNCATE_CUMEMSETD_FIRST_PARAM(char, 'a') // Truncate char to 8 bits
TEST_TRUNCATE_CUMEMSETD_FIRST_PARAM(bool, true) // Truncate bool to 1 bit


// A test case for data truncation of the first parameter of cuStreamDestroy
TEST_F(CudaDriverApiTest, DataTruncateCuStreamDestroyFirstParam) {
  // Create a stream
  CUstream stream;
  CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  // Try to destroy an invalid stream using char type
  char invalid_stream = -1; // Truncate the stream to 8 bits
  CUresult err = cuStreamDestroy((CUstream)invalid_stream);

  // Expect the destroy to fail with CUDA_ERROR_INVALID_HANDLE
  EXPECT_EQ(err, CUDA_ERROR_INVALID_HANDLE) << "CUDA driver API error: " << err;

  // Destroy the original stream
  CUDA_CHECK(cuStreamDestroy(stream));
}

// A test case for data truncation of the first parameter of cuEventCreate
TEST_F(CudaDriverApiTest, DataTruncateCuEventCreateFirstParam) {
  // Try to create an event with a NULL pointer as the first parameter
  CUevent* pevent = NULL; // Truncate the pointer to 32 bits
  CUresult err = cuEventCreate(pevent, CU_EVENT_DEFAULT);

  // Expect the creation to fail with CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << "CUDA driver API error: " << err;
}

// A test case for data truncation of the second parameter of cuEventCreate
TEST_F(CudaDriverApiTest, DataTruncateCuEventCreateSecondParam) {
  // Try to create an event with an invalid flags value using char type
  char flags = -1; // Truncate the flags to 8 bits
  CUevent event;
  CUresult err = cuEventCreate(&event, (unsigned int)flags);

  // Expect the creation to fail with CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << "CUDA driver API error: " << err;
}

// A test case for data truncation of the first parameter of cuEventDestroy
TEST_F(CudaDriverApiTest, DataTruncateCuEventDestroyFirstParam) {
  // Create an event
  CUevent event;
  CUDA_CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));

  // Try to destroy an invalid event using char type
  char invalid_event = -1; // Truncate the event to 8 bits
  CUresult err = cuEventDestroy((CUevent)invalid_event);

  // Expect the destroy to fail with CUDA_ERROR_INVALID_HANDLE
  EXPECT_EQ(err, CUDA_ERROR_INVALID_HANDLE) << "CUDA driver API error: " << err;

  // Destroy the original event
  CUDA_CHECK(cuEventDestroy(event));
}
