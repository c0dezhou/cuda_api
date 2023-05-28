// 使用数据截断（data truncation）对cuda driver api的输入参数进行修改，例如将cuMemAlloc()的第二个参数从size_t类型改为int类型，观察是否会导致内存分配失败或错误

#include "test_utils.h"

class TruncatuinTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cuInit(0);
        cuDeviceGet(&device, 0);
        cuCtxCreate(&context, 0, device);
    }

    void TearDown() override { cuCtxDestroy(context); }

    CUcontext context;
    CUresult res;
    CUdevice device;
};

TEST_F(TruncatuinTest, DataTruncateCuMemAllocSecondParam) {
    size_t total_mem, free_mem;
    checkError(cuMemGetInfo(&free_mem, &total_mem));

    int size = total_mem + 1;  // May cause integer overflow
    CUdeviceptr dptr;
    CUresult err = cuMemAlloc(&dptr, size);

    EXPECT_EQ(err, CUDA_ERROR_OUT_OF_MEMORY)
        << "CUDA driver API error: " << err;
}

TEST_F(TruncatuinTest, DataTruncateCuMemcpyHtoDThirdParam) {
  // Allocate a host memory buffer of 8 bytes
  char* hbuf = (char*)malloc(8);
  ASSERT_NE(hbuf, nullptr) << "Host memory allocation failed";


  CUdeviceptr dbuf;
  checkError(cuMemAlloc(&dbuf, 8));

  // Try to copy more than 8 bytes from host to device using short type
  short size = 9; // Truncate the size to 16 bits
  CUresult err = cuMemcpyHtoD(dbuf, hbuf, size);

  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << "CUDA driver API error: " << err;

  free(hbuf);
  checkError(cuMemFree(dbuf));
}


TEST_F(TruncatuinTest, DataTruncateCuMemAllocFirstParam) {
  float size = 8.5; // Truncate the size to 32 bits
  CUdeviceptr dptr;
  CUresult err = cuMemAlloc(&dptr, (size_t)size);

  EXPECT_EQ(err, CUDA_SUCCESS) << "CUDA driver API error: " << err;

  checkError(cuMemFree(dptr));
}

TEST_F(TruncatuinTest, DataTruncateCuMemFreeFirstParam) {
  // Allocate a device memory buffer of 8 bytes
  CUdeviceptr dbuf;
  checkError(cuMemAlloc(&dbuf, 8));

  // Try to free an invalid device pointer using int type
  int invalid_ptr = -1; // Truncate the pointer to 32 bits
  CUresult err = cuMemFree((CUdeviceptr)invalid_ptr);

  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << "CUDA driver API error: " << err;

  checkError(cuMemFree(dbuf));
}

TEST_F(TruncatuinTest, DataTruncateCuMemcpyDtoHThirdParam) {
  CUdeviceptr dbuf;
  checkError(cuMemAlloc(&dbuf, 8));

  char* hbuf = (char*)malloc(8);
  ASSERT_NE(hbuf, nullptr) << "Host memory allocation failed";

  short size = 9; // Truncate the size to 16 bits
  CUresult err = cuMemcpyDtoH(hbuf, dbuf, size);

  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << "CUDA driver API error: " << err;

  checkError(cuMemFree(dbuf));
  free(hbuf);
}


TEST_F(TruncatuinTest, DataTruncateCuStreamCreateSecondParam1) {
  // Try to create a stream with an invalid flags value using uint64_t type
  uint64_t flags = 0xFFFFFFFFFFFFFFFF; // Truncate the flags to 64 bits
  CUstream stream;
  CUresult err = cuStreamCreate(&stream, (unsigned int)flags);

  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << "CUDA driver API error: " << err;
}


TEST_F(TruncatuinTest, DataTruncateCuMemAllocFirstParam11) {
  // Try to allocate a device memory buffer using float type
  float size = 8.5; // Truncate the size to 32 bits
  CUdeviceptr dptr;
  CUresult err = cuMemAlloc(&dptr, (size_t)size);

  EXPECT_EQ(err, CUDA_SUCCESS) << "CUDA driver API error: " << err;

  checkError(cuMemFree(dptr));
}


TEST_F(TruncatuinTest, DataTruncateCuMemsetD8FirstParam) {
  CUdeviceptr dbuf;
  checkError(cuMemAlloc(&dbuf, 8));

  // Try to set the device memory buffer to a value using float type
  float value = 1.5; // Truncate the value to 32 bits
  CUresult err = cuMemsetD8(dbuf, (unsigned char)value, 8);

  EXPECT_EQ(err, CUDA_SUCCESS) << "CUDA driver API error: " << err;

  checkError(cuMemFree(dbuf));
}

TEST_F(TruncatuinTest, DataTruncateCuMemsetD8SecondParam) {
  CUdeviceptr dbuf;
  checkError(cuMemAlloc(&dbuf, 8));

  int value = -1; // Truncate the value to 32 bits
  CUresult err = cuMemsetD8(dbuf, (unsigned char)value, 8);

  EXPECT_EQ(err, CUDA_SUCCESS) << "CUDA driver API error: " << err;

  checkError(cuMemFree(dbuf));
}


TEST_F(TruncatuinTest, DataTruncateCuMemsetD8ThirdParam) {
  CUdeviceptr dbuf;
  checkError(cuMemAlloc(&dbuf, 8));

  short size = 9; // Truncate the size to 16 bits
  CUresult err = cuMemsetD8(dbuf, 0, size);

  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << "CUDA driver API error: " << err;

  checkError(cuMemFree(dbuf));
}


// template <typename T>
// void TestTruncateCuMemsetDFirstParam(T value) {
//   // Allocate a device memory buffer of 8 bytes
//   CUdeviceptr dbuf;
//   checkError(cuMemAlloc(&dbuf, 8));

//   // Try to set the device memory buffer to a value using T type
//   CUresult err = cuMemsetD(dbuf, value, sizeof(T));

//   EXPECT_EQ(err, CUDA_SUCCESS) << "CUDA driver API error: " << err;

//   checkError(cuMemFree(dbuf));
// }

// // A macro to instantiate the template function for different types and values
// #define TEST_TRUNCATE_CUMEMSETD_FIRST_PARAM(type, value) \
//   TEST_F(TruncatuinTest, DataTruncateCuMemsetD##type##FirstParam) { \
//     TestTruncateCuMemsetDFirstParam<type>(value); \
//   }

// TEST_TRUNCATE_CUMEMSETD_FIRST_PARAM(float, 1.5) // Truncate float to 32 bits
// TEST_TRUNCATE_CUMEMSETD_FIRST_PARAM(double, 2.5) // Truncate double to 64 bits
// TEST_TRUNCATE_CUMEMSETD_FIRST_PARAM(uint64_t, 0xFFFFFFFFFFFFFFFF) // Truncate uint64_t to 64 bits
// TEST_TRUNCATE_CUMEMSETD_FIRST_PARAM(int, -1) // Truncate int to 32 bits
// TEST_TRUNCATE_CUMEMSETD_FIRST_PARAM(char, 'a') // Truncate char to 8 bits
// TEST_TRUNCATE_CUMEMSETD_FIRST_PARAM(bool, true) // Truncate bool to 1 bit


// A test case for data truncation of the first parameter of cuStreamDestroy
TEST_F(TruncatuinTest, DataTruncateCuStreamDestroyFirstParam) {
  // Create a stream
  CUstream stream;
  checkError(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  // Try to destroy an invalid stream using char type
  char invalid_stream = -1; // Truncate the stream to 8 bits
  CUresult err = cuStreamDestroy((CUstream)invalid_stream);

  // Expect the destroy to fail with CUDA_ERROR_INVALID_HANDLE
  EXPECT_EQ(err, CUDA_ERROR_INVALID_HANDLE) << "CUDA driver API error: " << err;

  // Destroy the original stream
  checkError(cuStreamDestroy(stream));
}

// A test case for data truncation of the first parameter of cuEventCreate
TEST_F(TruncatuinTest, DataTruncateCuEventCreateFirstParam) {
  // Try to create an event with a NULL pointer as the first parameter
  CUevent* pevent = NULL; // Truncate the pointer to 32 bits
  CUresult err = cuEventCreate(pevent, CU_EVENT_DEFAULT);

  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << "CUDA driver API error: " << err;
}

// A test case for data truncation of the second parameter of cuEventCreate
TEST_F(TruncatuinTest, DataTruncateCuEventCreateSecondParam) {
  char flags = -1; // Truncate the flags to 8 bits
  CUevent event;
  CUresult err = cuEventCreate(&event, (unsigned int)flags);

  // Expect the creation to fail with CUDA_ERROR_INVALID_VALUE
  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << "CUDA driver API error: " << err;
}

// A test case for data truncation of the first parameter of cuEventDestroy
TEST_F(TruncatuinTest, DataTruncateCuEventDestroyFirstParam) {
  // Create an event
  CUevent event;
  checkError(cuEventCreate(&event, CU_EVENT_DEFAULT));

  // Try to destroy an invalid event using char type
  char invalid_event = -1; // Truncate the event to 8 bits
  CUresult err = cuEventDestroy((CUevent)invalid_event);

  EXPECT_EQ(err, CUDA_ERROR_INVALID_HANDLE) << "CUDA driver API error: " << err;

  checkError(cuEventDestroy(event));
}
