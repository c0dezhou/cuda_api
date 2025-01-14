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

    int size = total_mem + 1;  // overflow
    CUdeviceptr dptr;
    CUresult err = cuMemAlloc(&dptr, size);

    EXPECT_EQ(err, CUDA_ERROR_OUT_OF_MEMORY)  << err;
}

TEST_F(TruncatuinTest, DataTruncateCuMemcpyHtoDThirdParam) {
  char* hbuf = (char*)malloc(8);
  ASSERT_NE(hbuf, nullptr) << "Host memory allocation failed";


  CUdeviceptr dbuf;
  checkError(cuMemAlloc(&dbuf, 8));

  short size = 9; //short 16 bit
  CUresult err = cuMemcpyHtoD(dbuf, hbuf, size);

  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << err;

  free(hbuf);
  checkError(cuMemFree(dbuf));
}


TEST_F(TruncatuinTest, DataTruncateCuMemAllocFirstParam) {
  float size = 8.5; // float 32bit
  CUdeviceptr dptr;
  CUresult err = cuMemAlloc(&dptr, (size_t)size);

  EXPECT_EQ(err, CUDA_SUCCESS)  << err;

  checkError(cuMemFree(dptr));
}

TEST_F(TruncatuinTest, DataTruncateCuMemFreeFirstParam) {
  CUdeviceptr dbuf;
  checkError(cuMemAlloc(&dbuf, 8));  // 8 bytes

  int invalid_ptr = -1; //32 bit
  CUresult err = cuMemFree((CUdeviceptr)invalid_ptr);

  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE)  << err;

  checkError(cuMemFree(dbuf));
}

TEST_F(TruncatuinTest, DataTruncateCuMemcpyDtoHThirdParam) {
  CUdeviceptr dbuf;
  checkError(cuMemAlloc(&dbuf, 8));

  char* hbuf = (char*)malloc(8);
  ASSERT_NE(hbuf, nullptr) << "Host memory allocation failed";

  short size = 9; //16 bits
  CUresult err = cuMemcpyDtoH(hbuf, dbuf, size);

  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << err;

  checkError(cuMemFree(dbuf));
  free(hbuf);
}


TEST_F(TruncatuinTest, DataTruncateCuStreamCreateSecondParam1) {
  uint64_t flags = 0xFFFFFFFFFFFFFFFF; // 64 bits
  CUstream stream;
  CUresult err = cuStreamCreate(&stream, (unsigned int)flags);

  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE) << err;
}


TEST_F(TruncatuinTest, DataTruncateCuMemAllocFirstParam11) {
  float size = 8.5; //  32 bits
  CUdeviceptr dptr;
  CUresult err = cuMemAlloc(&dptr, (size_t)size);

  EXPECT_EQ(err, CUDA_SUCCESS)  << err;

  checkError(cuMemFree(dptr));
}


TEST_F(TruncatuinTest, DataTruncateCuMemsetD8FirstParam) {
  CUdeviceptr dbuf;
  checkError(cuMemAlloc(&dbuf, 8));

  float value = 1.5; // 32 bits
  CUresult err = cuMemsetD8(dbuf, (unsigned char)value, 8);

  EXPECT_EQ(err, CUDA_SUCCESS)  << err;

  checkError(cuMemFree(dbuf));
}

TEST_F(TruncatuinTest, DataTruncateCuMemsetD8SecondParam) {
  CUdeviceptr dbuf;
  checkError(cuMemAlloc(&dbuf, 8));

  int value = -1; //32 bits
  CUresult err = cuMemsetD8(dbuf, (unsigned char)value, 8);

  EXPECT_EQ(err, CUDA_SUCCESS)  << err;

  checkError(cuMemFree(dbuf));
}


TEST_F(TruncatuinTest, DataTruncateCuMemsetD8ThirdParam) {
  CUdeviceptr dbuf;
  checkError(cuMemAlloc(&dbuf, 8));

  short size = 9; // 16 bits
  CUresult err = cuMemsetD8(dbuf, 0, size);

  EXPECT_EQ(err, CUDA_ERROR_INVALID_VALUE)  << err;

  checkError(cuMemFree(dbuf));
}

template <typename T, typename Tsmall>
void TestTruncation(T value) {
  CUdeviceptr dbuf;
  checkError(cuMemAlloc(&dbuf, 8));

  CUresult err;

  if (sizeof(Tsmall) == 1) {
    err = cuMemsetD8(dbuf, static_cast<unsigned char>(value), 8);
  } else if (sizeof(Tsmall) == 2) {
    err = cuMemsetD16(dbuf, static_cast<unsigned short>(value), 4);
  } else if (sizeof(Tsmall) == 4) {
    err = cuMemsetD32(dbuf, static_cast<unsigned int>(value), 2);
  } else {
    err = CUDA_ERROR_INVALID_VALUE;
  }

  EXPECT_EQ(err, CUDA_SUCCESS)  << err;

  if (err == CUDA_SUCCESS) {
    Tsmall* hbuf = new Tsmall[8 / sizeof(Tsmall)];

    checkError(cuMemcpyDtoH(hbuf, dbuf, 8));

    Tsmall truncated_value = static_cast<Tsmall>(value);
    for (int i = 0; i < 8 / sizeof(Tsmall); ++i) {
      EXPECT_EQ(hbuf[i], truncated_value);
    }

    delete[] hbuf;
  }

  checkError(cuMemFree(dbuf));
}

#define TEST_TRUNCATION(type, type_small, value)        \
  TEST_F(TruncatuinTest, DataTruncateCuMemsetD##type##type_small) { \
    TestTruncation<type, type_small>(value);            \
  }

TEST_TRUNCATION(int, char, 0x12345678)
TEST_TRUNCATION(int, short, 0x12345678)
TEST_TRUNCATION(uint64_t,
                int,
                0x123456789abcdef0LL)
