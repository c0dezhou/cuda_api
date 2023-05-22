#include <cuda.h>
#include "gtest/gtest.h"
#include "stream_tests.h"

TEST_F(CuStreamTests, AC_BA_StreamDestroy_DestroyStream) {
    CUstream stream;
    EXPECT_EQ(cuStreamCreate(&stream, CU_STREAM_DEFAULT), CUDA_SUCCESS);
    EXPECT_EQ(cuStreamDestroy(stream), CUDA_SUCCESS);
}

TEST_F(CuStreamTests, AC_BA_StreamDestroy_DestroyStreamTwice) {
    CUstream stream;
    EXPECT_EQ(cuStreamCreate(&stream, CU_STREAM_DEFAULT), CUDA_SUCCESS);
    EXPECT_EQ(cuStreamDestroy(stream), CUDA_SUCCESS);
    EXPECT_EQ(cuStreamDestroy(stream), 400);
}

TEST_F(CuStreamTests, AC_BA_StreamDestroy_DestroyStreamTwicewithCtx) {
    CUstream stream;
    CUcontext Context;
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&Context, 0, cuDevice);
    cuCtxSetCurrent(Context);
    EXPECT_EQ(cuStreamCreate(&stream, CU_STREAM_DEFAULT), CUDA_SUCCESS);
    EXPECT_EQ(cuStreamDestroy(stream), CUDA_SUCCESS);
    EXPECT_EQ(cuStreamDestroy(stream), 400);
    cuCtxDestroy(Context);
}