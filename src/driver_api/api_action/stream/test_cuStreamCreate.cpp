#include <cuda.h>
#include "gtest/gtest.h"
#include "stream_tests.h"

TEST_F(CuStreamTests, AC_BA_StreamCreate_TestDefaultFlags) {
    CUstream new_stream;
    CUresult result = cuStreamCreate(&new_stream, CU_STREAM_DEFAULT);

    EXPECT_EQ(result, CUDA_SUCCESS)
        << "cuStreamCreate failed with CU_STREAM_DEFAULT flag.";
}


TEST_F(CuStreamTests, AC_INV_StreamCreate_InvalidPhStreamPointer) {
    CUstream* invalid_stream_ptr = nullptr;
    CUresult result = cuStreamCreate(invalid_stream_ptr, CU_STREAM_DEFAULT);

    EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE)
        << "cuStreamCreate should have failed with "
           "an invalid phStream pointer.";
}
