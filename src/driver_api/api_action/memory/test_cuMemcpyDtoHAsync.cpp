#include "memory_tests.h"

#define INIT_MEMD2HAsync()                             \
    CUdeviceptr dptr;                                  \
    void* hptr;                                        \
    size_t size = 5;                                \
    int init_value = 1234;                             \
    cuMemAllocHost(&hptr, size);                       \
    cuMemAlloc(&dptr, size);                           \
    CUstream stream;                                   \
    cuMemsetD32(dptr, init_value, size / sizeof(int)); \
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);

#define DEL_MEMD2HAsync()    \
    cuStreamDestroy(stream); \
    cuMemFree(dptr);         \
    cuMemFreeHost(hptr);

TEST_F(CuMemTest, AC_BA_MemcpyDtoHAsync_BasicBehavior) {
    INIT_MEMD2HAsync();
    CUresult result = cuMemcpyDtoHAsync(hptr, dptr, size, stream);
    EXPECT_EQ(result, CUDA_SUCCESS);

    cuStreamSynchronize(stream);

    for (size_t i = 0; i < size / sizeof(int); i++) {
        EXPECT_EQ(static_cast<int*>(hptr)[i], init_value);
    }
    DEL_MEMD2HAsync();
}

TEST_F(CuMemTest, AC_INV_MemcpyDtoHAsync_InvalidD2HExcessiveSize) {
    INIT_MEMD2HAsync();
    CUresult result =
        cuMemcpyDtoHAsync(hptr, dptr, static_cast<size_t>(-1), stream);
    EXPECT_NE(result, CUDA_SUCCESS);
    DEL_MEMD2HAsync();
}

TEST_F(CuMemTest, AC_INV_MemcpyDtoHAsync_InvalidD2HNullDst) {
    INIT_MEMD2HAsync();
    CUresult result = cuMemcpyDtoHAsync(nullptr, dptr, size, stream);
    EXPECT_NE(result, CUDA_SUCCESS);
    DEL_MEMD2HAsync();
}

TEST_F(CuMemTest, AC_INV_MemcpyDtoHAsync_InvalidD2HNullSrc) {
    INIT_MEMD2HAsync();
    CUresult result = cuMemcpyDtoHAsync(hptr, 0, size, stream);
    EXPECT_NE(result, CUDA_SUCCESS);
    DEL_MEMD2HAsync();
}

TEST_F(CuMemTest, AC_INV_MemcpyDtoHAsync_InvalidD2HStream) {
    INIT_MEMD2HAsync();
    CUresult result = cuMemcpyDtoHAsync(hptr, dptr, size, (CUstream)(-1));
    EXPECT_EQ(result, CUDA_ERROR_INVALID_HANDLE);
    DEL_MEMD2HAsync();
}

TEST_F(CuMemTest, AC_EG_MemcpyDtoHAsync_D2HZeroBytes) {
    INIT_MEMD2HAsync();
    CUresult result = cuMemcpyDtoHAsync(hptr, dptr, 0, stream);
    EXPECT_EQ(result, CUDA_SUCCESS);

    cuStreamSynchronize(stream);
    DEL_MEMD2HAsync();
}

TEST_F(CuMemTest, AC_SA_MemcpyDtoHAsync_AsyncBehaviorD2H) {
    INIT_MEMD2HAsync();

    CUresult result = cuMemcpyDtoHAsync(hptr, dptr, size, stream);
    EXPECT_EQ(result, CUDA_SUCCESS);

    for (size_t i = 0; i < size / sizeof(int); i++) {
        EXPECT_NE(static_cast<int*>(hptr)[i], init_value);
    }

    cuStreamSynchronize(stream);

    for (size_t i = 0; i < size / sizeof(int); i++) {
        EXPECT_EQ(static_cast<int*>(hptr)[i], init_value);
    }

    DEL_MEMD2HAsync();
}
