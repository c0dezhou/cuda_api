#include "memory_tests.h"

#define INIT_MEMD2H()            \
    CUdeviceptr dptr;            \
    void* hptr;                  \
    size_t size = 5;          \
    int init_value = 1234;       \
    cuMemAllocHost(&hptr, size); \
    cuMemAlloc(&dptr, size);     \
    cuMemsetD32(dptr, init_value, size / sizeof(int));

#define DEL_MEMD2H() \
    cuMemFree(dptr); \
    cuMemFreeHost(hptr);

TEST_F(CuMemTest, AC_BA_MemcpyDtoH_BasicBehavior) {
    INIT_MEMD2H();
    CUresult result = cuMemcpyDtoH(hptr, dptr, size);
    EXPECT_EQ(result, CUDA_SUCCESS);

    for (size_t i = 0; i < size / sizeof(int); i++) {
        EXPECT_EQ(static_cast<int*>(hptr)[i], init_value);
    }
    DEL_MEMD2H();
}

TEST_F(CuMemTest, AC_INV_MemcpyDtoH_InvalidDtoHExcessiveSize) {
    INIT_MEMD2H();
    CUresult result = cuMemcpyDtoH(hptr, dptr, static_cast<size_t>(-1));
    EXPECT_NE(result, CUDA_SUCCESS);
    DEL_MEMD2H();
}

TEST_F(CuMemTest, AC_INV_MemcpyDtoH_InvalidDtoHNullDst) {
    INIT_MEMD2H();
    CUresult result = cuMemcpyDtoH(nullptr, dptr, size);
    EXPECT_NE(result, CUDA_SUCCESS);
    DEL_MEMD2H();
}

TEST_F(CuMemTest, AC_INV_MemcpyDtoH_InvalidDtoHNullSrc) {
    INIT_MEMD2H();
    CUresult result = cuMemcpyDtoH(hptr, 0, size);
    EXPECT_NE(result, CUDA_SUCCESS);
    DEL_MEMD2H();
}

TEST_F(CuMemTest, AC_EG_MemcpyDtoH_ZeroBytes) {
    INIT_MEMD2H();
    CUresult result = cuMemcpyDtoH(hptr, dptr, 0);
    EXPECT_EQ(result, CUDA_SUCCESS);
    DEL_MEMD2H();
}