#include "memory_tests.h"
#include "cuda_runtime.h"

#define INIT_MEMD2DAsync()                          \
    CUdeviceptr d_src;                              \
    CUdeviceptr d_dst;                              \
    CUstream stream;                                \
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);     \
    size_t size = 5;                               \
    cuMemAlloc(&d_src, size * sizeof(int));         \
    cuMemAlloc(&d_dst, size * sizeof(int));         \
    int h_src[size];                                \
    int h_dst[size];                                \
    int h_ref[size];                                \
    int h_new[size];                                \
    for (int i = 0; i < size; i++) {                \
        h_src[i] = i;                               \
        h_ref[i] = h_src[i];                        \
    }                                               \
    for (int i = 0; i < size; i++) {                \
        std::cout << h_src[i] << " ";               \
    }                                               \
    cuMemcpyHtoD(d_src, h_src, size * sizeof(int)); \
    cuMemsetD32(d_src, 0, size);

#define DEL_MEMD2DAsync() \
    cuMemFree(d_src);    \
    cuMemFree(d_dst);    \
    cuStreamDestroy(stream);

TEST_F(CuMemTest, AC_BA_MemcpyDtoDAsync_BasicBehavior) {
    // TODO：待确认
    // error due to init step
    INIT_MEMD2DAsync();
    cuMemcpyDtoDAsync(d_dst, d_src, size * sizeof(int), stream);
    cuMemcpyDtoD(d_dst, d_src, size * sizeof(int));

    cuStreamSynchronize(stream);
    cuMemcpyDtoH(h_dst, d_dst, size * sizeof(int));

    for (int i = 0; i < size; i++) {
        EXPECT_EQ(h_dst[i], h_ref[i]);
    }

    DEL_MEMD2DAsync();
}

TEST_F(CuMemTest, AC_INV_MemcpyDtoDAsync_InvalidDevicePtr) {
    INIT_MEMD2DAsync();
    CUresult result;

    result = cuMemcpyDtoDAsync(0, d_src, size * sizeof(int), 0);
    EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE);

    result = cuMemcpyDtoDAsync(d_dst, 0, size * sizeof(int), 0);
    EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE);
    DEL_MEMD2DAsync();
}

TEST_F(CuMemTest, AC_EG_MemcpyDtoDAsync_InvalidByteCount) {
    INIT_MEMD2DAsync();
    CUresult result;

    result = cuMemcpyDtoDAsync(d_dst, d_src, 0, 0);
    EXPECT_EQ(result, CUDA_SUCCESS);
    DEL_MEMD2DAsync();
}

TEST_F(CuMemTest, AC_INV_MemcpyDtoDAsync_InvalidStream) {
    INIT_MEMD2DAsync();
    CUresult result;
    result = cuMemcpyDtoDAsync(d_dst, d_src, size * sizeof(int), (CUstream)-1);
    EXPECT_EQ(result, CUDA_ERROR_INVALID_HANDLE);
    DEL_MEMD2DAsync();
}

TEST_F(CuMemTest, AC_EG_MemcpyDtoDAsync_Max) {
    // TODO：待确认
    GTEST_SKIP();  // due to core dump
    size_t size_max = 0;
    cuMemGetInfo(nullptr, &size_max);
    CUdeviceptr d_src;
    CUdeviceptr d_dst;
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
    cuMemAlloc(&d_src, size_max);
    cuMemAlloc(&d_dst, size_max);
    int h_src[size_max];
    int h_dst[size_max];
    int h_ref[size_max];
    for (int i = 0; i < size_max; i++) {
        h_src[i] = rand();
        h_ref[i] = h_src[i];
    }
    cuMemcpyHtoD(d_src, h_src, size_max * sizeof(int));
    cuMemcpyDtoDAsync(d_dst, d_src, size_max * sizeof(int), stream);

    cuStreamSynchronize(stream);

    cuMemcpyDtoH(h_dst, d_dst, size_max * sizeof(int));
    for (int i = 0; i < size_max; i++) {
        EXPECT_EQ(h_dst[i], h_ref[i]);
    }
}

TEST_F(CuMemTest, AC_EG_MemcpyDtoDAsync_Min) {
    INIT_MEMD2DAsync();
 
    cuMemcpyDtoDAsync(d_dst, d_src, 1, stream);
    cuStreamSynchronize(stream);

    cuMemcpyDtoH(h_dst, d_dst, size * sizeof(int));
    EXPECT_EQ(h_dst[0], h_ref[0]);
    DEL_MEMD2DAsync();
}

TEST_F(CuMemTest, AC_SA_MemcpyDtoDAsync_AsyncBehavior) {
    INIT_MEMD2DAsync();
    CUstream stream1;
    cuStreamCreate(&stream1, CU_STREAM_NON_BLOCKING);

    // 用NONB流
    cuMemcpyDtoDAsync(d_dst, d_src, size * sizeof(int), stream1);

    cuMemcpyDtoH(h_dst, d_dst, size * sizeof(int));

    for (int i = 0; i < size; i++) {
        EXPECT_NE(h_dst[i], h_ref[i]);
    }

    cuStreamSynchronize(stream1);
    cuMemcpyDtoH(h_dst, d_dst, size * sizeof(int));

    for (int i = 0; i < size; i++) {
        EXPECT_EQ(h_dst[i], h_ref[i]);
    }

    DEL_MEMD2DAsync();
}

TEST_F(CuMemTest, AC_OT_MemcpyDtoDAsync_RepeatedCalls) {
    // TODO：待确认
    INIT_MEMD2DAsync();

    cuMemcpyDtoDAsync(d_dst, d_src, size * sizeof(int), stream); 

    // 将src的前半部分复制到dst的后半部分
    cuMemcpyDtoDAsync(d_dst + size / 2, d_src, size / 2 * sizeof(int), stream);
    // ddst {0,0,0,0,0,5,6,7,8,9}

    cuStreamSynchronize(stream);

    cuMemcpyDtoH(h_dst, d_dst, size * sizeof(int));
    for (int i = 0; i < size/2; i++) {
        EXPECT_EQ(h_dst[i], 0);
    }
    for (int i = size /2 ; i < size; i++) {
        EXPECT_EQ(h_dst[i], h_ref[i]);
    }

    DEL_MEMD2DAsync();
}

// 检查函数是否能正确处理不同的内存对齐方式
TEST_F(CuMemTest, AC_OT_MemcpyDtoDAsync_MemoryAlignment) {
    INIT_MEMD2DAsync();

    cuMemcpyDtoDAsync(d_dst + sizeof(int), d_src, sizeof(int), stream);

    h_ref[1] = h_src[0];

    cuStreamSynchronize(stream);

    cuMemcpyDtoH(h_dst, d_dst, size * sizeof(int));

    EXPECT_EQ(h_dst[1], h_ref[1]);
}