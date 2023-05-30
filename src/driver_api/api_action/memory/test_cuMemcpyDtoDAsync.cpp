#include "memory_tests.h"
#include "cuda_runtime.h"

#define INIT_MEMD2DAsync()                      \
    CUdeviceptr d_src;                          \
    CUdeviceptr d_dst;                          \
    CUstream stream;                            \
    cuStreamCreate(&stream, CU_STREAM_DEFAULT); \
    size_t N = 102404;                   \
    cuMemAlloc(&d_src, N * sizeof(int));     \
    cuMemAlloc(&d_dst, N * sizeof(int));     \
    int h_src[N];                            \
    int h_dst[N];                            \
    int h_ref[N];                            \
    int h_new[N];                            \
    for (int i = 0; i < N; i++) {            \
        h_src[i] = i + 1;                       \
        h_ref[i] = h_src[i];                    \
    }                                           \
    cuMemsetD32(d_src, 0, N * sizeof(int));  \
    cuMemsetD32(d_dst, 0, N * sizeof(int));  \
    cuMemcpyHtoD(d_src, h_src, N * sizeof(int));

#define DEL_MEMD2DAsync() \
    cuMemFree(d_src);     \
    cuMemFree(d_dst);     \
    cuStreamDestroy(stream);

TEST_F(CuMemTest, AC_BA_MemcpyDtoDAsync_BasicBehavior) {
    // TODO：解决
    CUdeviceptr d_src;
    CUdeviceptr d_dst;
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    size_t N = 10;
    cuMemAlloc(&d_src, N * sizeof(int));
    cuMemAlloc(&d_dst, N * sizeof(int));
    int h_src[N];
    int h_dst[N];
    int h_ref[N];
    int h_new[N];
    for (int i = 0; i < N; i++) {
        h_src[i] = i;
        h_ref[i] = h_src[i];
    }
    for (int i = 0; i < N; i++) {
        std::cout << h_src[i] << " ";
    }
    cuMemcpyHtoD(d_src, h_src, N * sizeof(int));

    cuMemcpyDtoDAsync(d_dst, d_src, N * sizeof(int), stream);

    cuStreamSynchronize(stream);

    cuMemcpyDtoH(h_dst, d_dst, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        EXPECT_EQ(h_dst[i], h_ref[i]);
    }

    cuMemFree(d_src);
    cuMemFree(d_dst);
    cuStreamDestroy(stream);
}

TEST_F(CuMemTest, AC_INV_MemcpyDtoDAsync_InvalidDevicePtr) {
    INIT_MEMD2DAsync();
    CUresult result;

    result = cuMemcpyDtoDAsync(0, d_src, N * sizeof(int), 0);
    EXPECT_EQ(result, CUDA_ERROR_INVALID_VALUE);

    result = cuMemcpyDtoDAsync(d_dst, 0, N * sizeof(int), 0);
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
    result = cuMemcpyDtoDAsync(d_dst, d_src, N * sizeof(int), (CUstream)-1);
    EXPECT_EQ(result, CUDA_ERROR_INVALID_HANDLE);
    DEL_MEMD2DAsync();
}

TEST_F(CuMemTest, AC_EG_MemcpyDtoDAsync_Max) {
    // TODO：解决
    // GTEST_SKIP();
    // 尝试在大小为 size_max 的堆栈上创建数组 h_src、h_dst 和 h_ref。 在 C++
    // 中，数组大小需要保持不变并且在编译时已知，此外，在堆栈上创建如此大的数组几乎肯定会导致堆栈溢出。
    size_t free_mem, total_mem;
    cuMemGetInfo(&free_mem, &total_mem);

    size_t size_max = free_mem * 0.1 / sizeof(int);

    CUdeviceptr d_src;
    CUdeviceptr d_dst;
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);

    CUresult res_src = cuMemAlloc(&d_src, size_max * sizeof(int));
    CUresult res_dst = cuMemAlloc(&d_dst, size_max * sizeof(int));
    EXPECT_EQ(res_src, CUDA_SUCCESS);
    EXPECT_EQ(res_dst, CUDA_SUCCESS);

    int* h_src = new int[size_max];
    int* h_dst = new int[size_max];
    int* h_ref = new int[size_max];

    for (size_t i = 0; i < size_max; i++) {
        h_src[i] = rand();
        h_ref[i] = h_src[i];
    }

    cuMemcpyHtoD(d_src, h_src, size_max * sizeof(int));
    cuMemcpyDtoDAsync(d_dst, d_src, size_max * sizeof(int), stream);

    cuStreamSynchronize(stream);

    cuMemcpyDtoH(h_dst, d_dst, size_max * sizeof(int));
    for (size_t i = 0; i < size_max; i++) {
        EXPECT_EQ(h_dst[i], h_ref[i]);
    }

    delete[] h_src;
    delete[] h_dst;
    delete[] h_ref;

    cuMemFree(d_src);
    cuMemFree(d_dst);
    cuStreamDestroy(stream);
}

TEST_F(CuMemTest, AC_EG_MemcpyDtoDAsync_Min) {
    INIT_MEMD2DAsync();
 
    cuMemcpyDtoDAsync(d_dst, d_src, 1, stream);
    cuStreamSynchronize(stream);

    cuMemcpyDtoH(h_dst, d_dst, N * sizeof(int));
    EXPECT_EQ(h_dst[0], h_ref[0]);
    DEL_MEMD2DAsync();
}

TEST_F(CuMemTest, AC_SA_MemcpyDtoDAsync_AsyncBehavior) {
    INIT_MEMD2DAsync();
    CUstream stream1;
    cuStreamCreate(&stream1, CU_STREAM_NON_BLOCKING);

    // 用NONB流
    cuMemcpyDtoDAsync(d_dst, d_src, N * sizeof(int), stream1);

    cuMemcpyDtoH(h_dst, d_dst, N * sizeof(int));

    for (int i = N; i >0; i--) {
        ASSERT_NE(h_dst[i], h_ref[i]) << i;
    }

    cuStreamSynchronize(stream1);
    cuMemcpyDtoH(h_dst, d_dst, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        EXPECT_EQ(h_dst[i], h_ref[i]);
    }

    DEL_MEMD2DAsync();
}

TEST_F(CuMemTest, AC_OT_MemcpyDtoDAsync_RepeatedCalls) {
    // TODO：解决
    INIT_MEMD2DAsync();
    
    // // ddst {0,0,0,0,0,5,6,7,8,9}

    cuMemcpyDtoDAsync(d_dst, d_src, N / 2 * sizeof(int), stream);
    cuMemcpyDtoDAsync(d_dst + (N / 2) * sizeof(int),
                      d_src + (N / 2) * sizeof(int), N / 2 * sizeof(int),
                      stream);
    cuStreamSynchronize(stream);

    // another way：
    // CUstream stream1, stream2;
    // cuStreamCreate(&stream1, CU_STREAM_DEFAULT);
    // cuStreamCreate(&stream2, CU_STREAM_DEFAULT);

    // cuMemcpyDtoDAsync(d_dst, d_src, N / 2 * sizeof(int), stream1);
    // cuMemcpyDtoDAsync(d_dst + N / 2, d_src + N / 2, N / 2 *
    // sizeof(int), stream2);

    // cuStreamSynchronize(stream1);
    // cuStreamSynchronize(stream2);


    cuMemcpyDtoH(h_dst, d_dst, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        EXPECT_EQ(h_dst[i], h_ref[i]);
    }

    DEL_MEMD2DAsync();
}

TEST_F(CuMemTest, AC_OT_MemcpyDtoDAsync_Half) {
    // TODO：解决
    INIT_MEMD2DAsync();

    // // ddst {0,0,0,0,0,5,6,7,8,9}

    cuMemcpyDtoDAsync(d_dst + (N / 2) * sizeof(int),
                      d_src + (N / 2) * sizeof(int), N / 2 * sizeof(int),
                      stream);
    cuStreamSynchronize(stream);

    cuMemcpyDtoH(h_dst, d_dst, N * sizeof(int));
    for (int i = 0; i < N / 2; i++) {
        EXPECT_EQ(h_dst[i], 0);
    }
    for (int i = N / 2; i < N; i++) {
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

    cuMemcpyDtoH(h_dst, d_dst, N * sizeof(int));

    EXPECT_EQ(h_dst[1], h_ref[1]);
}