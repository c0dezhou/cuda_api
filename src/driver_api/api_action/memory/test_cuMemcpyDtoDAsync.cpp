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

TEST_F(CuMemTest, MemcpyDtoDAsync_BasicBehavior) {
    // TODO: 解决
    INIT_MEMD2DAsync();
    
    cuMemcpyDtoDAsync(d_dst, d_src, size * sizeof(int), stream);

    // Wait for the async operation to complete before launching another copy operation
    cuStreamSynchronize(stream);

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
// TODO：解决
// 尝试在大小为 size_max 的堆栈上创建数组 h_src、h_dst 和 h_ref。 在 C++ 中，数组大小需要保持不变并且在编译时已知，此外，在堆栈上创建如此大的数组几乎肯定会导致堆栈溢出。
    size_t free_mem, total_mem;
    cuMemGetInfo(&free_mem, &total_mem);

    // Allocating slightly less than the total free memory to leave some room for system operations
    size_t size_max = free_mem * 0.9 / sizeof(int); 

    CUdeviceptr d_src;
    CUdeviceptr d_dst;
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);

    CUresult res_src = cuMemAlloc(&d_src, size_max * sizeof(int));
    CUresult res_dst = cuMemAlloc(&d_dst, size_max * sizeof(int));
    EXPECT_EQ(res_src, CUDA_SUCCESS);
    EXPECT_EQ(res_dst, CUDA_SUCCESS);

    // Using dynamic allocation to avoid stack overflow
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
    // TODO：解决
    INIT_MEMD2DAsync();

    // cuMemcpyDtoDAsync(d_dst, d_src, size * sizeof(int), stream); 

    // // 将src的前半部分复制到dst的后半部分
    // cuMemcpyDtoDAsync(d_dst + size / 2, d_src, size / 2 * sizeof(int), stream);
    // // ddst {0,0,0,0,0,5,6,7,8,9}

    cuMemcpyDtoDAsync(d_dst, d_src, size / 2 * sizeof(int), stream);
cuStreamSynchronize(stream);
cuMemcpyDtoDAsync(d_dst + size / 2, d_src + size / 2, size / 2 * sizeof(int), stream);

// another solution：
// CUstream stream1, stream2;
// cuStreamCreate(&stream1, CU_STREAM_DEFAULT);
// cuStreamCreate(&stream2, CU_STREAM_DEFAULT);

// cuMemcpyDtoDAsync(d_dst, d_src, size / 2 * sizeof(int), stream1);
// cuMemcpyDtoDAsync(d_dst + size / 2, d_src + size / 2, size / 2 * sizeof(int), stream2);

// cuStreamSynchronize(stream1);
// cuStreamSynchronize(stream2);


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