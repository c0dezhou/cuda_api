#include "module_tests.h"

TEST_F(cuModuleTest, AC_BA_ModuleLoadFatBinary_LoadValidFatBinary) {
    // TODO：待确认
    const void* fatCubin = fatbin_path;
    CUresult res = cuModuleLoadFatBinary(&module_fatbin, fatCubin);

    EXPECT_EQ(CUDA_SUCCESS, res);
    EXPECT_NE(nullptr, module_fatbin);
}

TEST_F(cuModuleTest, AC_INV_ModuleLoadFatBinary_LoadNullFatBinary) {
    const void* fatCubin = nullptr;
    CUresult res = cuModuleLoadFatBinary(&module_fatbin, fatCubin);
    EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, res);
}

TEST_F(cuModuleTest, AC_INV_ModuleLoadFatBinary_LoadInvalidFatBinary) {
    const void*
        fatCubin =
            "/data/system/yunfan/cuda_api/common/cuda_kernel/"
            "invalid_fatbin.fatbin";  // 损坏的数据等
    CUresult res = cuModuleLoadFatBinary(&module_fatbin, fatCubin);
    EXPECT_EQ(CUDA_ERROR_INVALID_IMAGE, res);
}

TEST_F(cuModuleTest, AC_INV_ModuleLoadFatBinary_LoadNullModule) {
    const void* fatCubin = fatbin_path;
    CUresult res = cuModuleLoadFatBinary(nullptr, fatCubin);
    EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, res);
}

TEST_F(cuModuleTest, AC_OT_ModuleLoadFatBinary_LoadRepeatedFatBinary) {
    // TODO：待确认
    const void* fatCubin = fatbin_path;
    CUresult res = cuModuleLoadFatBinary(&module_fatbin, fatCubin);
    EXPECT_EQ(CUDA_SUCCESS, res);
    EXPECT_NE(nullptr, module_fatbin);
    CUmodule module2;
    res = cuModuleLoadFatBinary(&module2, fatCubin);
    EXPECT_EQ(CUDA_ERROR_ALREADY_MAPPED, res);
}

TEST_F(cuModuleTest, AC_SA_ModuleLoadFatBinary_LoadAsyncFatBinary) {
    const void* fatCubin = fatbin_path;
    CUstream stream;
    CUresult res = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
    EXPECT_EQ(CUDA_SUCCESS, res);
    res = cuModuleLoadFatBinary(&module_fatbin, fatCubin);
    EXPECT_EQ(CUDA_SUCCESS, res);
    EXPECT_NE(nullptr, module_fatbin);
    res = cuStreamSynchronize(stream);
    EXPECT_EQ(CUDA_SUCCESS, res);
    res = cuStreamDestroy(stream);
    EXPECT_EQ(CUDA_SUCCESS, res);
}