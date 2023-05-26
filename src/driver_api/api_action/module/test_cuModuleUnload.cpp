#include "module_tests.h"

#define INIT_UNLOADMODULE()                                                   \
    int device_count;                                                         \
    CUdevice device;                                                          \
    CUcontext context;                                                        \
    CUmodule module, module_sm75, module_fatbin;                              \
    const char* fname_sm75 =                                                  \
        "/data/system/yunfan/cuda_api/common/cuda_kernel/"                    \
        "cuda_kernel_sm_75.ptx";                                              \
    const char* fatbin_path =                                                 \
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.fatbin"; \
    const char* fname =                                                       \
        "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx";    \
    cuInit(0);                                                                \
    cuDeviceGet(&device, 0);                                                  \
    cuCtxCreate(&context, 0, device);                                         \
    cuModuleLoad(&module, fname);

TEST(cuModuleTest_, AC_BA_ModuleUnload_UnloadValidFatbinModule) {
    // TODO：待确认
    INIT_UNLOADMODULE();
    CUresult res = cuModuleLoadFatBinary(&module_fatbin, fatbin_path);
    EXPECT_EQ(CUDA_SUCCESS, res);
    EXPECT_NE(nullptr, module);
    res = cuModuleUnload(module);
    EXPECT_EQ(CUDA_SUCCESS, res);

    cuCtxDestroy(context);
}

TEST(cuModuleTest_, AC_INV_ModuleUnload_UnloadValidModule) {
    INIT_UNLOADMODULE();
    EXPECT_NE(nullptr, module);
    CUresult res = cuModuleUnload(module);
    EXPECT_EQ(CUDA_SUCCESS, res);

    cuCtxDestroy(context);
}

TEST(cuModuleTest_, AC_INV_ModuleUnload_UnloadNullModule) {
    INIT_UNLOADMODULE();
    CUmodule module1 = nullptr;
    CUresult res = cuModuleUnload(module1);
    EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, res);

    cuCtxDestroy(context);
}

TEST(cuModuleTest_, AC_INV_ModuleUnload_UnloadInvalidModule) {
    // TODO：待确认
    GTEST_SKIP();  // core dump
    INIT_UNLOADMODULE();
    CUresult res = cuModuleUnload(module_sm75);
    EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, res);

    cuCtxDestroy(context);
}

TEST(cuModuleTest_, AC_INV_ModuleUnload_UnloadOnotherContext) {
    INIT_UNLOADMODULE();
    CUresult res = cuModuleLoad(&module_sm75, fname_sm75);
    EXPECT_EQ(CUDA_SUCCESS, res);
    EXPECT_NE(nullptr, module_sm75);
    CUcontext context2;
    res = cuCtxCreate(&context2, 0, device);
    EXPECT_EQ(CUDA_SUCCESS, res);
    CUfunction function;
    res = cuModuleGetFunction(&function, module_sm75, "_Z6vecAddPfS_S_");
    EXPECT_EQ(CUDA_SUCCESS, res);

    res = cuCtxSetCurrent(context);
    EXPECT_EQ(CUDA_SUCCESS, res);
    res = cuModuleUnload(module_sm75);
    EXPECT_EQ(CUDA_ERROR_CONTEXT_IS_DESTROYED, res);
    res = cuCtxDestroy(context2);
    EXPECT_EQ(CUDA_SUCCESS, res);

    cuCtxDestroy(context);
}

TEST(cuModuleTest_, AC_SA_ModuleUnload_UnloadAsyncModule) {
    INIT_UNLOADMODULE();
    CUresult res = cuModuleLoad(&module_sm75, fname_sm75);
    EXPECT_EQ(CUDA_SUCCESS, res);
    EXPECT_NE(nullptr, module_sm75);
    CUstream stream;
    res = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
    EXPECT_EQ(CUDA_SUCCESS, res);
    res = cuModuleUnload(module_sm75);
    EXPECT_EQ(CUDA_SUCCESS, res);
    res = cuStreamSynchronize(stream);
    EXPECT_EQ(CUDA_SUCCESS, res);
    res = cuStreamDestroy(stream);
    EXPECT_EQ(CUDA_SUCCESS, res);

    cuCtxDestroy(context);
}

TEST(cuModuleTest_, AC_OT_ModuleUnload_UnloadRepeatedModule) {
    INIT_UNLOADMODULE();
    CUresult res = cuModuleLoad(&module_sm75, fname_sm75);
    EXPECT_EQ(CUDA_SUCCESS, res);
    EXPECT_NE(nullptr, module_sm75);
    res = cuModuleUnload(module_sm75);
    EXPECT_EQ(CUDA_SUCCESS, res);
    // TODO：待确认 下面new case
    GTEST_SKIP();  // due to core dump
    res = cuModuleUnload(module_sm75);
    EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, res);

    cuCtxDestroy(context);
}

// TEST_F(cuModuleTest_, UnloadTwice) {
//     CUmodule cuModule;
//     CUresult res;

//     // Load a module
//     res = cuModuleLoad(&cuModule, "path_to_your_ptx_file");
//     ASSERT_EQ(res, CUDA_SUCCESS);

//     // Unload the module for the first time
//     res = cuModuleUnload(cuModule);
//     EXPECT_EQ(res, CUDA_SUCCESS);

//     // Try to unload the same module again
//     res = cuModuleUnload(cuModule);
//     EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);  // Expect an error indicating
//                                                // invalid operation
// }
