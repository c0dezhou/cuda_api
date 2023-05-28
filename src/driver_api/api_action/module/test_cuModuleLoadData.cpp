#include "module_tests.h"

TEST_F(cuModuleTest, AC_BA_ModuleLoadData_BasicBehavior) {
    // TODO：解决
    FILE* fp = fopen(fname_sm80, "rb");
    EXPECT_NE(fp, nullptr);
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void* image = malloc(size);
    EXPECT_NE(image, nullptr);
    fread(image, size, 1, fp);
    fclose(fp);

    CUmodule module;
    CUresult res = cuModuleLoadData(&module, image);
    EXPECT_EQ(res, CUDA_SUCCESS);
    if (res == CUDA_SUCCESS) {
        CUfunction function;
        res = cuModuleGetFunction(&function, module, "_Z6vecAddPfS_S_");
        EXPECT_EQ(res, CUDA_SUCCESS);
        if (res == CUDA_SUCCESS) {
            int N = 1024;
            float* h_A = new float[N];
            float* h_B = new float[N];
            float* h_C = new float[N];
            for (int i = 0; i < N; i++) {
                h_A[i] = i;
                h_B[i] = i * 2;
            }
            CUdeviceptr d_A, d_B, d_C;
            cuMemAlloc(&d_A, N * sizeof(float));
            cuMemAlloc(&d_B, N * sizeof(float));
            cuMemAlloc(&d_C, N * sizeof(float));
            cuMemcpyHtoD(d_A, h_A, N * sizeof(float));
            cuMemcpyHtoD(d_B, h_B, N * sizeof(float));

            void* args[] = {&d_A, &d_B, &d_C, &N};
            int blockSize = 256;
            int gridSize = (N + blockSize - 1) / blockSize;
            cuLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1, 0,
                           nullptr, args, nullptr);

            cuCtxSynchronize();

            cuMemcpyDtoH(h_C, d_C, N * sizeof(float));
            for (int i = 0; i < N; i++) {
                EXPECT_FLOAT_EQ(h_C[i], h_A[i] + h_B[i]);
            }

            delete[] h_A;
            delete[] h_B;
            delete[] h_C;
            res = cuMemFree(d_A);
            EXPECT_EQ(res, CUDA_SUCCESS);
            res = cuMemFree(d_B);
            EXPECT_EQ(res, CUDA_SUCCESS);
            res = cuMemFree(d_C);
            EXPECT_EQ(res, CUDA_SUCCESS);
        }
        res = cuModuleUnload(module);
        EXPECT_EQ(res, CUDA_SUCCESS);
        free(image);
    }
}

TEST_F(cuModuleTest, AC_INV_ModuleLoadData_InvalidImag) {
    // TODO：待确认
    GTEST_SKIP();  // due to coredump
    CUmodule module;
    CUresult res = cuModuleLoadData(&module, nullptr);
    EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE);
}

TEST_F(cuModuleTest, AC_EG_ModuleLoadData_LargeFile) {
    // TODO: 使用一个最大的文件大小作为参数
    const char* fname_sm80 = "large_file.cubin";
}
