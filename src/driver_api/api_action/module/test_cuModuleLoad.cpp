#include "module_tests.h"

TEST_F(cuModuleTest, AC_BA_ModuleLoad_BasicBehavior) {
    CUresult res = cuModuleLoad(&module_sm75, fname_sm75);
    EXPECT_EQ(res, CUDA_SUCCESS);
    if (res == CUDA_SUCCESS) {
        CUfunction function;
        res = cuModuleGetFunction(&function, module_sm75, "_Z6vecAddPfS_S_");
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
            res = cuMemAlloc(&d_A, N * sizeof(float));
            EXPECT_EQ(res, CUDA_SUCCESS);
            res = cuMemAlloc(&d_B, N * sizeof(float));
            EXPECT_EQ(res, CUDA_SUCCESS);
            res = cuMemAlloc(&d_C, N * sizeof(float));
            EXPECT_EQ(res, CUDA_SUCCESS);
            res = cuMemcpyHtoD(d_A, h_A, N * sizeof(float));
            EXPECT_EQ(res, CUDA_SUCCESS);
            res = cuMemcpyHtoD(d_B, h_B, N * sizeof(float));
            EXPECT_EQ(res, CUDA_SUCCESS);

            void* args[] = {&d_A, &d_B, &d_C, &N};
            int blockSize = 256;
            int gridSize = (N + blockSize - 1) / blockSize;

            res = cuLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1, 0,
                                 nullptr, args, nullptr);
            EXPECT_EQ(res, CUDA_SUCCESS);

            if (res == CUDA_SUCCESS) {
                res = cuMemcpyDtoH(h_C, d_C, N * sizeof(float));
                EXPECT_EQ(res, CUDA_SUCCESS);
                for (int i = 0; i < N; i++) {
                    EXPECT_FLOAT_EQ(h_C[i], h_A[i] + h_B[i]);
                }
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
        res = cuModuleUnload(module_sm75);
        EXPECT_EQ(res, CUDA_SUCCESS);
    }
}

TEST_F(cuModuleTest, AC_INV_ModuleLoad_INvalidfname) {
    const char* fname = "invalid_file";

    CUresult res = cuModuleLoad(&module_sm75, fname);
    EXPECT_EQ(res, CUDA_ERROR_FILE_NOT_FOUND);
}

TEST_F(cuModuleTest, AC_EG_ModuleLoad_Longfilename) {
    const char* fname =
        "/data/system/yunfan/cuda_api/common/cuda_kernel/"
        "cuda_kernel_long_long_long_long_long_long_long_long_long_long_long_"
        "long_long.ptx";

    CUresult res = cuModuleLoad(&module_sm75, fname);
    if (res == CUDA_SUCCESS) {
        CUfunction function;
        res = cuModuleGetFunction(&function, module_sm75, "_Z6vecAddPfS_S_");
        EXPECT_EQ(res, CUDA_SUCCESS);
        res = cuModuleUnload(module_sm75);
        EXPECT_EQ(res, CUDA_SUCCESS);
    } else {
        CUfunction function;
        res = cuModuleGetFunction(&function, module_sm75, "_Z6vecAddPfS_S_");
        EXPECT_NE(res, CUDA_SUCCESS);
    }
}

TEST_F(cuModuleTest, AC_SA_ModuleLoad_SyncBehavior) {
    CUresult res = cuModuleLoad(&module_sm75, fname_sm75);
    EXPECT_EQ(res, CUDA_SUCCESS);
    if (res == CUDA_SUCCESS) {
        CUfunction function;
        res = cuModuleGetFunction(&function, module_sm75, "_Z6vecAddPfS_S_");
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
            res = cuMemAlloc(&d_A, N * sizeof(float));
            EXPECT_EQ(res, CUDA_SUCCESS);
            res = cuMemAlloc(&d_B, N * sizeof(float));
            EXPECT_EQ(res, CUDA_SUCCESS);
            res = cuMemAlloc(&d_C, N * sizeof(float));
            EXPECT_EQ(res, CUDA_SUCCESS);
            res = cuMemcpyHtoD(d_A, h_A, N * sizeof(float));
            EXPECT_EQ(res, CUDA_SUCCESS);
            res = cuMemcpyHtoD(d_B, h_B, N * sizeof(float));
            EXPECT_EQ(res, CUDA_SUCCESS);
            void* args[] = {&d_A, &d_B, &d_C, &N};
            int blockSize = 256;
            int gridSize = (N + blockSize - 1) / blockSize;
            CUdeviceptr d_D;
            res = cuMemAlloc(&d_D, N * sizeof(float));
            EXPECT_EQ(res, CUDA_SUCCESS);

            res = cuLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1, 0,
                                 nullptr, args, nullptr);
            EXPECT_EQ(res, CUDA_SUCCESS);
            if (res == CUDA_SUCCESS) {
                res = cuMemcpyDtoH(h_C, d_C, N * sizeof(float));
                EXPECT_EQ(res, CUDA_SUCCESS);
                for (int i = 0; i < N; i++) {
                    EXPECT_FLOAT_EQ(h_C[i], h_A[i] + h_B[i]);
                }
            }
            res = cuMemsetD8(d_D, 0, N * sizeof(float));
            EXPECT_EQ(res, CUDA_SUCCESS);
            float* h_D = new float[N];
            res = cuMemcpyDtoH(h_D, d_D, N * sizeof(float));
            EXPECT_EQ(res, CUDA_SUCCESS);
            for (int i = 0; i < N; i++) {
                EXPECT_FLOAT_EQ(h_D[i], 0);
            }

            delete[] h_A;
            delete[] h_B;
            delete[] h_C;
            delete[] h_D;
            res = cuMemFree(d_A);
            EXPECT_EQ(res, CUDA_SUCCESS);
            res = cuMemFree(d_B);
            EXPECT_EQ(res, CUDA_SUCCESS);
            res = cuMemFree(d_C);
            EXPECT_EQ(res, CUDA_SUCCESS);
            res = cuMemFree(d_D);
            EXPECT_EQ(res, CUDA_SUCCESS);
        }
        res = cuModuleUnload(module_sm75);
        EXPECT_EQ(res, CUDA_SUCCESS);
    }
}

TEST_F(cuModuleTest, AC_OT_ModuleLoad_RepeatedCall) {
    CUmodule module1, module2;
    CUresult res = cuModuleLoad(&module1, fname_sm75);
    EXPECT_EQ(res, CUDA_SUCCESS);
    res = cuModuleLoad(&module2, fname_sm75);
    EXPECT_EQ(res, CUDA_SUCCESS);
    if (res == CUDA_SUCCESS) {
        CUfunction function1, function2;
        res = cuModuleGetFunction(&function1, module1, "_Z6vecAddPfS_S_");
        EXPECT_EQ(res, CUDA_SUCCESS);
        res = cuModuleGetFunction(&function2, module2, "_Z3addiiPi");
        EXPECT_EQ(res, CUDA_SUCCESS);
        EXPECT_NE(function1, function2);
        res = cuModuleUnload(module1);
        EXPECT_EQ(res, CUDA_SUCCESS);
        res = cuModuleUnload(module2);
        EXPECT_EQ(res, CUDA_SUCCESS);
        const char* fname1 = fname_sm75;
        const char* fname2 =
            "/data/system/yunfan/cuda_api/common/cuda_kernel/"
            "cuda_kernel_long_long_long_long_long_long_long_long_long_long_"
            "long_long_long.cubin";
        CUmodule module1, module2;
        res = cuModuleLoad(&module1, fname1);
        EXPECT_EQ(res, CUDA_SUCCESS);
        res = cuModuleLoad(&module2, fname2);
        EXPECT_EQ(res, 300);

        if (res == CUDA_SUCCESS) {
            CUfunction function1, function2;
            res = cuModuleGetFunction(&function1, module1, "_Z6vecAddPfS_S_");
            EXPECT_EQ(res, CUDA_SUCCESS);
            res = cuModuleGetFunction(&function2, module2, "add");
            EXPECT_EQ(res, 300);
            EXPECT_NE(function1, function2);
            res = cuModuleUnload(module1);
            EXPECT_EQ(res, CUDA_SUCCESS);
            res = cuModuleUnload(module2);
            EXPECT_EQ(res, CUDA_SUCCESS);
        }
    }
}