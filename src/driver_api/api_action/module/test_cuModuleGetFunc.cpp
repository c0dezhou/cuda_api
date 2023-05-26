#include "module_tests.h"

TEST_F(cuModuleTest, AC_BA_ModuleGetFunction_BasicBehavior) {
    CUfunction function;

    CUresult result =
        cuModuleGetFunction(&function, module, "_Z18arraySelfIncrementPii");
    EXPECT_EQ(result, CUDA_SUCCESS);
    EXPECT_NE(function, nullptr);
}

TEST_F(cuModuleTest, AC_INV_ModuleGetFunction_InvalidhFunc) {
    EXPECT_EQ(cuModuleGetFunction(nullptr, module, "_Z18arraySelfIncrementPii"),
              CUDA_ERROR_INVALID_HANDLE);
}

TEST_F(cuModuleTest, AC_INV_ModuleGetFunction_Invalidhmod) {
    CUfunction function;

    EXPECT_EQ(
        cuModuleGetFunction(&function, nullptr, "_Z18arraySelfIncrementPii"),
        CUDA_ERROR_INVALID_HANDLE);
}

TEST_F(cuModuleTest, AC_INV_ModuleGetFunction_Invalidfname) {
    CUfunction function;

    EXPECT_EQ(cuModuleGetFunction(&function, module, nullptr),
              CUDA_ERROR_INVALID_VALUE);
}

TEST_F(cuModuleTest, AC_INV_ModuleGetFunction_Invalidfuncname) {
    CUfunction function;

    EXPECT_EQ(cuModuleGetFunction(&function, module, "not_exist"),
              CUDA_ERROR_NOT_FOUND);
}

TEST_F(cuModuleTest, AC_EG_ModuleGetFunction_nullstr) {
    CUfunction function;

    EXPECT_EQ(cuModuleGetFunction(&function, module, ""),
              CUDA_ERROR_INVALID_VALUE);
}

TEST_F(cuModuleTest, AC_OT_ModuleGetFunction_DifferentModulesAndFunctions) {
    CUfunction function1, function2, function3;

    CUresult result1 =
        cuModuleGetFunction(&function1, module, "_Z9addKernelPiPKiS1_");
    CUresult result2 = cuModuleGetFunction(&function2, module, "_Z3addiiPi");

    EXPECT_EQ(result1, CUDA_SUCCESS);
    EXPECT_EQ(result2, CUDA_SUCCESS);
    EXPECT_NE(function1, nullptr);
    EXPECT_NE(function2, nullptr);
    EXPECT_NE(function1, function2);

    CUmodule module2;
    cuModuleLoad(&module2,
                 "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                 "cuda_kernel_sm_75.ptx");

    CUresult result3 =
        cuModuleGetFunction(&function3, module2, "_Z6vecAddPfS_S_");

    EXPECT_EQ(result3, CUDA_SUCCESS);
    EXPECT_NE(function3, nullptr);
    EXPECT_NE(function3, function1);
    EXPECT_NE(function3, function2);

    cuModuleUnload(module2);
}

TEST_F(cuModuleTest, AC_OT_ModuleGetFunction_RepeatedCalls) {
    CUfunction function1, function2, function3;

    CUresult result1 =
        cuModuleGetFunction(&function1, module, "_Z9addKernelPiPKiS1_");

    EXPECT_EQ(result1, CUDA_SUCCESS);
    EXPECT_NE(function1, nullptr);

    CUresult result2 =
        cuModuleGetFunction(&function2, module, "_Z9addKernelPiPKiS1_");

    EXPECT_EQ(result2, CUDA_SUCCESS);
    EXPECT_NE(function2, nullptr);
    EXPECT_EQ(function2, function1);

    CUresult result3 = cuModuleGetFunction(&function3, module, "_Z3addiiPi");

    EXPECT_EQ(result3, CUDA_SUCCESS);
    EXPECT_NE(function3, nullptr);
    EXPECT_NE(function3, function1);
    EXPECT_NE(function3, function2);
}
