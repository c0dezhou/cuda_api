#include "module_tests.h"

TEST_F(cuModuleTest, AC_BA_ModuleLoadFatBinary_LoadValidFatBinary) {
    // TODO：待确认
    const void* fatCubin = fatbin_path;
    CUresult res = cuModuleLoadFatBinary(&module_fatbin, fatCubin);

    EXPECT_EQ(CUDA_SUCCESS, res);
    EXPECT_NE(nullptr, module_fatbin);
}

// The CUDA Fatbin file is a container file format that bundles together multiple versions of a GPU program into a single file. This file can then be loaded by `cuModuleLoadFatBinary()`. To generate a Fatbin file, you can use the NVIDIA `nvcc` compiler, along with the `-fatbin` option.

// Here's the step-by-step process to do that:

// 1. First, you need to write your CUDA code in a `.cu` file. For instance, `cuda_kernel.cu`.

// 2. After that, use the `nvcc` compiler with the `-fatbin` option to generate the Fatbin file. Here's the command you would use:

//     ```
//     nvcc -fatbin -o cuda_kernel.fatbin cuda_kernel.cu
//     ```

//     This command will generate a Fatbin file named `cuda_kernel.fatbin` from the source file `cuda_kernel.cu`.

// 3. You can then use `cuModuleLoadFatBinary()` to load the Fatbin file in your program. Here's a simple example of how you might do that:

//     ```cpp
//     CUmodule cuModule;
//     CUresult res;
//     // Load the fatbin file
//     char* fatbin;
//     size_t fatbinSize;
//     // ... Load the fatbin file into the fatbin variable ...

//     res = cuModuleLoadFatBinary(&cuModule, fatbin);
//     if (res != CUDA_SUCCESS) {
//         printf("Failed to load module: %d\n", res);
//         return;
//     }
//     ```

//     In the above code, you need to load the Fatbin file into the `fatbin` variable. You could use a function like `fread()` to do that.

// Please note that `-fatbin` is used to create a standalone GPU binary, but the host code (if any) is ignored. If your `.cu` file contains both host and device code, and you want to include host code in your binary, you might need to split your file into separate host and device parts, compile them separately, and then link them together.

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

TEST_F(cuModuleTest, AC_INV_ModuleLoadFatBinary_LoadMismatchedFatBinary) {
    // TODO：待确认
    // TODO: 一个不匹配的fat binary对象，它不包含当前设备的目标架构
    const void* fatCubin = "";
    CUresult res = cuModuleLoadFatBinary(&module_fatbin, fatCubin);
    EXPECT_EQ(CUDA_ERROR_NO_BINARY_FOR_GPU, res);

//     NVIDIA's nvcc compiler allows you to specify multiple target architectures for which you want to compile PTX or cubin code, and embeds them all into a single 'fatbinary'. This means you can generate binaries that target multiple different GPU architectures, not just the one present on the machine where you're compiling.

// Here's how you would generate a fat binary for multiple target architectures:

// ```bash
// nvcc -fatbin -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -o cuda_kernel.fatbin cuda_kernel.cu
// ```

// This command will generate a fat binary (`cuda_kernel.fatbin`) that includes code optimized for both the `sm_60` and `sm_70` architectures, from the source file `cuda_kernel.cu`.

// The `gencode` option allows you to specify a list of architectures for which to generate code. The `arch` option specifies the virtual architecture for which to generate PTX, and the `code` option specifies the real architecture for which to generate cubin.

// Note: You would replace `sm_60` and `sm_70` with the desired compute capabilities of the target architectures. For the most recent list of compute capabilities, you can refer to the CUDA Programming Guide. 

// When your CUDA program runs, the CUDA driver will select the most suitable version of the binary for the GPU on which the program is running. If no suitable version is available (i.e., if the fat binary doesn't contain a version compatible with the GPU), then you will receive an error when you attempt to load the module. 

// Therefore, if you are compiling a fat binary that does not match the target architecture of the current device, you must make sure that the target device is compatible with one of the architectures specified during the compilation process.
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