#include <cuda.h>
#include <gtest/gtest.h>
#include <thread>

// CUDA operation 1
void cudaOperation1(CUdevice device, CUcontext context) {
    // Set the current CUDA context
    cuCtxSetCurrent(context);

    // CUDA operations using device 1

    // Device-specific initialization
    CUmodule module;
    CUfunction kernel;
    cuModuleLoad(&module, "/path/to/kernel1.ptx");
    cuModuleGetFunction(&kernel, module, "vecAdd");

    // Example data
    const int N = 100;
    const int size = N * sizeof(float);
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    CUdeviceptr d_A, d_B, d_C;
    cuMemAlloc(&d_A, size);
    cuMemAlloc(&d_B, size);
    cuMemAlloc(&d_C, size);

    // Copy data from host to device
    cuMemcpyHtoD(d_A, h_A, size);
    cuMemcpyHtoD(d_B, h_B, size);

    // Launch kernel
    void* args[] = {&d_A, &d_B, &d_C};
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    cuLaunchKernel(kernel, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, NULL, args, NULL);

    // Copy result back to host
    cuMemcpyDtoH(h_C, d_C, size);

    // Verify result
    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(h_C[i], h_A[i] + h_B[i]);
    }

    // Cleanup
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cuModuleUnload(module);
}

// CUDA operation 2
void cudaOperation2(CUdevice device, CUcontext context) {
    // Set the current CUDA context
    cuCtxSetCurrent(context);

    // CUDA operations using device 2

    // Device-specific initialization
    CUmodule module;
    CUfunction kernel;
    cuModuleLoad(&module, "/path/to/kernel2.ptx");
    cuModuleGetFunction(&kernel, module, "vecAdd");

    // Example data
    const int N = 100;
    const int size = N * sizeof(float);
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.5f;
        h_B[i] = 2.5f;
    }

    // Allocate device memory
    CUdeviceptr d_A, d_B, d_C;
    cuMemAlloc(&d_A, size);
    cuMemAlloc(&d_B, size);
    cuMemAlloc(&d_C, size);

    // Copy data from
// host to device
cuMemcpyHtoD(d_A, h_A, size);
cuMemcpyHtoD(d_B, h_B, size);
// Launch kernel
void* args[] = {&d_A, &d_B, &d_C};
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
cuLaunchKernel(kernel, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, NULL, args, NULL);

// Copy result back to host
cuMemcpyDtoH(h_C, d_C, size);

// Verify result
for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(h_C[i], h_A[i] + h_B[i]);
}

// Cleanup
cuMemFree(d_A);
cuMemFree(d_B);
cuMemFree(d_C);
free(h_A);
free(h_B);
free(h_C);
cuModuleUnload(module);
}

// CUDA operation 3
void cudaOperation3(CUdevice device, CUcontext context) {
// Set the current CUDA context
cuCtxSetCurrent(context);
// CUDA operations using device 1 or 2

// Device-specific initialization
CUmodule module;
CUfunction kernel;
cuModuleLoad(&module, "/path/to/kernel1.ptx"); // You can load kernel1 or kernel2 PTX file here
cuModuleGetFunction(&kernel, module, "vecAdd");

// Example data
const int N = 100;
const int size = N * sizeof(float);
float* h_A = (float*)malloc(size);
float* h_B = (float*)malloc(size);
float* h_C = (float*)malloc(size);

// Initialize input data
for (int i = 0; i < N; ++i) {
    h_A[i] = 2.0f;
    h_B[i] = 3.0f;
}

// Allocate device memory
CUdeviceptr d_A, d_B, d_C;
cuMemAlloc(&d_A, size);
cuMemAlloc(&d_B, size);
cuMemAlloc(&d_C, size);

// Copy data from host to device
cuMemcpyHtoD(d_A, h_A, size);
cuMemcpyHtoD(d_B, h_B, size);

// Launch kernel
void* args[] = {&d_A, &d_B, &d_C};
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
cuLaunchKernel(kernel, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, NULL, args, NULL);

// Copy result back to host
cuMemcpyDtoH(h_C, d_C, size);

// Verify result
for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(h_C[i], h_A[i] + h_B[i]);
}

// Cleanup
cuMemFree(d_A);
cuMemFree(d_B);
cuMemFree(d_C);
free(h_A);
free(h_B);
free(h_C);
cuModuleUnload(module);
}

// Test fixture class
class CudaMultiThreadTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA and create the shared context
        cuInit(0);
        cuCtxCreate(&context, CU_CTX_SCHED_AUTO, 0);

        // Create CUDA devices
        cuDeviceGet(&device1, 0);
        cuDeviceGet(&device2, 1);
    }

    void TearDown() override {
        // Destroy the shared context
        cuCtxDestroy(context);
    }

    CUcontext context;    // Shared CUDA context
    CUdevice device1;     // CUDA device 1
    CUdevice device2;     // CUDA device 2
};

// Test case 1: Perform CUDA operation 1 on device 1
TEST_F(CudaMultiThreadTest, CudaOperation1) {
    std::thread thread1(cudaOperation1, device1, context);
    thread1.join();
}

// Test case 2: Perform CUDA operation 2 on device 2
TEST_F(CudaMultiThreadTest, CudaOperation2) {
    std::thread thread2(cudaOperation2, device2, context);
    thread2.join();
}

// Test case 3: Perform CUDA operation 3 using both devices
TEST_F(CudaMultiThreadTest, CudaOperation3) {
    std::thread thread1(cudaOperation3, device1, context);
    std::thread thread2(cudaOperation3, device2, context);
    thread1.join();
    thread2.join();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}