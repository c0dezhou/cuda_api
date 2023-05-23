#include <cuda.h>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

// CUDA kernel
__global__ void vecAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Result checker
void checkResult(const float* h_C, const float* h_A, const float* h_B, int N) {
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        EXPECT_FLOAT_EQ(h_C[i], expected) << "Error at index " << i;
    }
}


// CUDA operation
void cudaOperation(CUdevice device, CUcontext context, int threadId) {
    // Set the current CUDA context
    cuCtxSetCurrent(context);

    // CUDA operations

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

    // Create CUDA stream
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);

    // Copy data from host to device
    cuMemcpyHtoDAsync(d_A, h_A, size, stream);
    cuMemcpyHtoDAsync(d_B, h_B, size, stream);

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
    cuLaunchKernel(
        vecAdd,
        blocksPerGrid.x, 1, 1,
        threadsPerBlock.x, 1, 1,
        0, stream,
        reinterpret_cast<void**>(&d_A),
        reinterpret_cast<void**>(&d_B),
        reinterpret_cast<void**>(&d_C)
    );

    // Copy result back to host
    cuMemcpyDtoHAsync(h_C, d_C, size, stream);

    // Synchronize stream
    cuStreamSynchronize(stream);

    // Verify result
    // for (int i = 0; i < N; ++i) {
    //     EXPECT_FLOAT_EQ(h_C[i], h_A[i] + h_B[i]);
    // }
    checkResult(h_C, h_A, h_B, N);
    

    // Cleanup
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cuStreamDestroy(stream);
}

// Test fixture class
class CudaMultiThreadTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA and create the shared context
        cuInit(0);
        cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);

        // Create CUDA stream for each thread
        streams.resize(numThreads);
        for (int i = 0; i < numThreads; ++i) {
            cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING);
        }
    }

    void TearDown() override {
        // Destroy CUDA streams
        for (int i = 0; i < numThreads; ++i) {
            cuStreamDestroy(streams[i]);
        }

        // Destroy the shared context
        cuCtxDestroy(context);
    }

    CUdevice device = 0;
    CUcontext context;
    const int numThreads = 4;
    std::vector<CUstream> streams;
};

// Test case to run CUDA operations concurrently using multiple threads
TEST_F(CudaMultiThreadTest, RunConcurrentOperations) {
    std::vector<std::thread> threads;

    // Start threads
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(std::thread(cudaOperation, device, context, i));
    }

    // Wait for threads to finish
    for (auto& thread : threads) {
        thread.join();
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
