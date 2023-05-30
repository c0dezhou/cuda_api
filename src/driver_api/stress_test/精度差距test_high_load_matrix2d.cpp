#include "test_utils.h"

#define N 1024

class MatrixMultiplicationTest : public ::testing::Test {
   protected:
    float* A;
    float* B;
    float* C_CPU;
    float* C_GPU;
    size_t matrixSize;
    CUdeviceptr d_A;
    CUdeviceptr d_B;
    CUdeviceptr d_C;
    CUcontext cuContext;
    CUdevice cuDevice;

    void SetUp() override {
        matrixSize = N * N * sizeof(float);

        A = new float[N * N];
        B = new float[N * N];
        C_CPU = new float[N * N];
        C_GPU = new float[N * N];

        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        for (int i = 0; i < N * N; ++i) {
            A[i] = distribution(generator);
            B[i] = distribution(generator);
            C_CPU[i] = 0.0f;
            C_GPU[i] = 0.0f;
        }

        checkError(cuInit(0));
        checkError(cuDeviceGet(&cuDevice, 0));
        checkError(cuCtxCreate(&cuContext, 0, cuDevice));

        checkError(cuMemAlloc(&d_A, matrixSize));
        checkError(cuMemAlloc(&d_B, matrixSize));
        checkError(cuMemAlloc(&d_C, matrixSize));

        checkError(cuMemcpyHtoD(d_A, A, matrixSize));
        checkError(cuMemcpyHtoD(d_B, B, matrixSize));
    }

    void TearDown() override {
        delete[] A;
        delete[] B;
        delete[] C_CPU;
        delete[] C_GPU;
        checkError(cuMemFree(d_A));
        checkError(cuMemFree(d_B));
        checkError(cuMemFree(d_C));
        checkError(cuCtxDestroy(cuContext));
    }
};

TEST_F(MatrixMultiplicationTest, MatrixMultiplyCPUvsGPUTest) {
        auto cpuStartTime = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C_CPU[i * N + j] = sum;
            }
        }
        auto cpuEndTime = std::chrono::high_resolution_clock::now();
        auto cpuDuration =
            std::chrono::duration_cast<std::chrono::microseconds>(cpuEndTime -
                                                                  cpuStartTime)
                .count();

        const char*
            ptx =
                "/data/system/yunfan/cuda_api/common/cuda_kernel/"
                "stress_kernel.ptx";

        CUmodule cuModule;
        checkError(cuModuleLoad(&cuModule, ptx));

        CUfunction cuFunction;
        checkError(cuModuleGetFunction(&cuFunction, cuModule,
                                       "_Z20matrixMultiplyKernelPKfS0_Pfiii"));

        int size = N;
        void* params[] = {&d_A, &d_B, &d_C, &size, &size, &size};
        auto gpuStartTime = std::chrono::high_resolution_clock::now();
        checkError(cuLaunchKernel(cuFunction, N / 32, N / 32, 1, 32, 32, 1, 0, 0, params,
                       0));
        auto gpuEndTime = std::chrono::high_resolution_clock::now();
        auto gpuDuration =
            std::chrono::duration_cast<std::chrono::microseconds>(gpuEndTime -
                                                                  gpuStartTime)
                .count();

        checkError(cuMemcpyDtoH(C_GPU, d_C, matrixSize));

        checkError(cuModuleUnload(cuModule));

        for (int i = 0; i < N * N; ++i) {
            EXPECT_FLOAT_EQ(C_CPU[i], C_GPU[i]);
        }

        std::cout << "Memory Usage:\n";
        std::cout << "  CPU: " << N * N * sizeof(float) * 3 / (1024 * 1024)
                  << " MB\n";
        std::cout << "  GPU: " << N * N * sizeof(float) * 3 / (1024 * 1024)
                  << " MB\n";
        std::cout << "Calculation Time:\n";
        std::cout << "  CPU: " << cpuDuration << " microseconds\n";
        std::cout << "  GPU: " << gpuDuration << " microseconds\n";

}
