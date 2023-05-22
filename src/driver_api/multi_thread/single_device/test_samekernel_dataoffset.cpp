// #include <cuda.h>
// #include <gtest/gtest.h>
// #include <thread>

// void initData(float* data, int size, float value) {
//     for (int i = 0; i < size; i++) {
//         data[i] = value;
//     }
// }

// void checkData(float* data, int size, float expected) {
//     for (int i = 0; i < size; i++) {
//         EXPECT_FLOAT_EQ(data[i], expected) << "error index: " << i;
//     }
// }

// class CudaTest : public ::testing::Test {
//    protected:
//     void SetUp() {
//         CUresult result = cuInit(0);
//         ASSERT_EQ(result, CUDA_SUCCESS);
//         result = cuDeviceGet(&device, 0);
//         ASSERT_EQ(result, CUDA_SUCCESS);

//         result = cuCtxCreate(&context, 0, device);
//         ASSERT_EQ(result, CUDA_SUCCESS);
//         result = cuCtxSetCurrent(context);
//         ASSERT_EQ(result, CUDA_SUCCESS);
//         result = cuModuleLoad(&module,
//                               "/data/system/yunfan/cuda_api/common/cuda_kernel/"
//                               "cuda_kernel_sm_75.ptx");
//         ASSERT_EQ(result, CUDA_SUCCESS);

//         result = cuModuleGetFunction(&function, module, "_Z6vecAddPfS_S_");
//         ASSERT_EQ(result, CUDA_SUCCESS);

//         result = cuMemAlloc(&d_A, N * sizeof(float));
//         ASSERT_EQ(result, CUDA_SUCCESS);
//         result = cuMemAlloc(&d_B, N * sizeof(float));
//         ASSERT_EQ(result, CUDA_SUCCESS);
//         result = cuMemAlloc(&d_C, N * sizeof(float));
//         ASSERT_EQ(result, CUDA_SUCCESS);
//     }

//     void TearDown() {
//         CUresult result = cuMemFree(d_A);
//         ASSERT_EQ(result, CUDA_SUCCESS);
//         result = cuMemFree(d_B);
//         ASSERT_EQ(result, CUDA_SUCCESS);
//         result = cuMemFree(d_C);
//         ASSERT_EQ(result, CUDA_SUCCESS);

//         result = cuModuleUnload(module);
//         ASSERT_EQ(result, CUDA_SUCCESS);
//         result = cuCtxDestroy(context);
//         ASSERT_EQ(result, CUDA_SUCCESS);
//     }

//     // void SetUp() override {
//     //     CUresult result = cuCtxSetCurrent(context);
//     //     ASSERT_EQ(result, CUDA_SUCCESS);
//     // }

//     // void TearDown() override {
//     //     CUresult result = cuCtxSynchronize();
//     //     ASSERT_EQ(result, CUDA_SUCCESS);
//     // }

//     // int N = 1000000;  // size of data
//     int N = 4;    // num of data
//     int M = 256;  // number of threads per block

//     CUdevice device;      // device handle
//     CUcontext context;    // context handle
//     CUmodule module;      // module handle
//     CUfunction function;  // kernel function
//     CUdeviceptr d_A;      // device memory for vector A
//     CUdeviceptr d_B;      // device memory for vector B
//     CUdeviceptr d_C;      // device memory for vector C
// };

// // define the static members of the test fixture class
// // CUdevice CudaTest::device;
// // CUcontext CudaTest::context;
// // CUmodule CudaTest::module;
// // CUfunction CudaTest::function;
// // CUdeviceptr CudaTest::d_A;
// // CUdeviceptr CudaTest::d_B;
// // CUdeviceptr CudaTest::d_C;

// TEST_F(CudaTest, SingleThreadSingleDevice) {
//     float* h_A = (float*)malloc(N * sizeof(float));
//     float* h_B = (float*)malloc(N * sizeof(float));
//     float* h_C = (float*)malloc(N * sizeof(float));

//     initData(h_A, N, 1.0f);
//     initData(h_B, N, 2.0f);
//     initData(h_C, N, 0.0f);

//     CUresult result = cuMemcpyHtoD(d_A, h_A, N * sizeof(float));
//     ASSERT_EQ(result, CUDA_SUCCESS);
//     result = cuMemcpyHtoD(d_B, h_B, N * sizeof(float));
//     ASSERT_EQ(result, CUDA_SUCCESS);

//     void* args[] = {&d_A, &d_B, &d_C};
//     int gridDim = (N + M - 1) / M;
//     result =
//         cuLaunchKernel(function, gridDim, 1, 1, M, 1, 1, 0, NULL, args, NULL);
//     ASSERT_EQ(result, CUDA_SUCCESS);

//     // copy data from device to host
//     result = cuMemcpyDtoH(h_C, d_C, N * sizeof(float));
//     ASSERT_EQ(result, CUDA_SUCCESS);

//     // check data
//     checkData(h_C, N, 3.0f);

//     // free host memory
//     free(h_A);
//     free(h_B);
//     free(h_C);
// }

// TEST_F(CudaTest, MultiThreadSingleDevice) {
//     const int numThreads = 2;
//     // const int segmentSize = N / numThreads;
//     const int segmentSize = 2;

//     float* h_A = (float*)malloc(N * sizeof(float));
//     float* h_B = (float*)malloc(N * sizeof(float));
//     float* h_C = (float*)malloc(N * sizeof(float));

//     initData(h_A, N, 1.0f);
//     initData(h_B, N, 2.0f);
//     initData(h_C, N, 0.0f);

//     std::thread threads[numThreads];
//     CUevent events[numThreads];

//     for (int i = 0; i < numThreads; i++) {
//         CUresult result = cuEventCreate(&events[i], CU_EVENT_DEFAULT);
//         ASSERT_EQ(result, CUDA_SUCCESS);
//         threads[i] = std::thread([=]() {
//             std::cout << "in thread :" << i << std::endl;
//             CUresult result = cuDeviceGet(&device, 0);
//             ASSERT_EQ(result, CUDA_SUCCESS);
//             CUcontext ctx;
//             result = cuCtxCreate(&ctx, 0, device);
//             ASSERT_EQ(result, CUDA_SUCCESS);
//             result = cuCtxSetCurrent(ctx);
//             ASSERT_EQ(result, CUDA_SUCCESS);
//             CUstream stream;
//             result = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
//             ASSERT_EQ(result, CUDA_SUCCESS);
//             result = cuStreamGetCtx(stream, &ctx);
//             ASSERT_EQ(result, CUDA_SUCCESS);

//             CUmodule modulei;      // module handle
//             CUfunction functioni;  // kernel function

//             result = cuModuleLoad(&modulei,
//                                   "/data/system/yunfan/cuda_api/common/"
//                                   "cuda_kernel/cuda_kernel.ptx");
//             ASSERT_EQ(result, CUDA_SUCCESS);

//             result =
//                 cuModuleGetFunction(&functioni, modulei, "_Z6vecAddPfS_S_");
//             ASSERT_EQ(result, CUDA_SUCCESS);

//             int offset = i * segmentSize;  // 0 4
//             // int size = (i == numThreads - 1) ? (N - offset) : segmentSize;
//             int size = segmentSize;

//             // result = cuMemcpyHtoD(d_A + offset, h_A + offset,
//             //                            size * sizeof(float));
//             // ASSERT_EQ(result, CUDA_SUCCESS);
//             // result = cuMemcpyHtoD(d_B + offset, h_B + offset,
//             //                            size * sizeof(float));
//             // ASSERT_EQ(result, CUDA_SUCCESS);
//             result = cuMemcpyHtoDAsync(d_A + offset, h_A + offset,
//                                        size * sizeof(float), stream);
//             ASSERT_EQ(result, CUDA_SUCCESS);
//             result = cuMemcpyHtoDAsync(d_B + offset, h_B + offset,
//                                        size * sizeof(float), stream);
//             ASSERT_EQ(result, CUDA_SUCCESS);

//             result = cuStreamSynchronize(stream);
//             ASSERT_EQ(result, CUDA_SUCCESS);

//             void* args[] = {&d_A, &d_B, &d_C, &N};
//             int gridDim = (size + M - 1) / M;
//             result = cuLaunchKernel(functioni, gridDim, 1, 1, M, 1, 1, 0,
//                                     stream, args, nullptr);
//             ASSERT_EQ(result, CUDA_SUCCESS);

//             result = cuStreamSynchronize(stream);
//             ASSERT_EQ(result, CUDA_SUCCESS);

//             result = cuEventRecord(events[i], stream);
//             ASSERT_EQ(result, CUDA_SUCCESS);

//             result =
//                 cuMemcpyDtoH(h_C + offset, d_C + offset, size * sizeof(float));
//             ASSERT_EQ(result, CUDA_SUCCESS);

//             result = cuModuleUnload(modulei);
//             ASSERT_EQ(result, CUDA_SUCCESS);
//             result = cuStreamDestroy(stream);
//             ASSERT_EQ(result, CUDA_SUCCESS);
//             result = cuCtxDestroy(ctx);
//             ASSERT_EQ(result, CUDA_SUCCESS);
//         });
//     }

//     for (int i = 0; i < numThreads; i++) {
//         threads[i].join();
//     }

//     checkData(h_C, N, 3.0f);

//     free(h_A);
//     free(h_B);
//     free(h_C);
// }