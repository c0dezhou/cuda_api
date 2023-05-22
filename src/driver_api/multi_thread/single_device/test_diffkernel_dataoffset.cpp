

// void initData(float* data, int size, float value) {
//     for (int i = 0; i < size; i++) {
//         data[i] = value;
//     }
// }

// void checkData(float* data, int size, float expected) {
//     // for (int i = 0; i < size; i++) {
//         EXPECT_FLOAT_EQ(*data, expected);
//     // }
// }



// TEST_F(MthTest, MultiThreadSingleDeviceMultiStreamInterleaved) {
//     checkError(cuCtxSetCurrent(context));

//     checkError(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

//     int size = N * sizeof(float);
//     const int numThreads = 2;
//     size_t segmentSize = sizeof(float);

//     float* h_A = (float*)malloc(N * sizeof(float));
//     float* h_B = (float*)malloc(N * sizeof(float));
//     float* h_C = (float*)malloc(N * sizeof(float));

//     initData(h_A, N, 1.0f);
//     initData(h_B, N, 2.0f);
//     initData(h_C, N, 0.0f);

//     std::thread threads[numThreads];
//     CUevent events[numThreads];

//     for (int i = 0; i < numThreads; i++) {
//         checkError(cuEventCreate(&events[i], CU_EVENT_DEFAULT));

//         threads[i] = std::thread([=]() {
//             CUcontext ctx;
//             checkError(cuCtxSetCurrent(context));
//             checkError(cuCtxGetCurrent(&ctx));
//             ASSERT_EQ(ctx, context);

//             CUstream strm;
//             checkError(cuStreamCreate(&strm, CU_STREAM_NON_BLOCKING));
//             checkError(cuStreamGetCtx(strm, &context));

//             size_t offset = i * 1;  // 0, 1, 2, 3
//             // int size = (i == numThreads - 1) ? (N - offset) : segmentSize;
//             // size_t segmentSize = sizeof(float);

//             checkError(cuMemcpyHtoDAsync(d_A + offset, h_A + offset,
//                                          segmentSize, strm));
//             checkError(cuMemcpyHtoDAsync(d_B + offset, h_B + offset,
//                                          segmentSize, strm));

//             void* args1[] = {&d_A, &d_B, &d_C};
//             int gridDim1 = (size + M - 1) / M;

//             int gridDimX = 1, gridDimY = 1, gridDimZ = 1;
//             int blockDimX = 2, blockDimY = 1, blockDimZ = 1;
//             checkError(cuLaunchKernel(vecAdd, gridDimX, gridDimY, gridDimZ,
//                                       blockDimX, blockDimY, blockDimZ, 0, strm,
//                                       args1, NULL));

//             float factor = i + 1.0f;
//             void* args2[] = {&d_C, &factor};
//             int gridDim2 = (size + M - 1) / M;
//             checkError(cuLaunchKernel(
//                 vecScal, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, strm,
//                 args2, NULL));

//             checkError(cuMemcpyDtoHAsync(h_C + offset, d_C + offset,
//                                          segmentSize, strm));
//             checkError(cuEventRecord(events[i], strm));
//             checkError(cuStreamSynchronize(strm));
//             checkError(cuStreamDestroy(strm));
//         });
//     }

//     for (int i = 0; i < numThreads; i++) {
//         threads[i].join();
//         CUresult result = cuEventDestroy(events[i]);
//         ASSERT_EQ(result, CUDA_SUCCESS);
//     }

//     for (int i = 0; i < numThreads; i++) {
//         int offset = i * segmentSize;
//         float expected = (i + 1.0f) * (i + 3.0f);  // factor * (A[i] + B[i])
//         checkData(h_C + offset, segmentSize, expected);
//     }

    
//     checkError(cuStreamDestroy(stream));
// }
