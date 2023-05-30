
#include "test_utils.h"

class StreamOverlapTest : public ::testing::Test {
   protected:
    void SetUp() override {
        checkError(cuInit(0));
        checkError(cuDeviceGet(&device_, 0));
        checkError(cuCtxCreate(&context_, 0, device_));
        checkError(cuModuleLoad(
            &module_,
            "/data/system/yunfan/cuda_api/common/cuda_kernel/cuda_kernel.ptx"));

        checkError(cuEventCreate(&startEvent1, CU_EVENT_DEFAULT));
        checkError(cuEventCreate(&endEvent1, CU_EVENT_DEFAULT));
        checkError(cuEventCreate(&startEvent2, CU_EVENT_DEFAULT));
        checkError(cuEventCreate(&endEvent2, CU_EVENT_DEFAULT));
    }

    void TearDown() override {
        // Destroy the events
        checkError(cuEventDestroy(startEvent1));
        checkError(cuEventDestroy(endEvent1));
        checkError(cuEventDestroy(startEvent2));
        checkError(cuEventDestroy(endEvent2));
        checkError(cuModuleUnload(module_));
        checkError(cuCtxDestroy(context_));
    }

    CUdevice device_;
    CUcontext context_;
    CUmodule module_;
    CUevent startEvent1, startEvent2, endEvent1, endEvent2;
};

TEST_F(StreamOverlapTest, ScenarioOne) {
    CUstream stream1, stream2;
    checkError(cuStreamCreate(&stream1, CU_STREAM_DEFAULT));
    checkError(cuStreamCreate(&stream2, CU_STREAM_DEFAULT));
    CUfunction kernel1, kernel2;
    checkError(cuModuleGetFunction(&kernel1, module_, "_Z9vec_add_3Pfi"));
    checkError(cuModuleGetFunction(&kernel2, module_, "_Z7vec_powPfi"));
    int N = 1000;
    float *h_data1, *h_data2, *d_data1, *d_data2;
    checkError(cuMemAllocHost((void**)&h_data1, N * sizeof(float)));
    checkError(cuMemAllocHost((void**)&h_data2, N * sizeof(float)));
    checkError(cuMemAlloc((CUdeviceptr*)&d_data1, N * sizeof(float)));
    checkError(cuMemAlloc((CUdeviceptr*)&d_data2, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        h_data1[i] = i;
        h_data2[i] = i + 1;
    }
    checkError(cuMemcpyHtoDAsync((CUdeviceptr)d_data1, h_data1,
                                 N * sizeof(float), stream1));
    checkError(cuMemcpyHtoDAsync((CUdeviceptr)d_data2, h_data2,
                                 N * sizeof(float), stream2));

    void* args1[] = {&d_data1, &N};
    void* args2[] = {&d_data2, &N};
    checkError(cuEventRecord(startEvent1, NULL));
    checkError(cuLaunchKernel(kernel1, N / 256 +1, 1, 1, 256, 1, 1, 0, stream1,
                              args1, NULL));
    checkError(cuEventRecord(endEvent1, NULL));
    checkError(cuEventSynchronize(endEvent1));

    checkError(cuEventRecord(startEvent2, NULL));
    checkError(cuLaunchKernel(kernel2, N / 256 +1, 1, 1, 256, 1, 1, 0, stream2,
                              args2, NULL));
    checkError(cuEventRecord(endEvent2, NULL));
    checkError(cuEventSynchronize(endEvent2));

    float timeKernel1, timeKernel2, totalTime;
    checkError(cuEventElapsedTime(&timeKernel1, startEvent1, endEvent1));
    checkError(cuEventElapsedTime(&timeKernel2, startEvent2, endEvent2));

    float overlaptotalTime = std::max(timeKernel1, timeKernel2);
    float actualtotalTime;
    checkError(cuEventElapsedTime(&actualtotalTime, endEvent2, startEvent1));

    std::cout << "Kernel 1 execution time: " << timeKernel1 << " ms"
              << std::endl;
    std::cout << "Kernel 2 execution time: " << timeKernel2 << " ms"
              << std::endl;

    std::cout << "overlaptotalTime execution time: " << overlaptotalTime
              << " ms" << std::endl;
    std::cout << "actualtotalTime execution time: " << overlaptotalTime << " ms"
              << std::endl;

    if (overlaptotalTime > actualtotalTime) {
        std::cout << "Kernels overlapped" << std::endl;
    } else {
        std::cout << "Kernels did not overlap" << std::endl;
    }

    checkError(cuMemcpyDtoHAsync(h_data1, (CUdeviceptr)d_data1,
                                 N * sizeof(float), stream1));
    checkError(cuMemcpyDtoHAsync(h_data2, (CUdeviceptr)d_data2,
                                 N * sizeof(float), stream2));

    checkError(cuStreamSynchronize(stream1));
    checkError(cuStreamSynchronize(stream2));

    for (int i = 0; i < N; i++) {
        ASSERT_FLOAT_EQ(h_data1[i], i + 3) << h_data1[i-1];
        ASSERT_FLOAT_EQ(h_data2[i], (i + 1) * (i + 1));
    }

    checkError(cuMemFreeHost(h_data1));
    checkError(cuMemFreeHost(h_data2));
    checkError(cuMemFree((CUdeviceptr)d_data1));
    checkError(cuMemFree((CUdeviceptr)d_data2));

    checkError(cuStreamDestroy(stream1));
    checkError(cuStreamDestroy(stream2));
}

TEST_F(StreamOverlapTest, ScenarioTwo) {
    cuCtxSetCurrent(context_)
    CUstream stream;
    checkError(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    CUfunction kernel;
    checkError(cuModuleGetFunction(&kernel, module_,
                                   "_Z22vec_multiply_2_withidxPfii"));

    int N = 1000;
    int M = 256;
    float *h_data, *d_data;
    checkError(cuMemAllocHost((void**)&h_data, N * sizeof(float)));
    checkError(cuMemAlloc((CUdeviceptr*)&d_data, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }

    for (int i = 0; i < N; i += M) {
        checkError(cuMemcpyHtoDAsync((CUdeviceptr)d_data + i, h_data + i,
                                     M * sizeof(float), stream));
        void* args[] = {&d_data, &i, &M};
        checkError(cuLaunchKernel(kernel, M / 256, 1, 1, 256, 1, 1, 0, stream,
                                  args, NULL));
    }
    checkError(cuMemcpyDtoHAsync(h_data, (CUdeviceptr)d_data, N * sizeof(float),
                                 stream));

    checkError(cuStreamSynchronize(stream));

    for (int i = 0; i < N; i++) {
        ASSERT_FLOAT_EQ(h_data[i], i * i);
    }

    checkError(cuMemFreeHost(h_data));
    checkError(cuMemFree((CUdeviceptr)d_data));

    checkError(cuStreamDestroy(stream));
}

TEST_F(StreamOverlapTest, ScenarioThree) {
    const int K = 4;
    CUstream streams[K];
    CUevent stevents[K];
    CUevent edevents[K];
    float elapsedtims[K];
    for (int i = 0; i < K; i++) {
        checkError(cuStreamCreate(&streams[i], 0));
        checkError(cuEventCreate(&stevents[i], 0));
        checkError(cuEventCreate(&edevents[i], 0));
    }
    CUfunction kernels[K];
    for (int i = 0; i < K; i++) {
        checkError(cuModuleGetFunction(&kernels[i], module_, "_Z7vec_powPfi"));
    }
    int N = 1000;
    float *h_input, *h_output, *d_input, *d_output;
    checkError(cuMemAllocHost((void**)&h_input, N * sizeof(float)));
    checkError(cuMemAllocHost((void**)&h_output, N * sizeof(float)));
    checkError(cuMemAlloc((CUdeviceptr*)&d_input, N * sizeof(float)));
    checkError(cuMemAlloc((CUdeviceptr*)&d_output, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        h_input[i] = i;
    }

    checkError(cuMemcpyHtoDAsync((CUdeviceptr)d_input, h_input,
                                 N * sizeof(float), streams[0]));

    for (int i = 0; i < K; i++) {
        void* args[] = {&d_input, &N};
        checkError(cuEventRecord(stevents[i], streams[i]));
        checkError(cuLaunchKernel(kernels[i], N / 256+1, 1, 1, 256, 1, 1, 0,
                                  streams[i], args, NULL));
        checkError(cuEventRecord(edevents[i], streams[i]));
        checkError(cuStreamSynchronize(streams[i]));
        checkError(
            cuEventElapsedTime(&elapsedtims[i], stevents[i],edevents[i]));
    }

    float actualtime;
    checkError(cuEventElapsedTime(&actualtime, stevents[0], edevents[K-1]));
    float overlaptime = *(std::max_element(elapsedtims, elapsedtims + K));

    std::cout << "overlaptotalTime execution time: " << overlaptime << " ms"
              << std::endl;
    std::cout << "actualtotalTime execution time: " << actualtime << " ms"
              << std::endl;

    if (overlaptime > actualtime) {
        std::cout << "Kernels overlapped" << std::endl;
    } else {
        std::cout << "Kernels did not overlap" << std::endl;
    }

    checkError(cuMemcpyDtoHAsync(h_output, (CUdeviceptr)d_input,
                                 N * sizeof(float), streams[0]));

    checkError(cuStreamSynchronize(streams[0]));

    for (int i = 0; i < N; i++) {
        float expected = h_input[i];
        for (int j = 0; j < K; j++) {
            expected = expected * expected;
        }
        ASSERT_FLOAT_EQ(h_output[i], expected);
    }

    checkError(cuMemFreeHost(h_input));
    checkError(cuMemFreeHost(h_output));
    checkError(cuMemFree((CUdeviceptr)d_input));
    checkError(cuMemFree((CUdeviceptr)d_output));


    for (int i = 0; i < K; i++) {
        checkError(cuStreamDestroy(streams[i]));
        checkError(cuEventDestroy(stevents[i]));
        checkError(cuEventDestroy(edevents[i]));
    }
}