#include "mths_tests.h"

void run_kernel(int thread_id,
                int N,
                CUcontext context,
                CUstream stream,
                CUevent event,
                CUfunction add_kernel) {
    printf("Thread %d started.\n", thread_id);

    checkError(cuCtxSetCurrent(context));

    CUdeviceptr d_a;
    CUdeviceptr d_b;
    CUdeviceptr d_c;
    size_t size = sizeof(int) * N;
    checkError(cuMemAlloc(&d_a, size));
    checkError(cuMemAlloc(&d_b, size));
    checkError(cuMemAlloc(&d_c, size));

    int h_a[N];
    int h_b[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = i + thread_id;  // 每个线程的数组元素不同
        h_b[i] = i + thread_id;
    }
    checkError(cuMemcpyHtoD(d_a, h_a, size));
    checkError(cuMemcpyHtoD(d_b, h_b, size));

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    void* args[] = {&d_a, &d_b, &d_c, &N};
    // 使用事件来保证只有一个线程可以启动内核函数
    checkError(cuStreamWaitEvent(stream, event, 0));
    int gridDimX = 2, gridDimY = 2, gridDimZ = 1;
    int blockDimX = 16, blockDimY = 16, blockDimZ = 1;
    int sharedMemBytes = 256;
    checkError(cuLaunchKernel(add_kernel, gridDimX, gridDimY, gridDimZ,
                              blockDimX, blockDimY, blockDimZ, sharedMemBytes,
                              stream, args, nullptr));
    checkError(cuEventRecord(event, stream));

    checkError(cuStreamSynchronize(stream));

    int h_c[N];
    checkError(cuMemcpyDtoH(h_c, d_c, size));

    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    checkError(cuMemFree(d_a));
    checkError(cuMemFree(d_b));
    checkError(cuMemFree(d_c));

    printf("Thread %d finished.\n", thread_id);
}

TEST_F(MthsTest, MTH_Single_Device_1_stream_n_ths) {
    checkError(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    checkError(cuEventCreate(&event, CU_EVENT_DEFAULT));

    const int M = 4;

    std::thread threads[M];
    for (int i = 0; i < M; i++) {
        threads[i] = std::thread(run_kernel, i, N, context,
                                 stream, event, add_kernel);  // 线程ID为i
    }

    for (int i = 0; i < M; i++) {
        threads[i].join();
        printf("Thread %d joined.\n", i);
    }

    checkError(cuEventDestroy(event));

    checkError(cuStreamDestroy(stream));
}
