// #include "test_utils.h"

// int N = 5;

// CUdevice device = 0;

// CUmodule module;

// CUfunction kernel;

// CUcontext context;

// void run_kernel(int thread_id) {
//     printf("Thread %d started.\n", thread_id);

//     CUcontext context_new;
//     checkError(cuCtxCreate(&context_new, 0, device));
//     checkError(cuCtxSetCurrent(context_new));

//     CUdeviceptr d_a = NULL;
//     CUdeviceptr d_b = NULL;
//     CUdeviceptr d_c = NULL;
//     size_t size = sizeof(int) * N;
//     checkError(cuMemAlloc(&d_a, size));
//     checkError(cuMemAlloc(&d_b, size));
//     checkError(cuMemAlloc(&d_c, size));

//     int h_a[N];
//     int h_b[N];
//     for (int i = 0; i < N; i++) {
//         h_a[i] = i + thread_id;
//         h_b[i] = i + thread_id;
//     }
//     checkError(cuMemcpyHtoD(d_a, h_a, size));
//     checkError(cuMemcpyHtoD(d_b, h_b, size));

//     checkError(cuModuleLoad(&module,
//                             "/data/system/yunfan/cuda_api/common/cuda_kernel/"
//                             "cuda_kernel.ptx"));

//     checkError(cuModuleGetFunction(&kernel, module, "_Z6vecAddPfS_S_"));

//     int block_size = 256;
//     int grid_size = (N + block_size - 1) / block_size;
//     int gridDimX = 2, gridDimY = 2, gridDimZ = 1;
//     int blockDimX = 16, blockDimY = 16, blockDimZ = 1;
//     int sharedMemBytes = 256;
//     void* args[] = {&d_a, &d_b, &d_c, &N};
//     checkError(cuLaunchKernel(kernel, gridDimX, gridDimY, gridDimZ, blockDimX,
//                                blockDimY, blockDimZ, sharedMemBytes, 0, args,
//                                nullptr));

//     checkError(cuCtxSynchronize());
//     checkError(cuStreamSynchronize(0));

//     int h_c[N];
//     checkError(cuMemcpyDtoH(h_c, d_c, size));
//     for (int i = 0; i < N; i++) {
//         printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
//     }

//     checkError(cuMemFree(d_a));
//     checkError(cuMemFree(d_b));
//     checkError(cuMemFree(d_c));
//     checkError(cuCtxDestroy(context_new));

//     printf("Thread %d finished.\n", thread_id);
// }

// TEST(MTHCOMPLEX, runsamekernel) {
//     checkError(cuInit(0));
//     checkError(cuDeviceGet(&device, 0));
//     checkError(cuCtxCreate(&context, 0, device));

//     const int M = 4;

//     std::thread threads[M];
//     for (int i = 0; i < M; i++) {
//         threads[i] = std::thread(run_kernel, i);
//     }

//     for (int i = 0; i < M; i++) {
//         threads[i].join();
//         printf("Thread %d joined.\n", i);
//     }
//     checkError(cuCtxDestroy(context));
// }
