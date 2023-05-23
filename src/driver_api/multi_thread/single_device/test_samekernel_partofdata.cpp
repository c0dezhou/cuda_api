// CUDA operation performed by each thread
void cudaOperation(int threadId, CUdeviceptr d_A, CUdeviceptr d_B, CUdeviceptr d_C, int size, int numThreads) {
    // Set CUDA device
    cuCtxSetCurrent(NULL);

    // Initialize CUDA context
    CUcontext context;
    cuCtxCreate(&context, 0, 0);


    // dataSizePerThread 计算每个线程应处理的元素数。 它将数据的总大小 (size) 除以线程数 (numThreads)。 如果 size 可以被 numThreads 整除，这可以确保每个线程获得相等的数据份额。

// startIndex 确定当前线程的数据段的起始索引。 它将线程ID（threadId）乘以dataSizePerThread，得到线程应该处理的数据的初始位置。

// endIndex 指定当前线程的数据段的结束索引。 它是通过将 dataSizePerThread 添加到 startIndex 来计算的。 这决定了线程将操作的索引范围。

// if 条件检查当前线程是否是最后一个线程 (threadId == numThreads - 1)。 如果为真，则意味着可能还有一些剩余元素需要处理。 在这种情况下，通过添加 size % numThreads 调整 endIndex 以包含其他元素。 这确保了当总大小不能在线程之间均匀分割时，最后一个线程处理任何剩余的元素。

    // Calculate the portion of data for this thread
    int dataSizePerThread = size / numThreads;
    int startIndex = threadId * dataSizePerThread;
    int endIndex = startIndex + dataSizePerThread;
    if (threadId == numThreads - 1) {
        // Last thread takes care of the remaining elements if size % numThreads != 0
        endIndex += size % numThreads;
    }

    // Allocate memory on the device
    CUresult result;
    result = cuMemAlloc(&d_A, size * sizeof(float));
    assert(result == CUDA_SUCCESS);
    result = cuMemAlloc(&d_B, size * sizeof(float));
    assert(result == CUDA_SUCCESS);
    result = cuMemAlloc(&d_C, size * sizeof(float));
    assert(result == CUDA_SUCCESS);

    // Initialize input data for this thread
    std::vector<float> h_A(size);
    std::vector<float> h_B(size);
    for (int i = 0; i < size; ++i) {
        h_A[i] = i;
        h_B[i] = i;
    }

    // 在 cuMemcpyHtoD 函数中，d_A 是 CUdeviceptr 类型，它是指向设备内存的指针。 该函数需要数据应复制到设备内存中的内存地址。 由于 `d_A` 表示分配给当前线程的设备内存段的基地址，您需要将其偏移量乘以元素的数量（即 `startIndex`）乘以每个元素的大小（`sizeof(float)` ) 以正确指向当前线程的段的起始位置。

// 另一方面，`h_A.data()` 返回一个指向为整个数据数组分配的主机内存的指针。 由于 h_A 已经是指向数组开头的指针，所以只需要在其上添加 startIndex 来指示当前线程的段起始位置。 此偏移量直接应用于指针本身，无需乘以元素大小。

    // Copy a portion of input data from host to device
    result = cuMemcpyHtoD(d_A + startIndex * sizeof(float), h_A.data() + startIndex, (endIndex - startIndex) * sizeof(float));
    assert(result == CUDA_SUCCESS);
    result = cuMemcpyHtoD(d_B + startIndex * sizeof(float), h_B.data() + startIndex, (endIndex - startIndex) * sizeof(float));
    assert(result == CUDA_SUCCESS);

    // Launch the CUDA kernel
    CUmodule module;
    result = cuModuleLoad(&module, "cuda_kernel.ptx");
    assert(result == CUDA_SUCCESS);

    CUfunction function;
    result = cuModuleGetFunction(&function, module, "vecAdd");
    assert(result == CUDA_SUCCESS);

    int blockSize = 256;
    int gridSize = ((endIndex - startIndex) + blockSize - 1) / blockSize;
    void* kernelParams[] = { &d_A, &d_B, &d_C, &size };
    result = cuLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1, 0, NULL, kernelParams, NULL);
    assert(result == CUDA_SUCCESS);

    // Copy the result back from the device to host
    std::vector<float> h_C(size);
    result = cuMemcpyDtoH(h_C.data() + startIndex, d_C + startIndex * sizeof(float), (endIndex - startIndex) * sizeof(float));
    assert(result == CUDA_SUCCESS);

    // Free device memory
    result = cuMemFree(d_A);
    assert(result == CUDA_SUCCESS);
    result = cuMemFree(d_B);
    assert(result == CUDA_SUCCESS);
    result = cuMemFree(d_C);
    assert(result == CUDA_SUCCESS);

    // Destroy CUDA context
    cuCtxDestroy(context);
}

int main() {
    // Initialize CUDA driver API
    cuInit(0);

    const int numThreads = 20; // Number of threads
    const int dataSize = 1000; // Size of data

    std::vector<std::thread> threads(numThreads);
    std::vector<CUdeviceptr> d_A(numThreads);
    std::vector<CUdeviceptr> d_B(numThreads);
    std::vector<CUdeviceptr> d_C(numThreads);

    // Launch CUDA operations in multiple threads
    for (int i = 0; i < numThreads; ++i) {
        threads[i] = std::thread(cudaOperation, i, d_A[i], d_B[i], d_C[i], dataSize, numThreads);
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "All threads finished executing CUDA operations." << std::endl;

    // Verify the result
    for (int i = 0; i < dataSize; ++i) {
        float expected = i + i;
        float actual = h_C[i];
        if (actual != expected) {
            std::cout << "Verification failed at index " << i << ": expected " << expected << ", actual " << actual << std::endl;
            return 1;
        }
    }

    std::cout << "Verification succeeded. The final result is reasonable." << std::endl;

    return 0;
}
