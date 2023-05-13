// 定义一个宏，用来生成不同数据类型的测试用例
#define CUMEMSETD_TEST(T, N) \
  // 分配设备内存 \
  CUdeviceptr d_ptr; \
  size_t size = N * sizeof(T); \
  cuMemAlloc(&d_ptr, size); \
  \
  // 分配主机内存 \
  T* h_ptr = new T[N]; \
  \
  // 基本行为测试 \
  // 设置设备内存为最大值 \
  T val = std::numeric_limits<T>::max(); \
  cuMemsetD##N##Async(d_ptr, val, N, NULL); \
  \
  // 同步设备 \
  cuDeviceSynchronize(); \
  \
  // 拷贝设备内存到主机内存 \
  cuMemcpyDtoH(h_ptr, d_ptr, size); \
  \
  // 检查主机内存中的值是否都为最大值 \
  for (size_t i = 0; i < N; i++) { \
    EXPECT_EQ(h_ptr[i], val); \
  } \ 
   \ 
   // 异常测试 \ 
   // 设置一个无效的目标设备指针 \ 
   CUdeviceptr d_ptr_invalid = NULL; \ 
   \ 
   // 设置一个无效的值 \ 
   val = -1; \ 
   \ 
   // 调用cuMemsetD##N##Async，期望返回CUDA_ERROR_INVALID_VALUE \ 
   CUresult res = cuMemsetD##N##Async(d_ptr_invalid, val, N, NULL); \ 
   EXPECT_EQ(res, CUDA_ERROR_INVALID_VALUE); \ 
   \ 
   // 边界值测试 \ 
   // 设置设备内存为最小值 \ 
   val = std::numeric_limits<T>::min(); \ 
   cuMemsetD##N##Async(d_ptr, val, N, NULL); \ 
   \ 
   // 同步设备 \ 
   cuDeviceSynchronize(); \ 
   \ 
   // 拷贝设备内存到主机内存 \ 
   cuMemcpyDtoH(h_ptr, d_ptr, size); \ 
   \ 
   // 检查主机内存中的值是否都为最小值 \ 
   for (size_t i = 0; i < N; i++) { \ 
     EXPECT_EQ(h_ptr[i], val); \ 
   } \ 
    \ 
    // 同步或异步行为测试 \ 
    // 设置设备内存为最大值 \ 
    val = std::numeric_limits<T>::max(); \ 
    cuMemsetD##N##Async(d_ptr, val, N, NULL); \ 
    \ 
    // 不同步设备，直接拷贝设备内存到主机内存 \ 
    cuMemcpyDtoH(h_ptr, d_ptr, size); \ 
    \ 
    // 检查主机内存中的值是否都为最大值，期望失败，因为cuMemsetD##N##Async是异步的 \ 
    for (size_t i = 0; i < N; i++) { \ 
      EXPECT_NE(h_ptr[i], val); \ 
    } \ 
     \ 
     // 同步设备 \ 
     cuDeviceSynchronize(); \ 
     \ 
     // 再次拷贝设备内存到主机内存
     // 同步或异步行为测试
// 设置设备内存为最大值
val = std::numeric_limits<T>::max();
cuMemsetD##N##Async(d_ptr, val, N, NULL);

// 不同步设备，直接拷贝设备内存到主机内存
cuMemcpyDtoH(h_ptr, d_ptr, size);

// 检查主机内存中的值是否都为最大值，期望失败，因为cuMemsetD##N##Async是异步的
for (size_t i = 0; i < N; i++) {
  EXPECT_NE(h_ptr[i], val);
}

// 同步设备
cuDeviceSynchronize();

// 再次拷贝设备内存到主机内存
cuMemcpyDtoH(h_ptr, d_ptr, size);

// 检查主机内存中的值是否都为最大值，期望成功，因为cuMemsetD##N##Async已经完成
for (size_t i = 0; i < N; i++) {
  EXPECT_EQ(h_ptr[i], val);
}

// 重复调用测试
// 设置设备内存为最小值
val = std::numeric_limits<T>::min();
cuMemsetD##N##Async(d_ptr, val, N, NULL);

// 同步设备
cuDeviceSynchronize();

// 拷贝设备内存到主机内存
cuMemcpyDtoH(h_ptr, d_ptr, size);

// 检查主机内存中的值是否都为最小值
for (size_t i = 0; i < N; i++) {
  EXPECT_EQ(h_ptr[i], val);
}

// 再次设置设备内存为最大值
val = std::numeric_limits<T>::max();
cuMemsetD##N##Async(d_ptr, val, N, NULL);

// 同步设备
cuDeviceSynchronize();

// 再次拷贝设备内存到主机内存
cuMemcpyDtoH(h_ptr, d_ptr, size);

// 检查主机内存中的值是否都为最大值
for (size_t i = 0; i < N; i++) {
  EXPECT_EQ(h_ptr[i], val);
}

// 其他你能想到的测试
// 设置一个CUDA流
CUstream hStream;
cuStreamCreate(&hStream, CU_STREAM_DEFAULT);

// 设置设备内存为最大值，使用CUDA流
val = std::numeric_limits<T>::max();
cuMemsetD##N##Async(d_ptr, val, N, hStream);

// 在CUDA流上执行一个简单的核函数，将设备内存中的每个值加1
dim3 gridDim((N + 255) / 256);
dim3 blockDim(256);
addOne<<<gridDim, blockDim, 0, hStream>>>(d_ptr_, N);

// 同步CUDA流
cuStreamSynchronize(hStream);

// 拷贝设备内存到主机内存
cuMemcpyDtoH(h_ptr_, d_ptr_, size);

// 检查主机内存中的值是否都为最小值，因为最大值 + 1 = 最小值（溢出）
for (size_t i = 0; i < N; i++) {
  EXPECT_EQ(h_ptr_[i], std::numeric_limits<T>::min());
}

// 销毁CUDA流
cuStreamDestroy(hStream);

// 释放内存
delete[] h_ptr;
cuMemFree(d_ptr);
