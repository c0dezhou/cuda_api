// 高负载测试一：你想要在一个GPU上执行一个加密解密任务，该任务需要对多个文件进行AES-256算法的加密和解密，并且每次操作都需要随机生成不同的密钥和初始向量。你可以使用cuda driver api来创建一个stream，并在该stream上循环执行以下步骤：分配主机和设备内存，读取文件内容，拷贝内容到设备内存，随机生成密钥和初始向量，调用加密和解密的kernel函数，拷贝结果回主机内存，保存结果到文件，释放主机和设备内存。你可以使用gtest来计时每次执行的时间，并检查结果的正确性。
// 高负载测试二：你想要在多个GPU上执行一个排序搜索任务，该任务需要对多个大规模的数组进行快速排序算法的排序和二分查找算法的搜索，并且每次操作都需要随机生成不同的数组和目标值。你可以使用cuda driver api来创建多个进程，每个进程负责一个GPU，并在每个进程中创建多个stream，每个stream负责一个数组。你可以使用gtest来循环执行以下步骤：分配主机和设备内存，生成随机数组和目标值，拷贝数组到设备内存，调用排序和搜索的kernel函数，拷贝结果回主机内存，验证结果，释放主机和设备内存。你可以使用gtest来计时每次执行的时间，并检查结果的正确性。
// Include the necessary headers
#include <cuda.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>


// Define a test fixture class for cuda driver api
class CudaDriverTest : public ::testing::Test {
 protected:
  // Set up the test environment
  void SetUp() override {
    // Initialize the cuda driver api
    checkError(cuInit(0));
    // Get the first device handle
    checkError(cuDeviceGet(&device_, 0));
    // Create a context for the device
    checkError(cuCtxCreate(&context_, 0, device_));
    // Load the module containing the kernel functions
    checkError(cuModuleLoad(&module_, "kernel.ptx"));
  }

  // Tear down the test environment
  void TearDown() override {
    // Unload the module
    checkError(cuModuleUnload(module_));
    // Destroy the context
    checkError(cuCtxDestroy(context_));
  }

  // Declare some common variables
  CUdevice device_;
  CUcontext context_;
  CUmodule module_;
};

// Define a test case for stress test one
TEST_F(CudaDriverTest, StressTestOne) {
  // Create a stream
  CUstream stream;
  checkError(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  // Get the kernel function handles
  CUfunction kernel1, kernel2;
  checkError(cuModuleGetFunction(&kernel1, module_, "aes_encrypt"));
  checkError(cuModuleGetFunction(&kernel2, module_, "aes_decrypt"));
  // Allocate host and device memory for input and output files
  const int N = 1024;
  unsigned char *h_input, *h_output, *d_input, *d_output;
  checkError(cuMemAllocHost((void**)&h_input, N * sizeof(unsigned char)));
  checkError(cuMemAllocHost((void**)&h_output, N * sizeof(unsigned char)));
  checkError(cuMemAlloc(&d_input, N * sizeof(unsigned char)));
  checkError(cuMemAlloc(&d_output, N * sizeof(unsigned char)));
  // Repeat the encryption and decryption for 100 times
  for (int i = 0; i < 100; i++) {
    // Read the input file
    std::ifstream fin("input.txt", std::ios::binary);
    ASSERT_TRUE(fin) << "Cannot open input file";
    fin.read((char*)h_input, N);
    fin.close();
    // Copy input file to device memory on stream
    checkError(cuMemcpyHtoDAsync(d_input, h_input, N * sizeof(unsigned char), stream));
    // Generate random key and iv
    unsigned char key[16], iv[16];
    for (int j = 0; j < 16; j++) {
      key[j] = rand() % 256;
      iv[j] = rand() % 256;
    }
    // Set up kernel parameters
    void *args1[] = {&d_input, &d_output, &key, &iv};
    void *args2[] = {&d_output, &d_input, &key, &iv};
    // Launch kernel function for encryption on stream
    checkError(cuLaunchKernel(kernel1, N / 256, 1, 1, 256, 1, 1, 0, stream, args1, NULL));
    // Launch kernel function for decryption on stream
    checkError(cuLaunchKernel(kernel2, N / 256, 1, 1, 256, 1, 1, 0, stream, args2, NULL));
    // Copy output file back to host memory on stream
    checkError(cuMemcpyDtoHAsync(h_output, d_input, N * sizeof(unsigned char), stream));
    // Synchronize stream
    checkError(cuStreamSynchronize(stream));
    // Save the output file
    std::ofstream fout("output.txt", std::ios::binary);
    ASSERT_TRUE(fout) << "Cannot open output file";
    fout.write((char*)h_output, N);
    fout.close();
    // Verify the results
    for (int j = 0; j < N; j++) {
      ASSERT_EQ(h_output[j], h_input[j]);
    }
  }
  // Free host and device memory
  checkError(cuMemFreeHost(h_input));
  checkError(cuMemFreeHost(h_output));
  checkError(cuMemFree(d_input));
  checkError(cuMemFree(d_output));
  // Destroy stream
  checkError(cuStreamDestroy(stream));
}

// Define a test case for stress test two
TEST_F(CudaDriverTest, StressTestTwo) {
  // Create multiple streams
  const int K = 4;
  CUstream streams[K];
  for (int i = 0; i < K; i++) {
    checkError(cuStreamCreate(&streams[i], CU_STREAM_DEFAULT));
  }
  // Get the kernel function handles
  CUfunction kernel1, kernel2;
  checkError(cuModuleGetFunction(&kernel1, module_, "quicksort"));
  checkError(cuModuleGetFunction(&kernel2, module_, "binary_search"));
  // Allocate host and device memory for input and output arrays
  const int N = 1000000;
  int *h_input[K], *h_output[K], *d_input[K], *d_output[K];
  for (int i = 0; i < K; i++) {
    checkError(cuMemAllocHost((void**)&h_input[i], N * sizeof(int)));
    checkError(cuMemAllocHost((void**)&h_output[i], sizeof(int)));
    checkError(cuMemAlloc(&d_input[i], N * sizeof(int)));
    checkError(cuMemAlloc(&d_output[i], sizeof(int)));
  }
  // Repeat the sorting and searching for 100 times
  for (int i = 0; i < 100; i++) {
    // Generate random arrays and target values
    int target[K];
    for (int j = 0; j < K; j++) {
      for (int k = 0; k < N; k++) {
        h_input[j][k] = rand() % N;
      }
      target[j] = rand() % N;
    }
    // Copy input arrays to device memory on streams
    for (int j = 0; j < K; j++) {
      checkError(cuMemcpyHtoDAsync(d_input[j], h_input[j], N * sizeof(int), streams[j]));
    }
    // Set up kernel parameters
    void *args1[K][3];
    void *args2[K][4];
    for (int j = 0; j < K; j++) {
      args1[j][0] = &d_input[j];
      args1[j][1] = &N;
      args1[j][2] = &N;
      args2[j][0] = &d_input[j];
      args2[j][1] = &N;
      args2[j][2] = &target[j];
      args2[j][3] = &d_output[j];
    }
    // Launch kernel function for sorting on streams
    for (int j = 0; j < K; j++) {
      checkError(cuLaunchKernel(kernel1, 1, 1, 1, 1, 1, 1, 0, streams[j], args1[j], NULL));
    }
    // Launch kernel function for searching on streams
    for (int j = 0; j < K; j++) {
      checkError(cuLaunchKernel(kernel2, N / 256, 1, 1, 256, 1, 1, 0, streams[j], args2[j], NULL));
    }
    // Copy output arrays back to host memory on streams
    for (int j = 0; j < K; j++) {
      checkError(cuMemcpyDtoHAsync(h_output[j], d_output[j], sizeof(int), streams[j]));
    }
    // Synchronize streams
    for (int j = 0; j < K; j++) {
      checkError(cuStreamSynchronize(streams[j]));
    }
    // Verify the results
    for (int j = 0; j < K; j++) {
      int index = h_output[j][0];
      if (index == -1) {
        // Target value not found
        for (int k = 0; k < N; k++) {
          ASSERT_NE(h_input[j][k], target[j]);
        }
      } else {
        // Target value found
        ASSERT_EQ(h_input[j][index], target[j]);
      }
    }
  }
  // Free host and device memory
  for (int i = 0; i < K; i++) {
    checkError(cuMemFreeHost(h_input[i]));
    checkError(cuMemFreeHost(h_output[i]));
    checkError(cuMemFree(d_input[i]));
    checkError(cuMemFree(d_output[i]));
  }
  // Destroy streams
  for (int i = 0; i < K; i++) {
    checkError(cuStreamDestroy(streams[i]));
  }
}


// kernel
// 这些函数的头文件可能取决于您使用的编程语言和库。一般来说，您可能需要包含一个提供AES-256加密算法的头文件，例如<aes.h>或<openssl/aes.h>。1
// Define a kernel function for AES-256 encryption
__global__ void aes_encrypt(unsigned char *input, unsigned char *output, unsigned char *key, unsigned char *iv) {
  // Get the thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Check the thread index is within the data range
  if (i < 1024) {
    // Get the input and output pointers
    unsigned char *in = input + i * 16;
    unsigned char *out = output + i * 16;
    // Declare some variables for AES-256 encryption
    unsigned char state[4][4];
    unsigned char round_key[240];
    int round = 0;
    // Copy the input to the state array
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        state[j][k] = in[j + k * 4];
      }
    }
    // Add the initial round key to the state before starting the rounds
    KeyExpansion(key, round_key);
    AddRoundKey(0, state, round_key);
    // Perform 14 rounds of encryption
    for (round = 1; round < 14; round++) {
      SubBytes(state);
      ShiftRows(state);
      MixColumns(state);
      AddRoundKey(round, state, round_key);
    }
    // Perform the final round without MixColumns
    SubBytes(state);
    ShiftRows(state);
    AddRoundKey(14, state, round_key);
    // Copy the state to the output array
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        out[j + k * 4] = state[j][k];
      }
    }
  }
}

// Define a kernel function for AES-256 decryption
__global__ void aes_decrypt(unsigned char *input, unsigned char *output, unsigned char *key, unsigned char *iv) {
  // Get the thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Check the thread index is within the data range
  if (i < 1024) {
    // Get the input and output pointers
    unsigned char *in = input + i * 16;
    unsigned char *out = output + i * 16;
    // Declare some variables for AES-256 decryption
    unsigned char state[4][4];
    unsigned char round_key[240];
    int round = 0;
    // Copy the input to the state array
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        state[j][k] = in[j + k * 4];
      }
    }
    // Add the initial round key to the state before starting the rounds
    KeyExpansion(key, round_key);
    AddRoundKey(14, state, round_key);
    // Perform 14 rounds of decryption
    for (round = 13; round > 0; round--) {
      InvShiftRows(state);
      InvSubBytes(state);
      AddRoundKey(round, state, round_key);
      InvMixColumns(state);
    }
    // Perform the final round without InvMixColumns
    InvShiftRows(state);
    InvSubBytes(state);
    AddRoundKey(0, state, round_key);
    // Copy the state to the output array
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        out[j + k * 4] = state[j][k];
      }
    }
  }
}

// Define a kernel function for quicksort
__global__ void quicksort(int *array, int left, int right) {
  // Get the thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Check the thread index is within the array range
  if (i < right - left) {
    // Declare some variables for quicksort
    int pivot = array[left];
    int l = left + 1;
    int r = right;
    int temp;
    // Partition the array
    while (l <= r) {
      while (l <= r && array[l] <= pivot) {
        l++;
      }
      while (l <= r && array[r] > pivot) {
        r--;
      }
      if (l < r) {
        temp = array[l];
        array[l] = array[r];
        array[r] = temp;
      }
    }
    // Swap the pivot and the rightmost element
    temp = array[left];
    array[left] = array[r];
    array[r] = temp;
    // Recursively sort the left and right subarrays
    if (left < r - 1) {
      quicksort<<<1, 1>>>(array, left, r - 1);
    }
    if (r + 1 < right) {
      quicksort<<<1, 1>>>(array, r + 1, right);
    }
  }
}

// Define a kernel function for binary search
__global__ void binary_search(int *array, int size, int target, int *result) {
  // Get the thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Check the thread index is within the array range
  if (i < size) {
    // Declare some variables for binary search
    int left = 0;
    int right = size - 1;
    int mid;
    // Search the target value in the sorted array
    while (left <= right) {
      mid = (left + right) / 2;
      if (array[mid] == target) {
        // Target value found, store the index in the result
        result[0] = mid;
        return;
      } else if (array[mid] < target) {
        // Target value is in the right subarray
        left = mid + 1;
      } else {
        // Target value is in the left subarray
        right = mid - 1;
      }
    }
    // Target value not found, store -1 in the result
    result[0] = -1;
  }
}
