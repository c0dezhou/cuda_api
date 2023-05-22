// #include "memory_tests.h"

// TEST_F(CuMemTest, MemoryLeakDetection) {
//     GTEST_SKIP();
//     const size_t alloc_size = 1024 * 1024;  // 1 MB
//     const int num_iter = 100;

//     long long sys_mem_usage = getSystemMemoryUsage();

//     for(int i = 0; i< num_iter; i++){
//         int* p;
//         CUresult res = cuMemAllocHost((void**)&p, alloc_size * sizeof(int));

//         if (res == CUDA_SUCCESS) {
//             performOperations(p, alloc_size);

//             cuMemFreeHost(p);
//         }
//     }

//     long long final_mem_usage = getSystemMemoryUsage();
//     EXPECT_LE(final_mem_usage, sys_mem_usage);
// }