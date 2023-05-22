#include "memory_tests.h"

TEST_F(CuMemTest, LoopAllocation) {
    GTEST_SKIP();
    const int ths_num = 10;
    const int alloc_size = 1024;
    int* p;

    for (int i = 0; i < ths_num; i++) {
        p += 8;
        cuMemAllocHost((void**)&p, alloc_size * sizeof(int));
        performOperations(p, alloc_size);
        cuMemFreeHost(p);
    }
}

TEST_F(CuMemTest, LoopAllocationMultiths) {
    GTEST_SKIP();
    const int ths_num = 10;
    const int alloc_size = 1024;

    std::vector<std::thread> ths;

    for (int i = 0; i < ths_num; i++) {
        ths.emplace_back([alloc_size]() {
            int* p;
            cuMemAllocHost((void**)&p, alloc_size * sizeof(int));
            performOperations(p, alloc_size);
            cuMemFreeHost(p);
        });
    }

    for (auto& th : ths) {
        th.join();
    }
}

TEST_F(CuMemTest, LoopAllocationMakeErrors) {
    const int ths_num = 10;
    const size_t alloc_size = 1ULL << 32;  // 4 GB

    int* p;
    for (int i = 0; i < ths_num; ++i) {
        p += 8;
        CUresult res = cuMemAllocHost((void**)&p, alloc_size);
        if (res == CUDA_SUCCESS) {
            std::cout << "in th " << i << ": meemallochost successful\n";
            cuMemFreeHost(p);
        } else {
            std::cerr << "th " << i
                      << ": meemallochost failed , errcode: " << res
                      << std::endl;
        }
    }
}