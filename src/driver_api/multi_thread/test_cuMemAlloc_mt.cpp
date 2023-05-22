// #include "memory_tests.h"


// TEST_F(CuMemTest, MultiThreadedScenario) {
//     GTEST_SKIP();
//     const int ths_num = 10;
//     std::vector<std::thread> ths;

//     for (int i = 0; i < ths_num; i++) {
//         ths.emplace_back([]() { testAllocHost<int>(42); });
//     }

//     for (auto& th : ths){
//         th.join();
//     }
// }

// TEST_F(CuMemTest, MultiThreadedAllocation) {
//     const int ths_num = 10;
//     const int alloc_size = 1024;

//     std::vector<std::thread> ths;

//     for (auto i = 0; i < ths_num; i++) {
//         ths.emplace_back([ths_num]() { 
//             int* p;
//             CUresult res = cuMemAllocHost((void**)&p, alloc_size * sizeof(int));
//             EXPECT_EQ(res, CUDA_SUCCESS);

//             for (auto j = 0; j < alloc_size; j++){
//                 p[j] = j;
//                 EXPECT_EQ(p[j], j);
//             }
//             cuMemFreeHost(p);
//         });
//     }

//     for(auto& th : ths){
//         th.join();
//     }
// }

// TEST_F(CuMemTest, SynchronizedThreadsAllocation) {
//     const int ths_num = 10;
//     const int alloc_size = 1024;

//     // TODO: if need lock? or use stream sync?
// }

// TEST_F(CuMemTest, ErrorOccurred){
//     // TODO: make a dead lock?
// }

// TEST_F(CuMemTest, CombinedMultistreamAndThread) {
//     const int ths_num = 4;
//     const int stream_num = 2;
//     const int alloc_size = 1024;

//     std::vector<std::thread> ths;

//     std::vector<CUstream> streams(stream_num);
//     for (int i = 0; i < stream_num; ++i) {
//         cuStreamCreate(&streams[i], 0);
//     }

//     for (int i = 0; i < ths_num; ++i) {
//         ths.emplace_back([&, i]() {
//             CUstream stream = streams[i % stream_num];

//             int* h_A;
//             CUdeviceptr d_A;
//             cuMemAlloc(&d_A, alloc_size * sizeof(int));
//             cuMemAllocHost((void**)&h_A, alloc_size * sizeof(int));
//             initarray(h_A, alloc_size);

//             cuMemcpyHtoDAsync(d_A, (void**)&h_A, alloc_size * sizeof(int), stream);
//             // launch kernel 做一个计算
//             cuStreamSynchronize(stream);
            
//             // 和预期比较
//             cuMemFreeHost(h_A);
//         });
//     }

//     for (auto& th : ths) {
//         th.join();
//     }

//     for (auto& s : streams) {
//         cuStreamDestroy(s);
//     }
// }