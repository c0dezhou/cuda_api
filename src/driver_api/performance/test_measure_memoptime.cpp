// #include "memory_tests.h"

// TEST_F(CuMemTest, PerformanceBasicMeasureAllocHostTime) {
//     const size_t alloc_size =
//         1024 * 1024 * sizeof(int);  // 1 MB

//     CUstream s1;
//     cuStreamCreate(&s1, 0);

//     CUevent start_e, end_e;
//     cuEventCreate(&start_e, 0);
//     cuEventCreate(&end_e, 0);

//     cuEventRecord(start_e,s1);

//     int* p;
//     cuMemAllocHost((void**)&p, alloc_size);

//     cuEventRecord(end_e, s1);
//     cuEventSynchronize(end_e);

//     float elapsedTime;
//     cuEventElapsedTime(&elapsedTime, start_e, end_e);

//     cuMemFreeHost(p);

//     cuEventDestroy(start_e);
//     cuEventDestroy(end_e);

//     std::cout << "cuMemAllocHost Alloc Time: " << elapsedTime << " ms"
//               << std::endl;
// }

// TEST_F(CuMemTest, PerformanceHighLoad) {
//     const int ths_num = 4;
//     const size_t alloc_size = 1024 * 1024;  // 1 MB
//     const int alloc_interval = 100;         // ms
//     const int test_duration = 5000;         // ms

//     std::vector<int*> alloc_p;

//     std::vector<std::thread> threads;
//     for (int i = 0; i < ths_num; ++i) {
//         threads.emplace_back([&, i]() {
//             while (true) {
//                 int* p;
//                 cuMemAllocHost((void**)&p, alloc_size);

//                 performOperations(p, alloc_size);
//                 alloc_p.push_back(p);
//                 std::this_thread::sleep_for(
//                     std::chrono::milliseconds(alloc_interval));

//                 cuMemFreeHost(p);
//             }
//         });
//     }


//     std::this_thread::sleep_for(std::chrono::milliseconds(test_duration));

//     for (auto& thread : threads) {
//         thread.join();
//     }

//     std::cout << "Number of Allocations: " << alloc_p.size()
//               << std::endl;

//     for (auto& p : alloc_p) {
//         cuMemFreeHost(p);
//     }
// }
