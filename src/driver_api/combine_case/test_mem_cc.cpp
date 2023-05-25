// #include "memory_tests.h"

// TEST_F(CuMemTest, MultiStreamAllocation) {
//     GTEST_SKIP();
//     const int alloc_size = 1024;

//     CUstream s1, s2;
//     cuStreamCreate(&s1, 0);
//     cuStreamCreate(&s2, CU_STREAM_NON_BLOCKING);

//     int *p1, p2;
//     cuMemAllocHost((void**)&p1, alloc_size * sizeof(int));
//     cuMemAllocHost((void**)&p2, alloc_size * sizeof(int));

//     // TODO: cpy mem in diff stream ,
//     // 加个timerecord 以验证allochost 对cpy的优化
// }

// TEST_F(CuMemTest, OverlappingStreamOperations) {
//     // 数据传输，验证流重叠（加上计算的kernel？）
// }

// TEST_F(CuMemTest, EventSyncwithMem) {
//     const int alloc_size = 1024;

//     CUevent event;
//     cuEventCreate(&event, 0);

//     int* p;
//     cuMemAllocHost((void**)&p, alloc_size * sizeof(int));

//     cuEventRecord(event, 0);

//     performOperations(p, alloc_size);
//     cuEventSynchronize(event);

//     for (int i = 0; i < alloc_size; ++i) {
//         EXPECT_EQ(p[i], i);
//     }

//     cuMemFreeHost(p);
//     cuEventDestroy(event);
// }

// TEST_F(CuMemTest, StreamEventSynchronization) {
//     const int alloc_size = 1024;

//     CUstream s1, s2;
//     cuStreamCreate(&s1, 0);
//     cuStreamCreate(&s2, CU_STREAM_NON_BLOCKING);

//     int p;
//     int h_A = 8;
//     cuMemAllocHost((void**)&p, alloc_size * sizeof(int));

//     CUevent event1, event2;
//     cuEventCreate(&event1, 0);
//     cuEventCreate(&event2, 0);

//     cuMemcpyHtoDAsync((CUdevice)p, &h_A, alloc_size * sizeof(int), s1);
//     cuEventRecord(event1, s1);

//     // TODO: 在s2上做一个依赖s1的操作， 然后wait事件
//     // 再在s1上做一个p的计算操作
//     // 同步事件之后查看p值是否正确

//     cuEventDestroy(event1);
//     cuEventDestroy(event2);
//     cuStreamDestroy(s1);
//     cuStreamDestroy(s2);
// }