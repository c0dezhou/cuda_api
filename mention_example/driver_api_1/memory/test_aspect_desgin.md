
1. **Single-threaded tests**
   
    a. **Basic functionality**: Validate if cuMemAllocHost correctly allocates pinned host memory of various sizes and data types. Ensure that the allocated memory addresses are valid.

    b. **Edge cases**: Test allocating zero bytes or a negative amount of memory, and check how the function reacts when trying to allocate more memory than available.

    c. **Error handling**: Verify that appropriate error codes are returned for incorrect byte counts, invalid pointers, etc.

2. **Multi-threaded tests**

    a. **Basic functionality**: Test cuMemAllocHost's behavior when called from different threads, ensuring each thread can independently allocate memory without impacting others.

    b. **Synchronization**: Test cuMemAllocHost with synchronized threads where each thread waits for the previous one to finish before starting.

    c. **Error handling**: Validate how cuMemAllocHost handles errors in a multi-threaded scenario, such as trying to allocate more memory than is available or allocating zero bytes.

3. **Multi-process tests**

    a. **Basic functionality**: Each process should independently allocate memory using cuMemAllocHost, and the test should verify successful allocations that don't impact other processes.

    b. **Inter-process communication**: Test behavior when memory allocated in one process is accessed by another.

    c. **Error handling**: Test how cuMemAllocHost handles errors in a multi-process scenario, like trying to allocate more memory than is available, allocating zero bytes, etc.

4. **Multi-stream tests**

    a. **Basic functionality**: Verify that cuMemAllocHost can correctly allocate memory used in different streams without impacting other streams.

    b. **Overlapping and synchronization**: Test if cuMemAllocHost can handle overlapping operations and synchronization of operations in different streams.

5. **Loop tests**

    a. **Basic functionality**: Test cuMemAllocHost in a loop scenario, where allocation and deallocation are performed repeatedly across different threads, processes, or streams.

    b. **Error handling**: Test how cuMemAllocHost handles errors in loop scenarios, such as what happens if you attempt to allocate more memory than is available repeatedly.

6. **Event-related tests**

    a. **Basic functionality**: Test cuMemAllocHost in scenarios where CUDA events are used for synchronization. Test if cuMemAllocHost can correctly allocate memory that can be accessed after a specific event.

    b. **Multi-stream synchronization**: Use CUDA events to synchronize operations in different streams that are dependent on the memory allocated by cuMemAllocHost.

    c. **Performance measurement**: Use CUDA events to measure the time taken by cuMemAllocHost to allocate memory.

    d. **Error handling**: Test how cuMemAllocHost handles errors in event-related scenarios.

7. **Combined scenarios**

    a. **Multi-threading, multi-processing, and multi-streaming**: Test cuMemAllocHost in combined scenarios, such as multi-threading and multi-streaming, or multi-processing and multi-streaming.

    b. **Event, multi-threading, multi-processing, and multi-streaming**: Test cuMemAllocHost in combined scenarios, such as multi-threading and multi-streaming with events, or multi-processing and multi-streaming with events.

8. **API interactions**

    a. **Interaction with other CUDA driver APIs**: Test how cuMemAllocHost interacts with other CUDA driver APIs. Test if cuMemAllocHost can correctly allocate memory that can be accessed by cuMemcpy or other CUDA APIs.

9. **Performance**

    a. **Memory allocation speed (continued)**: Measure the performance of cuMemAllocHost under different conditions, such as different memory sizes, different numbers of threads/processes/streams, and with or without synchronization events.

10. **Stress Tests**

    a. **High Load**: Test how cuMemAllocHost performs under high load, such as continuous memory allocation and deallocation over an extended period.

    b. **Resource Limitation**: Test the behavior of cuMemAllocHost when system resources are limited or being heavily used by other processes.

11. **Security Tests**

    a. **Access Violation**: Test if cuMemAllocHost properly prevents unauthorized access to the allocated memory.

    b. **Data Integrity**: Test if cuMemAllocHost maintains data integrity, i.e., if the data written to the allocated memory can be read back without any corruption.

12. **Memory Leak Tests**

    a. **Memory Release**: After the allocated memory is no longer needed, test if it is correctly freed and if the system memory usage goes back to the expected level.

    b. **Repeated Allocation and Deallocation**: Repeatedly allocate and deallocate memory, and monitor system memory usage to check for potential memory leaks.

重复调用API可能会出现异常的情况有：

如果API的参数或返回值与之前的调用不一致，可能会导致内存泄漏，内存损坏，或者错误的结果。
如果API的调用与其他CUDA操作之间没有正确地同步，可能会导致竞态条件，死锁，或者性能下降。
如果API的调用超出了设备或系统的资源限制，比如内存容量，寄存器数量，线程数等，可能会导致失败，异常，或者崩溃。


有一种方法可以重复运行你的测试用例，就是使用**–gtest_repeat**标志。这个标志可以让你指定重复的次数，例如：

$ foo_test --gtest_repeat=100
复制
这样就会重复运行foo_test中的所有测试用例100次。如果你只想重复某些测试用例，你可以结合使用**–gtest_filter**标志，例如：

$ foo_test --gtest_repeat=100 --gtest_filter=ServerTest.*
复制
这样就会重复运行foo_test中的ServerTest测试套件中的所有测试用例100次。如果你想要测试稳定性，你还可以使用**–gtest_break_on_failure**标志，这样当有一个测试用例失败时，程序就会停止运行，方便你调试。


如果你想要进行并行的测试，你可以使用一个脚本来执行不同的测试用例，例如：

$ ./test --gtest_filter=ServerTest.* &
$ ./test --gtest_filter=ClientTest.* &
$ ./test --gtest_filter=NetworkTest.* &
复制
这样就会在后台运行三个不同的测试用例，分别是ServerTest，ClientTest和NetworkTest。你需要确保这些测试用例是独立的，不会相互干扰。你也可以使用一个第三方的工具来帮助你并行运行测试用例，例如：https://github.com/google/gtest-parallel/

gtest-parallel是一个脚本，可以并行地执行GoogleTest的二进制文件，从而提高测试的速度。你可以使用以下方式来使用gtest-parallel：

$ ./gtest-parallel path/to/binary...
复制
这样就会在后台运行你指定的所有测试二进制文件，并将测试结果输出到屏幕上。你还可以使用一些选项来控制gtest-parallel的行为，例如：

$ ./gtest-parallel --workers=4 --output_dir=/tmp/test_results path/to/binary...
复制
这样就会限制最多有4个进程同时运行，并将测试结果保存到/tmp/test_results目录下。你可以使用–help选项来查看所有可用的选项。


// 多线程：
```
// 定义一个测试用例
TEST(ServerTest, TwoRequests) {
  // 创建一个Server对象
  Server server;

  // 创建两个线程，分别发送请求
  std::thread t1([&server]() {
    Result r1 = server.AcceptClientRequest(request1);
    EXPECT_EQ(r1, correctResultFor1);
  });
  std::thread t2([&server]() {
    Result r2 = server.AcceptClientRequest(request2);
    EXPECT_EQ(r2, correctResultFor2);
  });

  // 等待两个线程结束
  t1.join();
  t2.join();
}

```



// 多进程
// 定义一个测试用例
```TEST(ServerTest, TwoRequests) {
  // 创建一个Server对象
  Server server;

  // 创建两个子进程，分别发送请求
  pid_t pid1 = fork();
  if (pid1 == 0) {
    // 子进程1
    Result r1 = server.AcceptClientRequest(request1);
    EXPECT_EQ(r1, correctResultFor1);
    exit(0); // 结束子进程
  }
  pid_t pid2 = fork();
  if (pid2 == 0) {
    // 子进程2
    Result r2 = server.AcceptClientRequest(request2);
    EXPECT_EQ(r2, correctResultFor2);
    exit(0); // 结束子进程
  }

  // 等待两个子进程结束
  int status;
  waitpid(pid1, &status, 0);
  waitpid(pid2, &status, 0);
}```


根据我从网上搜索到的信息，cuda driver api和cuda runtime api可以混用，但是要注意一些规则12：

如果使用driver api创建并设置了当前上下文，那么后续的runtime api调用会使用这个上下文，而不会创建新的上下文。
如果使用runtime api隐式初始化了上下文（例如通过cudaSetDevice），那么可以用cuCtxGetCurrent获取这个上下文，并在driver api调用中使用它。
如果要在同一个程序中混用driver api和runtime api，那么必须保证它们使用的是同一个CUDA版本，并且在链接时指定相应的库。
在你的代码中，你在同步或异步行为测试中混用了driver api和runtime api，这是可以的，只要你遵循了上述规则。你可以用cuCtxGetApiVersion检查你的上下文是否是正确的版本。

https://stackoverflow.com/questions/60132426/how-can-i-mix-cuda-driver-api-with-cuda-runtime-api

https://zhuanlan.zhihu.com/p/544492099

https://stackoverflow.com/questions/19240658/cuda-kernel-launch-parameters-explained-right

