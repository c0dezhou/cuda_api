/*
创建两个或多个stream，分别在不同的stream上执行不同的kernel，使用event来同步kernel的完成情况，然后使用memcpy来复制kernel的输出到主机内存。
创建一个stream，使用memcpyAsync来复制数据从主机内存到设备内存，然后在同一个stream上执行一个或多个kernel，使用event来记录kernel的开始和结束时间，最后使用memcpyAsync来复制kernel的输出到主机内存。
创建一个或多个stream，使用cudaStreamAttachMemAsync来将一块主机内存附加到stream上，然后在stream上执行一个或多个kernel，使用event来记录kernel的开始和结束时间，最后使用cudaStreamSynchronize来等待stream上的所有操作完成。
创建一个或多个stream，使用cudaMallocManaged来分配一块统一内存，然后在不同的stream上执行不同的kernel，使用event来记录kernel的开始和结束时间，最后使用cudaDeviceSynchronize来等待所有设备上的操作完成。
*/

/*
测试在同一个stream中执行多个kernel，以及在不同的stream中执行多个kernel，比较它们的执行时间和结果是否一致。
测试在同一个stream中执行多个memcpy，以及在不同的stream中执行多个memcpy，比较它们的执行时间和结果是否一致。
测试在同一个stream中混合使用kernel和memcpy，以及在不同的stream中混合使用kernel和memcpy，比较它们的执行时间和结果是否一致。
测试使用event来记录kernel或memcpy的开始和结束时间，以及使用cudaStreamWaitEvent来等待某个event完成，比较它们的执行时间和结果是否正确。
测试使用cudaStreamSynchronize或cudaDeviceSynchronize来同步某个stream或设备上的所有操作，比较它们的执行时间和结果是否正确。
测试使用cudaStreamCreate或cudaStreamCreateWithFlags来创建不同类型的stream，比较它们的性能和特性是否符合预期。
测试使用cudaStreamDestroy来销毁某个stream，比较它是否会影响其他stream或设备上的操作。
测试使用cudaMemcpyAsync或cudaMemcpyPeerAsync来异步地拷贝数据，比较它们的执行时间和结果是否正确。
测试使用cudaMemcpy或cudaMemcpyPeer来同步地拷贝数据，比较它们的执行时间和结果是否正确。
测试使用cudaMemcpy2DAsync或cudaMemcpy3DAsync来异步地拷贝二维或三维数据，比较它们的执行时间和结果是否正确。
测试使用cudaMemcpy2D或cudaMemcpy3D来同步地拷贝二维或三维数据，比较它们的执行时间和结果是否正确。
测试使用cudaMalloc或cudaMallocManaged来分配设备内存，比较它们的性能和特性是否符合预期。
测试使用cudaFree来释放设备内存，比较它是否会影响其他内存或设备上的操作。
测试使用多线程来并发地操作同一个设备或不同的设备，比较它们的执行时间和结果是否正确。
测试使用多进程来并发地操作同一个设备或不同的设备，比较它们的执行时间和结果是否正确。
测试使用cudaSetDevice或cudaGetDevice来切换或获取当前活动的设备，比较它们是否会影响其他设备上的操作。
*/

// 多线程和多进程的情况下，如何管理和转移context

/*
创建两个或多个流，分别在不同的设备上执行不同的核函数，并使用事件来同步它们的执行顺序。
创建一个流，将其分成多个子流，并在每个子流上执行不同的核函数或内存操作。使用事件来控制子流之间的依赖关系。
创建一个图形对象，包含多个节点，每个节点对应一个核函数、内存操作或主机函数。在图形中使用事件来标记节点的开始和结束，并在不同的流上实例化和执行图形。
创建一个循环，每次迭代都创建一个新的流，并在该流上执行一个核函数。使用事件来检测每个流的完成状态，并在所有流完成后退出循环。
创建一个动态并行场景，其中一个核函数会在运行时启动其他核函数，并使用事件来通知主机线程它们的完成状态。
*/

/*
流的重叠是指在不同的流上同时执行不同类型的操作，以提高 GPU 的利用率。例如，可以在一个流上执行核函数，而在另一个流上执行内存拷贝，或者在多个流上执行不同的核函数。

下面是一些可能的测试场景：

创建两个或多个流，分别在它们上面执行核函数和内存拷贝，并使用事件或流查询来检测它们的重叠程度。
创建一个流，将其分成多个子流，并在每个子流上执行不同类型的操作。使用 cudaStreamAttachMemAsync 函数来将内存区域附加到特定的子流，以减少同步开销。
创建一个图形对象，包含多个节点，每个节点对应一个核函数、内存操作或主机函数。在图形中使用 cudaGraphAddDependencies 函数来指定节点之间的依赖关系，并在不同的流上实例化和执行图形。
创建一个动态并行场景，其中一个核函数会在运行时启动其他核函数，并使用 cudaStreamCreateWithFlags 函数来为每个子核函数创建一个非阻塞的流。使用 cudaStreamIsCapturing 函数来检测当前是否处于图形捕获模式，并根据需要调整流的创建方式。

创建两个或多个流，分别在它们上面执行核函数和内存拷贝，并使用 cudaStreamSynchronize 或 cudaStreamWaitEvent 函数来同步它们。使用 cudaStreamQuery 或 cudaEventQuery 函数来检测它们的异步状态，并使用 cudaGetLastError 函数来检测是否有错误发生。
创建一个流，将其分成多个子流，并在每个子流上执行不同类型的操作。使用 cudaStreamAttachMemAsync 函数来将内存区域附加到特定的子流，以减少同步开销。使用 cudaStreamSynchronize 或 cudaStreamWaitEvent 函数来同步子流，并使用 cudaStreamQuery 或 cudaEventQuery 函数来检测它们的异步状态。
创建一个图形对象，包含多个节点，每个节点对应一个核函数、内存操作或主机函数。在图形中使用 cudaGraphAddDependencies 函数来指定节点之间的依赖关系，并在不同的流上实例化和执行图形。使用 cudaGraphExecSynchronize 或 cudaGraphExecWaitEvent 函数来同步图形执行，并使用 cudaGraphExecQuery 或 cudaGraphExecEventQuery 函数来检测图形执行的异步状态。
创建一个动态并行场景，其中一个核函数会在运行时启动其他核函数，并使用 cudaStreamCreateWithFlags 函数来为每个子核函数创建一个非阻塞的流。使用 cudaDeviceSynchronize 或 cudaEventSynchronize 函数来同步设备，并使用 cudaDeviceQuery 或 cudaEventQuery 函数来检测设备的异步状态。
*/

/*
创建多个主机线程，每个线程创建一个或多个流，并在它们上面执行核函数和内存拷贝。使用 cudaStreamCreateWithPriority 函数来为每个流指定不同的优先级，并使用 cudaStreamGetPriority 函数来获取它们的优先级。使用 cudaEventCreateWithFlags 函数来为每个操作创建一个或多个事件，并使用 cudaEventRecord 函数来记录它们的发生时间。使用 cudaEventElapsedTime 函数来计算不同事件之间的时间差，并使用 cudaEventDestroy 函数来销毁事件。
创建多个主机线程，每个线程创建一个或多个流，并在它们上面执行核函数和内存拷贝。使用 cudaStreamCreateWithFlags 函数来为每个流指定不同的标志，例如 cudaStreamNonBlocking 或 cudaStreamPerThread，并使用 cudaStreamGetFlags 函数来获取它们的标志。使用 cudaEventCreateWithFlags 函数来为每个操作创建一个或多个事件，并使用 cudaEventRecord 函数来记录它们的发生时间。使用 cudaStreamWaitEvent 函数来让一个流等待另一个流上的事件，并使用 cudaEventSynchronize 函数来同步事件。
创建多个主机线程，每个线程创建一个或多个图形对象，并在它们上面添加不同类型的节点。使用 cudaGraphAddDependencies 函数来指定节点之间的依赖关系，并使用 cudaGraphInstantiate 函数来实例化图形对象。使用 cudaGraphLaunch 函数来在不同的流上执行图形对象，并使用 cudaGraphExecDestroy 函数来销毁图形执行对象。使用 cudaGraphDestroy 函数来销毁图形对象。
创建多个主机线程，每个线程创建一个或多个动态并行场景，并在它们上面执行核函数和内存拷贝。使用 cudaStreamCreateWithFlags 函数来为每个子核函数创建一个非阻塞的流，并使用 cudaStreamIsCapturing 函数来检测当前是否处于图形捕获模式。使用 cudaLaunchHostFunc 函数来在不同的流上执行主机函数，并使用 cudaLaunchKernel 函数来在不同的流上执行设备函数。使用 cudaDeviceSynchronize 函数来同步设备，并使用 cudaGetLastError 函数来检测是否有错误发生。
*/

// https://github.com/NVIDIA/DCGM/blob/master/nvvs/plugin_src/memtest/Memtest.cpp
// https://github.com/ComputationalRadiationPhysics/cuda_memtest
// 首先，它会检测系统中有多少个可用的 CUDA 设备，并根据用户的输入选择要测试的设备。
// 然后，它会为每个要测试的设备创建一个 CUDA 上下文（context），并分配一定大小的 GPU 内存。
// 接着，它会执行一系列的测试，每个测试都有一个特定的模式（pattern）和一个特定的核函数（kernel）。每个测试都会在 GPU 内存中写入或读取一些数据，并与预期的结果进行比较。如果发现任何不匹配，就会报告错误。
// 最后，它会释放 GPU 内存和 CUDA 上下文，并输出测试的结果和统计信息。

// https://github.com/QINZHAOYU/CudaSteps


/*
选择不同的kernel函数，比如单精度、双精度、半精度或整数的乘加操作1，并比较它们的执行时间和吞吐量。
选择不同的kernel配置，比如网格大小，块大小，共享内存大小等23，并观察它们对性能的影响。
选择不同的内存访问模式，比如全局内存，纹理内存，常量内存等4，并分析它们的带宽和延迟。
选择不同的编译方式，比如离线静态编译或在线动态编译23，并比较它们的编译时间和生成的PTX代码。
选择不同的context管理策略，比如单context或多context23，并评估它们对内存分配，数据传输，kernel加载等开销的影响。

不同精度的运算需要不同的硬件资源和指令集，例如单精度和双精度的浮点运算需要不同的寄存器和算术逻辑单元12，而半精度和整数的运算可以利用张量核心3来加速。
不同精度的运算对内存带宽和延迟的要求也不同，例如双精度的运算需要更多的内存访问次数和传输字节数，而半精度的运算可以减少内存压力和提高吞吐量12。
不同精度的运算对数值稳定性和精确度的影响也不同，例如双精度的运算可以提供更高的精确度和范围，而半精度的运算可能会导致溢出或下溢12。
不同精度的运算对编译器优化和调试的难易程度也不同，例如双精度的运算可能需要更多的编译选项和标志，而半精度的运算可能需要更多的测试和验证12。
*/

/*
执行速度：指的是kernel函数或整个程序的运行时间，通常用秒或毫秒来衡量。执行速度越快，说明性能越好。
执行效率：指的是kernel函数或整个程序的运算能力，通常用每秒浮点运算次数（FLOPS）或每秒内存传输字节数（GB/s）来衡量。执行效率越高，说明性能越好。
当然，还有一些其他的性能指标，例如：

执行稳定性：指的是kernel函数或整个程序的运行结果的可靠性和一致性，通常用数值误差或正确率来衡量。执行稳定性越高，说明性能越好。
执行可扩展性：指的是kernel函数或整个程序的运行性能随着问题规模或设备规模的变化情况，通常用加速比或效率比来衡量。执行可扩展性越高，说明性能越好。

执行功耗：指的是kernel函数或整个程序的运行过程中消耗的电能，通常用瓦特或焦耳来衡量。执行功耗越低，说明性能越好。
执行效能：指的是kernel函数或整个程序的运算能力和功耗的比值，通常用每秒浮点运算次数每瓦特（FLOPS/W）或每秒内存传输字节数每瓦特（GB/s/W）来衡量。执行效能越高，说明性能越好。
执行可移植性：指的是kernel函数或整个程序的运行性能在不同的设备或平台上的兼容性和适应性，通常用可移植性指数或可移植性比例来衡量。执行可移植性越高，说明性能越好。
执行可维护性：指的是kernel函数或整个程序的运行代码的可读性和可修改性，通常用代码行数，注释率，复杂度等指标来衡量。执行可维护性越高，说明性能越好。

性能指标：cuda driver api提供了一些内置的性能指标，例如power和perf_per_watt，可以用nvprof --query-metrics命令查看2。
性能工具：cuda driver api提供了一些性能工具，例如CUDA事件（CUDA event）和CUDA计时器（CUDA timer），可以用来测量kernel函数或整个程序的执行时间和功耗23。
另一种方法是使用一些第三方的性能工具，例如：

NVIDIA Nsight Systems：这是一个系统级的性能分析工具，可以显示应用程序的CPU和GPU活动的时间线，并提供一些性能指标和建议4。
NVIDIA Nsight Compute：这是一个内核级的性能分析工具，可以显示kernel函数的详细信息，并提供一些性能指标和建议4。
*/


/*
好的，我可以为你设计一些使用场景，包括以上所有 api，除了图管理和互操作性之外。😊

以下是一些可能的使用场景：

- **使用设备管理、上下文管理、内存管理、模块管理和执行控制 api 来加载和运行一个简单的 GPU 内核**¹²：

  - 首先，需要使用 `cuInit` 函数来初始化 cuda driver api，并使用 `cuDeviceGet` 函数来获取第一个可用的 GPU 设备。
  - 然后，需要使用 `cuCtxCreate` 函数来创建一个 GPU 上下文，并使用 `cuCtxSetCurrent` 函数来将其设置为当前上下文。
  - 接着，需要使用 `cuModuleLoad` 函数来加载一个包含 GPU 内核的 ptx 文件，并使用 `cuModuleGetFunction` 函数来获取内核函数的句柄。
  - 然后，需要使用 `cuMemAlloc` 函数来分配 GPU 上的内存，并使用 `cuMemcpyHtoD` 函数来将主机上的数据复制到 GPU 上。
  - 接着，需要使用 `cuLaunchKernel` 函数来配置和启动 GPU 内核，并传入内核函数的句柄、网格和块的维度、共享内存的大小、流的句柄和内核参数的指针。
  - 最后，需要使用 `cuCtxSynchronize` 函数来等待 GPU 上的所有操作完成，并使用 `cuMemcpyDtoH` 函数来将 GPU 上的结果复制到主机上。然后，需要使用 `cuMemFree`、`cuModuleUnload` 和 `cuCtxDestroy` 函数来释放 GPU 上的资源。

- **使用设备管理、上下文管理、内存管理、模块管理、执行控制和流管理 api 来实现异步执行和重叠计算与数据传输**¹²：

  - 首先，需要使用 `cuInit`、`cuDeviceGet` 和 `cuCtxCreate` 函数来初始化 cuda driver api 并创建一个 GPU 上下文。
  - 然后，需要使用 `cuStreamCreate` 函数来创建多个 GPU 流，并将它们分配给不同的任务，例如计算或数据传输。
  - 接着，需要使用 `cuModuleLoad` 和 `cuModuleGetFunction` 函数来加载并获取 GPU 内核函数的句柄。
  - 然后，需要使用 `cuMemAllocHost` 和 `cuMemAlloc` 函数来分配主机和 GPU 上的内存，并将它们分成多个块。
  - 接着，需要在一个循环中执行以下操作：
    - 使用 `cuMemcpyHtoDAsync` 函数将主机上的一个数据块异步复制到 GPU 上，并指定一个流作为参数。
    - 使用 `cuLaunchKernel` 函数在同一个流中异步启动 GPU 内核，并传入相应的参数。
    - 使用 `cuMemcpyDtoHAsync` 函数将 GPU 上的一个结果块异步复制到主机上，并指定另一个流作为参数。
  - 最后，需要使用 `cuStreamSynchronize` 或 `cuCtxSynchronize` 函数来等待所有流中的操作完成，并使用 `cuMemFreeHost`、`cuMemFree`、`cuModuleUnload`、`cuStreamDestroy` 和 `cuCtxDestroy` 函数来释放资源。

- **使用设备管理、上下文管理、内存管理、模块管理、执行控制和事件管理 api 来测量 GPU 内核的执行时间**¹²：

  - 首先，需要使用 `cuInit`、`cuDeviceGet` 和 `cuCtxCreate` 函数来初始化 cuda driver api 并创建一个 GPU 上下文。
  - 然后，需要使用 `cuEventCreate` 函数来创建两个 GPU 事件，并设置它们为计时模式。
  - 接着，需要使用 `cuModuleLoad` 和 `cuModuleGetFunction` 函数来加载并获取 GPU 内核函数的句柄。
  - 然后，需要使用 `cuEventRecord` 函数在默认流中记录第一个事件，并传入开始标志作为参数。
  - 接着，需要使用 `cuLaunchKernel` 函数在默认流中启动 GPU 内核，并传入相应的参数。
  - 然后，需要使用 `cuEventRecord` 函数在默认流中记录第二个事件，并传入结束标志作为参数。
  - 接着，需要使用 `cuEventSynchronize` 或 `cuCtxSynchronize` 函数来等待第二个事件完成。
  - 最后，需要使用 `cuEventElapsedTime` 函数来获取两个事件之间经过的时间，并打印出结果。然后，需要使用 `cuEventDestroy`、
    `cuModuleUnload` 和 `cuCtxDestroy` 函数来释放资源。

(1) CUDA Driver API :: CUDA Toolkit Documentation. https://docs.nvidia.com/cuda/cuda-driver-api/index.html.
(2) 浅谈Cuda driver API - 知乎. https://zhuanlan.zhihu.com/p/111602648.
(3) CUDA 编程手册系列 附录L – CUDA底层驱动API（一） - 知乎. https://zhuanlan.zhihu.com/p/561961012.
*/