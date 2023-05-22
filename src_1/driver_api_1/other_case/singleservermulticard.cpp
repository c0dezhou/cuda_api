/*
设计一些单机多卡的使用场景 以下是一些可能的使用场景：

- **使用设备管理、上下文管理、内存管理、模块管理、执行控制和流管理 api 来实现单机多卡的数据并行**¹²：

  - 首先，需要使用 `cuInit` 函数来初始化 cuda driver api，并使用 `cuDeviceGetCount` 和 `cuDeviceGet` 函数来获取可用的 GPU 设备数量和句柄。
  - 然后，需要使用 `cuCtxCreate` 函数来为每个 GPU 设备创建一个 GPU 上下文，并将它们保存在一个数组中。
  - 接着，需要使用 `cuModuleLoad` 函数来为每个 GPU 上下文加载一个包含 GPU 内核的 ptx 文件，并使用 `cuModuleGetFunction` 函数来获取内核函数的句柄。
  - 然后，需要使用 `cuMemAlloc` 函数来为每个 GPU 上下文分配 GPU 上的内存，并使用 `cuMemcpyHtoD` 函数来将主机上的数据复制到 GPU 上。
  - 接着，需要使用 `cuStreamCreate` 函数来为每个 GPU 上下文创建一个 GPU 流，并将它们保存在一个数组中。
  - 然后，需要在一个循环中执行以下操作：
    - 使用 `cuCtxSetCurrent` 函数来切换到当前循环的 GPU 上下文。
    - 使用 `cuLaunchKernel` 函数在当前循环的 GPU 流中启动 GPU 内核，并传入内核函数的句柄、网格和块的维度、共享内存的大小和内核参数的指针。
  - 最后，需要在另一个循环中执行以下操作：
    - 使用 `cuCtxSetCurrent` 函数来切换到当前循环的 GPU 上下文。
    - 使用 `cuStreamSynchronize` 函数来等待当前循环的 GPU 流中的操作完成，并使用 `cuMemcpyDtoH` 函数来将 GPU 上的结果复制到主机上。然后，需要使用 `cuMemFree`、`cuModuleUnload`、`cuStreamDestroy` 和 `cuCtxDestroy` 函数来释放当前循环的 GPU 上下文的资源。

- **使用设备管理、上下文管理、内存管理、模块管理、执行控制和事件管理 api 来实现单机多卡的模型并行**¹²：

  - 首先，需要使用 `cuInit` 函数来初始化 cuda driver api，并使用 `cuDeviceGetCount` 和 `cuDeviceGet` 函数来获取可用的 GPU 设备数量和句柄。
  - 然后，需要使用 `cuCtxCreate` 函数来为每个 GPU 设备创建一个 GPU 上下文，并将它们保存在一个数组中。
  - 接着，需要使用 `cuModuleLoad` 函数来为每个 GPU 上下文加载一个包含不同部分（例如不同层）的 GPU 内核的 ptx 文件，并使用 `cuModuleGetFunction` 函数来获取内核函数的句柄。
  - 然后，需要使用 `cuMemAlloc` 函数来为每个 GPU 上下文分配 GPU 上的内存，并使用 `cuMemcpyHtoD` 函数来将主机上的数据复制到 GPU 上。
  - 接着，需要使用 `cuEventCreate` 函数来为每个 GPU 上下文创建两个 GPU 事件，并设置它们为默认模式。
  - 然后，需要在一个循环中执行以下操作：
    - 使用 `cuCtxSetCurrent` 函数来切换到当前循环的 GPU 上下文。
    - 使用 `cuEventRecord` 函数在默认流中记录第一个事件，并传入开始标志作为参数。
    - 使用 `cuLaunchKernel` 函数在默认流中启动当前循环的 GPU 内核，并传入相应的参数。
    - 使用 `cuEventRecord` 函数在默认流中记录第二个事件，并传入结束标志作为参数。
    - 如果当前循环不是最后一个循环，则使用 `cuMemcpyPeerAsync` 函数在默认流中将当前循环的 GPU 上下文的输出异步复制到下一个循环的 GPU 上下文的输入，并传入相应的参数。
  - 最后，需要在另一个循环中执行以下操作：
    - 使用 `cuCtxSetCurrent` 函数来切换到当前循环的 GPU 上下文。
    - 使用 `cuEventSynchronize` 或 `cuCtxSynchronize` 函数来等待第二个事件完成。如果当前循环是最后一个循环，则使用 `cuMemcpyDtoH` 函数来将最终结果复制到主机上。然后，需要使用 `cuMemFree`、
      `cuModuleUnload`、`cuEventDestroy` 和 `cuCtxDestroy` 函数来释放当前循环的 GPU 上下文的资源。

(1) 附录L - CUDA 底层驱动 API - NVIDIA 技术博客. https://developer.nvidia.com/zh-cn/blog/cuda-low-level-driver-api-cn/.
(2) Pytorch的并行化-单机多GPU卡 - 知乎 - 知乎专栏. https://zhuanlan.zhihu.com/p/343891349.
(3) 浅谈Cuda driver API - 知乎. https://zhuanlan.zhihu.com/p/111602648.
*/

/*
有一些更复杂的场景，例如：

- **使用设备管理、上下文管理、内存管理、模块管理、执行控制和虚拟内存管理 api 来实现单机多卡的动态内存分配**¹：

  - 首先，需要使用 `cuInit` 函数来初始化 cuda driver api，并使用 `cuDeviceGetCount` 和 `cuDeviceGet` 函数来获取可用的 GPU 设备数量和句柄。
  - 然后，需要使用 `cuCtxCreate` 函数来为每个 GPU 设备创建一个 GPU 上下文，并将它们保存在一个数组中。
  - 接着，需要使用 `cuModuleLoad` 函数来为每个 GPU 上下文加载一个包含 GPU 内核的 ptx 文件，并使用 `cuModuleGetFunction` 函数来获取内核函数的句柄。
  - 然后，需要使用 `cuMemCreate` 函数来为每个 GPU 上下文创建一个虚拟内存对象，并设置其大小和属性。这个虚拟内存对象可以在不同的 GPU 上下文之间共享和迁移。
  - 接着，需要使用 `cuMemMap` 函数来将虚拟内存对象映射到每个 GPU 上下文的地址空间，并获取其设备指针。然后，可以使用 `cuMemcpyHtoD` 函数来将主机上的数据复制到虚拟内存对象中。
  - 然后，需要在一个循环中执行以下操作：
    - 使用 `cuCtxSetCurrent` 函数来切换到当前循环的 GPU 上下文。
    - 使用 `cuLaunchKernel` 函数在默认流中启动 GPU 内核，并传入内核函数的句柄、网格和块的维度、共享内存的大小和内核参数的指针。其中，内核参数的指针应该指向虚拟内存对象的设备指针。
    - 如果当前循环不是最后一个循环，则使用 `cuMemSetAccess` 函数来修改虚拟内存对象的访问策略，使其可以被下一个循环的 GPU 上下文访问。这样，就可以实现虚拟内存对象在不同的 GPU 上下文之间的动态迁移。
  - 最后，需要在另一个循环中执行以下操作：
    - 使用 `cuCtxSetCurrent` 函数来切换到当前循环的 GPU 上下文。
    - 如果当前循环是最后一个循环，则使用 `cuMemcpyDtoH` 函数来将虚拟内存对象中的结果复制到主机上。然后，需要使用 `cuMemUnmap`、
      `cuMemRelease`、`cuModuleUnload` 和 `cuCtxDestroy` 函数来释放当前循环的 GPU 上下文的资源。

(1) 浅谈Cuda driver API - 知乎. https://zhuanlan.zhihu.com/p/111602648.
(2) CUDA Driver API :: CUDA Toolkit Documentation. https://docs.nvidia.com/cuda/cuda-driver-api/index.html.
(3) 针对深度学习的GPU共享 - 知乎 - 知乎专栏. https://zhuanlan.zhihu.com/p/285994980.
*/

/*
设计一些验证卡间通信的案例，例如：

- **使用设备管理、上下文管理、内存管理、模块管理、执行控制和对等上下文内存访问 api 来实现单机多卡的直接内存访问**¹：

  - 首先，需要使用 `cuInit` 函数来初始化 cuda driver api，并使用 `cuDeviceGetCount` 和 `cuDeviceGet` 函数来获取可用的 GPU 设备数量和句柄。
  - 然后，需要使用 `cuCtxCreate` 函数来为每个 GPU 设备创建一个 GPU 上下文，并将它们保存在一个数组中。
  - 接着，需要使用 `cuModuleLoad` 函数来为每个 GPU 上下文加载一个包含 GPU 内核的 ptx 文件，并使用 `cuModuleGetFunction` 函数来获取内核函数的句柄。
  - 然后，需要使用 `cuMemAlloc` 函数来为每个 GPU 上下文分配 GPU 上的内存，并使用 `cuMemcpyHtoD` 函数来将主机上的数据复制到 GPU 上。
  - 接着，需要使用 `cuDeviceCanAccessPeer` 函数来检查每对 GPU 设备之间是否可以进行直接内存访问，并使用 `cuCtxEnablePeerAccess` 函数来启用对等上下文内存访问。
  - 然后，需要在一个循环中执行以下操作：
    - 使用 `cuCtxSetCurrent` 函数来切换到当前循环的 GPU 上下文。
    - 使用 `cuLaunchKernel` 函数在默认流中启动 GPU 内核，并传入内核函数的句柄、网格和块的维度、共享内存的大小和内核参数的指针。其中，内核参数的指针应该指向其他 GPU 上下文的内存地址，以实现直接内存访问。
  - 最后，需要在另一个循环中执行以下操作：
    - 使用 `cuCtxSetCurrent` 函数来切换到当前循环的 GPU 上下文。
    - 使用 `cuMemcpyDtoH` 函数来将当前循环的 GPU 上下文的结果复制到主机上。然后，需要使用 `cuCtxDisablePeerAccess`、
      `cuMemFree`、`cuModuleUnload` 和 `cuCtxDestroy` 函数来释放当前循环的 GPU 上下文的资源。

源: 与必应的对话， 2023/5/19
(1) CUDA Driver API :: CUDA Toolkit Documentation. https://docs.nvidia.com/cuda/cuda-driver-api/index.html.
(2) 附录L - CUDA 底层驱动 API - NVIDIA 技术博客. https://developer.nvidia.com/zh-cn/blog/cuda-low-level-driver-api-cn/.
(3) 针对深度学习的GPU共享 - 知乎 - 知乎专栏. https://zhuanlan.zhihu.com/p/285994980.
*/

/*
可能会触发异常但不一定会触发的案例，那么你可以尝试以下的场景：

- **使用设备管理、上下文管理、内存管理、模块管理、执行控制和对等上下文内存访问 api 来实现单机多卡的直接内存访问，但在不同的流中进行内存访问，可能会导致数据竞争或内存不一致**：

  - 首先，需要使用 `cuInit` 函数来初始化 cuda driver api，并使用 `cuDeviceGetCount` 和 `cuDeviceGet` 函数来获取可用的 GPU 设备数量和句柄。
  - 然后，需要使用 `cuCtxCreate` 函数来为每个 GPU 设备创建一个 GPU 上下文，并将它们保存在一个数组中。
  - 接着，需要使用 `cuModuleLoad` 函数来为每个 GPU 上下文加载一个包含 GPU 内核的 ptx 文件，并使用 `cuModuleGetFunction` 函数来获取内核函数的句柄。
  - 然后，需要使用 `cuMemAlloc` 函数来为每个 GPU 上下文分配 GPU 上的内存，并使用 `cuMemcpyHtoD` 函数来将主机上的数据复制到 GPU 上。
  - 接着，需要使用 `cuDeviceCanAccessPeer` 函数来检查每对 GPU 设备之间是否可以进行直接内存访问，并使用 `cuCtxEnablePeerAccess` 函数来启用对等上下文内存访问。
  - 然后，需要为每个 GPU 上下文创建一个非默认流，并将它们保存在一个数组中。
  - 然后，需要在一个循环中执行以下操作：
    - 使用 `cuCtxSetCurrent` 函数来切换到当前循环的 GPU 上下文。
    - 使用 `cuLaunchKernel` 函数在当前循环的非默认流中启动 GPU 内核，并传入内核函数的句柄、网格和块的维度、共享内存的大小和内核参数的指针。其中，内核参数的指针应该指向其他 GPU 上下文的内存地址，以实现直接内存访问。
  - 最后，需要在另一个循环中执行以下操作：
    - 使用 `cuCtxSetCurrent` 函数来切换到当前循环的 GPU 上下文。
    - 使用 `cuMemcpyDtoHAsync` 函数来将当前循环的 GPU 上下文的结果异步复制到主机上，并指定当前循环的非默认流作为复制操作所属的流。然后，需要使用 `cuStreamSynchronize` 函数来等待复制操作完成。然后，需要使用 `cuCtxDisablePeerAccess`、
      `cuMemFree`、`cuModuleUnload`、`cuStreamDestroy` 和 `cuCtxDestroy` 函数来释放当前循环的 GPU 上下文的资源。

这个场景可能会触发异常，因为不同的流之间没有隐式同步，所以如果一个流中的内核正在修改某个内存地址，而另一个流中的内核或者主机端代码正在读取或者写入同一个内存地址，就可能会导致数据竞争或者内存不一致。要避免这种情况，可以使用一些同步机制，如事件、信号量或者互斥锁等。
*/

/*
一些多线程情景，例如：

- **使用设备管理、上下文管理、内存管理、模块管理、执行控制和对等上下文内存访问 api 来实现单机多卡的直接内存访问，但在不同的 CPU 线程中创建和销毁 GPU 上下文，可能会导致上下文切换或者资源泄露**：

  - 首先，需要使用 `cuInit` 函数来初始化 cuda driver api，并使用 `cuDeviceGetCount` 和 `cuDeviceGet` 函数来获取可用的 GPU 设备数量和句柄。
  - 然后，需要创建一个线程池，并为每个 GPU 设备分配一个 CPU 线程。
  - 接着，在每个 CPU 线程中执行以下操作：
    - 使用 `cuCtxCreate` 函数来为当前线程分配的 GPU 设备创建一个 GPU 上下文，并将它保存在一个局部变量中。
    - 使用 `cuModuleLoad` 函数来为当前 GPU 上下文加载一个包含 GPU 内核的 ptx 文件，并使用 `cuModuleGetFunction` 函数来获取内核函数的句柄。
    - 使用 `cuMemAlloc` 函数来为当前 GPU 上下文分配 GPU 上的内存，并使用 `cuMemcpyHtoD` 函数来将主机上的数据复制到 GPU 上。
    - 使用 `cuDeviceCanAccessPeer` 函数来检查当前 GPU 设备与其他 GPU 设备之间是否可以进行直接内存访问，并使用 `cuCtxEnablePeerAccess` 函数来启用对等上下文内存访问。
    - 使用 `cuLaunchKernel` 函数在默认流中启动 GPU 内核，并传入内核函数的句柄、网格和块的维度、共享内存的大小和内核参数的指针。其中，内核参数的指针应该指向其他 GPU 上下文的内存地址，以实现直接内存访问。
    - 使用 `cuMemcpyDtoH` 函数来将当前 GPU 上下文的结果复制到主机上。然后，需要使用 `cuCtxDisablePeerAccess`、
      `cuMemFree`、`cuModuleUnload` 和 `cuCtxDestroy` 函数来释放当前 GPU 上下文的资源。

这个场景可能会触发异常，因为不同的 CPU 线程之间没有隐式同步，所以如果一个线程中的 GPU 上下文正在修改某个内存地址，而另一个线程中的 GPU 上下文或者主机端代码正在读取或者写入同一个内存地址，就可能会导致数据竞争或者内存不一致。要避免这种情况，可以使用一些同步机制，如互斥锁或者条件变量等。另外，如果一个线程中创建了一个 GPU 上下文，但没有正确地销毁它，就可能会导致资源泄露或者上下文切换。要避免这种情况，可以使用一些异常处理机制，如 try-catch-finally 或者 RAII 等。

(1) 附录L - CUDA 底层驱动 API - NVIDIA 技术博客. https://developer.nvidia.com/zh-cn/blog/cuda-low-level-driver-api-cn/.
(2) CUDA Driver API :: CUDA Toolkit Documentation. https://docs.nvidia.com/cuda/cuda-driver-api/index.html.
(3) 总体流程设计（1）-CUDA程序的等级结构 - 知乎 - 知乎专栏. https://zhuanlan.zhihu.com/p/129375374.
(4) 浅谈Cuda driver API - 知乎. https://zhuanlan.zhihu.com/p/111602648.
*/