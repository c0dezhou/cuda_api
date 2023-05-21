/*

根据我在网上找到的信息，如果你现在有cuda kernel的.cu文件，你可以使用以下步骤来编译成fatbin二进制文件：

1. 使用nvcc命令将.cu文件编译成.ptx和.cubin文件，例如：`nvcc -arch=compute_35 -code=sm_35 test.cu -ptx -cubin`
2. 使用fatbinary命令将.ptx和.cubin文件合并成.fatbin文件，例如：`fatbinary --create=test.fatbin -64 --image=profile=sm_35,file=test.cubin --image=profile=compute_35,file=test.ptx`
3. 使用clang-offload-bundler命令将.fatbin文件嵌入到.fatbin.c文件中，例如：`clang-offload-bundler -type=o -targets=hip-amdgcn-amd-amdhsa--gfx906 -inputs=test.fatbin -outputs=test.fatbin.c`
4. 使用c++编译器将.fatbin.c文件和其他host代码链接成可执行文件或共享库，例如：`g++ test.cpp test.fatbin.c -o test`



(1) AMD 编译概述 & Fatbin 文件生成 & HIP Runtime API（启动 CUDA 核函数） - 知乎. https://zhuanlan.zhihu.com/p/390617013.
(2) CUDA编译过程_cuda 代码编译的过程_s.feng的博客-CSDN博客. https://blog.csdn.net/feng__shuai/article/details/111319185.
(3) CUDA以及NVCC编译流程_cuda编译_NewCoder_BL的博客-CSDN博客. https://blog.csdn.net/u011934885/article/details/72463090.
(4) 在LLVM中实现多架构编译 - 知乎 - 知乎专栏. https://zhuanlan.zhihu.com/p/67032644.

如果你想查询当前机器gpu的arch，你可以使用以下方法之一：

1. 使用deviceQuery程序来输出gpu的信息，包括CUDA Capability Major/Minor version number，例如：`./deviceQuery`。这个程序可以从cuda samples中编译或者从网上下载¹²。
2. 使用nvidia-smi命令来查看gpu的型号，例如：`nvidia-smi -L`。然后根据型号在网上查询对应的arch，例如：`https://developer.nvidia.com/cuda-gpus`³。
3. 如果是在jetson平台，可以使用jtop命令来查看gpu的信息，包括arch，例如：`jtop`。

(1) Intel® Arc™ Graphics Drivers. https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/software/drivers.html.
(2) NVIDIA之获得显卡GPU_ARCHS值的方法 - CSDN博客. https://blog.csdn.net/zong596568821xp/article/details/106411024.
(3) 【转载】各种 NVIDIA 架构所匹配的 arch 和 gencode - CSDN博客. https://blog.csdn.net/panda1234lee/article/details/110091930.
*/
