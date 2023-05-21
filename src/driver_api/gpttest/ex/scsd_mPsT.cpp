在一个GPU上执行多个任务的一个可能的方法是，使用CUDA Multi-Process Server (MPS)来将来自不同进程的CUDA操作合并到一个上下文中，从而提高GPU的并发性和利用率。MPS可以让多个进程共享一个GPU，而不需要进行上下文切换或内存复制。MPS的使用方法和限制可以参考CUDA文档²。另一个可能的方法是，使用CUDA stream来将不同任务的内核分配到不同的流中，然后使用流优先级或流回调来控制流之间的执行顺序和依赖关系。这种方法可以让多个任务在同一个进程和上下文中并行执行，但是需要注意内存和资源的管理和分配。使用CUDA stream的方法和示例可以参考CUDA文档³。

(1) Running more than one CUDA applications on one GPU. https://stackoverflow.com/questions/31643570/running-more-than-one-cuda-applications-on-one-gpu.
(2) How to run multiple jobs on a GPU grid with CUDA using SLURM. https://stackoverflow.com/questions/51867273/how-to-run-multiple-jobs-on-a-gpu-grid-with-cuda-using-slurm.
(3) USING CUDA AND X | NVIDIA. https://nvidia.custhelp.com/app/answers/detail/a_id/3029/~/using-cuda-and-x.

这是一个可能的代码示例，使用MPS来运行多个非MPI的CUDA应用程序。假设我们有一个简单的向量加法的CUDA程序，我们想要用多个进程同时运行它，并且利用MPS来提高GPU的并发性和利用率。我们可以使用以下步骤来实现：

1. 编译向量加法的CUDA程序，假设生成的可执行文件名为vecAdd。
2. 启动MPS服务，使用nvidia-smi命令设置GPU的计算模式为默认模式，并且指定MPS管道目录和日志目录。例如：
```bash
nvidia-smi -c DEFAULT # set compute mode to default
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps # set MPS pipe directory
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log # set MPS log directory
nvidia-cuda-mps-control -d # start MPS daemon
```
3. 在多个终端中分别启动一个进程，运行vecAdd程序，并且指定不同的输入和输出文件。例如：
```bash
./vecAdd input1.txt output1.txt # run vecAdd in process 1
./vecAdd input2.txt output2.txt # run vecAdd in process 2
./vecAdd input3.txt output3.txt # run vecAdd in process 3
./vecAdd input4.txt output4.txt # run vecAdd in process 4
```
4. 检查输出文件，验证结果是否正确。
5. 停止MPS服务，使用nvidia-cuda-mps-control命令并输入quit。例如：
```bash
nvidia-cuda-mps-control # enter interactive mode
quit # quit MPS control
```

源: 与必应的对话， 2023/5/21
(1) Multi-Process Service :: GPU Deployment and Management Documentation. https://docs.nvidia.com/deploy/mps/index.html.
(2) gpu - How do I use Nvidia Multi-process Service (MPS) to run multiple .... https://stackoverflow.com/questions/34709749/how-do-i-use-nvidia-multi-process-service-mps-to-run-multiple-non-mpi-cuda-app.
(3) GitHub - NVIDIA/cuda-samples: Samples for CUDA Developers which .... https://github.com/NVIDIA/cuda-samples.