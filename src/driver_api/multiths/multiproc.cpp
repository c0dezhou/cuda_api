// A function to run a Handle object in a process
void runHandle(Handle handle) {
  handle(); // Call the Handle object
}

// A function to create and wait for processes for each Handle object in the array
void testHandles() {
  int n = sizeof(handles) / sizeof(handles[0]); // Get the number of Handle objects in the array
  pid_t pids[n]; // Create an array of process IDs
  for (int i = 0; i < n; i++) {
    pids[i] = fork(); // Create a child process for each Handle object
    if (pids[i] == 0) {
      runHandle(handles[i]); // Run the Handle object in the child process
      exit(0); // Exit the child process
    }
    else if (pids[i] < 0) {
      printf("fork failed: %d\n", pids[i]); // Print an error message if fork failed
      exit(1); // Exit the parent process with an error code
    }
  }
  for (int i = 0; i < n; i++) {
    waitpid(pids[i], NULL, 0); // Wait for each child process to finish
  }
}

/*
MPS的使用并没有在这段代码中体现，而是在运行这段代码之前需要做的一些配置和启动。MPS是一种CUDA服务，可以让多个进程共享一个GPU，而不需要每个进程都创建自己的上下文。这样可以减少上下文切换的开销，提高GPU的利用率。要使用MPS，你需要：

- 确保你的GPU支持MPS，并且安装了最新的CUDA驱动和工具包。
- 设置环境变量CUDA_MPS_PIPE_DIRECTORY和CUDA_MPS_LOG_DIRECTORY，指定MPS的管道和日志的目录。
- 启动MPS守护进程（nvidia-cuda-mps-control），并输入start命令。
- 在你的代码中，使用cuCtxCreate_v2函数来创建上下文，并指定CU_CTX_SCHED_AUTO或CU_CTX_SCHED_SPIN作为标志。
- 在你的代码结束后，停止MPS守护进程（nvidia-cuda-mps-control），并输入quit命令。

你可以参考这个链接来了解更多关于MPS的信息：https://docs.nvidia.com/deploy/mps/index.html
*/

// spawn

// A function to run a Handle object in a process
void runHandle(Handle handle) {
  handle(); // Call the Handle object
}

// A function to create and wait for processes for each Handle object in the array
void testHandles() {
  int n = sizeof(handles) / sizeof(handles[0]); // Get the number of Handle objects in the array
  pid_t pids[n]; // Create an array of process IDs
  for (int i = 0; i < n; i++) {
    pids[i] = posix_spawn(NULL, NULL, NULL, NULL, runHandle, handles[i]); // Create a child process for each Handle object and pass it as an argument
    if (pids[i] < 0) {
      printf("posix_spawn failed: %d\n", pids[i]); // Print an error message if posix_spawn failed
      exit(1); // Exit the parent process with an error code
    }
  }
  for (int i = 0; i < n; i++) {
    posix_waitpid(pids[i], NULL, 0); // Wait for each child process to finish
  }
}
