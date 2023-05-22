  checkDriver(cuLaunchKernel(kernel,
                             grid_size,
                             block_size,
                             args,
                             NULL,
                             NULL));

   // 等待内核执行完成
   checkDriver(cuCtxSynchronize());

   // 复制设备数据到主机
   int h_c[N];
   checkDriver(cuMemcpyDtoH(h_c,d_c,size));

   // 打印结果
   for(int i=0;i<N;i++){
     printf("%d + %d = %d\n",h_a[i],h_b[i],h_c[i]);
   }

   //释放设备内存
   checkDriver(cuMemFree(d_a));
   checkDriver(cuMemFree(d_b));
   checkDriver(cuMemFree(d_c));

   // 销毁上下文
   checkDriver(cuCtxDestroy(context));

   printf("Process %d finished.\n", process_id);
}

int main(){
  // 检查cuda driver的初始化
  checkDriver(cuInit(0));

  // 获取设备句柄
  checkDriver(cuDeviceGet(&device, 0));

  // 加载PTX或CUBIN文件
  checkDriver(cuModuleLoad(&module, "kernel.ptx"));

  // 获取内核函数句柄
  checkDriver(cuModuleGetFunction(&kernel, module, "add"));

  // 定义一个常量M，表示进程的数量
  const int M = 4;

  // 创建M个子进程
  for(int i=0;i<M;i++){
    pid_t pid = fork();
    if(pid == 0){
      // 子进程中执行内核函数
      run_kernel(i); // 进程ID为i
      // 子进程退出
      exit(0);
    }
    else if(pid > 0){
      // 父进程中打印子进程的ID
      printf("Process %d created.\n", pid);
    }
    else{
      // fork失败，打印错误信息
      perror("fork");
    }
  }

  // 父进程中等待所有子进程结束
  for(int i=0;i<M;i++){
    int status;
    pid_t pid = wait(&status);
    printf("Process %d finished with status %d.\n", pid, status);
  }

  // 卸载模块
  checkDriver(cuModuleUnload(module));

  return 0;
}
