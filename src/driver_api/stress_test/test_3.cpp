#include <stdio.h>
#include <cuda.h>

// 定义一些常量
#define NUM_THREADS 1024 // 每个块的线程数
#define NUM_BLOCKS 65536 // 每个网格的块数
#define NUM_ITERATIONS 1000 // 每个线程的迭代次数
#define MEM_SIZE 1024 * 1024 * 1024 // 要分配的内存大小

// 一个简单的核函数，执行一些计算并写入全局内存
__global__ void stress_kernel(int *data) {
    // 获取线程索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 初始化一个局部变量
    int x = tid;
    // 循环一定次数
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        // 执行一些计算
        x = (x * x + i) % MEM_SIZE;
        // 写入全局内存
        data[x] = tid;
    }
}

int main() {
    // 声明一个指向设备内存的指针
    int *d_data;
    // 分配设备内存
    cudaError_t err = cudaMalloc(&d_data, MEM_SIZE);
    // 检查错误
    if (err != cudaSuccess) {
        printf("分配设备内存出错: %s\n", cudaGetErrorString(err));
        return -1;
    }
    // 使用大量的块和线程启动核函数
    stress_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_data);
    // 检查错误
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("启动核函数出错: %s\n", cudaGetErrorString(err));
        return -1;
    }
    // 同步设备
    err = cudaDeviceSynchronize();
    // 检查错误
    if (err != cudaSuccess) {
        printf("同步设备出错: %s\n", cudaGetErrorString(err));
        return -1;
    }
    // 释放设备内存
    err = cudaFree(d_data);
    // 检查错误
    if (err != cudaSuccess) {
        printf("释放设备内存出错: %s\n", cudaGetErrorString(err));
        return -1;
    }
    // 返回成功
    return 0;
}
