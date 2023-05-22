// 假设你已经安装了cuda toolkit，并且有一个cuda device可用
#include <iostream>
#include <thread>
#include <vector>
#include <cuda.h>

// 定义一个函数指针的类，用于存储不同的cuda kernel函数
class KernelFunc {
public:
    // 构造函数，接受一个指向cuda kernel函数的指针
    KernelFunc(void (*func)(int*, int*, int)) {
        this->func = func;
    }

    // 调用函数指针，传入参数
    void operator()(int* a, int* b, int n) {
        func(a, b, n);
    }

private:
    // 函数指针，指向一个接受两个int数组和一个int大小的cuda kernel函数
    void (*func)(int*, int*, int);
};

// 定义一个简单的cuda kernel函数，用于计算两个数组的和
__global__ void add(int* a, int* b, int n) {
    // 获取线程索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 检查边界
    if (index < n) {
        // 计算和
        a[index] = a[index] + b[index];
    }
}

// 定义一个简单的cuda kernel函数，用于计算两个数组的差
__global__ void sub(int* a, int* b, int n) {
    // 获取线程索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 检查边界
    if (index < n) {
        // 计算差
        a[index] = a[index] - b[index];
    }
}

// 定义一个函数，用于创建并执行多个线程，每个线程调用不同的cuda kernel函数
void run_threads(std::vector<KernelFunc> funcs, int* a, int* b, int n) {
    // 获取可用的cuda device数量
    int device_count;
    cudaGetDeviceCount(&device_count);
    std::cout << "Available cuda devices: " << device_count << std::endl;

    // 创建一个线程数组，大小为函数数组的大小
    std::thread threads[funcs.size()];

    // 遍历每个函数
    for (int i = 0; i < funcs.size(); i++) {
        // 设置当前线程使用的cuda device，假设每个线程使用不同的device
        cudaSetDevice(i % device_count);
        std::cout << "Thread " << i << " using device " << i % device_count << std::endl;

        // 创建一个线程，调用当前函数，并传入参数
        threads[i] = std::thread(funcs[i], a, b, n);
    }

    // 等待所有线程完成
    for (int i = 0; i < funcs.size(); i++) {
        threads[i].join();
    }
}

// 主函数，测试多线程并发调用不同的cuda kernel函数
int main() {
    // 定义数组大小
    const int N = 10;

    // 分配主机内存，并初始化数据
    int* h_a = new int[N];
    int* h_b = new int[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // 分配设备内存，并拷贝数据
    int* d_a;
    int* d_b;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // 定义两个函数指针的类对象，分别指向add和sub函数
    KernelFunc add_func(add);
    KernelFunc sub_func(sub);

    // 创建一个函数指针的类对象数组，存储两个对象
    std::vector<KernelFunc> funcs;
    funcs.push_back(add_func);
    funcs.push_back(sub_func);

    // 调用run_threads函数，传入函数对象数组和参数
    run_threads(funcs, d_a, d_b, N);

    // 拷贝结果回主机内存，并打印
    cudaMemcpy(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);
    
for (int i = 0; i < N; i++) {
        std::cout << h_a[i] << std::endl;
}

// 释放内存，并返回
delete[] h_a;
delete[] h_b;
cudaFree(d_a);
cudaFree(d_b);
return 0;
}
