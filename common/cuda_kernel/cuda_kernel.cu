#include <cuda.h>
#include <vector>
#include <math.h>

#define CLOCK_RATE 1695000  // not sure, use cudaGetDeviceProperties to ensure that
__device__ void sleep_device(float t) {
    clock_t t0 = clock64();
    clock_t t1 = t0;
    while ((t1 - t0) / (CLOCK_RATE * 1000.0f) < t)
        t1 = clock64();
}

__global__ void vecAdd(float* A, float* B, float* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];
}

__global__ void vecScale(float* A, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    A[i] = A[i] * factor;
}

__global__ void add(int a, int b, int *result) {
    *result = a + b;
}

__global__ void addKernel(int* c, const int* a, const int* b) {
    sleep_device(3);
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
    
}

__global__ void delay_device(float seconds) {
    sleep_device(seconds);
}

__global__ void arraySelfIncrement(int* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i]++;
    }
};

__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void mul_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void sub_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

__global__ void div_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && b[i] != 0) {
        c[i] = a[i] / b[i];
    }
}

__global__ void vec_multiply_2(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] *= 2.0f;
    }
}

__global__ void vec_add_3(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] += 3.0f;
    }
}

__global__ void vec_sub_1(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] -= 1.0f;
    }
}

__global__ void vec_multiply_2_withidx(float* data, int startIndex, int endIndex) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= startIndex && tid < endIndex) {
        data[tid] *= 2.0f;
    }
}

__global__ void vec_add_3_withidx(float* data, int startIndex, int endIndex) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= startIndex && tid < endIndex) {
        data[tid] += 3.0f;
    }
}