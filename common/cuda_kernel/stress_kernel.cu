#include <cuda.h>
#include <math.h>
#include <vector>

// 输入是一个左矩阵MxK，右矩阵KxN，输出矩阵MxN
__global__ void matrixMultiplyKernel(const float* A,
                                                const float* B,
                                                float* C,
                                                int M,
                                                int N,
                                                int K) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (ty < M && tx < N) {
        float c = 0;
        for (int i = 0; i < K; ++i) {
            c += A[ty * K + i] * B[i * N + tx];
        }
        C[ty * N + tx] = c;
    }
}


__global__ void quickSort(int* array, int left, int right) {

    if (left < right) {
        int pivotIndex;
        [&array, &left, &right, &pivotIndex]() {
            int pivot = array[right];
            int i = left - 1;

            for (int j = left; j <= right - 1; j++) {
                if (array[j] < pivot) {
                    i++;
                    int temp = array[i];
                    array[i] = array[j];
                    array[j] = temp;
                }
            }
            int temp = array[i + 1];
            array[i + 1] = array[right];
            array[right] = temp;

            pivotIndex =  i + 1;
        };

        quickSort<<<1, 1>>>(array, left, pivotIndex - 1);
        quickSort<<<1, 1>>>(array, pivotIndex + 1, right);
    }
}

__global__ void binarySearch(const int* array, int target, int* result, int array_size) {
    int left = 0;
    int right = array_size - 1;

    while (left <= right) {
        int mid = (left + right) / 2;

        if (array[mid] == target) {
            *result = mid;
            return;
        } else if (array[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    *result = -1;
}

__global__ void forward(float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = input[i] * input[i];
    }
}

__global__ void backward(float* output, float* input, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        input[i] = cbrtf(output[i]);
    }
}
