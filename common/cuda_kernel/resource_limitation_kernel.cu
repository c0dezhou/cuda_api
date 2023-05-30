#include <cuda.h>
__global__ void dummy_kernel() {
}

__global__ void infinite_kernel() {
    // Loop forever
    while (true) {
    }
}