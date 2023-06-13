#include "test_utils.h"

#define SIZE_2MB  (2<<20)
#define SIZE_10MB (10<<20)
#define SIZE_4GB  ((size_t)4<<30)
#define SIZE_8GB  ((size_t)8<<30)
#define SIZE_15GB  ((size_t)15<<30)

void fillAndCheck(SDdeviceptr d_src, SDdeviceptr d_dest, size_t size) {
    unsigned char pattern = 0xAB; // Any pattern
    std::vector<unsigned char> h_data(size);

    checkError(cuMemsetD8(d_src, pattern, size));
    checkError(cuMemcpyDtoD(d_dest, d_src, size));

    checkError(cuMemcpyDtoH(h_data.data(), d_dest, size));
    for (size_t i = 0; i < size; i++) {
        if (h_data[i] != pattern) {
            std::cerr << "Verification failed at position " << i << std::endl;
            exit(1);
        }
    }
}

void mem_d2h_d2d_set(int d) {
    checkError(cuInit(0));

    SDdevice cuDevice;
    checkError(cuDeviceGet(&cuDevice, d));

    SDcontext cuContext;
    checkError(cuCtxCreate(&cuContext, 0, cuDevice));

    // Perform allocation, copy, check, release in a loop
    for (int N = 1; N <= 5; N++) {
        // Sizes 2MB, 10MB, 2MB * N, 4GB, 8GB
        std::vector<size_t> sizes = {SIZE_2MB, SIZE_10MB, SIZE_2MB * (size_t)N, SIZE_4GB, SIZE_8GB};

        for (auto size : sizes) {
            SDdeviceptr d_src, d_dest;
            checkError(cuMemAlloc(&d_src, size));
            checkError(cuMemAlloc(&d_dest, size));

            fillAndCheck(d_src, d_dest, size);

            checkError(cuMemFree(d_src));
            checkError(cuMemFree(d_dest));
        }
    }

    checkError(cuCtxDestroy(cuContext));
}

TEST(STRESS_NEW, mem_d2h_d2d_set) {
    int dev_count;
    checkError(cuInit(0));
    checkError(cuDeviceGetCount(&dev_count));

    for (int d = 0; d < dev_count; d++) {
        mem_d2h_d2d_set(d);
    }
}
