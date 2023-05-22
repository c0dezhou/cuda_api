#include "test_utils.h"

void performOperations(int* p, size_t alloc_size) {
    for (auto i = 0; i < alloc_size; i++) {
        p[i] = static_cast<int>(i);
    }
};

void initarray(int* p, size_t alloc_size) {
    for (auto i = 0; i < alloc_size; i++) {
        p[i] = static_cast<int>(i);
    }
};

void compareOPerations(int* p, size_t alloc_size) {
    for (auto i = 0; i < alloc_size; ++i) {
        EXPECT_EQ(p[i], i);
    }
};

long long getSystemMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;  // in kilobytes
}

float calculateElapsedTime(const CUevent& start, const CUevent& stop) {
    float elapsedTime;
    cuEventElapsedTime(&elapsedTime, start, stop);
    return elapsedTime;
}

void get_random(int* num, int a, int b) {  // [a, b]
    srand((unsigned)time(NULL));
    *num = (rand() % (b - a + 1)) + a;
}


bool __check_cuda_error(CUresult code,
                         const char* op,
                         const char* file,
                         int line) {
    if (code != CUresult::CUDA_SUCCESS) {
        const char* err_name = nullptr;
        const char* err_message = nullptr;
        cuGetErrorName(code, &err_name);
        cuGetErrorString(code, &err_message);
        printf("%s:%d %s failed. \n code = %s, message = %s\n", file, line, op,
               err_name, err_message);
        return false;
    }
    return true;
}