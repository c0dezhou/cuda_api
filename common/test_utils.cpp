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

bool __checkError_error(CUresult code,
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

bool verifyResult(const std::vector<float>& data,
                  float expectedValue,
                  int startIndex,
                  int endIndex) {
    bool flag = true;
    for (int i = startIndex; i < endIndex; i++) {
        if (data[i] != expectedValue) {
            std::cout << "unexpected val:" << data[i] << " at " << i;
            flag = false;
        }
    }
    return flag;
}

bool verifyResult(const std::vector<float>& data, float expectedValue) {
    bool flag = true;
    for (int i = 0; i < data.size(); i++) {
        if (std::abs(data[i] - expectedValue) > 1e-5) {
            std::cout << "error index: " << i << "expect " << data[i]
                      << std::endl;
            flag = false;
        }
    }
    return flag;
}

bool verifyResult(const std::vector<float>& h_A,
                  const std::vector<float>& h_B,
                  const std::vector<float>& h_C) {
    bool flag = true;
    for (size_t i = 0; i < h_C.size(); ++i) {
        float expected = h_A[i] + h_B[i];
        std::cout << expected << " " << std::endl;
        if (h_C[i] != expected) {
            std::cout << "Verification failed at index " << i << std::endl;
            flag = false;
        }
    }
    return flag;
}

void cpuComputation(std::vector<float>& data) {
    for (float& value : data) {
        value *= 2.0f;
    }
}