#include "stream_tests.h"


TEST_F(CuStreamTests, AC_BA_LaunchHostFunc_ValidHostFunction) {
    auto hostFunc = [](void* userData) {
        std::atomic<int>* counter = static_cast<std::atomic<int>*>(userData);
        (*counter)++;
    };

    std::atomic<int> counter(0);

    CUresult result = cuLaunchHostFunc(cuStream, hostFunc, &counter);
    EXPECT_EQ(result, CUDA_SUCCESS);

    cuStreamSynchronize(cuStream);

    EXPECT_EQ(counter.load(), 1);
}

TEST_F(CuStreamTests, AC_INV_LaunchHostFunc_InvalidHostFunction) {
    auto invalidHostFunc = [](void* userData) {
        CUdevice cuDevice;
        cuDeviceGet(&cuDevice, 0); 
    };

    CUresult result = cuLaunchHostFunc(cuStream, invalidHostFunc, nullptr);
    EXPECT_EQ(result, CUDA_SUCCESS);

    cuStreamSynchronize(cuStream);
}

std::atomic<int> orderCounter(0);


void hostFunc1(void* userData) {
    int* orderArray = static_cast<int*>(userData);
    orderArray[0] = ++orderCounter;
};

void hostFunc2(void* userData) {
    int* orderArray = static_cast<int*>(userData);
    orderArray[1] = ++orderCounter;
};

TEST_F(CuStreamTests, AC_BA_LaunchHostFunc_HostFunctionOrder) {
    int orderArray[2] = {0, 0};

    CUresult result1 = cuLaunchHostFunc(cuStream, hostFunc1, orderArray);
    CUresult result2 = cuLaunchHostFunc(cuStream, hostFunc2, orderArray);

    EXPECT_EQ(result1, CUDA_SUCCESS);
    EXPECT_EQ(result2, CUDA_SUCCESS);

    cuStreamSynchronize(cuStream);

    // inorderd
    EXPECT_EQ(orderArray[0], 1);
    EXPECT_EQ(orderArray[1], 2);
}
