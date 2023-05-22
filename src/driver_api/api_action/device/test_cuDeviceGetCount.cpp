#include "test_utils.h"

TEST(CuDeviceTest_, AC_BA_DeviceGetCount_BasicBehavior) {
    cuInit(0);
    int count;
    CUresult status = cuDeviceGetCount(&count);
    EXPECT_EQ(status, CUDA_SUCCESS);
}

TEST(CuDeviceTest_, AC_BA_DeviceGetCount_InvalidValue) {
    cuInit(0);
    CUresult status = cuDeviceGetCount(nullptr);
    EXPECT_EQ(status, CUDA_ERROR_INVALID_VALUE);
}

TEST(CuDeviceTest_, AC_BA_DeviceGetCount_MultipleDevices) {
    cuInit(0);
    int count;
    CUresult status = cuDeviceGetCount(&count);
    EXPECT_EQ(status, CUDA_SUCCESS);
    EXPECT_GE(count, 2);
}

TEST(CuDeviceTest_, PERF_Device_DeviceGetCount_masuretime) {
    cuInit(0);
    auto start = std::chrono::high_resolution_clock::now();
    int count;
    CUresult status = cuDeviceGetCount(&count);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    EXPECT_LT(duration, 10);
}

TEST(CuDeviceTest_, AC_BA_DeviceGetCount_RepeatedCalls) {
    cuInit(0);
    int valid_count;
    cuDeviceGetCount(&valid_count);
    for (int i = 0; i < 100; i++) {
        int count;
        CUresult status = cuDeviceGetCount(&count);
        EXPECT_EQ(status, CUDA_SUCCESS);
        EXPECT_EQ(count, valid_count);
    }
}

TEST(CuDeviceTest_, AC_BA_DeviceGetCount_DeviceEnumeration) {
    cuInit(0);
    int count;
    CUresult status = cuDeviceGetCount(&count);
    EXPECT_EQ(status, CUDA_SUCCESS);
    EXPECT_GT(count, 0);
    std::vector<CUdevice> devices;
    for (int i = 0; i < count; i++) {
        CUdevice device;
        status = cuDeviceGet(&device, i);
        EXPECT_EQ(status, CUDA_SUCCESS);
        devices.push_back(device);
    }
    EXPECT_EQ(devices.size(), count);
}
