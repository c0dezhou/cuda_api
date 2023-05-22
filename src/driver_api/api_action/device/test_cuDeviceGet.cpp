#include "test_utils.h"

TEST(CuDeviceTest_, AC_BA_DeviceGet_BasicBehavior) {
    cuInit(0);
    CUdevice device;
    CUresult status = cuDeviceGet(&device, 0);
    EXPECT_EQ(status, CUDA_SUCCESS);
}

TEST(CuDeviceTest_, AC_INV_DeviceGet_InvalidValue) {
    cuInit(0);
    CUdevice device;
    CUresult status = cuDeviceGet(nullptr, 0);
    EXPECT_EQ(status, CUDA_ERROR_INVALID_VALUE);
}

TEST(CuDeviceTest_, AC_INV_DeviceGet_InvalidDevice) {
    cuInit(0);
    CUdevice device;
    CUresult status = cuDeviceGet(&device, 100);
    EXPECT_EQ(status, CUDA_ERROR_INVALID_DEVICE);
}

TEST(CuDeviceTest_, AC_INV_DeviceGet_InvalidNegative) {
    cuInit(0);
    CUdevice device;
    CUresult status = cuDeviceGet(&device, -1);
    EXPECT_EQ(status, CUDA_ERROR_INVALID_DEVICE);
}

TEST(CuDeviceTest_, AC_OT_DeviceGet_MultipleDevicesDeviceGet) {
    cuInit(0);
    CUdevice device;
    CUresult status;
    status = cuDeviceGet(&device, 0);
    EXPECT_EQ(status, CUDA_SUCCESS);
    int count;
    cuDeviceGetCount(&count);
    status = cuDeviceGet(&device, count - 1);
    EXPECT_EQ(status, CUDA_SUCCESS);
}

TEST(CuDeviceTest_, AC_BA_DeviceGet_NoDeviceDeviceGet) {
    int count;
    cuDeviceGetCount(&count);
    if(count == 0){
        CUdevice device;
        CUresult status = cuDeviceGet(&device, 0);
        EXPECT_EQ(status, CUDA_ERROR_NO_DEVICE);
    }
}

TEST(CuDeviceTest_, AC_SA_DeviceGet_SynchronousBehaviorDeviceGet) {
    auto start = std::chrono::high_resolution_clock::now();
    CUdevice device;
    CUresult status = cuDeviceGet(&device, 0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    EXPECT_LT(duration, 10);
}

TEST(CuDeviceTest_, MTH_Device_DeviceGet) {
    auto start_out = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> ths;

    for (int i = 0; i < 10; i++) {
        ths.emplace_back([]() {
            auto start = std::chrono::high_resolution_clock::now();
            CUdevice device;
            CUresult status = cuDeviceGet(&device, 0);
            EXPECT_EQ(status, CUDA_SUCCESS);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                      start)
                    .count();
            EXPECT_LT(duration, 10);
        });
    }

    for (auto& th : ths) {
        th.join();
    }
    auto end_out = std::chrono::high_resolution_clock::now();
    auto duration_out =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_out - start_out)
            .count();
    EXPECT_LT(duration_out, 11);
}

TEST(CuDeviceTest_, AC_OT_DeviceGet_RepeatedCalls) {
    for (int i = 0; i < 100; i++) {
        CUdevice device;
        CUresult status = cuDeviceGet(&device, 0);
        EXPECT_EQ(status, CUDA_SUCCESS);
    }
}