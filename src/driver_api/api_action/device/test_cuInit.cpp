#include "test_utils.h"

TEST(CuDeviceTest_, AC_BA_Init_BasicBehaviorcu) {
    CUresult status = cuInit(0);
    EXPECT_EQ(status, CUDA_SUCCESS);
}

TEST(CuDeviceTest_, AC_EG_Init_Double) {
    cuInit(0);
    CUresult status = cuInit(0);
    EXPECT_EQ(status, CUDA_SUCCESS);
}

TEST(CuDeviceTest_, AC_INV_Init_InvalidCount) {
    CUresult status = cuInit(100);
    EXPECT_EQ(status, CUDA_ERROR_INVALID_VALUE);
}

TEST(CuDeviceTest_, AC_INV_Init_NotInitcuInit) {
    CUdevice device;
    CUresult status = cuDeviceGet(&device, 0);
    EXPECT_EQ(status, CUDA_ERROR_NOT_INITIALIZED);
}

TEST(CuDeviceTest_, AC_INV_Init_NoDrivercuInit) {
    // if no driver
    GTEST_SKIP();
    CUresult status = cuInit(0);
    EXPECT_EQ(status, CUDA_ERROR_UNKNOWN);
}

TEST(CuDeviceTest_, MTH_Device_INit_AsynchronousBehavior) {
    auto start_out = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> ths;

    for (int i = 0; i < 10; i++) {
        ths.emplace_back([]() {
            auto start = std::chrono::high_resolution_clock::now();
            CUdevice device;
            CUresult status = cuInit(0);
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

TEST(CuDeviceTest_, AC_OT_Init_RepeatedCalls) {
    for (int i = 0; i < 100; i++) {
        CUdevice device;
        CUresult status = cuInit(0);
        EXPECT_EQ(status, CUDA_SUCCESS);
    }
}