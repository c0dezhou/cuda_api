#include "device_tests.h"

TEST_F(CuDeviceTest, AC_BA_DeviceGetAttribute_BasicBehavior) {
    for (int i = 0; i < device_count; i++) {
        CUdevice devicei;
        EXPECT_EQ(CUDA_SUCCESS, cuDeviceGet(&devicei, i));

        int max_threads, max_grid_x, max_shared_mem;
        EXPECT_EQ(CUDA_SUCCESS,
                  cuDeviceGetAttribute(
                      &max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                      devicei));
        EXPECT_EQ(
            CUDA_SUCCESS,
            cuDeviceGetAttribute(&max_grid_x,
                                 CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, devicei));
        EXPECT_EQ(
            CUDA_SUCCESS,
            cuDeviceGetAttribute(
                &max_shared_mem,
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, devicei));

        EXPECT_GT(max_threads, 0);
        EXPECT_GT(max_grid_x, 0);
        EXPECT_GT(max_shared_mem, 0);
    }
}

TEST_F(CuDeviceTest, AC_INV_DeviceGetAttribute_ExceptionHandling) {
    int* pi = nullptr;
    EXPECT_EQ(CUDA_ERROR_INVALID_VALUE,
              cuDeviceGetAttribute(
                  pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));

    device = -1;
    pi = new int;
    EXPECT_EQ(CUDA_ERROR_INVALID_DEVICE,
              cuDeviceGetAttribute(
                  pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
    delete pi;

    EXPECT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, 0));
    pi = new int;
    EXPECT_EQ(
        CUDA_ERROR_INVALID_VALUE,
        cuDeviceGetAttribute(
            pi, (CUdevice_attribute_enum)(CU_DEVICE_ATTRIBUTE_MAX + 1), device));
    delete pi;
}

TEST_F(CuDeviceTest, AC_BA_DeviceGetAttribute_DeviceDifferences) {
    // GTEST_SKIP();  // due to core dump
    if (device_count >= 2) {
        CUdevice device1, device2;
        EXPECT_EQ(CUDA_SUCCESS, cuDeviceGet(&device1, 0));
        EXPECT_EQ(CUDA_SUCCESS, cuDeviceGet(&device2, 1));

        int pi1;
        int pi2;

        EXPECT_EQ(
            CUDA_SUCCESS,
            cuDeviceGetAttribute(
                &pi1, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device1));
        EXPECT_EQ(
            CUDA_SUCCESS,
            cuDeviceGetAttribute(
                &pi2, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device2));
        int major1 = pi1;
        int major2 = pi2;
        EXPECT_EQ(
            CUDA_SUCCESS,
            cuDeviceGetAttribute(
                &pi1, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device1));
        EXPECT_EQ(
            CUDA_SUCCESS,
            cuDeviceGetAttribute(
                &pi2, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device2));
        int minor1 = pi1;
        int minor2 = pi2;
        EXPECT_EQ(major1, major2);
        EXPECT_EQ(minor1, minor2);

        if (major1 != major2 || minor1 != minor2) {
            EXPECT_EQ(
                CUDA_SUCCESS,
                cuDeviceGetAttribute(
                    &pi1, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device1));
            EXPECT_EQ(
                CUDA_SUCCESS,
                cuDeviceGetAttribute(
                    &pi2, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device2));
            EXPECT_NE(pi1, pi2);

            EXPECT_EQ(
                CUDA_SUCCESS,
                cuDeviceGetAttribute(
                    &pi1, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                    device1));
            EXPECT_EQ(
                CUDA_SUCCESS,
                cuDeviceGetAttribute(
                    &pi2, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                    device2));
            EXPECT_NE(pi1, pi2);

            EXPECT_EQ(
                CUDA_SUCCESS,
                cuDeviceGetAttribute(
                    &pi1, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                    device1));
            EXPECT_EQ(
                CUDA_SUCCESS,
                cuDeviceGetAttribute(
                    &pi2, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                    device2));
            EXPECT_NE(pi1, pi2);
        }
    }
}

void* thread_func(void* arg) {
    int thread_id = *(int*)arg;
    int device_count;
    cuDeviceGetCount(&device_count);
    int device_id = thread_id % device_count;

    CUdevice device;
    EXPECT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, device_id));

    int pi;

    EXPECT_EQ(CUDA_SUCCESS,
              cuDeviceGetAttribute(
                  &pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
    EXPECT_GT(pi, 0);

    EXPECT_EQ(
        CUDA_SUCCESS,
        cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device));
    EXPECT_GT(pi, 0);

    EXPECT_EQ(CUDA_SUCCESS,
              cuDeviceGetAttribute(
                  &pi, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));
    EXPECT_GT(pi, 0);

    return nullptr;
}
TEST_F(CuDeviceTest, MTH_Device_DeviceGetAttribute_MultiThreadingDeviceGetAttr) {
    if (device_count >= 2) {
        
        const int num_threads = 4;
        int thread_ids[num_threads];
        pthread_t threads[num_threads];

        for (int i = 0; i < num_threads; i++) {
            thread_ids[i] = i;
            EXPECT_EQ(0, pthread_create(&threads[i], nullptr, thread_func,
                                        &thread_ids[i]));
        }

        for (int i = 0; i < num_threads; i++) {
            EXPECT_EQ(0, pthread_join(threads[i], nullptr));
        }
    }
}

void process_func(int process_id) {
    int device_count;
    EXPECT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&device_count));
    int device_id = process_id % device_count;
    CUdevice device;
    EXPECT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, device_id));
    int pi;
    EXPECT_EQ(CUDA_SUCCESS,
              cuDeviceGetAttribute(
                  &pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
    EXPECT_GT(pi, 0);
    EXPECT_EQ(
        CUDA_SUCCESS,
        cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device));
    EXPECT_GT(pi, 0);
    EXPECT_EQ(CUDA_SUCCESS,
              cuDeviceGetAttribute(
                  &pi, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));
    EXPECT_GT(pi, 0);
    exit(0);
}

TEST_F(CuDeviceTest, MPROC_Device_DeviceGetAttribute_MultiProcessing) {
    GTEST_SKIP(); // fork process not init
    if (device_count >= 2) {
        const int num_processes = 4;
        pid_t pids[num_processes];
        for (int i = 0; i < num_processes; i++) {
            pids[i] = fork();
            EXPECT_NE(-1, pids[i]);
            if (pids[i] == 0) {
                process_func(i);
            }
        }
        for (int i = 0; i < num_processes; i++) {
            int status;
            EXPECT_EQ(pids[i], waitpid(pids[i], &status, 0));
            EXPECT_EQ(0, status);
        }
    }
}

TEST_F(CuDeviceTest, AC_OT_DeviceGetAttribute_RepeatedCallsFixedAttribute) {
    // GTEST_SKIP();  // due to core dump
    for (int i = 0; i < device_count; i++) {
        CUdevice device;
        EXPECT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, i));
        int pi;
        CUdevice_attribute attrib =
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
        EXPECT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(&pi, attrib, device));
        int first_result = pi;
        for (int j = 0; j < 10; j++) {
            EXPECT_EQ(CUDA_SUCCESS,
                      cuDeviceGetAttribute(&pi, attrib, device));
            EXPECT_EQ(first_result, pi);
        }
    }
}

TEST_F(CuDeviceTest, AC_EG_DeviceGetAttribute_BoundaryValuesActual) {
    // GTEST_SKIP();  // due to core dump
    for (int i = 0; i < device_count; i++) {
        CUdevice device;
        EXPECT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, i));
        int pi;
        int min_value = std::numeric_limits<int>::max();
        int max_value = std::numeric_limits<int>::min();
        for (int j = 1; j < CU_DEVICE_ATTRIBUTE_MAX -1; j++) {
            CUdevice_attribute attrib = static_cast<CUdevice_attribute>(j);

            EXPECT_EQ(CUDA_SUCCESS,
                      cuDeviceGetAttribute(&pi, attrib, device)) << "error index :" << j;
            min_value = std::min(min_value, pi);
            max_value = std::max(max_value, pi);
        }
        EXPECT_EQ(
            CUDA_SUCCESS,
            cuDeviceGetAttribute(
                &pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
        EXPECT_GE(pi, min_value);
        EXPECT_LE(pi, max_value);
        EXPECT_EQ(CUDA_SUCCESS,
                  cuDeviceGetAttribute(
                      &pi, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device));
        EXPECT_GE(pi, min_value);
        EXPECT_LE(pi, max_value);
        EXPECT_EQ(CUDA_SUCCESS,
                  cuDeviceGetAttribute(
                      &pi, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                      device));
        EXPECT_GE(pi, min_value);
        EXPECT_LE(pi, max_value);

    }
}