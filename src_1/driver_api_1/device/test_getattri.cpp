TEST_F(CuDeviceGetAttributeTest, BasicBehavior) {
  // Get the number of devices
  int device_count;
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&device_count));

  // Loop over the devices
  for (int i = 0; i < device_count; i++) {
    // Get the device handle
    CUdevice device;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, i));

    // Get some attributes
    int max_threads, max_grid_x, max_shared_mem;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(&max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(&max_grid_x, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device));
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(&max_shared_mem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));

    // Check the values are positive
    EXPECT_GT(max_threads, 0);
    EXPECT_GT(max_grid_x, 0);
    EXPECT_GT(max_shared_mem, 0);
  }
}

TEST_F(CuDeviceGetAttributeTest, ExceptionHandling) {
  // Get the number of devices
  int device_count;
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&device_count));

  // Get the first device handle
  CUdevice device;
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, 0));

  // Declare an integer pointer
  int *pi;

  // Test with a null pointer
  pi = nullptr;
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));

  // Test with an invalid device
  device = -1;
  pi = new int;
  EXPECT_EQ(CUDA_ERROR_INVALID_DEVICE, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
  delete pi;

  // Test with an invalid attribute
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, 0));
  pi = new int;
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX - 1, device));
  delete pi;
}

TEST_F(CuDeviceGetAttributeTest, SyncOrAsyncBehavior) {
  // Get the number of devices
  int device_count;
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&device_count));

  // Get the first device handle
  CUdevice device;
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, 0));

  // Create an event
  CUevent event;
  ASSERT_EQ(CUDA_SUCCESS, cuEventCreate(&event, CU_EVENT_DEFAULT));

  // Create a stream
  CUstream stream;
  ASSERT_EQ(CUDA_SUCCESS, cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  // Declare an integer pointer
  int *pi = new int;

  // Test the synchronous behavior
  // Call the function before recording the event
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
  ASSERT_EQ(CUDA_SUCCESS, cuEventRecord(event, stream));
  // Wait for the event to complete
  ASSERT_EQ(CUDA_SUCCESS, cuEventSynchronize(event));
  // Check the value is positive
  EXPECT_GT(*pi, 0);

  // Test the asynchronous behavior
  // Call the function after recording the event
  ASSERT_EQ(CUDA_SUCCESS, cuEventRecord(event, stream));
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
  // Wait for the event to complete
  ASSERT_EQ(CUDA_SUCCESS, cuEventSynchronize(event));
  // Check the value is positive
  EXPECT_GT(*pi, 0);

  // Clean up
  delete pi;
  ASSERT_EQ(CUDA_SUCCESS, cuStreamDestroy(stream));
  ASSERT_EQ(CUDA_SUCCESS, cuEventDestroy(event));
}

TEST_F(CuDeviceGetAttributeTest, SyncOrAsyncBehavior) {
  // Get the number of devices
  int device_count;
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&device_count));

  // Get the first device handle
  CUdevice device;
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, 0));

  // Create an event
  CUevent event;
  ASSERT_EQ(CUDA_SUCCESS, cuEventCreate(&event, CU_EVENT_DEFAULT));

  // Create a stream
  CUstream stream;
  ASSERT_EQ(CUDA_SUCCESS, cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  // Declare an integer pointer
  int *pi = new int;

  // Test the synchronous behavior
  // Call the function before recording the event
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
  ASSERT_EQ(CUDA_SUCCESS, cuEventRecord(event, stream));
  // Wait for the event to complete
  ASSERT_EQ(CUDA_SUCCESS, cuEventSynchronize(event));
  // Check the value is positive
  EXPECT_GT(*pi, 0);

  // Test the asynchronous behavior
  // Call the function after recording the event
  ASSERT_EQ(CUDA_SUCCESS, cuEventRecord(event, stream));
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
  // Wait for the event to complete
  ASSERT_EQ(CUDA_SUCCESS, cuEventSynchronize(event));
  // Check the value is positive
  EXPECT_GT(*pi, 0);

  // Clean up
  delete pi;
  ASSERT_EQ(CUDA_SUCCESS, cuStreamDestroy(stream));
  ASSERT_EQ(CUDA_SUCCESS, cuEventDestroy(event));
}

TEST_F(CuDeviceGetAttributeTest, DeviceDifferences) {
  // Get the number of devices
  int device_count;
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&device_count));

  // Check if there are at least two devices
  if (device_count >= 2) {
    // Get the first and second device handles
    CUdevice device1, device2;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&device1, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&device2, 1));

    // Declare two integer pointers
    int *pi1 = new int;
    int *pi2 = new int;

    // Get the compute capability of the devices
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi1, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device1));
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi2, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device2));
    int major1 = *pi1;
    int major2 = *pi2;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi1, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device1));
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi2, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device2));
    int minor1 = *pi1;
    int minor2 = *pi2;

    // Check if the devices have different compute capabilities
    if (major1 != major2 || minor1 != minor2) {
      // Get some attributes that depend on the compute capability
      ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi1, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device1));
      ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi2, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device2));
      EXPECT_NE(*pi1, *pi2);

      ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi1, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device1));
      ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi2, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device2));
      EXPECT_NE(*pi1, *pi2);

      ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi1, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device1));
      ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi2, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device2));
      EXPECT_NE(*pi1, *pi2);
    }

    // Clean up
    delete pi1;
    delete pi2;
  }
}

// 检查函数是否能正确处理多线程或多进程的情况，比如在不同的线程或进程中获取同一个或不同的设备的属性。可以使用pthread或fork来创建多个线程或进程，并在每个线程或进程中调用函数，比如
TEST_F(CuDeviceGetAttributeTest, MultiThreading) {
  // Get the number of devices
  int device_count;
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&device_count));

  // Check if there are at least two devices
  if (device_count >= 2) {
    // Define a function to run in each thread
    void* thread_func(void* arg) {
      // Get the thread id and the device id
      int thread_id = *(int*)arg;
      int device_id = thread_id % device_count;

      // Get the device handle
      CUdevice device;
      EXPECT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, device_id));

      // Declare an integer pointer
      int *pi = new int;

      // Get some attributes
      EXPECT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
      EXPECT_GT(*pi, 0);

      EXPECT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device));
      EXPECT_GT(*pi, 0);

      EXPECT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));
      EXPECT_GT(*pi, 0);

      // Clean up
      delete pi;

      // Return
      return nullptr;
    }

    // Define the number of threads
    const int num_threads = 4;

    // Create an array of thread ids
    int thread_ids[num_threads];

    // Create an array of thread handles
    pthread_t threads[num_threads];

    // Create and run the threads
    for (int i = 0; i < num_threads; i++) {
      thread_ids[i] = i;
      ASSERT_EQ(0, pthread_create(&threads[i], nullptr, thread_func, &thread_ids[i]));
    }

    // Wait for the threads to finish
    for (int i = 0; i < num_threads; i++) {
      ASSERT_EQ(0, pthread_join(threads[i], nullptr));
    }
}

TEST_F(CuDeviceGetAttributeTest, MultiProcessing) {
  // Get the number of devices
  int device_count;
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&device_count));

  // Check if there are at least two devices
  if (device_count >= 2) {
    // Define a function to run in each process
    void process_func(int process_id) {
      // Get the device id
      int device_id = process_id % device_count;

      // Get the device handle
      CUdevice device;
      EXPECT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, device_id));

      // Declare an integer pointer
      int *pi = new int;

      // Get some attributes
      EXPECT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
      EXPECT_GT(*pi, 0);

      EXPECT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device));
      EXPECT_GT(*pi, 0);

      EXPECT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));
      EXPECT_GT(*pi, 0);

      // Clean up
      delete pi;

      // Exit
      exit(0);
    }

    // Define the number of processes
    const int num_processes = 4;

    // Create an array of process ids
    pid_t pids[num_processes];

    // Create and run the processes
    for (int i = 0; i < num_processes; i++) {
      pids[i] = fork();
      ASSERT_NE(-1, pids[i]);
      if (pids[i] == 0) {
        process_func(i);
      }
    }

    // Wait for the processes to finish
    for (int i = 0; i < num_processes; i++) {
      int status;
      ASSERT_EQ(pids[i], waitpid(pids[i], &status, 0));
      ASSERT_EQ(0, status);
    }
  }
}


TEST_F(CuDeviceGetAttributeTest, RepeatedCallsFixedAttribute) {
  // Get the number of devices
  int device_count;
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&device_count));

  // Loop over the devices
  for (int i = 0; i < device_count; i++) {
    // Get the device handle
    CUdevice device;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, i));

    // Declare an integer pointer
    int *pi = new int;

    // Get a fixed attribute
    CUdevice_attribute attrib = CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;

    // Call the function once and store the result
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, attrib, device));
    int first_result = *pi;

    // Call the function multiple times and compare the results
    for (int j = 0; j < 10; j++) {
      ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, attrib, device));
      EXPECT_EQ(first_result, *pi);
    }

    // Clean up
    delete pi;
  }
}

TEST_F(CuDeviceGetAttributeTest, RepeatedCallsFixedDevice) {
  // Get the number of devices
  int device_count;
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&device_count));

  // Check if there is at least one device
  if (device_count >= 1) {
    // Get the first device handle
    CUdevice device;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, 0));

    // Declare an integer pointer
    int *pi = new int;

    // Loop over the attributes
    for (int i = 0; i < CU_DEVICE_ATTRIBUTE_MAX; i++) {
      // Get an attribute
      CUdevice_attribute attrib = static_cast<CUdevice_attribute>(i);

      // Call the function once and store the result
      ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, attrib, device));
      int first_result = *pi;

      // Call the function multiple times and compare the results
      for (int j = 0; j < 10; j++) {
        ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, attrib, device));
        EXPECT_EQ(first_result, *pi);
      }
    }

    // Clean up
  }
}

TEST_F(CuDeviceGetAttributeTest, BoundaryValuesActual) {
  // Get the number of devices
  int device_count;
  ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&device_count));

  // Loop over the devices
  for (int i = 0; i < device_count; i++) {
    // Get the device handle
    CUdevice device;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, i));

    // Declare an integer pointer
    int *pi = new int;

    // Get the actual minimum and maximum attribute values
    int min_value = std::numeric_limits<int>::max();
    int max_value = std::numeric_limits<int>::min();
    for (int j = 0; j < CU_DEVICE_ATTRIBUTE_MAX; j++) {
      // Get an attribute
      CUdevice_attribute attrib = static_cast<CUdevice_attribute>(j);

      // Get the attribute value
      ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, attrib, device));

      // Update the minimum and maximum values
      min_value = std::min(min_value, *pi);
      max_value = std::max(max_value, *pi);
    }

    // Check the function with the minimum and maximum values
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
    EXPECT_GE(*pi, min_value);
    EXPECT_LE(*pi, max_value);

    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device));
    EXPECT_GE(*pi, min_value);
    EXPECT_LE(*pi, max_value);

    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetAttribute(pi, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));
    EXPECT_GE(*pi, min_value);
    EXPECT_LE(*pi, max_value);

    // Clean up
  }
}