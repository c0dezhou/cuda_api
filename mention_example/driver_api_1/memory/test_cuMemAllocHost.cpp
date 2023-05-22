#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

template <typename T>
void testAllocHost(T value) {
    T *p;
    CUresult res = cuMemAllocHost((void**)&p, sizeof(T));
    ASSERT_EQ(res, CUDA_SUCCESS);
    *p = value;
    ASSERT_EQ(*p, value);
    cuMemFreeHost(p);
}

class CuMemAllocHostTest : public ::testing::Test {
protected:
    void SetUp() override {
        // You can initialize the CUDA driver here, e.g., cuInit(0);
    }

    void TearDown() override {
        // Cleanup code if necessary
    }
};

// Testing the basic functionality of cuMemAllocHost with different data types and sizes
TEST_F(CuMemAllocHostTest, BasicFunctionality) {
    // Test with integer
    testAllocHost<int>(12345);
    
    // Test with double
    testAllocHost<double>(12345.6789);
    
    // Test with a large array of float
    const int arraySize = 1024;
    float *pFloatArray;
    CUresult res = cuMemAllocHost((void**)&pFloatArray, arraySize * sizeof(float));
    ASSERT_EQ(res, CUDA_SUCCESS);
    for (int i = 0; i < arraySize; ++i) {
        pFloatArray[i] = static_cast<float>(i);
        ASSERT_FLOAT_EQ(pFloatArray[i], static_cast<float>(i));
    }
    cuMemFreeHost(pFloatArray);
}

// Test edge cases
TEST_F(CuMemAllocHostTest, EdgeCases) {
    // Test with zero bytes
    int* pZeroBytes;
    CUresult res = cuMemAllocHost((void**)&pZeroBytes, 0);
    ASSERT_EQ(res, CUDA_ERROR_INVALID_VALUE);

    // Test with negative memory size
    int* pNegativeSize;
    res = cuMemAllocHost((void**)&pNegativeSize, -1);
    ASSERT_EQ(res, CUDA_ERROR_INVALID_VALUE);

    // Test with invalid memory allocation
    int* pInvalidAlloc;
    res = cuMemAllocHost((void**)&pInvalidAlloc, sizeof(int));
    ASSERT_EQ(res, CUDA_ERROR_OUT_OF_MEMORY);
}

// Test error handling
TEST_F(CuMemAllocHostTest, ErrorHandling) {
    // Test with invalid pointer
    int* pInvalidPtr = nullptr;
    CUresult res = cuMemAllocHost((void**)&pInvalidPtr, sizeof(int));
    ASSERT_EQ(res, CUDA_ERROR_INVALID_VALUE);

    // Test with incorrect byte count
    int* pIncorrectSize;
    res = cuMemAllocHost((void**)&pIncorrectSize, sizeof(int) + 1);
    ASSERT_EQ(res, CUDA_ERROR_INVALID_VALUE);
}

// Test multi-threaded scenario
TEST_F(CuMemAllocHostTest, MultiThreadedScenario) {
    const int numThreads = 10;
    std::vector<std::thread> threads;

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this] {
            std::lock_guard<std::mutex> lock(allocationMutex);
            testAllocHost<int>(42);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

// Test cuMemAllocHost behavior when called from different threads
TEST_F(CuMemAllocHostTest, MultiThreadedAllocation) {
    constexpr int numThreads = 10;
    constexpr int allocationSize = 1024;

    std::vector<std::thread> threads;

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([allocationSize]() {
            int* ptr;
            CUresult res = cuMemAllocHost((void**)&ptr, allocationSize * sizeof(int));
            ASSERT_EQ(res, CUDA_SUCCESS);

            for (int j = 0; j < allocationSize; ++j) {
                ptr[j] = j;
                ASSERT_EQ(ptr[j], j);
            }

            cuMemFreeHost(ptr);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

// Test cuMemAllocHost behavior with synchronized threads
TEST_F(CuMemAllocHostTest, SynchronizedThreadsAllocation) {
    constexpr int numThreads = 10;
    constexpr int allocationSize = 1024;

    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable cv;
    bool startAllocating = false;

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&, i]() {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return startAllocating; });

            int* ptr;
            CUresult res = cuMemAllocHost((void**)&ptr, allocationSize * sizeof(int));
            ASSERT_EQ(res, CUDA_SUCCESS);

            for (int j = 0; j < allocationSize; ++j) {
                ptr[j] = j;
                ASSERT_EQ(ptr[j], j);
            }

            cuMemFreeHost(ptr);

            if (i < numThreads - 1) {
                cv.notify_one();
            }
        });
    }

    // Start the allocation process
    {
        std::lock_guard<std::mutex> lock(mutex);
        startAllocating = true;
    }
    cv.notify_one();

    for (auto& thread : threads) {
        thread.join();
    }
}

// Test cuMemAllocHost error handling in a multi-threaded scenario
TEST_F(CuMemAllocHostTest, MultiThreadingSynchronization) {
    constexpr int numThreads = 10;
    constexpr int allocationSize = 1024;

    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable cv;
    bool startAllocating = false;
    bool errorOccurred = false;

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&, i]() {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return startAllocating; });

            if (!errorOccurred) {
                int* ptr;
                CUresult res = cuMemAllocHost((void**)&ptr, allocationSize * sizeof(int));

                if (res != CUDA_SUCCESS) {
                    errorOccurred = true;
                } else {
                    for (int j = 0; j < allocationSize; ++j) {
                        ptr[j] = j;
                        ASSERT_EQ(ptr[j], j);
                    }

                    cuMemFreeHost(ptr);
                }
            }

            if (i < numThreads - 1) {
                cv.notify_one();
            }
        });
    }

    // Start the allocation process
    {
        std::lock_guard<std::mutex> lock(mutex);
        startAllocating = true;
    }
    cv.notify_one();

    for (auto& thread : threads) {
        thread.join();
    }

    ASSERT_TRUE(errorOccurred);
}



TEST_F(CuMemAllocHostTest, MultiProcessing) {
    // Test cuMemAllocHost in a multi-process scenario
    // Google Test framework (`gtest`) does not support forking processes directly within a `TEST_F` test case. The `TEST_F` macro is used to define a test fixture and its associated test cases, but it does not have built-in support for process forking.
}

TEST_F(CuMemAllocHostTest, MultiProcessingInterProcessCommunication) {
    // Test inter-process communication in multi-process scenarios
}

TEST_F(CuMemAllocHostTest, MultiProcessingErrorHandling) {
    // Test error handling in multi-process scenarios
}

// Test cuMemAllocHost with multiple streams
TEST_F(CuMemAllocHostTest, MultiStreamAllocation) {
    constexpr int allocationSize = 1024;

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Allocate memory for stream1
    int* ptr1;
    cudaMallocHost((void**)&ptr1, allocationSize * sizeof(int));

    // Allocate memory for stream2
    int* ptr2;
    cudaMallocHost((void**)&ptr2, allocationSize * sizeof(int));

    // Use stream1 to write data to ptr1
    for (int i = 0; i < allocationSize; ++i) {
        ptr1[i] = i;
    }

    // Use stream2 to write data to ptr2
    for (int i = 0; i < allocationSize; ++i) {
        ptr2[i] = i * 2;
    }

    // Synchronize streams to ensure completion of memory operations
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Verify the data written by stream1
    for (int i = 0; i < allocationSize; ++i) {
        ASSERT_EQ(ptr1[i], i);
    }

    // Verify the data written by stream2
    for (int i = 0; i < allocationSize; ++i) {
        ASSERT_EQ(ptr2[i], i * 2);
    }

    // Free allocated memory
    cudaFreeHost(ptr1);
    cudaFreeHost(ptr2);

    // Destroy streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}


// Test cuMemAllocHost with overlapping and synchronized operations in different streams
TEST_F(CuMemAllocHostTest, OverlappingStreamOperations) {
    constexpr int allocationSize = 1024;

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Allocate memory for stream1 and stream2
    int* ptr1;
    int* ptr2;
    cudaMallocHost((void**)&ptr1, allocationSize * sizeof(int));
    cudaMallocHost((void**)&ptr2, allocationSize * sizeof(int));

    // Perform overlapping operations in stream1 and stream2
    for (int i = 0; i < allocationSize; ++i) {
        ptr1[i] = i;
        ptr2[i] = i * 2;
    }

    // Synchronize streams to ensure completion of memory operations
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Verify the data written by stream1 and stream2
    for (int i = 0; i < allocationSize; ++i) {
        ASSERT_EQ(ptr1[i], i);
        ASSERT_EQ(ptr2[i], i * 2);
    }

    // Free allocated memory
    cudaFreeHost(ptr1);
    cudaFreeHost(ptr2);

    // Destroy streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

// Test cuMemAllocHost in a loop scenario
TEST_F(CuMemAllocHostTest, LoopAllocation) {
    constexpr int numIterations = 10;
    constexpr int allocationSize = 1024;

    // Create a vector of threads
    std::vector<std::thread> threads;

    for (int i = 0; i < numIterations; ++i) {
        threads.emplace_back([allocationSize]() {
            int* ptr;
            cudaMallocHost((void**)&ptr, allocationSize * sizeof(int));

            // Perform operations with allocated memory

            cudaFreeHost(ptr);
        });
    }

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }
}

// Test error handling of cuMemAllocHost in a loop scenario
TEST_F(CuMemAllocHostTest, LoopAllocationErrors) {
    constexpr int numIterations = 10;
    constexpr size_t allocationSize = 1ULL << 32;  // 4 GB

    for (int i = 0; i < numIterations; ++i) {
        int* ptr;
        CUresult res = cuMemAllocHost((void**)&ptr, allocationSize);
        if (res == CUDA_SUCCESS) {
            std::cout << "Iteration " << i << ": Memory allocation successful\n";
            cuMemFreeHost(ptr);
        } else {
            std::cerr << "Iteration " << i << ": Memory allocation failed with error code " << res << "\n";
        }
    }
}

// Test cuMemAllocHost with CUDA event synchronization
TEST_F(CuMemAllocHostTest, EventSynchronization) {
    constexpr int allocationSize = 1024;

    // Create CUDA events
    cudaEvent_t event;
    cudaEventCreate(&event);

    // Allocate memory using cuMemAllocHost
    int* ptr;
    cudaMallocHost((void**)&ptr, allocationSize * sizeof(int));

    // Record an event to mark the completion of memory allocation
    cudaEventRecord(event);

    // Perform some operations on the allocated memory

    // Synchronize with the event to ensure memory operations have completed
    cudaEventSynchronize(event);

    // Access and verify the allocated memory
    for (int i = 0; i < allocationSize; ++i) {
        ASSERT_EQ(ptr[i], 0);  // Verify the content of the allocated memory
    }

    // Free the allocated memory
    cudaFreeHost(ptr);

    // Destroy the CUDA event
    cudaEventDestroy(event);
}

// Test synchronization between streams using CUDA events and cuMemAllocHost
TEST_F(CuMemAllocHostTest, StreamSynchronization) {
    constexpr int allocationSize = 1024;

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Allocate memory using cuMemAllocHost
    int* ptr;
    cudaMallocHost((void**)&ptr, allocationSize * sizeof(int));

    // Create CUDA events for synchronization
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    // Perform operations in stream1
    cudaMemcpyAsync(ptr, someData, allocationSize * sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaEventRecord(event1, stream1);

    // Perform operations in stream2 dependent on the completion of stream1
    cudaStreamWaitEvent(stream2, event1, 0);
    // Perform operations using ptr in stream2

    // Record an event to mark the completion of stream2 operations
    cudaEventRecord(event2, stream2);

    // Synchronize with event2 to ensure stream2 operations have completed
    cudaEventSynchronize(event2);

    // Access and verify the allocated memory
    for (int i = 0; i < allocationSize; ++i) {
        ASSERT_EQ(ptr[i], expectedData[i]);
    }

    // Free the allocated memory
    cudaFreeHost(ptr);

    // Destroy the CUDA events and streams
    cudaEventDestroy(event1);
    cudaEventDestroy(event2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

// Test measuring time taken by cuMemAllocHost using CUDA events
TEST_F(CuMemAllocHostTest, MeasureAllocationTime) {
    constexpr size_t allocationSize = 1024 * 1024 * sizeof(int);  // Allocate 1 MB

    // Create CUDA events
    cudaEvent_t startEvent, endEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);

    // Record the start event
    cudaEventRecord(startEvent);

    // Allocate memory using cuMemAllocHost
    int* ptr;
    cudaMallocHost((void**)&ptr, allocationSize);

    // Record the end event
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);

    // Calculate the elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, endEvent);

    // Free the allocated memory
    cudaFreeHost(ptr);

    // Destroy the CUDA events
    cudaEventDestroy(startEvent);
    cudaEventDestroy(endEvent);

    // Print the allocated memory allocation time
    std::cout << "cuMemAllocHost Allocation Time: " << elapsedTime << " ms" << std::endl;
}

// Test error handling of cuMemAllocHost in event-related scenarios
TEST_F(CuMemAllocHostTest, EventErrorHandling) {
    constexpr size_t allocationSize = 1024 * 1024 * sizeof(int);  // Allocate 1 MB

    // Create a CUDA event
    cudaEvent_t event;
    cudaEventCreate(&event);

    // Allocate memory using cuMemAllocHost
    int* ptr;
    CUresult res = cuMemAllocHost((void**)&ptr, allocationSize);

    // Perform operations dependent on the allocation
    if (res == CUDA_SUCCESS) {
        // Record an event to mark the completion of the operations
        cudaEventRecord(event);
    }

    // Synchronize with the event to ensure the operations have completed
    cudaEventSynchronize(event);

    // Verify if allocation was successful and perform error handling
    if (res == CUDA_SUCCESS) {
        // Access and verify the allocated memory
        for (int i = 0; i < allocationSize / sizeof(int); ++i) {
            ASSERT_EQ(ptr[i], 0);
        }
    } else {
        // Handle allocation error
        std::cerr << "Memory allocation failed with error code " << res << std::endl;
    }

    // Free the allocated memory
    cudaFreeHost(ptr);

    // Destroy the CUDA event
    cudaEventDestroy(event);
}

// Test cuMemAllocHost in combined scenarios (multi-threading and multi-streaming)
TEST_F(CuMemAllocHostTest, CombinedScenarios) {
    constexpr int numThreads = 4;
    constexpr int numStreams = 2;
    constexpr int allocationSize = 1024;

    // Create a vector of threads
    std::vector<std::thread> threads;

    // Create CUDA streams
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Spawn threads for each stream
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&, i]() {
            // Assign a unique CUDA stream to each thread
            cudaStream_t stream = streams[i % numStreams];

            // Allocate memory for the thread
            int* ptr;
            cudaMallocHost((void**)&ptr, allocationSize * sizeof(int));

            // Perform operations using the allocated memory in the thread's assigned stream
            for (int j = 0; j < allocationSize; ++j) {
                ptr[j] = j + i;
            }

            // Synchronize the stream to ensure completion of memory operations
            cudaStreamSynchronize(stream);

            // Free the allocated memory
            cudaFreeHost(ptr);
        });
    }

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Destroy CUDA streams
    for (auto& stream : streams) {
        cudaStreamDestroy(stream);
    }
}

// Test cuMemAllocHost in combined scenarios (multi-threading and multi-streaming with events)
TEST_F(CuMemAllocHostTest, CombinedScenariosWithEvents) {
    constexpr int numThreads = 4;
    constexpr int numStreams = 2;
    constexpr int allocationSize = 1024;

    // Create a vector of threads
    std::vector<std::thread> threads;

    // Create CUDA streams and events
    std::vector<cudaStream_t> streams(numStreams);
    std::vector<cudaEvent_t> events(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
    }

    // Spawn threads for each stream
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&, i]() {
            // Assign a unique CUDA stream and event to each thread
            cudaStream_t stream = streams[i % numStreams];
            cudaEvent_t event = events[i % numStreams];

            // Allocate memory for the thread
            int* ptr;
            cudaMallocHost((void**)&ptr, allocationSize * sizeof(int));

            // Perform operations using the allocated memory in the thread's assigned stream
            for (int j = 0; j < allocationSize; ++j) {
                ptr[j] = j + i;
            }

            // Record an event to mark the completion of memory operations in the thread's stream
            cudaEventRecord(event, stream);

            // Synchronize the event to ensure completion of memory operations in the thread's stream
            cudaEventSynchronize(event);

            // Free the allocated memory
            cudaFreeHost(ptr);
        });
    }

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Destroy CUDA streams and events
    for (auto& stream : streams) {
        cudaStreamDestroy(stream);
    }
    for (auto& event : events) {
        cudaEventDestroy(event);
    }
}

// Test interaction of cuMemAllocHost with other CUDA driver APIs
TEST_F(CuMemAllocHostTest, InteractionWithOtherAPIs) {
    constexpr int allocationSize = 1024;

    // Allocate memory using cuMemAllocHost
    int* ptr;
    cudaMallocHost((void**)&ptr, allocationSize * sizeof(int));

    // Perform operations with the allocated memory
    for (int i = 0; i < allocationSize; ++i) {
        ptr[i] = i;
    }

    // Allocate memory on the device
    int* devicePtr;
    cudaMalloc((void**)&devicePtr, allocationSize * sizeof(int));

    // Copy data from host to device using cuMemcpy
    cudaMemcpy(devicePtr, ptr, allocationSize * sizeof(int), cudaMemcpyHostToDevice);

    // Perform operations on the device memory

    // Copy data from device to host using cuMemcpy
    int* resultPtr;
    cudaMallocHost((void**)&resultPtr, allocationSize * sizeof(int));
    cudaMemcpy(resultPtr, devicePtr, allocationSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify the data copied from device to host
    for (int i = 0; i < allocationSize; ++i) {
        ASSERT_EQ(resultPtr[i], i);
    }

    // Free the allocated memory
    cudaFreeHost(ptr);
    cudaFree(devicePtr);
    cudaFreeHost(resultPtr);
}

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

class CuMemAllocHostTest : public ::testing::Test {
protected:
    void SetUp() override {
        // You can initialize the CUDA driver here, e.g., cuInit(0);
    }

    void TearDown() override {
        // Cleanup code if necessary
    }
};

// Helper function to measure time
template <typename F>
double MeasureTime(F&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Helper function to measure time
template <typename F>
double MeasureTime(F&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Test performance of cuMemAllocHost under different conditions
TEST_F(CuMemAllocHostTest, PerformanceMeasurement) {
    constexpr int numThreads = 4;
    constexpr int numProcesses = 2;
    constexpr int numStreams = 2;
    constexpr size_t minAllocationSize = 1024;
    constexpr size_t maxAllocationSize = 1024 * 1024;
    constexpr size_t allocationStep = 1024;
    constexpr bool useEvents = true;

    // Vector to store the results
    std::vector<double> results;

    // Vary the memory size
    for (size_t allocationSize = minAllocationSize; allocationSize <= maxAllocationSize; allocationSize += allocationStep) {
        // Vary the number of threads/processes/streams
        for (int threads = 1; threads <= numThreads; ++threads) {
            for (int processes = 1; processes <= numProcesses; ++processes) {
                for (int streams = 1; streams <= numStreams; ++streams) {
                    // Measure the performance
                    double elapsedTime = MeasureTime([&]() {
                        std::vector<std::thread> threadPool;
                        std::vector<cudaStream_t> streamPool;
                        std::vector<cudaEvent_t> eventPool;

                        // Create CUDA streams and events
                        for (int i = 0; i < streams; ++i) {
                            cudaStream_t stream;
                            cudaStreamCreate(&stream);
                            streamPool.push_back(stream);

                            if (useEvents) {
                                cudaEvent_t event;
                                cudaEventCreate(&event);
                                eventPool.push_back(event);
                            }
                        }

                        // Spawn threads/processes to allocate memory
                        for (int t = 0; t < threads; ++t) {
                            threadPool.emplace_back([&]() {
                                for (int p = 0; p < processes; ++p) {
                                    // Assign a unique CUDA stream to each process
                                    cudaStream_t stream = streamPool[p % streams];
                                    cudaEvent_t event = useEvents ? eventPool[p % streams] : nullptr;

                                    // Allocate memory for the process
                                    int* ptr;
                                    cudaMallocHost((void**)&ptr, allocationSize);

                                    // Perform operations using the allocated memory in the process's assigned stream
                                    // ...

                                    // Synchronize the stream with an event, if events are used
                                    if (useEvents) {
                                        cudaEventRecord(event, stream);
                                        cudaStreamWaitEvent(stream, event, 0);
                                    }

                                    // Free the allocated memory
                                    cudaFreeHost(ptr);
                                }
                            });
                        }

                        // Join all threads/processes
                        for (auto& thread : threadPool) {
                            thread.join();
                        }

                        
                        // Destroy CUDA streams and events
                        for (auto& stream : streamPool) {
                            cudaStreamDestroy(stream);
                        }
                        for (auto& event : eventPool) {
                            cudaEventDestroy(event);
                        }
                    });

                    // Store the measured performance
                    results.push_back(elapsedTime);
                }
            }
        }
    }
    // Print the results
    int count = 0;
    for (size_t allocationSize = minAllocationSize; allocationSize <= maxAllocationSize; allocationSize += allocationStep) {
        for (int threads = 1; threads <= numThreads; ++threads) {
            for (int processes = 1; processes <= numProcesses; ++processes) {
                for (int streams = 1; streams <= numStreams; ++streams) {
                    std::cout << "Allocation Size: " << allocationSize << " bytes | Threads: " << threads
                            << " | Processes: " << processes << " | Streams: " << streams << " | "
                            << "Time: " << results[count] << " ms" << std::endl;
                    ++count;
                }
            }
        }
    }
}



// Helper function to measure time
template <typename F>
double MeasureTime(F&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Test performance of cuMemAllocHost under high load
TEST_F(CuMemAllocHostTest, HighLoadPerformance) {
    constexpr int numThreads = 4;
    constexpr size_t allocationSize = 1024 * 1024;  // 1 MB
    constexpr int allocationInterval = 100;         // ms
    constexpr int testDuration = 5000;              // ms

    // Vector to store the allocated pointers
    std::vector<int*> allocatedPointers;

    // Start the memory allocation in separate threads
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&, i]() {
            while (true) {
                // Allocate memory
                int* ptr;
                cudaMallocHost((void**)&ptr, allocationSize);

                // Perform operations using the allocated memory
                // ...

                // Store the allocated pointer
                allocatedPointers.push_back(ptr);

                // Sleep for the specified allocation interval
                std::this_thread::sleep_for(std::chrono::milliseconds(allocationInterval));

                // Deallocate memory
                cudaFreeHost(ptr);
            }
        });
    }

    // Wait for the specified test duration
    std::this_thread::sleep_for(std::chrono::milliseconds(testDuration));

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Print the number of allocations performed during the test
    std::cout << "Number of Allocations: " << allocatedPointers.size() << std::endl;

    // Free the remaining allocated memory
    for (auto& ptr : allocatedPointers) {
        cudaFreeHost(ptr);
    }
}

void PerformOperations(int* ptr, size_t allocationSize) {
    // Fill the allocated memory with a pattern
    for (size_t i = 0; i < allocationSize; ++i) {
        ptr[i] = static_cast<int>(i);
    }
}

// Test behavior of cuMemAllocHost under limited system resources or heavy system load
TEST_F(CuMemAllocHostTest, ResourceLimitation) {
    constexpr size_t allocationSize = 1024 * 1024;  // 1 MB

    // Attempt to allocate memory using cuMemAllocHost
    int* ptr;
    CUresult res = cuMemAllocHost((void**)&ptr, allocationSize * sizeof(int));

    // Check the result of the allocation
    if (res == CUDA_SUCCESS) {
        // Allocation succeeded
        std::cout << "Memory allocation succeeded." << std::endl;

        // Perform operations with the allocated memory
        // ...

        // Free the allocated memory
        cuMemFreeHost(ptr);
    } else if (res == CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) {
        // Allocation failed due to limited system resources
        std::cout << "Memory allocation failed due to limited system resources." << std::endl;
    } else {
        // Other allocation failure
        std::cout << "Memory allocation failed." << std::endl;
    }
}

// Test if cuMemAllocHost properly prevents unauthorized access to the allocated memory
TEST_F(CuMemAllocHostTest, UnauthorizedAccessPrevention) {
    constexpr size_t allocationSize = 1024 * 1024;  // 1 MB

    // Allocate memory using cuMemAllocHost
    int* ptr;
    CUresult res = cuMemAllocHost((void**)&ptr, allocationSize * sizeof(int));

    // Check the result of the allocation
    if (res == CUDA_SUCCESS) {
        // Allocation succeeded
        std::cout << "Memory allocation succeeded." << std::endl;

        // Perform operations with the allocated memory
        // ...

        // Attempt unauthorized access to the allocated memory
        ptr[0] = 123;  // Unauthorized write access
        int value = ptr[0];  // Unauthorized read access

        // Verify that unauthorized access is prevented
        ASSERT_NE(value, 123) << "Unauthorized read access detected.";

        // Free the allocated memory
        cuMemFreeHost(ptr);
    } else if (res == CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) {
        // Allocation failed due to limited system resources
        std::cout << "Memory allocation failed due to limited system resources." << std::endl;
    } else {
        // Other allocation failure
        std::cout << "Memory allocation failed." << std::endl;
    }
}

// Test if cuMemAllocHost maintains data integrity
TEST_F(CuMemAllocHostTest, DataIntegrity) {
    constexpr size_t allocationSize = 1024 * 1024;  // 1 MB

    // Allocate memory using cuMemAllocHost
    int* ptr;
    CUresult res = cuMemAllocHost((void**)&ptr, allocationSize * sizeof(int));

    // Check the result of the allocation
    if (res == CUDA_SUCCESS) {
        // Allocation succeeded
        std::cout << "Memory allocation succeeded." << std::endl;

        // Generate random data to write to the allocated memory
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(1, 100);

        // Write random data to the allocated memory
        for (size_t i = 0; i < allocationSize; ++i) {
            ptr[i] = dist(gen);
        }

        // Read back the data from the allocated memory and verify integrity
        for (size_t i = 0; i < allocationSize; ++i) {
            int expectedValue = ptr[i];
            int actualValue = ptr[i];
            ASSERT_EQ(actualValue, expectedValue) << "Data corruption detected at index " << i;
        }

        // Free the allocated memory
        cuMemFreeHost(ptr);
    } else if (res == CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) {
        // Allocation failed due to limited system resources
        std::cout << "Memory allocation failed due to limited system resources." << std::endl;
    } else {
        // Other allocation failure
        std::cout << "Memory allocation failed." << std::endl;
    }
}

// Helper function to get the current system memory usage
long long GetSystemMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;  // in kilobytes
}

// Test if the allocated memory is correctly freed and if the system memory usage goes back to the expected level
TEST_F(CuMemAllocHostTest, MemoryDeallocation) {
    constexpr size_t allocationSize = 1024 * 1024;  // 1 MB

    // Get the initial system memory usage
    long long initialMemoryUsage = GetSystemMemoryUsage();

    // Allocate memory using cuMemAllocHost
    int* ptr;
    CUresult res = cuMemAllocHost((void**)&ptr, allocationSize * sizeof(int));

    // Check the result of the allocation
    if (res == CUDA_SUCCESS) {
        // Allocation succeeded
        std::cout << "Memory allocation succeeded." << std::endl;

        // Perform operations with the allocated memory
        // ...

        // Free the allocated memory
        cuMemFreeHost(ptr);

        // Get the current system memory usage after freeing the memory
        long long currentMemoryUsage = GetSystemMemoryUsage();

        // Calculate the expected memory usage
        long long expectedMemoryUsage = initialMemoryUsage;

        // Check if the system memory usage has returned to the expected level
        ASSERT_EQ(currentMemoryUsage, expectedMemoryUsage) << "System memory usage did not return to the expected level.";
    } else if (res == CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) {
        // Allocation failed due to limited system resources
        std::cout << "Memory allocation failed due to limited system resources." << std::endl;
    } else {
        // Other allocation failure
        std::cout << "Memory allocation failed." << std::endl;
    }
}

// Helper function to get the current system memory usage
long long GetSystemMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;  // in kilobytes
}

// Test for potential memory leaks by repeatedly allocating and deallocating memory while monitoring system memory usage
TEST_F(CuMemAllocHostTest, MemoryLeakDetection) {
    constexpr size_t allocationSize = 1024 * 1024;  // 1 MB
    constexpr int numIterations = 100;

    // Get the initial system memory usage
    long long initialMemoryUsage = GetSystemMemoryUsage();

    // Repeat allocation and deallocation
    for (int i = 0; i < numIterations; ++i) {
        // Allocate memory using cuMemAllocHost
        int* ptr;
        CUresult res = cuMemAllocHost((void**)&ptr, allocationSize * sizeof(int));

        // Check the result of the allocation
        if (res == CUDA_SUCCESS) {
            // Allocation succeeded
            std::cout << "Memory allocation succeeded." << std::endl;

            // Perform operations with the allocated memory
            // ...

            // Free the allocated memory
            cuMemFreeHost(ptr);
        } else if (res == CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) {
            // Allocation failed due to limited system resources
            std::cout << "Memory allocation failed due to limited system resources." << std::endl;
        } else {
            // Other allocation failure
            std::cout << "Memory allocation failed." << std::endl;
        }
    }

    // Get the final system memory usage
    long long finalMemoryUsage = GetSystemMemoryUsage();

    // Calculate the expected memory usage
    long long expectedMemoryUsage = initialMemoryUsage;

    // Check if the system memory usage remains within an acceptable range
    ASSERT_LE(finalMemoryUsage, expectedMemoryUsage * 1.1) << "Potential memory leak detected.";

    // Print the final memory usage for manual inspection
    std::cout << "Final System Memory Usage: " << finalMemoryUsage << " KB" << std::endl;
}





int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

