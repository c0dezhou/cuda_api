// #include "test_utils.h"

// bandwidthTest.cu
// This application provides the memcopy bandwidth of the GPU and memcpy bandwidth across PCI-e.
// This application is capable of measuring device to device copy bandwidth, host to device
// copy bandwidth for pageable and page-locked memory, and device to host copy bandwidth for
// pageable and page-locked memory.

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <helper_cuda_drvapi.h>

#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x, size) ( ((size_t)x+(size-1))&(~(size-1)) )
#define MAX(a,b) ((a > b) ? a : b)

// Define the CUDA driver API error check macro
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
    if (CUDA_SUCCESS != err)
    {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

// Define the structure for the different memory copy tests
typedef enum
{
    DEVICE_TO_HOST,
    HOST_TO_DEVICE,
    DEVICE_TO_DEVICE,
    PAGEABLE,
    PINNED,
    SHARED,
    WRITE_COMBINED = PINNED,
    MEMORY_TYPE_COUNT
} memoryCopyKind;

// Define the names for the different memory copy tests
const char *sMemoryCopyKind[] =
{
    "Device to Host",
    "Host to Device",
    "Device to Device",
    "Pageable memory",
    "Pinned memory",
};

// Define the structure for the different memory copy tests
typedef struct
{
    CUdeviceptr d_src;
    CUdeviceptr d_dst;
    unsigned char *h_src;
    unsigned char *h_dst;
} testCUDA;

// Define the structure for the different memory copy tests
typedef struct
{
    int iSize;
    int iNumIterations;
} testParams;

// Define the structure for the different memory copy tests
typedef struct
{
    testCUDA *cuda;
    testParams *params;
} testResults;

// Define the structure for the different memory copy tests
typedef struct
{
    bool bD2H;
    bool bH2D;
    bool bD2D;
} testMode;

// Define the structure for the different memory copy tests
typedef struct
{
    bool bPageable;
    bool bPinned;
} testMemory;

// Define the structure for the different memory copy tests
typedef struct
{
    bool bWC;
} testWC;

// Define the structure for the different memory copy tests
typedef struct
{
    bool bRange;
} testRange;

// Define the structure for the different memory copy tests
typedef struct
{
    int iStart;
} testStart;

// Define the structure for the different memory copy tests
typedef struct
{
    int iEnd;
} testEnd;

// Define the structure for the different memory copy tests
typedef struct
{
    int iIncrement;
} testIncrement;

// Define a function to print out help information
void printHelp(void)
{
  printf("Usage:  bandwidthTest [OPTION]...\n");
  printf("Test the bandwidth for device to host, host to device, and device to device transfers\n");
  printf("\n");
  printf("Example: measure the bandwidth of device to host pinned memory copies in the range 1024 Bytes to 102400 Bytes in 1024 Byte increments\n");
  printf("./bandwidthTest --memory=pinned --mode=range --start=1024 --end=102400 --increment=1024 --dtoh\n");
  printf("\n");
  printf("Options:\n");
  printf("--help\t\tDisplay this help menu\n");
  printf("--csv\t\tPrint results as a CSV\n");
  printf("--device=[deviceno]\tSpecify the device device to be used\n");
  printf("  all - compute cumulative bandwidth on all the devices\n");
  printf("  0,1,2,...,n - Specify any particular device to be used\n");
  printf("--memory=[MEMMODE]\tSpecify which memory mode to use\n");
  printf("  pageable - pageable memory\n");
  printf("  pinned   - non-pageable system memory\n");
  printf("--mode=[MODE]\tSpecify the mode to use\n");
  printf("  quick - performs a quick measurement\n");
  printf("  range - measures a user-specified range of values\n");
  printf("  shmoo - performs an intense shmoo of a large range of values\n");
  printf("--htod\t\tMeasure host to device transfers\n");
  printf("--dtoh\t\tMeasure device to host transfers\n");
  printf("--dtod\t\tMeasure device to device transfers\n");
  printf("--wc\t\tAllocate pinned memory as write-combined\n");
  printf("--cputiming\tForce CPU-based timing always\n");

// Range mode options
printf("\n--start=[SIZE]\tStarting transfer size in bytes\n");
printf("--end=[SIZE]\tEnding transfer size in bytes\n");
printf("--increment=[SIZE]\tIncrement size in bytes\n");

}

// Define a function to parse the command line arguments
void parseCommandLineArguments(int argc, char **argv, testMode *mode, testMemory *memMode, testWC *wc,
                              testRange *range, testStart *start, testEnd *end, testIncrement *inc,
                              int *deviceNo)
{
    int i;

    for (i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0)
        {
            printHelp();
            exit(EXIT_SUCCESS);
        }

        if (strcmp(argv[i], "--csv") == 0)
        {
            printf("bandwidthTest-mode,bandwidthTest-memory,bandwidthTest-size(MB),bandwidthTest-time(ms),bandwidthTest-bandwidth(GB/s)\n");
        }

        if (strcmp(argv[i], "--device") == 0)
        {
            if (argv[i + 1] == NULL || argv[i + 1][0] == '-')
            {
                printf("Invalid device number\n");
                exit(EXIT_FAILURE);
            }

            if (strcmp(argv[i + 1], "all") == 0)
            {
                *deviceNo = -1;
            }
            else
            {
                *deviceNo = atoi(argv[i + 1]);
                i++;
            }
        }

        if (strcmp(argv[i], "--memory") == 0)
        {
            if (argv[i + 1] == NULL || argv[i + 1][0] == '-')
            {
                printf("Invalid memory mode\n");
                exit(EXIT_FAILURE);
            }

            if (strcmp(argv[i + 1], "pageable") == 0)
            {
                memMode->bPageable = true;
                memMode->bPinned   = false;
                i++;
            }
            else if (strcmp(argv[i + 1], "pinned") == 0)
            {
                memMode->bPageable = false;
                memMode->bPinned   = true;
                i++;
            }
        }

        if (strcmp(argv[i], "--mode") == 0)
        {
            if (argv[i + 1] == NULL || argv[i + 1][0] == '-')
            {
                printf("Invalid mode\n");
                exit(EXIT_FAILURE);
            }

            if (strcmp(argv[i + 1], "quick") == 0)
            {
                range->bRange = false;
                i++;
            }
            else if (strcmp(argv[i + 1], "range") == 0)
            {
                range->bRange = true;
                i++;
            }
        }

        if (strcmp(argv[i], "--htod") == 0)
        {
            mode->bD2H = false;
            mode->bH2D = true;
            mode->bD2D = false;
        }

        if (strcmp(argv[i], "--dtoh") == 0)
        {
            mode->bD2H = true;
            mode->bH2D = false;
            mode->bD2D = false;
        }

        if (strcmp(argv[i], "--dtod") == 0)
        {
            mode->bD2H = false;
            mode->bH2D = false;
            mode->bD2D = true;
        }

        if (strcmp(argv[i], "--wc") == 0)
        {
            wc->bWC = true;
        }

        if (strcmp(argv[i], "--cputiming") == 0)
        {
            // No need to do anything
        }

        if (strcmp(argv[i], "--start") == 0)
        {
            if (argv[i + 1] == NULL || argv[i + 1][0] == '-')
            {
                printf("Invalid starting size\n");
                exit(EXIT_FAILURE);
            }

            start->iStart = atoi(argv[i + 1]);
            i++;
        }

        if (strcmp(argv[i], "--end") == 0)
        {
            if (argv[i + 1] == NULL || argv[i + 1][0] == '-')
            {
                printf("Invalid ending size\n");
                exit(EXIT_FAILURE);
            }

            end->iEnd = atoi(argv[i + 1]);
            i++;
        }

        if (strcmp(argv[i], "--increment") == 0)
        {
            if (argv[i + 1] == NULL || argv[i + 1][0] == '-')
            {
                printf("Invalid increment size\n");
                exit(EXIT_FAILURE);
            }

            inc->iIncrement = atoi(argv[i + 1]);
            i++;
        }
    }
}

// Define a function to allocate memory for the tests
void allocateMemory(testCUDA *cuda, testMemory *memMode, testWC *wc, int iSize)
{
    CUresult result;

    // Allocate host memory
    if (memMode->bPinned)
    {
        unsigned int flags;

        // Set the flags for pinned memory allocation
        if (wc->bWC)
        {
            flags = CU_MEMHOSTALLOC_WRITECOMBINED;
        }
        else
        {
            flags = CU_MEMHOSTALLOC_PORTABLE;
        }

        // Allocate pinned memory
        result = cuMemHostAlloc((void **)&cuda->h_src, iSize, flags);
        checkCudaErrors(result);

        result = cuMemHostAlloc((void **)&cuda->h_dst, iSize, flags);
        checkCudaErrors(result);
    }
    else
    {
        // Allocate pageable memory
        cuda->h_src = (unsigned char *)malloc(iSize);

        if (cuda->h_src == NULL)
        {
            fprintf(stderr, "Not enough memory available on host to run test!\n");
            exit(EXIT_FAILURE);
        }

        cuda->h_dst = (unsigned char *)malloc(iSize);

        if (cuda->h_dst == NULL)
        {
            fprintf(stderr, "Not enough memory available on host to run test!\n");
            exit(EXIT_FAILURE);
        }
    }

    // Allocate device memory
    result = cuMemAlloc(&cuda->d_src, iSize);
    checkCudaErrors(result);

    result = cuMemAlloc(&cuda->d_dst, iSize);
    checkCudaErrors(result);
}

// Define a function to free the allocated memory
void freeMemory(testCUDA *cuda, testMemory *memMode)
{
    CUresult result;

    // Free host memory
    if (memMode->bPinned)
    {
        result = cuMemFreeHost(cuda->h_src);
        checkCudaErrors(result);

        result = cuMemFreeHost(cuda->h_dst);
        checkCudaErrors(result);
    }
    else
    {
        free(cuda->h_src);
        free(cuda->h_dst);
    }

    // Free device memory
    result = cuMemFree(cuda->d_src);
    checkCudaErrors(result);

    result = cuMemFree(cuda->d_dst);
    checkCudaErrors(result);
}

// Define a function to initialize the host and device memory with some data
void initMemory(testCUDA *cuda, int iSize)
{
    CUresult result;

    // Initialize host memory
    for (int i = 0; i < iSize; i++)
    {
        cuda->h_src[i] = (unsigned char)(i & 0xff);
    }

    // Copy host memory to device memory
    result = cuMemcpyHtoD(cuda->d_src, cuda->h_src, iSize);
    checkCudaErrors(result);

    // Clear device memory
    result = cuMemsetD8(cuda->d_dst, 0, iSize);
    checkCudaErrors(result);
}

// Define a function to test the bandwidth of device to host transfers
float testDeviceToHostTransfer(testCUDA *cuda, testParams *params)
{
    CUresult result;
    CUevent start, stop;
    float elapsedTime;

    // Create CUDA events
    result = cuEventCreate(&start, CU_EVENT_DEFAULT);
    checkCudaErrors(result);

    result = cuEventCreate(&stop, CU_EVENT_DEFAULT);
    checkCudaErrors(result);

    // Record the start event
    result = cuEventRecord(start, NULL);
    checkCudaErrors(result);

    // Run the test loop
    for (int i = 0; i < params->iNumIterations; i++)
    {
        // Copy device memory to host memory
        result = cuMemcpyDtoH(cuda->h_dst, cuda->d_src, params->iSize);
        checkCudaErrors(result);
    }

    // Record the stop event
    result = cuEventRecord(stop, NULL);
    checkCudaErrors(result);

    // Wait for the stop event to complete
    result = cuEventSynchronize(stop);
    checkCudaErrors(result);

    // Calculate the elapsed time in milliseconds
    result = cuEventElapsedTime(&elapsedTime, start, stop);
    checkCudaErrors(result);

    // Destroy CUDA events
    result = cuEventDestroy(start);
    checkCudaErrors(result);

    result = cuEventDestroy(stop);
    checkCudaErrors(result);

    // Return the bandwidth in GB/s
    return (float)(params->iSize * params->iNumIterations) / (elapsedTime * 1024.0f * 1024.0f);
}

// Define a function to test the bandwidth of host to device transfers
float testHostToDeviceTransfer(testCUDA *cuda, testParams *params)
{
    CUresult result;
    CUevent start, stop;
    float elapsedTime;

    // Create CUDA events
    result = cuEventCreate(&start, CU_EVENT_DEFAULT);
    checkCudaErrors(result);

    result = cuEventCreate(&stop, CU_EVENT_DEFAULT);
    checkCudaErrors(result);

    // Record the start event
    result = cuEventRecord(start, NULL);
    checkCudaErrors(result);

    // Run the test loop
    for (int i = 0; i < params->iNumIterations; i++)
    {
        // Copy host memory to device memory
        result = cuMemcpyHtoD(cuda->d_dst, cuda->h_src, params->iSize);
        checkCudaErrors(result);
    }

    // Record the stop event
    result = cuEventRecord(stop, NULL);
    checkCudaErrors(result);

    // Wait for the stop event to complete
    result = cuEventSynchronize(stop);
    checkCudaErrors(result);

    // Calculate the elapsed time in milliseconds
    result = cuEventElapsedTime(&elapsedTime, start, stop);
    checkCudaErrors(result);

    // Destroy CUDA events
    result = cuEventDestroy(start);
    checkCudaErrors(result);

    result = cuEventDestroy(stop);
    checkCudaErrors(result);

    // Return the bandwidth in GB/s
    return (float)(params->iSize * params->iNumIterations) / (elapsedTime * 1024.0f * 1024.0f);
}

// Define a function to test the bandwidth of device to device transfers
float testDeviceToDeviceTransfer(testCUDA *cuda, testParams *params)
{
    CUresult result;
    CUevent start, stop;
    float elapsedTime;

    // Create CUDA events
    result = cuEventCreate(&start, CU_EVENT_DEFAULT);
    checkCudaErrors(result);

    result = cuEventCreate(&stop, CU_EVENT_DEFAULT);
    checkCudaErrors(result);

    // Record the start event
    result = cuEventRecord(start, NULL);
    checkCudaErrors(result);

    // Run the test loop
    for (int i = 0; i < params->iNumIterations; i++)
    {
        // Copy device memory to device memory
        result = cuMemcpyDtoD(cuda->d_dst, cuda->d_src, params->iSize);
        checkCudaErrors(result);
    }

    // Record the stop event
    result = cuEventRecord(stop, NULL);
    checkCudaErrors(result);

    // Wait for the stop event to complete
    result = cuEventSynchronize(stop);
    checkCudaErrors(result);

    // Calculate the elapsed time in milliseconds
    result = cuEventElapsedTime(&elapsedTime, start, stop);
    checkCudaErrors(result);

    // Destroy CUDA events
    result = cuEventDestroy(start);
    checkCudaErrors(result);

    result = cuEventDestroy(stop);
    checkCudaErrors(result);

    // Return the bandwidth in GB/s
    return (float)(params->iSize * params->iNumIterations) / (elapsedTime * 1024.0f * 1024.0f);
}

// Define a function to check the results of the memory transfers
bool testCheckResults(testCUDA *cuda, testParams *params)
{
    bool bTestResult = true;

    // Check host memory
    for (int i = 0; i < params->iSize; i++)
    {
        if (cuda->h_dst[i] != cuda->h_src[i])
        {
            bTestResult = false;
            break;
        }
    }

    // Check device memory
    CUresult result;
    unsigned char *h_buf = (unsigned char *)malloc(params->iSize);

    if (h_buf == NULL)
    {
        fprintf(stderr, "Not enough memory available on host to run test!\n");
        exit(EXIT_FAILURE);
    }

    result = cuMemcpyDtoH(h_buf, cuda->d_dst, params->iSize);
    checkCudaErrors(result);

    for (int i = 0; i < params->iSize; i++)
    {
        if (h_buf[i] != cuda->h_src[i])
        {
            bTestResult = false;
            break;
        }
    }

    free(h_buf);

    return bTestResult;
}

// Define a function to print the results of the tests
void printResults(testResults *results, testMode *mode, testMemory *memMode, int deviceNo)
{
    CUresult result;
    CUdevice device;
    char deviceName[256];
    int clockRate;
    int memoryClockRate;
    int memoryBusWidth;

    // Get the device properties
    if (deviceNo == -1)
    {
        result = cuDeviceGet(&device, 0);
        checkCudaErrors(result);
    }
    else
    {
        result = cuDeviceGet(&device, deviceNo);
        checkCudaErrors(result);
    }

    result = cuDeviceGetName(deviceName, 256, device);
    checkCudaErrors(result);

    result = cuDeviceGetAttribute(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device);
    checkCudaErrors(result);

    result = cuDeviceGetAttribute(&memoryClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device);
    checkCudaErrors(result);

    result = cuDeviceGetAttribute(&memoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device);
    checkCudaErrors(result);

    // Calculate the theoretical peak bandwidth
    float peakBandwidth = 2.0f * memoryClockRate * (memoryBusWidth / 8) / 1.0e6;

    // Print the device information
    printf("Running on...\n\n");
    printf(" Device %d: %s\n", deviceNo, deviceName);
    printf(" CUDA Driver Version: %d.%d\n", CUDA_VERSION / 1000, (CUDA_VERSION % 100) / 10);
    printf(" CUDA Capability Major/Minor version number: %d.%d\n", results->cuda->deviceProp.major, results->cuda->deviceProp.minor);
    printf(" Total amount of global memory: %.0f MBytes (%llu bytes)\n",
           (float)results->cuda->deviceProp.totalGlobalMem / 1048576.0f, (unsigned long long) results->cuda->deviceProp.totalGlobalMem);
    printf(" GPU Clock rate: %.0f MHz (%0.2f GHz)\n", clockRate * 1e-3f, clockRate * 1e-6f);
    printf(" Memory Clock rate: %.0f Mhz\n", memoryClockRate * 1e-3f);
    printf(" Memory Bus Width: %d-bit\n", memoryBusWidth);
    printf(" Peak Memory Bandwidth: %.2f GB/s\n\n", peakBandwidth);

    // Print the test mode
    if (mode->bD2H)
        printf(" %s Bandwidth Test\n", sMemoryCopyKind[DEVICE_TO_HOST]);
    else if (mode->bH2D)
        printf(" %s Bandwidth Test\n", sMemoryCopyKind[HOST_TO_DEVICE]);
    else if (mode->bD2D)
        printf(" %s Bandwidth Test\n", sMemoryCopyKind[DEVICE_TO_DEVICE]);

    // Print the memory mode
    if (memMode->bPinned)
        printf(" %s Memory Transfers\n\n", sMemoryCopyKind[PINNED]);
    else if (memMode->bPageable)
        printf(" %s Memory Transfers\n\n", sMemoryCopyKind[PAGEABLE]);

    // Print the test results
    printf(" Transfer Size (Bytes)\tBandwidth(MB/s)\tPercentage of Peak\n");
    printf(" %14d\t%11.2f\t%16.2f%%\n", results->params->iSize, results->bandwidth, results->bandwidth / peakBandwidth * 100.0f);
}

// Define a function to run the tests
void runTest(testResults *results, testMode *mode, testMemory *memMode, testWC *wc, testRange *range,
             testStart *start, testEnd *end, testIncrement *inc)
{
    CUresult result;
    CUdevice device;
    CUcontext context;

    // Initialize the device and the context
    result = cuInit(0);
    checkCudaErrors(result);

    result = cuDeviceGet(&device, 0);
    checkCudaErrors(result);

    result = cuCtxCreate(&context, 0, device);
    checkCudaErrors(result);

    // Get the device properties
    result = cuDeviceGetProperties(&results->cuda->deviceProp, device);
    checkCudaErrors(result);

    // Allocate memory for the tests
    allocateMemory(results->cuda, memMode, wc, results->params->iSize);

    // Initialize the memory with some data
    initMemory(results->cuda, results->params->iSize);

    // Run the tests
    if (mode->bD2H)
        results->bandwidth = testDeviceToHostTransfer(results->cuda, results->params);
    else if (mode->bH2D)
        results->bandwidth = testHostToDeviceTransfer(results->cuda, results->params);
    else if (mode->bD2D)
        results->bandwidth = testDeviceToDeviceTransfer(results->cuda, results->params);

    // Check the results
    bool bTestResult = testCheckResults(results->cuda, results->params);

    if (bTestResult)
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    // Free the memory
    freeMemory(results->cuda, memMode);

    // Destroy the context
    result = cuCtxDestroy(context);
    checkCudaErrors(result);
}

// Define a function to run the tests with gtest
void testBandwidth(int deviceNo, testMode *mode, testMemory *memMode, testWC *wc, testRange *range,
                   testStart *start, testEnd *end, testIncrement *inc)
{
    // Create the structures for the tests
    testCUDA cuda;
    testParams params;
    testResults results;

    // Initialize the structures
    memset(&cuda, 0, sizeof(cuda));
    memset(&params, 0, sizeof(params));
    memset(&results, 0, sizeof(results));

    results.cuda = &cuda;
    results.params = &params;

    // Set the number of iterations for each test
    params.iNumIterations = 100;

    // Set the size of each transfer
    if (range->bRange)
        params.iSize = start->iStart;
    else
        params.iSize = 1024 * 1024; // 1 MB

    // Run the tests for each transfer size
    while (params.iSize <= end->iEnd)
    {
        // Run the test
        runTest(&results, mode, memMode, wc, range, start, end, inc);

        // Print the results
        printResults(&results, mode, memMode, deviceNo);

        // Check the bandwidth and compare it with the peak bandwidth
        float peakBandwidth = 2.0f * results.cuda->deviceProp.memoryClockRate *
                              (results.cuda->deviceProp.memoryBusWidth / 8) / 1.0e6;
        EXPECT_GE(results.bandwidth / peakBandwidth * 100.0f, 90.0f) << "The bandwidth is less than 90% of the peak bandwidth";

        // Increment the transfer size
        if (range->bRange)
            params.iSize += inc->iIncrement;
        else
            break;
    }
}

// Define a macro to create a gtest for each memory copy kind and memory type
#define TEST_BANDWIDTH(MEMORY_COPY_KIND, MEMORY_TYPE) \
TEST(BandwidthTest, MEMORY_COPY_KIND ## _ ## MEMORY_TYPE) \
{ \
    int deviceNo = 0; \
    testMode mode; \
    testMemory memMode; \
    testWC wc; \
    testRange range; \
    testStart start; \
    testEnd end; \
    testIncrement inc; \
    \
    memset(&mode, 0, sizeof(mode)); \
    memset(&memMode, 0, sizeof(memMode)); \
    memset(&wc, 0, sizeof(wc)); \
    memset(&range, 0, sizeof(range)); \
    memset(&start, 0, sizeof(start)); \
    memset(&end, 0, sizeof(end)); \
    memset(&inc, 0, sizeof(inc)); \
    \
    mode.b ## MEMORY_COPY_KIND = true; \
    memMode.b ## MEMORY_TYPE = true; \
    range.bRange = true; \
    start.iStart = 1024; \
    end.iEnd = 1024 * 1024 * 100; \
    inc.iIncrement = 1024 * 1024; \
    \
    testBandwidth(deviceNo, &mode, &memMode, &wc, &range, &start, &end, &inc); \
}

// Create gtests for device to host transfers with pageable and pinned memory
TEST_BANDWIDTH(D2H, Pageable)
TEST_BANDWIDTH(D2H, Pinned)

// Create gtests for host to device transfers with pageable and pinned memory
TEST_BANDWIDTH(H2D, Pageable)
TEST_BANDWIDTH(H2D, Pinned)

// Create gtests for device to device transfers
TEST_BANDWIDTH(D2D, Pageable)
