// 这是一个可能的代码示例，使用fork和pipe来实现多进程多GPU的cuda应用程序。
// 假设我们有一个简单的向量加法的cuda程序，我们想要用多个GPU同时运行它，并且利用管道来传递输入和输出数据

// 编译向量加法的cuda程序，假设生成的可执行文件名为vecAdd。
// 编写一个主程序，使用cuda driver api的cuDeviceGetCount和cuDeviceGet函数来获取可用的GPU数量和句柄，
// 然后为每个GPU创建一个子进程，并在子进程中创建和销毁自己的cuda上下文。
// 使用pipe函数来创建一对管道，分别用于从主进程向子进程发送输入数据和从子进程向主进程接收输出数据。
// 使用fork函数来创建子进程，并在子进程中使用dup2函数来重定向标准输入和输出到管道。使用exec函数来执行vecAdd程序，并传递GPU句柄作为参数。
// 在主进程中，使用write和read函数来向管道写入和读取数据，并等待所有子进程结束。例如：

#include <cuda.h>
#include <unistd.h>
#include <sys/wait.h>
#include <iostream>

// Define a helper function to check CUDA errors
void checkCudaError(CUresult result, const char* msg)
{
    if (result != CUDA_SUCCESS) {
        const char* error;
        cuGetErrorName(result, &error);
        std::cerr << "CUDA error: " << msg << ": " << error << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv)
{
    // Initialize the CUDA driver API
    checkCudaError(cuInit(0), "cuInit");

    // Get the number and handles of available GPUs
    int deviceCount;
    checkCudaError(cuDeviceGetCount(&deviceCount), "cuDeviceGetCount");
    CUdevice devices[deviceCount];
    for (int i = 0; i < deviceCount; i++) {
        checkCudaError(cuDeviceGet(&devices[i], i), "cuDeviceGet");
    }

    // Create a child process for each GPU
    pid_t pids[deviceCount];
    int input_pipes[deviceCount][2];
    int output_pipes[deviceCount][2];
    for (int i = 0; i < deviceCount; i++) {
        // Create a pair of pipes for input and output data
        if (pipe(input_pipes[i]) == -1 || pipe(output_pipes[i]) == -1) {
            std::cerr << "Pipe error" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Fork a child process
        pids[i] = fork();
        if (pids[i] == -1) {
            std::cerr << "Fork error" << std::endl;
            exit(EXIT_FAILURE);
        }

        if (pids[i] == 0) {
            // In the child process

            // Close the unused ends of the pipes
            close(input_pipes[i][1]);
            close(output_pipes[i][0]);

            // Redirect the standard input and output to the pipes
            dup2(input_pipes[i][0], STDIN_FILENO);
            dup2(output_pipes[i][1], STDOUT_FILENO);

            // Execute the vecAdd program with the GPU handle as an argument
            char arg[10];
            sprintf(arg, "%d", devices[i]);
            execl("./vecAdd", "./vecAdd", arg, nullptr);

            // If exec returns, there is an error
            std::cerr << "Exec error" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // In the parent process

    // Close the unused ends of the pipes
    for (int i = 0; i < deviceCount; i++) {
        close(input_pipes[i][0]);
        close(output_pipes[i][1]);
    }

    // Allocate and initialize host input and output vectors
    int N = 256 * deviceCount;
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = 0.0f;
    }

    // Write the input vectors to the input pipes
        for (int i = 0; i < deviceCount; i++) {
        // Write the size of the vector
        int size = N / deviceCount;
        if (write(input_pipes[i][1], &size, sizeof(int)) == -1) {
            std::cerr << "Write error" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Write the vector A
        if (write(input_pipes[i][1], h_A + i * size, size * sizeof(float)) == -1) {
            std::cerr << "Write error" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Write the vector B
        if (write(input_pipes[i][1], h_B + i * size, size * sizeof(float)) == -1) {
            std::cerr << "Write error" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Read the output vectors from the output pipes
    for (int i = 0; i < deviceCount; i++) {
        // Read the size of the vector
        int size;
        if (read(output_pipes[i][0], &size, sizeof(int)) == -1) {
            std::cerr << "Read error" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Read the vector C
        if (read(output_pipes[i][0], h_C + i * size, size * sizeof(float)) == -1) {
            std::cerr << "Read error" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Verify the results
    for (int i = 0; i < N; i++) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            std::cerr << "Incorrect result at index " << i << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    // Wait for all child processes to finish
    for (int i = 0; i < deviceCount; i++) {
        int status;
        waitpid(pids[i], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            std::cerr << "Child process " << pids[i] << " exited abnormally" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return 0;
}
