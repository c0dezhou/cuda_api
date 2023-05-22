// #include <cuda.h>
// #include <sys/wait.h>
// #include <unistd.h>
// #include <iostream>
// #include <cmath>

// void checkCudaError(CUresult result, const char* msg) {
//     if (result != CUDA_SUCCESS) {
//         const char* error;
//         cuGetErrorName(result, &error);
//         std::cerr << "CUDA error: " << msg << ": " << error << std::endl;
//         exit(EXIT_FAILURE);
//     }
// }

// int main(int argc, char** argv) {
//     checkCudaError(cuInit(0), "cuInit");

//     int deviceCount;
//     checkCudaError(cuDeviceGetCount(&deviceCount), "cuDeviceGetCount");
//     CUdevice devices[deviceCount];
//     for (int i = 0; i < deviceCount; i++) {
//         checkCudaError(cuDeviceGet(&devices[i], i), "cuDeviceGet");
//     }

//     pid_t pids[deviceCount];
//     int input_pipes[deviceCount][2];
//     int output_pipes[deviceCount][2];
//     for (int i = 0; i < deviceCount; i++) {
//         if (pipe(input_pipes[i]) == -1 || pipe(output_pipes[i]) == -1) {
//             std::cerr << "Pipe error" << std::endl;
//             exit(EXIT_FAILURE);
//         }

//         pids[i] = fork();
//         if (pids[i] == -1) {
//             std::cerr << "Fork error" << std::endl;
//             exit(EXIT_FAILURE);
//         }

//         if (pids[i] == 0) {
//             std::cout << "In the child process" << std::endl;

//             close(input_pipes[i][1]);
//             close(output_pipes[i][0]);

//             dup2(input_pipes[i][0], STDIN_FILENO);
//             dup2(output_pipes[i][1], STDOUT_FILENO);

//             // 执行两个向量加法
//             char arg[10];
//             sprintf(arg, "%d", devices[i]);
//             execl("./vectorAdd", "./vectorAdd", arg, nullptr);

//             std::cerr << "Exec error" << std::endl;
//             exit(EXIT_FAILURE);
//         }
//     }

//     // In the parent process
//     // Close the unused ends of the pipes
//     for (int i = 0; i < deviceCount; i++) {
//         close(input_pipes[i][0]);
//         close(output_pipes[i][1]);
//     }

//     int N = 256 * deviceCount;
//     float* h_A = new float[N];
//     float* h_B = new float[N];
//     float* h_C = new float[N];

//     for (int i = 0; i < N; i++) {
//         h_A[i] = rand() / (float)RAND_MAX;
//         h_B[i] = rand() / (float)RAND_MAX;
//         h_C[i] = 0.0f;
//     }

//     // 向input pipes写数据
//     for (int i = 0; i < deviceCount; i++) {
//         int size = N / deviceCount;
//         if (write(input_pipes[i][1], &size, sizeof(int)) == -1) {
//             std::cerr << "Write error" << std::endl;
//             exit(EXIT_FAILURE);
//         }

//         if (write(input_pipes[i][1], h_A + i * size, size * sizeof(float)) ==
//             -1) {
//             std::cerr << "Write error" << std::endl;
//             exit(EXIT_FAILURE);
//         }

//         if (write(input_pipes[i][1], h_B + i * size, size * sizeof(float)) ==
//             -1) {
//             std::cerr << "Write error" << std::endl;
//             exit(EXIT_FAILURE);
//         }
//     }

//     // 从output pipes读数据
//     for (int i = 0; i < deviceCount; i++) {
//         int size;
//         if (read(output_pipes[i][0], &size, sizeof(int)) == -1) {
//             std::cerr << "Read error" << std::endl;
//             exit(EXIT_FAILURE);
//         }

//         if (read(output_pipes[i][0], h_C + i * size, size * sizeof(float)) ==
//             -1) {
//             std::cerr << "Read error" << std::endl;
//             exit(EXIT_FAILURE);
//         }
//     }

//     for (int i = 0; i < N; i++) {
//         if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
//             std::cerr << "Incorrect result at index " << i << std::endl;
//             exit(EXIT_FAILURE);
//         }
//     }

//     delete[] h_A;
//     delete[] h_B;
//     delete[] h_C;

//     for (int i = 0; i < deviceCount; i++) {
//         int status;
//         waitpid(pids[i], &status, 0);
//         if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
//             std::cerr << "Child process " << pids[i] << " exited abnormally"
//                       << std::endl;
//             exit(EXIT_FAILURE);
//         }
//     }

//     return 0;
// }
