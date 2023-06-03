#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <gtest/gtest.h>

#define SIZE_SMALL 10000
#define SIZE_MEDIUM 10000000
#define SIZE_LARGE 100000000
#define DEVICE_COUNT 2
#define ROUNDS 10

TEST(MPROC, spwan_multi) {
    char *sizes[] = {"small", "medium", "large"};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < DEVICE_COUNT; i++) {
        for (int j = 0; j < num_sizes; j++) {
            pid_t pid;
            char device_str[16], size_str[16];
            sprintf(device_str, "%d", i);
            sprintf(size_str, "%d", sizes[j]);
            char *args[] = { "./spwan_child", device_str, size_str, NULL };

            posix_spawn(&pid, "./spwan_child", NULL, NULL, args, NULL);

            waitpid(pid, NULL, 0);
        }
    }

}
