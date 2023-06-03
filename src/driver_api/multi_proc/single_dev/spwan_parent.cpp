#include <gtest/gtest.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>

// g++ spwan_child.cpp -o spwan_child -lcuda -I/usr/local/cuda/include -lpthread

TEST(MPROC, spwan_single) {
    pid_t pid1, pid2;
    char *args[] = { "./spwan_child", NULL };
    posix_spawn(&pid1, args[0], NULL, NULL, args, NULL);
    posix_spawn(&pid2, args[0], NULL, NULL, args, NULL);

    // Wait for both child processes to finish
    waitpid(pid1, NULL, 0);
    waitpid(pid2, NULL, 0);

}
