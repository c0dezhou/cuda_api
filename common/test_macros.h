#pragma once

#include "gtest/gtest.h"
#include <cuda.h>

#define EXPECT_CUDA_SUCCESS(call)                                                                              \
    do                                                                                                         \
    {                                                                                                          \
        CUresult _result = call;                                                                               \
        EXPECT_EQ(_result, CUDA_SUCCESS) << "Expected " #call " to return CUDA_SUCCESS, but got: " << _result; \
    } while (0)
