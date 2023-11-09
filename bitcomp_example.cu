/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Simple example to show how to use bitcomp's native lossy API to compress
// floating point data.
//
// Bitcomp's lossy compression performs an on-the-fly integer quantization
// and compresses the resulting integral values with the lossless encoder.
// A smaller delta used for the quantization will typically lower the
// compression ratio, but will increase precision.

#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>

#include <native/bitcomp.h>

#include "utils.h"

#define DATA_TYPE uint8_t

#define CUDA_CHECK(func)                                                        \
    do                                                                          \
    {                                                                           \
        cudaError_t rt = (func);                                                \
        if (rt != cudaSuccess)                                                  \
        {                                                                       \
            std::cout << "API call failure \"" #func "\" with " << rt << " at " \
                      << __FILE__ << ":" << __LINE__ << std::endl;              \
            throw;                                                              \
        }                                                                       \
    } while (0);

#define BITCOMP_CHECK(call)                                    \
    {                                                          \
        bitcompResult_t err = call;                            \
        if (BITCOMP_SUCCESS != err)                            \
        {                                                      \
            fprintf(                                           \
                stderr,                                        \
                "Bitcomp error %d in file '%s' in line %i.\n", \
                err,                                           \
                __FILE__,                                      \
                __LINE__);                                     \
            fflush(stderr);                                    \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    }

template <typename T>
T *compress(char *filePath)
{
    size_t fileSize = io::FileSize(filePath);
    T *inputHost = (T *)malloc(fileSize);
    io::read_binary_to_array<T>(filePath, inputHost, fileSize / sizeof(T));

    T *inputDevice;
    CUDA_CHECK(cudaMalloc(&inputDevice, fileSize));
    CUDA_CHECK(cudaMemcpy(inputDevice, inputHost, fileSize, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Let's execute all the GPU code in a non-default stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create a bitcomp plan to compress FP32 data using a signed integer
    // quantization, since the input data contains positive and negative values.
    bitcompHandle_t plan;
    BITCOMP_CHECK(bitcompCreatePlan(
        &plan,                  // Bitcomp handle
        fileSize,               // Size in bytes of the uncompressed data
        BITCOMP_UNSIGNED_8BIT,  // Data type
        BITCOMP_LOSSLESS,       // Compression type
        BITCOMP_DEFAULT_ALGO)); // Bitcomp algo, default or sparse

    // Query the maximum size of the compressed data (worst case scenario)
    // and allocate the compressed buffer
    size_t maxlen = bitcompMaxBuflen(fileSize);
    void *compbuf;
    CUDA_CHECK(cudaMalloc(&compbuf, maxlen));

    // Associate the bitcomp plan to the stream, otherwise the compression
    // or decompression would happen in the default stream
    BITCOMP_CHECK(bitcompSetStream(plan, stream));

    // Start recording on the specified stream
    cudaEventRecord(start, stream);

    // Compress the input data with the chosen quantization delta
    BITCOMP_CHECK(bitcompCompressLossless(plan, inputDevice, compbuf));

    // Stop recording on the specified stream
    cudaEventRecord(stop, stream);

    // Wait for the compression kernel to finish
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Query the compressed size
    size_t compsize;
    BITCOMP_CHECK(bitcompGetCompressedSize(compbuf, &compsize));
    float ratio = static_cast<float>(fileSize) / static_cast<float>(compsize);
    printf("Compression ratio = %.2f\n", ratio);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output the elapsed time
    std::cout << "Compression elapsed time: " << milliseconds << " ms" << std::endl;
    std::cout << "Compression throughput: " << static_cast<float>(fileSize) / 1024 / 1024 / milliseconds << " GB/s" << std::endl;

    char *compressedDataHost = (char *)malloc(compsize);
    CUDA_CHECK(cudaMemcpy(compressedDataHost, compbuf, compsize, cudaMemcpyDeviceToHost));
    std::string str(filePath);
    io::write_array_to_binary<char>(str + ".bitcomp", compressedDataHost, compsize);

    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Clean up
    BITCOMP_CHECK(bitcompDestroyPlan(plan));
    CUDA_CHECK(cudaFree(compbuf));
    CUDA_CHECK(cudaFree(inputDevice));

    // free(inputHost);
    free(compressedDataHost);
    return inputHost;
}

template <typename T>
T *decompress(char *filePath, size_t originalSize)
{
    size_t fileSize = io::FileSize(filePath);
    char *inputHost = (char *)malloc(fileSize);
    io::read_binary_to_array<char>(filePath, inputHost, fileSize / sizeof(char));

    char *inputDevice;
    CUDA_CHECK(cudaMalloc(&inputDevice, fileSize));
    CUDA_CHECK(cudaMemcpy(inputDevice, inputHost, fileSize, cudaMemcpyHostToDevice));

    // Let's execute all the GPU code in a non-default stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate a buffer for the decompressed data
    T *outputDevice;
    CUDA_CHECK(cudaMalloc(&outputDevice, originalSize));

    // Create a bitcomp plan to compress FP32 data using a signed integer
    // quantization, since the input data contains positive and negative values.
    bitcompHandle_t plan;
    BITCOMP_CHECK(bitcompCreatePlan(
        &plan,                  // Bitcomp handle
        originalSize,           // Size in bytes of the uncompressed data
        BITCOMP_UNSIGNED_8BIT,  // Data type
        BITCOMP_LOSSLESS,       // Compression type
        BITCOMP_DEFAULT_ALGO)); // Bitcomp algo, default or sparse

    // Associate the bitcomp plan to the stream, otherwise the compression
    // or decompression would happen in the default stream
    BITCOMP_CHECK(bitcompSetStream(plan, stream));

    // Start recording on the specified stream
    cudaEventRecord(start, stream);

    // Decompress the data
    BITCOMP_CHECK(bitcompUncompress(plan, inputDevice, outputDevice));

    // Stop recording on the specified stream
    cudaEventRecord(stop, stream);

    // Wait for the decompression to finish
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output the elapsed time
    std::cout << "Decompression elapsed time: " << milliseconds << " ms" << std::endl;
    std::cout << "Decompression throughput: " << static_cast<float>(originalSize) / 1024 / 1024 / milliseconds << " GB/s" << std::endl;

    T *outputHost = (T *)malloc(originalSize);
    CUDA_CHECK(cudaMemcpy(outputHost, outputDevice, originalSize, cudaMemcpyDeviceToHost));
    std::string str(filePath);
    io::write_array_to_binary<T>(str + ".decompressed", outputHost, originalSize / sizeof(T));

    // Clean up
    BITCOMP_CHECK(bitcompDestroyPlan(plan));
    CUDA_CHECK(cudaFree(inputDevice));
    CUDA_CHECK(cudaFree(outputDevice));

    free(inputHost);
    // free(outputHost);
    return outputHost;
}

template <typename T>
void roundTripVerification(char *filePath)
{
    size_t fileSize = io::FileSize(filePath);
    T *originalData = compress<T>(filePath);
    std::string str(filePath);
    str = str + ".bitcomp";
    char *compressedFilePath = (char *)str.c_str();
    T *reconstructedData = decompress<T>(compressedFilePath, fileSize);

    for (int i = 0; i < fileSize / sizeof(T); i++)
    {
        if (originalData[i] != reconstructedData[i])
        {
            std::cout << "Error: originalData[" << i << "] = " << originalData[i] << " != reconstructedData[" << i << "] = " << reconstructedData[i] << std::endl;
            return;
        }
    }

    free(originalData);
    free(reconstructedData);
    return;
}

int main(int argc, char *argv[])
{
    if (strcmp(argv[1], "-c") == 0)
    {
        DATA_TYPE *originalData = compress<DATA_TYPE>(argv[2]);
        free(originalData);
    }
    else if (strcmp(argv[1], "-d") == 0)
    {
        size_t originalSize = std::stoi(argv[3]);
        DATA_TYPE *reconstructedData = decompress<DATA_TYPE>(argv[2], originalSize);
        free(reconstructedData);
    }
    else if (strcmp(argv[1], "-r") == 0)
    {
        roundTripVerification<DATA_TYPE>(argv[2]);
    }

    return 0;
}