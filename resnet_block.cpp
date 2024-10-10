#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <random>
#include <ctime>
#include <sys/time.h>
#include "apis_c.h"

#define THREADSIZE 10

using namespace std;

void conv_block(float* M1, float* M1_out, float* kernel, int width, int height, int in_channels, int out_width, int out_height, int out_channels = 56, int padding = 3, int stride = 2, int kernel_size = 1) {
    // Placeholder convolution operation
    for (int c = 0; c < out_channels; ++c) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                float sum = 0.0f;
                for (int kc = 0; kc < in_channels; ++kc) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int h_offset = h * stride + kh - padding;
                            int w_offset = w * stride + kw - padding;
                            if (h_offset >= 0 && h_offset < height && w_offset >= 0 && w_offset < width) {
                                sum += M1[(kc * height + h_offset) * width + w_offset] * kernel[((c * in_channels + kc) * kernel_size + kh) * kernel_size + kw];
                            }
                        }
                    }
                }
                M1_out[(c * out_height + h) * out_width + w] = sum;
            }
        }
    }

    // Placeholder Batch Normalization and ReLU
    for (int c = 0; c < out_channels; ++c) {
        float mean = 0.0f;
        float stddev = 0.0f;
        // Compute mean
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                mean += M1_out[(c * out_height + h) * out_width + w];
            }
        }
        mean /= (out_height * out_width);

        // Compute standard deviation
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                float diff = M1_out[(c * out_height + h) * out_width + w] - mean;
                stddev += diff * diff;
            }
        }
        stddev = sqrt(stddev / (out_height * out_width));

        // Apply Batch Normalization and ReLU
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                int idx = (c * out_height + h) * out_width + w;
                M1_out[idx] = (M1_out[idx] - mean) / (stddev + 1e-5f); // Batch Norm
                M1_out[idx] = max(0.0f, M1_out[idx]); // ReLU
            }
        }
    }
}

void bnk1(float* M1, float* M_out, int width, int height, int in_channels, int out_channels, int padding, int stride = 2) {
    int kernel_size_first = 1;
    int out_channels_first = out_channels / 4;
    float* kernel = new float[in_channels * kernel_size_first * kernel_size_first * out_channels_first];

    // Initialize kernel with random values
    for (int i = 0; i < in_channels * kernel_size_first * kernel_size_first * out_channels_first; ++i) {
        kernel[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    int out_width = (width - kernel_size_first + 2 * padding) / stride + 1;
    int out_height = (height - kernel_size_first + 2 * padding) / stride + 1;
    float* M1_out = new float[out_width * out_height * out_channels_first];

    conv_block(M1, M1_out, kernel, width, height, in_channels, out_width, out_height, out_channels_first, padding, stride, kernel_size_first);

    // Residual connection
    int kernel_size_res = 1;
    int out_channels_res = out_channels;
    float* kernel_res = new float[in_channels * kernel_size_res * kernel_size_res * out_channels_res];

    // Initialize residual kernel with random values
    for (int i = 0; i < in_channels * kernel_size_res * kernel_size_res * out_channels_res; ++i) {
        kernel_res[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float* Mres_out = new float[out_width * out_height * out_channels_res];
    conv_block(M1, Mres_out, kernel_res, width, height, in_channels, out_width, out_height, out_channels_res, padding, stride, kernel_size_res);

    // Add residual output to main output
    for (int i = 0; i < out_width * out_height * out_channels_res; ++i) {
        M_out[i] = M1_out[i] + Mres_out[i];
    }

    delete[] kernel;
    delete[] kernel_res;
    delete[] M1_out;
    delete[] Mres_out;
}

int main(int argc, char** argv) {
    int idX = atoi(argv[1]);
    int idY = atoi(argv[2]);
    int batch = atoi(argv[3]);
    int width = 0, height = 0, channels = 0;
    int out_width = 0, out_height = 0, out_channels = 0;

    if (idY == 1) {
        width = 56;
        height = 56;
        channels = 64;
        out_width = 56;
        out_height = 56;
        out_channels = 256;
    } else if (idY == 2) {
        width = 56;
        height = 56;
        channels = 256;
        out_width = 28;
        out_height = 28;
        out_channels = 512;
    } else if (idY == 3) {
        width = 28;
        height = 28;
        channels = 512;
        out_width = 14;
        out_height = 14;
        out_channels = 1024;
    } else if (idY == 4) {
        width = 14;
        height = 14;
        channels = 1024;
        out_width = 7;
        out_height = 7;
        out_channels = 2048;
    } else {
        return 1;
    }

    float* M = new float[width * height * channels];
    float* M_out = new float[out_width * out_height * out_channels];

    for (int i = 0; i < batch; i++) {
        InterChiplet::receiveMessage(idX, idY, 0, 0, M, width * height * channels * sizeof(float));

        if (idY == 1) {
            bnk1(M, M_out, width, height, channels, out_channels, width / 2);
            float* M_out_sec = new float[out_width * out_height * out_channels];
            bnk1(M_out, M_out_sec, out_width, out_height, out_channels, out_channels, 0);
            bnk1(M_out_sec, M_out, out_width, out_height, out_channels, out_channels, 0);
            delete[] M_out_sec;
        } else if (idY == 2) {
            bnk1(M, M_out, width, height, channels, out_channels, 0);
            float* M_out_sec = new float[out_width * out_height * out_channels];
            float* M_out_third = new float[out_width * out_height * out_channels];
            bnk1(M_out, M_out_sec, out_width, out_height, out_channels, out_channels, 0);
            bnk1(M_out_sec, M_out_third, out_width, out_height, out_channels, out_channels, 0);
            bnk1(M_out_third, M_out, out_width, out_height, out_channels, out_channels, 0);
            delete[] M_out_sec;
            delete[] M_out_third;
        } else if (idY == 3) {
            bnk1(M, M_out, width, height, channels, out_channels, 0);
            float* M_out_sec = new float[out_width * out_height * out_channels];
            float* M_out_third = new float[out_width * out_height * out_channels];
            float* M_out_forth = new float[out_width * out_height * out_channels];
            float* M_out_fifth = new float[out_width * out_height * out_channels];
            bnk1(M_out, M_out_sec, out_width, out_height, out_channels, out_channels, 0);
            bnk1(M_out_sec, M_out_third, out_width, out_height, out_channels, out_channels, 0);
            bnk1(M_out_third, M_out_forth, out_width, out_height, out_channels, out_channels, 0);
            bnk1(M_out_forth, M_out_fifth, out_width, out_height, out_channels, out_channels, 0);
            bnk1(M_out_fifth, M_out, out_width, out_height, out_channels, out_channels, 0);
            delete[] M_out_sec;
            delete[] M_out_third;
            delete[] M_out_forth;
            delete[] M_out_fifth;
        } else if (idY == 4) {
            bnk1(M, M_out, width, height, channels, out_channels, 0);
            float* M_out_sec = new float[out_width * out_height * out_channels];
            bnk1(M_out, M_out_sec, out_width, out_height, out_channels, out_channels, 0);
            bnk1(M_out_sec, M_out, out_width, out_height, out_channels, out_channels, 0);
            delete[] M_out_sec;
        }

        InterChiplet::sendMessage(0, 0, idX, idY, M_out, out_width * out_height * out_channels * sizeof(float));
    }

    delete[] M;
    delete[] M_out;
    return 0;
}