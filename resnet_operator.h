#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
void conv(float* M1, float* kernel, float* M_out, int width, int height, int in_channels, int out_channels, int kernel_size, int stride, int padding = 3);
void BatchNormed(int in_channels, int width, int height, float* M1, float* mean_value, float* stddev);
void ReLu(int in_channels, int width, int height, float* M1);
void Maxpool(int in_channels, int width, int height, int out_width, int out_height, float* M1, float* M1_out, int kernel_size, int stride);
void compute_mean_stddev(float* M1_out, float* mean_value, float* stddev, int out_width, int out_height, int out_channels);
void Copy(float* shortcut, float* M_out, int width, int height, int channels);
void Initialize(float *M, int width, int height, int channel, int dim);
void add(float *M,float *res,float *M_out,int width,int height,int channel);
// void intTofloat(int64_t *M_int,float *M,int width,int height,int channel);
// void floatToint(int64_t *M_int,float *M,int width,int height,int channel);