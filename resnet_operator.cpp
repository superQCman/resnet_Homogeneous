#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

void conv(const vector<float>& M1, const vector<float>& kernel, vector<float>& M_out, int width, int height, int in_channels, int out_channels, int kernel_size, int stride, int padding = 3) {
    int out_width = (width - kernel_size + 2 * padding) / stride + 1;
    int out_height = (height - kernel_size + 2 * padding) / stride + 1;

    for (int z = 0; z < out_channels; ++z) {
        for (int y = 0; y < out_height; ++y) {
            for (int x = 0; x < out_width; ++x) {
                int x_start = x * stride - padding;
                int y_start = y * stride - padding;

                float sum = 0;
                for (int c = 0; c < in_channels; ++c) {
                    for (int i = 0; i < kernel_size; ++i) {
                        for (int j = 0; j < kernel_size; ++j) {
                            int input_x = x_start + j;
                            int input_y = y_start + i;

                            if (input_x >= 0 && input_x < width && input_y >= 0 && input_y < height) {
                                float val = M1[input_x + input_y * width + c * width * height];
                                float k_val = kernel[j + i * kernel_size + c * kernel_size * kernel_size + z * kernel_size * kernel_size * in_channels];
                                sum += val * k_val;
                            }
                        }
                    }
                }
                M_out[x + y * out_width + z * out_height * out_width] = sum;
            }
        }
    }
}

void BatchNormed(int in_channels, int width, int height, vector<float>& M1, const vector<float>& mean_value, vector<float>& stddev) {
    for (int z = 0; z < in_channels; ++z) {
        if (stddev[z] == 0) {
            stddev[z] = 1;
        }
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                M1[x + y * width + z * width * height] = (M1[x + y * width + z * width * height] - mean_value[z]) / stddev[z];
            }
        }
    }
}

void ReLu(int in_channels, int width, int height, vector<float>& M1) {
    for (int z = 0; z < in_channels; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (M1[x + y * width + z * width * height] < 0) {
                    M1[x + y * width + z * width * height] = 0;
                }
            }
        }
    }
}

void Maxpool(int in_channels, int width, int height, int out_width, int out_height, const vector<float>& M1, vector<float>& M1_out, int kernel_size, int stride) {
    for (int z = 0; z < in_channels; ++z) {
        for (int y = 0; y < out_height; ++y) {
            for (int x = 0; x < out_width; ++x) {
                int x_start = x * stride;
                int y_start = y * stride;
                float Max = M1[x_start + y_start * width + z * width * height];
                for (int i = 0; i < kernel_size; ++i) {
                    for (int j = 0; j < kernel_size; ++j) {
                        int x_input = x_start + i;
                        int y_input = y_start + j;

                        if (x_input >= 0 && x_input < width && y_input >= 0 && y_input < height) {
                            if (Max < M1[x_input + y_input * width + z * width * height]) {
                                Max = M1[x_input + y_input * width + z * width * height];
                            }
                        }
                    }
                }
                M1_out[x + y * out_width + z * out_width * out_height] = Max;
            }
        }
    }
}

void compute_mean_stddev(const vector<float>& M1_out, vector<float>& mean_value, vector<float>& stddev, int out_width, int out_height, int out_channels) {
    for (int x = 0; x < out_channels; ++x) {
        float sum = 0;
        for (int j = 0; j < out_height; ++j) {
            for (int z = 0; z < out_width; ++z) {
                sum += M1_out[z + j * out_width + x * out_width * out_height];
            }
        }
        mean_value[x] = sum / (out_height * out_width);

        float dev = 0;
        for (int j = 0; j < out_height; ++j) {
            for (int z = 0; z < out_width; ++z) {
                float diff = M1_out[z + j * out_width + x * out_width * out_height] - mean_value[x];
                dev += diff * diff;
            }
        }
        stddev[x] = sqrtf(dev / (out_height * out_width));
    }
}

void Copy(const vector<float>& shortcut, vector<float>& M_out, int width, int height, int channels) {
    for (int z = 0; z < channels; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                M_out[x + y * width + z * width * height] = shortcut[x + y * width + z * width * height];
            }
        }
    }
}

void Initialize(vector<float>& M, int width, int height, int channel, int dim) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(0, 1);

    for (int z = 0; z < channel; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int i = 0; i < dim; ++i) {
                    M[x + y * width + z * height * width + i * width * height * channel] = d(gen);
                }
            }
        }
    }
}

void add(const vector<float>& M, const vector<float>& res, vector<float>& M_out, int width, int height, int channel) {
    for (int z = 0; z < channel; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                M_out[x + y * width + z * height * width] = M[x + y * width + z * height * width] + res[x + y * width + z * height * width];
            }
        }
    }
}

// void intTofloat(const vector<int64_t>& M_int, vector<float>& M, int width, int height, int channel) {
//     float time = 1e4;
//     for (int z = 0; z < channel; ++z) {
//         for (int y = 0; y < height; ++y) {
//             for (int x = 0; x < width; ++x) {
//                 M[x + y * width + z * height * width] = M_int[x + y * width + z * height * width] / time;
//             }
//         }
//     }
// }

// void floatToint(vector<int64_t>& M_int, const vector<float>& M, int width, int height, int channel) {
//     float time = 1e4;
//     for (int z = 0; z < channel; ++z) {
//         for (int y = 0; y < height; ++y) {
//             for (int x = 0; x < width; ++x) {
//                 M_int[x + y * width + z * height * width] = M[x + y * width + z * height * width] * time;
//             }
//         }
//     }
// }