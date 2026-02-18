#include <array>
#include <cmath>
#include <algorithm>
#include "waveletDenoiser.hpp"

denoiser::denoiser() {
    constexpr double pi = 3.1415926;
    for (int n = 0; n < windowSize; ++n) {
        win_[n] = 0.5 - 0.5 * std::cos(2.0 * pi * n / (windowSize - 1));
    }
}

void denoiser::shift_left_hop_(std::array<double, windowSize>& acc,
                               std::array<double, windowSize>& wsum)
{
    // push data forward
    for (int i = 0; i < windowSize - hop; ++i) {
        acc[i]  = acc[i + hop];
        wsum[i] = wsum[i + hop];
    }
    // set last data to 0
    for (int i = windowSize - hop; i < windowSize; ++i) {
        acc[i]  = 0.0;
        wsum[i] = 0.0;
    }
}

void denoiser::add_block_wola_(std::array<double, windowSize>& acc,
                               std::array<double, windowSize>& wsum,
                               const std::array<double, windowSize>& block)
{
    for (int n = 0; n < windowSize; ++n) {
        const double w = win_[n];
        acc[n]  += block[n] * w;
        wsum[n] += w;
    }
}

void denoiser::push(double t, double ax, double ay, double az)
{
    // overwrite oldest slot, then move idx forward
    t_[idx]  = t;
    ax_[idx] = ax;
    ay_[idx] = ay;
    az_[idx] = az;

    idx = (idx + 1) % windowSize;

    if (!full) {
        count++;
        if (count >= windowSize) full = true;
    }

    hop_counter++;
}

void denoiser::denoise_axis_(std::array<double, windowSize>& w)
{
    haar_dwt(w, levels);

    // Layout after DWT(levels=3): [A3(8) | D3(8) | D2(16) | D1(32)]
    const int D1_start = 32, D1_len = 32;
    const int D2_start = 16, D2_len = 16;
    const int D3_start =  8, D3_len =  8;

    double sigma = mad_sigma_from_detail(w, D1_start, D1_len);
    if (sigma > 0.0) {
        const double N = static_cast<double>(windowSize);
        const double T = sigma * std::sqrt(2.0 * std::log(N));

        const double T1 = 1.0 * T;
        const double T2 = 0.6 * T;
        const double T3 = 0.2 * T;

        soft_threshold_range(w, D1_start, D1_len, T1);
        soft_threshold_range(w, D2_start, D2_len, T2);
        soft_threshold_range(w, D3_start, D3_len, T3);
    }

    haar_idwt(w, levels);
}

bool denoiser::denoise()
{
    if (!full) return false;
    if (hop_counter < hop) return false;
    hop_counter = 0;

    // 1) Rebuild window in time order: oldest -> newest
    std::array<double, windowSize> wx{}, wy{}, wz{};
    int current = idx; // idx points to the oldest slot (next to be overwritten)
    for (int i = 0; i < windowSize; ++i) {
        wx[i] = ax_[current];
        wy[i] = ay_[current];
        wz[i] = az_[current];
        current = (current + 1) % windowSize;
    }

    // 2) Denoise each axis (wavelet thresholding)
    denoise_axis_(wx);
    denoise_axis_(wy);
    denoise_axis_(wz);

    // 3) WOLA: advance accumulators by hop
    shift_left_hop_(ola_x_acc_, ola_x_wsum_);
    shift_left_hop_(ola_y_acc_, ola_y_wsum_);
    shift_left_hop_(ola_z_acc_, ola_z_wsum_);

    // 4) Add current denoised block (weighted)
    add_block_wola_(ola_x_acc_, ola_x_wsum_, wx);
    add_block_wola_(ola_y_acc_, ola_y_wsum_, wy);
    add_block_wola_(ola_z_acc_, ola_z_wsum_, wz);

    // 5) Emit hop samples (normalized)
    for (int k = 0; k < hop; ++k) {
        out_x_[k] = (ola_x_wsum_[k] > 1e-12) ? (ola_x_acc_[k] / ola_x_wsum_[k]) : 0.0;
        out_y_[k] = (ola_y_wsum_[k] > 1e-12) ? (ola_y_acc_[k] / ola_y_wsum_[k]) : 0.0;
        out_z_[k] = (ola_z_wsum_[k] > 1e-12) ? (ola_z_acc_[k] / ola_z_wsum_[k]) : 0.0;
    }

    return true;
}

void denoiser::haar_dwt(std::array<double, 64>& x, int levels){
    std::array<double, 64> temp;
    int length = windowSize;

    for (int i = 0; i < levels; i++){
        int half = length / 2;
        for (int j = 0; j < half; j++){
            // approxiamtion
            temp[j] = (x[2 * j] + x[2 * j + 1]) / std::sqrt(2.0); 
            // detailed
            temp[half + j] = (x[2 * j] - x[2 * j + 1]) / std::sqrt(2.0);
        }
        length /= 2;
        x = temp;
    }
}

void denoiser::haar_idwt(std::array<double, 64>& x, int levels){
    std::array<double, 64> temp;
    int length = windowSize;
    for (int i = 0; i < levels - 1; i++)
        length /= 2;

    for (int i = 0; i < levels; i++){
        int half = length / 2;
        for (int j = 0; j < half; j++){
            temp[2 * j] = (x[j] + x[half + j]) / std::sqrt(2.0);
            temp[2 * j + 1] = (x[j] - x[half + j]) / std::sqrt(2.0);
        }
        for (int j = 0; j < length; j++)
            x[j] = temp[j];
        length *= 2;
    }
}

double denoiser::median(std::array<double, 64> v, std::size_t n) {
    // Compute median of first n entries in v (copy by value)
    if (n == 0) return 0.0;
    auto begin = v.begin();
    auto mid = begin + static_cast<std::ptrdiff_t>(n / 2);
    std::nth_element(begin, mid, begin + static_cast<std::ptrdiff_t>(n));
    double m = *mid;
    if (n % 2 == 0) {
        auto mid2 = begin + static_cast<std::ptrdiff_t>(n / 2 - 1);
        std::nth_element(begin, mid2, begin + static_cast<std::ptrdiff_t>(n));
        m = 0.5 * (m + *mid2);
    }
    return m;
}

double denoiser::mad_sigma_from_detail(const std::array<double, 64>& coeffs,
                                              int start, int len) {
    // sigma ≈ MAD / 0.6745
    // MAD = median(|d - median(d)|)
    std::array<double, 64> tmp{};
    if (len == 0) return 0.0;

    // calculate median
    for (std::size_t i = 0; i < len; ++i) tmp[i] = coeffs[start + i];
    double med = median(tmp, len);

    // calculate diff median
    for (std::size_t i = 0; i < len; ++i) tmp[i] = std::fabs(tmp[i] - med);
    double mad = median(tmp, len);

    return mad / 0.6745;
}

void denoiser::soft_threshold_range(std::array<double, 64>& coeffs,
                                           int start, int len,
                                           double T) {
    // Soft threshold: sign(x)*max(|x|-T,0)
    const int end = start + len;
    for (int i = start; i < end; ++i) {
        double x = coeffs[i];
        double ax = std::fabs(x);
        if (ax <= T) {
            coeffs[i] = 0.0;
        } else {
            coeffs[i] = (x > 0 ? 1.0 : -1.0) * (ax - T);
        }
    }
}