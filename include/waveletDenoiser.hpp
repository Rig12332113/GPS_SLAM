#include <array>
#include <cmath>
#include <algorithm>

class denoiser {
public:
    static constexpr int windowSize = 64;
    static constexpr int levels     = 3;
    static constexpr int hop        = 8;

    denoiser();  // will init window weights

    void push(double t, double ax, double ay, double az);

    // Call when you have added hop samples (or gate internally)
    // Returns true when it emitted hop samples into out_* buffers.
    bool denoise();

    // Access last emitted hop samples
    const std::array<double, hop>& out_x() const { return out_x_; }
    const std::array<double, hop>& out_y() const { return out_y_; }
    const std::array<double, hop>& out_z() const { return out_z_; }

private:
    // -------- input ring buffers --------
    std::array<double, windowSize> t_{}, ax_{}, ay_{}, az_{};
    int idx = 0;         // points to the oldest sample position
    int count = 0;       // number of samples received (cap at windowSize)
    int hop_counter = 0; // counts samples since last denoise
    bool full = false;

    // -------- WOLA state (Part 1) --------
    std::array<double, windowSize> win_{};       // Hann or rectangular weights
    std::array<double, windowSize> ola_x_acc_{}, ola_x_wsum_{};
    std::array<double, windowSize> ola_y_acc_{}, ola_y_wsum_{};
    std::array<double, windowSize> ola_z_acc_{}, ola_z_wsum_{};

    // emitted hop samples each denoise call
    std::array<double, hop> out_x_{}, out_y_{}, out_z_{};

    // -------- wavelet core --------
    void haar_dwt(std::array<double, windowSize>& x, int levels);
    void haar_idwt(std::array<double, windowSize>& x, int levels);

    double median(std::array<double, windowSize> v, std::size_t n);
    double mad_sigma_from_detail(const std::array<double, windowSize>& coeffs, int start, int length);
    void soft_threshold_range(std::array<double, windowSize>& coeffs, int start, int length, double T);

    // -------- helpers (Part 3) --------
    static void shift_left_hop_(std::array<double, windowSize>& acc,
                                std::array<double, windowSize>& wsum);
    void add_block_wola_(std::array<double, windowSize>& acc,
                         std::array<double, windowSize>& wsum,
                         const std::array<double, windowSize>& block);

    void denoise_axis_(std::array<double, windowSize>& w);
};
