#include <iostream>

class IMUsample {
public:
    IMUsample() = default;

    // setters
    void setTimestamp(double t);
    void setQuat(const double q[4]);
    void setAccG(const double a[3]);

    // getters (optional but good practice)
    double getTimestamp() const;
    const double* getQuat() const;
    const double* getAccG() const;

    // stream output
    friend std::ostream& operator<<(std::ostream& os, const IMUsample& sample);

private:
    double timestamp{};
    double quat[4]{};
    double acc_g[3]{};
};