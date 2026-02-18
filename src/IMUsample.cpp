#include <iostream>
#include "IMUsample.hpp"

void IMUsample::setTimestamp(double t){
    timestamp = t;
}
void IMUsample::setQuat(const double q[4]){
    for (int i = 0; i < 4; ++i) {
        quat[i] = q[i];
    }
}
void IMUsample::setAccG(const double a[3]){
    for (int i = 0; i < 3; i++){
        acc_g[i] = a[i];
    }
}
double IMUsample::getTimestamp() const {
    return timestamp;
}

const double* IMUsample::getQuat() const {
    return quat;
}

const double* IMUsample::getAccG() const {
    return acc_g;
}

std::ostream& operator<< (std::ostream& os, const IMUsample& sample){
    os << "Parsed IMUsample:\n";
    os << "  t    = " << sample.timestamp << "\n";
    os << "  quat = ["
              << sample.quat[0] << ", " << sample.quat[1] << ", " << sample.quat[2] << ", " << sample.quat[3] << "]\n";
    os << "  acc_g= ["
              << sample.acc_g[0] << ", " << sample.acc_g[1] << ", " << sample.acc_g[2] << "]\n";
    return os;
}