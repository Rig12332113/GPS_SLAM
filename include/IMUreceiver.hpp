#pragma once

#include <string>
#include "IMUsample.hpp"

// Keep these macros consistent with receiver.cpp / IMUserver.cpp usage.
#ifndef MAX
#define MAX 1024
#endif

#ifndef IMU_PORT
#define IMU_PORT 8888
#endif

#ifndef SA
#define SA struct sockaddr
#endif

namespace IMU{
    bool parse_one_quat_accg(const std::string& line, IMUsample& out);
    void process(int connfd);
}
