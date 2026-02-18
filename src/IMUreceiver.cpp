#include <iostream>
#include <stdio.h>
#include <netdb.h> 
#include <netinet/in.h> 
#include <stdlib.h> 
#include <string.h> 
#include <sys/socket.h> 
#include <sys/types.h> 
#include <unistd.h>
#include <nlohmann/json.hpp>
#include "waveletDenoiser.hpp"
#include "IMUreceiver.hpp"

namespace IMU{
    bool parse_one_quat_accg(const std::string& line, IMUsample& out) {
        using nlohmann::json;

        json j;
        try {
            j = json::parse(line);

            if (!j.contains("t") || !j.contains("quat") || !j.contains("acc_g")) {
                return false;
            }

            const auto& qj = j["quat"];
            const auto& aj = j["acc_g"];

            if (!qj.is_array() || qj.size() != 4) return false;
            if (!aj.is_array() || aj.size() != 3) return false;

            double q[4], a[3];
            for (int i = 0; i < 4; ++i) q[i] = qj[i].get<double>();
            for (int i = 0; i < 3; ++i) a[i] = aj[i].get<double>();
            out.setTimestamp(j["t"].get<double>());
            out.setQuat(q);
            out.setAccG(a);
            return true;
        }catch(...){
            return false;
        }
    }

    void process(int connfd)
    {
        std::string data(MAX, '\0');
        std::string accum;
        accum.reserve(8 * MAX);

        bool printed_debug_line = false;
        denoiser dn;

        while (true) {
            ssize_t byteCount = ::read(connfd, &data[0], data.size());
            if (byteCount == 0) {
                std::cout << "Client disconnected.\n";
                break;
            } else if (byteCount < 0) {
                std::perror("read");
                break;
            }

            accum.append(data.data(), static_cast<size_t>(byteCount));

            // Extract complete lines (newline-delimited JSON)
            while (true) {
                size_t pos = accum.find('\n');
                if (pos == std::string::npos) break;

                std::string line = accum.substr(0, pos);
                accum.erase(0, pos + 1);

                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;

                IMUsample sample;
                if (!parse_one_quat_accg(line, sample)) {
                    continue;
                }
                std::cout << sample;
                const auto a = sample.getAccG();
                dn.push(sample.getTimestamp(), a[0], a[1], a[2]);

                // Drain all available hop outputs (important on bursty reads)
                while (dn.denoise()) {
                    const auto& ox = dn.out_x();
                    const auto& oy = dn.out_y();
                    const auto& oz = dn.out_z();

                    for (int k = 0; k < denoiser::hop; ++k) {
                        std::cout << ox[k] << " " << oy[k] << " " << oz[k] << std::endl;
                    }
                }
            }
        }
    }
}