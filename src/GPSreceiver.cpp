#include <iostream>
#include <stdio.h>
#include <netdb.h> 
#include <netinet/in.h> 
#include <stdlib.h> 
#include <string.h> 
#include <sys/socket.h> 
#include <sys/types.h> 
#include <unistd.h>
#include <cmath>
#include <nlohmann/json.hpp>
#include "GPSreceiver.hpp"

namespace GPS{
    int EARTHRADIUS = 6371000;
    bool parse_GPS(const std::string& line, GPSsample& out){
        using nlohmann::json;

        json j;
        try{
            j = json::parse(line);

            if (!j.contains("t") || !j.contains("lat") || !j.contains("lon") || !j.contains("alt")
                || !j.contains("hAcc") || !j.contains("vAcc") || !j.contains("speed")
                || !j.contains("course") || !j.contains("t_gps")) {
                return false;
            }
            out.setTime(j["t"].get<double>());
            out.setLatitude(j["lat"].get<double>());
            out.setLongitude(j["lon"].get<double>());
            out.setAltitude(j["alt"].get<double>());
            out.setHAcc(j["hAcc"].get<double>());
            out.setVAcc(j["vAcc"].get<double>());
            out.setSpeed(j["speed"].get<double>());
            out.setCourse(j["course"].get<double>());
            out.setTGPS(j["t_gps"].get<double>());
            return true;

        }catch(...){
            return false;
        }
    }

    void process(int connfd){

        std::string data(MAX, '\0');
        std::string accum;
        accum.reserve(8 * MAX);

        bool printed_debug_line = false;
        GPSsample origin;
        bool setOrigin = true;

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

                GPSsample sample;
                if (!parse_GPS(line, sample)) {
                    continue;
                }
                std::cout << sample;
                if (setOrigin){
                    origin = sample;
                    setOrigin = false;
                }
                else{
                    std::cout << "---------local position------------" << std::endl;
                    auto localPosition = localCoordinate(sample, origin);
                    std::cout << "x: " << localPosition.first << ",y: " <<  localPosition.second << std::endl;
                }
            }
        }
    }

    std::pair<double, double> localCoordinate(GPSsample& current, GPSsample& origin){
        // convert to radius
        double dLat = (current.getLatitude() - origin.getLatitude()) * M_PI / 180;
        double dLong = (current.getLongitude() - origin.getLongitude()) * M_PI / 180;
        double x = GPS::EARTHRADIUS * dLong * cos(origin.getLatitude() * M_PI / 180);
        double y = GPS::EARTHRADIUS * dLat;
        return std::pair<double, double>(x, y);
    }
}