#include <array>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

// macOS sockets
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include "IMUreceiver.hpp"
#include "GPSreceiver.hpp"

// ----------------------
// XY ring buffer
// ----------------------
struct RingXY {
    static constexpr int N = 2000;
    std::array<double, N> x{};
    std::array<double, N> y{};
    int head = 0;
    bool full = false;

    void push(double xv, double yv) {
        x[head] = xv;
        y[head] = yv;
        head = (head + 1) % N;
        if (head == 0) full = true;
    }

    int count() const { return full ? N : head; }

    // snapshot in time order: oldest -> newest
    int snapshot(std::array<double, N>& xo, std::array<double, N>& yo) const {
        const int c = count();
        if (!full) {
            for (int i = 0; i < c; ++i) { xo[i] = x[i]; yo[i] = y[i]; }
            return c;
        }
        for (int i = 0; i < N; ++i) {
            int idx = (head + i) % N;
            xo[i] = x[idx];
            yo[i] = y[idx];
        }
        return N;
    }

    void clear() { head = 0; full = false; }
};

// ----------------------
// IMU path state
// ----------------------
struct ImuPathState {
    std::mutex m;
    bool has_state = false;

    // 2D dead-reckoning in EN plane (meters)
    double px = 0.0, py = 0.0;   // px=East, py=North
    double vx = 0.0, vy = 0.0;   // vx=East, vy=North
    double last_t = 0.0;

    RingXY path;
};

// ----------------------
// GPS UI state
// ----------------------
struct GpsUiState {
    std::mutex m;
    bool has_origin = false;
    bool has_fix = false;

    GPSsample origin;
    double x_m = 0.0;   // East
    double y_m = 0.0;   // North
    double hAcc_m = 0.0;
    double t = 0.0;

    RingXY path;
};

// ----------------------
// IMU receiver thread (NEU accel already, gravity excluded; unit=g)
// ----------------------
static void imu_receiver_thread(ImuPathState* imu, std::atomic<bool>* running) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        std::fprintf(stderr, "[imu_viewer] socket() failed\n");
        return;
    }

    int opt = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in servaddr{};
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(IMU_PORT);

    if (bind(sockfd, (sockaddr*)&servaddr, sizeof(servaddr)) != 0) {
        std::fprintf(stderr, "[imu_viewer] bind() failed on port %d (already in use?)\n", IMU_PORT);
        close(sockfd);
        return;
    }

    if (listen(sockfd, 1) != 0) {
        std::fprintf(stderr, "[imu_viewer] listen() failed\n");
        close(sockfd);
        return;
    }

    std::fprintf(stderr, "[imu_viewer] listening on TCP port %d ...\n", IMU_PORT);

    // Accept with select timeout
    int connfd = -1;
    while (running->load()) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(sockfd, &rfds);
        timeval tv{0, 200 * 1000};

        int ret = select(sockfd + 1, &rfds, nullptr, nullptr, &tv);
        if (ret < 0) break;
        if (ret == 0) continue;

        sockaddr_in cli{};
        socklen_t len = sizeof(cli);
        connfd = accept(sockfd, (sockaddr*)&cli, &len);
        if (connfd >= 0) break;
    }

    if (connfd < 0) {
        close(sockfd);
        std::fprintf(stderr, "[imu_viewer] stopped before client connected\n");
        return;
    }

    std::fprintf(stderr, "[imu_viewer] client connected\n");

    std::string accum;
    accum.reserve(4096);

    char buf_read[MAX];

    // Tuning knobs
    constexpr double G0 = 9.80665;      // m/s^2
    constexpr double MAX_DT = 0.05;     // clamp dt (50 ms)
    constexpr double ACC_CLAMP = 30.0;  // clamp accel to reduce spikes

    auto clamp = [](double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    };

    while (running->load()) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(connfd, &rfds);
        timeval tv{0, 200 * 1000};

        int ret = select(connfd + 1, &rfds, nullptr, nullptr, &tv);
        if (ret < 0) {
            std::fprintf(stderr, "[imu_viewer] select() failed during recv\n");
            break;
        }
        if (ret == 0) continue;

        ssize_t n = read(connfd, buf_read, sizeof(buf_read));
        if (n <= 0) {
            std::fprintf(stderr, "[imu_viewer] connection closed\n");
            break;
        }

        accum.append(buf_read, buf_read + n);

        // process complete lines
        size_t pos;
        while ((pos = accum.find('\n')) != std::string::npos) {
            std::string line = accum.substr(0, pos);
            accum.erase(0, pos + 1);

            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;

            IMUsample sample;
            if (!IMU::parse_one_quat_accg(line, sample)) continue;

            const double t = sample.getTimestamp();

            // accel already in NEU, gravity removed, in g
            // Expect ordering [N, E, U] in getAccG(); we only use N/E for 2D path.
            const auto a_g = sample.getAccG();

            const double aN = clamp((double)a_g[0] * G0, -ACC_CLAMP, ACC_CLAMP);
            const double aE = clamp((double)a_g[1] * G0, -ACC_CLAMP, ACC_CLAMP);

            std::lock_guard<std::mutex> lk(imu->m);

            if (!imu->has_state) {
                imu->has_state = true;
                imu->last_t = t;
                imu->px = imu->py = 0.0;
                imu->vx = imu->vy = 0.0;
                imu->path.clear();
                imu->path.push(0.0, 0.0);
                continue;
            }

            double dt = t - imu->last_t;
            imu->last_t = t;

            if (!(dt > 0.0)) continue;
            if (dt > MAX_DT) dt = MAX_DT;

            // Plot axes: X=East, Y=North
            imu->vx += aE * dt;
            imu->vy += aN * dt;

            imu->px += imu->vx * dt;
            imu->py += imu->vy * dt;

            imu->path.push(imu->px, imu->py);
        }
    }

    close(connfd);
    close(sockfd);
    std::fprintf(stderr, "[imu_viewer] receiver thread exit\n");
}

// ----------------------
// GPS receiver thread
// ----------------------
static void gps_receiver_thread(GpsUiState* gps, std::atomic<bool>* running) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        std::fprintf(stderr, "[gps_viewer] socket() failed\n");
        return;
    }

    int opt = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in servaddr{};
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(GPS_PORT);

    if (bind(sockfd, (sockaddr*)&servaddr, sizeof(servaddr)) != 0) {
        std::fprintf(stderr, "[gps_viewer] bind() failed on port %d (already in use?)\n", GPS_PORT);
        close(sockfd);
        return;
    }

    if (listen(sockfd, 1) != 0) {
        std::fprintf(stderr, "[gps_viewer] listen() failed\n");
        close(sockfd);
        return;
    }

    std::fprintf(stderr, "[gps_viewer] listening on TCP port %d ...\n", GPS_PORT);

    // Accept with timeout
    int connfd = -1;
    while (running->load()) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(sockfd, &rfds);
        timeval tv{0, 200 * 1000};

        int ret = select(sockfd + 1, &rfds, nullptr, nullptr, &tv);
        if (ret < 0) break;
        if (ret == 0) continue;

        sockaddr_in cli{};
        socklen_t len = sizeof(cli);
        connfd = accept(sockfd, (sockaddr*)&cli, &len);
        if (connfd >= 0) break;
    }

    if (connfd < 0) {
        close(sockfd);
        std::fprintf(stderr, "[gps_viewer] stopped before client connected\n");
        return;
    }

    std::fprintf(stderr, "[gps_viewer] client connected\n");

    std::string accum;
    accum.reserve(4096);
    char buf_read[MAX];

    // Gate origin: don't anchor to a bad initial fix
    constexpr double ORIGIN_HACC_MAX = 10.0; // meters
    constexpr double DROP_HACC_MAX   = 30.0; // meters

    while (running->load()) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(connfd, &rfds);
        timeval tv{0, 200 * 1000};

        int ret = select(connfd + 1, &rfds, nullptr, nullptr, &tv);
        if (ret < 0) {
            std::fprintf(stderr, "[gps_viewer] select() failed during recv\n");
            break;
        }
        if (ret == 0) continue;

        ssize_t n = read(connfd, buf_read, sizeof(buf_read));
        if (n <= 0) {
            std::fprintf(stderr, "[gps_viewer] connection closed\n");
            break;
        }

        accum.append(buf_read, buf_read + n);

        size_t pos;
        while ((pos = accum.find('\n')) != std::string::npos) {
            std::string line = accum.substr(0, pos);
            accum.erase(0, pos + 1);

            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;

            GPSsample sample;
            if (!GPS::parse_GPS(line, sample)) continue;

            const double hAcc = sample.getHAcc();
            const double t    = sample.getTimestamp();

            std::lock_guard<std::mutex> lk(gps->m);

            gps->t = t;
            gps->hAcc_m = hAcc;

            if (!gps->has_origin) {
                if (hAcc <= ORIGIN_HACC_MAX) {
                    gps->origin = sample;
                    gps->has_origin = true;
                    gps->has_fix = true;
                    gps->x_m = 0.0;
                    gps->y_m = 0.0;
                    gps->path.clear();
                    gps->path.push(0.0, 0.0);
                } else {
                    gps->has_fix = false;
                }
                continue;
            }

            if (hAcc > DROP_HACC_MAX) {
                gps->has_fix = false;
                continue;
            }

            auto xy = GPS::localCoordinate(sample, gps->origin);
            gps->x_m = xy.first;   // East
            gps->y_m = xy.second;  // North
            gps->has_fix = true;
            gps->path.push(gps->x_m, gps->y_m);
        }
    }

    close(connfd);
    close(sockfd);
    std::fprintf(stderr, "[gps_viewer] receiver thread exit\n");
}

int main() {
    // ----------------------
    // GLFW + OpenGL init
    // ----------------------
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1000, 720, "GPS + IMU Path Viewer", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // ----------------------
    // ImGui + ImPlot init
    // ----------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImPlot::CreateContext();

    // ----------------------
    // Data + receiver threads
    // ----------------------
    ImuPathState imu;
    GpsUiState gps;

    std::atomic<bool> running{true};
    std::thread imu_rx(imu_receiver_thread, &imu, &running);
    std::thread gps_rx(gps_receiver_thread, &gps, &running);

    // snapshots
    std::array<double, RingXY::N> gpsx_s{}, gpsy_s{};
    std::array<double, RingXY::N> imux_s{}, imuy_s{};
    int gps_count = 0, imu_count = 0;

    // UI toggles
    bool show_gps = true;
    bool show_imu = true;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // snapshot GPS
        bool gps_has_origin=false, gps_has_fix=false;
        double gps_x=0.0, gps_y=0.0, gps_hAcc=0.0, gps_t=0.0;
        {
            std::lock_guard<std::mutex> lk(gps.m);
            gps_has_origin = gps.has_origin;
            gps_has_fix = gps.has_fix;
            gps_x = gps.x_m;
            gps_y = gps.y_m;
            gps_hAcc = gps.hAcc_m;
            gps_t = gps.t;
            gps_count = gps.path.snapshot(gpsx_s, gpsy_s);
        }

        // snapshot IMU
        bool imu_has=false;
        double imu_x=0.0, imu_y=0.0;
        {
            std::lock_guard<std::mutex> lk(imu.m);
            imu_has = imu.has_state;
            imu_x = imu.px;
            imu_y = imu.py;
            imu_count = imu.path.snapshot(imux_s, imuy_s);
        }

        // frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ---- Control window ----
        ImGui::Begin("Status / Controls");
        ImGui::Text("IMU port: %d", IMU_PORT);
        ImGui::Text("GPS port: %d", GPS_PORT);
        ImGui::Separator();

        ImGui::Checkbox("Show GPS", &show_gps);
        ImGui::SameLine();
        ImGui::Checkbox("Show IMU", &show_imu);

        if (ImGui::Button("Clear GPS Path")) {
            std::lock_guard<std::mutex> lk(gps.m);
            gps.path.clear();
            if (gps.has_origin) gps.path.push(0.0, 0.0);
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset IMU Integrator")) {
            std::lock_guard<std::mutex> lk(imu.m);
            imu.has_state = false; // resets position/velocity on next sample
            imu.path.clear();
        }

        ImGui::Separator();
        ImGui::Text("GPS: origin=%s fix=%s hAcc=%.1fm t=%.3f",
                    gps_has_origin ? "YES" : "NO",
                    gps_has_fix ? "YES" : "NO",
                    gps_hAcc, gps_t);
        if (gps_has_fix && gps_has_origin) {
            ImGui::Text("GPS xy (E,N): (%.2f, %.2f) m", gps_x, gps_y);
            ImGui::Text("GPS points: %d", gps_count);
        } else {
            ImGui::Text("GPS points: %d", gps_count);
        }

        ImGui::Text("IMU: state=%s", imu_has ? "YES" : "NO");
        if (imu_has) {
            ImGui::Text("IMU xy (E,N): (%.2f, %.2f) m", imu_x, imu_y);
            ImGui::Text("IMU points: %d", imu_count);
        } else {
            ImGui::Text("IMU points: %d", imu_count);
        }
        ImGui::End();

        // ---- Plot window ----
        ImGui::Begin("Path (GPS + IMU)");
        if (ImPlot::BeginPlot("##pathxy", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("East / X (m)", "North / Y (m)",
                              ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

            if (show_gps && gps_has_origin && gps_count > 1) {
                ImPlot::PlotLine("GPS Path", gpsx_s.data(), gpsy_s.data(), gps_count);
                if (gps_has_fix) ImPlot::PlotScatter("GPS Now", &gps_x, &gps_y, 1);
            }

            if (show_imu && imu_has && imu_count > 1) {
                ImPlot::PlotLine("IMU Path", imux_s.data(), imuy_s.data(), imu_count);
                ImPlot::PlotScatter("IMU Now", &imu_x, &imu_y, 1);
            }

            ImPlot::EndPlot();
        }
        ImGui::End();

        // render
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // shutdown
    running.store(false);
    if (imu_rx.joinable()) imu_rx.join();
    if (gps_rx.joinable()) gps_rx.join();

    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
