#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include "kernel.cuh"
#include "json.hpp"
using json = nlohmann::json;

int main() {
    // 1) Parse config
    std::ifstream cfgFile("configFile.json");
    if (!cfgFile) {
        std::cerr << "Failed to open configFile.json\n";
        return 1;
    }
    json cfg; cfgFile >> cfg;
    int nNeuron   = cfg["nNeuron"];
    int T         = cfg["T"];
    float dt      = cfg["dt"];
    float tau     = cfg["tau"];
    float vmax    = cfg["vmax"];
    float vth     = cfg["vth"];
    float Lambda  = cfg["Lambda"];
    float C       = cfg["C"];

    // 2) Allocate host arrays
    size_t NN = size_t(nNeuron) * nNeuron;
    std::vector<float> Q_h(NN);
    std::vector<float> vp_h(nNeuron, -0.5f);
    std::vector<float> vn_h(nNeuron, -0.5f);

    // 3) Initialize Q with Gaussian(0,0.5)
    std::mt19937 gen(123);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    for (size_t i = 0; i < NN; ++i) Q_h[i] = dist(gen);

    // 4) Allocate device memory
    float *Q_d, *vp_d, *vn_d;
    cudaMalloc(&Q_d, NN * sizeof(float));
    cudaMalloc(&vp_d, nNeuron * sizeof(float));
    cudaMalloc(&vn_d, nNeuron * sizeof(float));

    // 5) Copy to device
    cudaMemcpy(Q_d, Q_h.data(), NN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vp_d, vp_h.data(), nNeuron * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vn_d, vn_h.data(), nNeuron * sizeof(float), cudaMemcpyHostToDevice);

    // 6) Launch parameters
    dim3 block(256);
    dim3 grid((nNeuron + block.x - 1) / block.x);

    std::cout << "Starting simulation: " << nNeuron << " neurons, " << T << " steps.\n";

    // 7) Run naive simulation loop
    for (int t = 0; t < T; ++t) {
        simulate_step<<<grid, block>>>(
            Q_d, vp_d, vn_d, nNeuron, dt, tau,
            vmax, vth, Lambda, C
        );
        cudaDeviceSynchronize();
    }

    // 8) Retrieve results
    cudaMemcpy(vp_h.data(), vp_d, nNeuron * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Simulation complete. Sample vp[0] = " << vp_h[0] << "\n";

    // 9) Cleanup
    cudaFree(Q_d);
    cudaFree(vp_d);
    cudaFree(vn_d);

    return 0;
}