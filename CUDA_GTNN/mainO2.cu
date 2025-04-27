#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <random>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <cuda_runtime.h>
#include "KernelO2.cuh"
#include "json.hpp"

using json = nlohmann::json;

void load_matrix_from_txt(const char* filename, std::vector<float>& matrix, int& rows, int& cols) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return;
    }

    matrix.clear();
    std::string line;
    rows = 0;
    cols = 0;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float value;
        std::vector<float> row;
        
        while (iss >> value) {
            row.push_back(value);
        }
        
        if (cols == 0) {
            cols = row.size();
        } else if (cols != row.size() && row.size() > 0) {
            std::cerr << "Error: Inconsistent number of columns in row " << rows << std::endl;
            return;
        }
        
        matrix.insert(matrix.end(), row.begin(), row.end());
        if (row.size() > 0) rows++;
    }
    
    std::cout << "Successfully loaded " << rows << "x" << cols << " matrix from " << filename << std::endl;
}

void initialize_random_matrix(std::vector<float>& matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    matrix.resize(rows * cols);
    for(size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] = 0.5f * dist(gen);
    }
    std::cout << "Initialized random " << rows << "x" << cols << " matrix\n";
}

int main() {
    // Parse config
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
    bool loadFromFile = cfg.value("loadFromFile", false);

    // Allocate host arrays
    size_t NN = size_t(nNeuron) * nNeuron;
    std::vector<float> Q_h(NN);
    std::vector<float> vp_h(nNeuron, -0.5f);
    std::vector<float> vn_h(nNeuron, -0.5f);

    // Additional device memory for spike counting
    int *spikesP_d, *spikesN_d;
    cudaMalloc(&spikesP_d, sizeof(int));
    cudaMalloc(&spikesN_d, sizeof(int));

    // Initialize sliding window for spike counting
    const int win = 900;
    std::deque<int> S_hist(win, 0);
    std::vector<float> S_av(T, 0.0f);
    float spikeEnergy = 0.0f;

    // Host variables for spike counts
    int spikesP_h, spikesN_h;

    std::cout << "Starting simulation: " << nNeuron << " neurons, " << T << " steps.\n";

    // Initialize Q matrix
    int rows = nNeuron, cols = nNeuron;
    if (loadFromFile) {
        load_matrix_from_txt("TestQ.txt", Q_h, rows, cols);
    } else {
        initialize_random_matrix(Q_h, rows, cols);
    }

    // Allocate device memory
    float *Q_d, *vp_d, *vn_d;
    cudaMalloc(&Q_d, NN * sizeof(float));
    cudaMalloc(&vp_d, nNeuron * sizeof(float));
    cudaMalloc(&vn_d, nNeuron * sizeof(float));

    // Copy to device
    cudaMemcpy(Q_d, Q_h.data(), NN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vp_d, vp_h.data(), nNeuron * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vn_d, vn_h.data(), nNeuron * sizeof(float), cudaMemcpyHostToDevice);

    // Launch parameters
    dim3 block(256);
    dim3 grid((nNeuron + block.x - 1) / block.x);
    
    // Calculate shared memory size for both Q tile and voltage differences
    size_t sharedMemSize = (TILE_SIZE * TILE_SIZE + nNeuron) * sizeof(float);

    // Run simulation loop with spike tracking
    for (int t = 0; t < T; ++t) {
        // Reset spike counters
        cudaMemset(spikesP_d, 0, sizeof(int));
        cudaMemset(spikesN_d, 0, sizeof(int));

        // Launch kernel with shared memory
        simulate_step<<<grid, block, sharedMemSize>>>(
            Q_d, vp_d, vn_d, spikesP_d, spikesN_d, 
            nNeuron, dt, tau, vmax, vth, Lambda, C
        );
        cudaDeviceSynchronize();

        // Get spike counts
        cudaMemcpy(&spikesP_h, spikesP_d, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&spikesN_h, spikesN_d, sizeof(int), cudaMemcpyDeviceToHost);

        // Update spike history and calculate energy
        int totalSpikes = spikesP_h + spikesN_h;
        S_hist.pop_front();
        S_hist.push_back(totalSpikes);
        
        float sumSpikes = 0.0f;
        for(const auto& spike : S_hist) {
            sumSpikes += spike;
        }
        spikeEnergy = sumSpikes / (2.0f * win * nNeuron);
        S_av[t] = spikeEnergy;

        if ((t + 1) % 100 == 0) {
            std::cout << "Step " << (t + 1) << "/" << T << ", Energy: " << spikeEnergy << "\n";
        }
    }

    // Calculate final mean spiking energy (last 50 timesteps)
    float finalEnergy = 0.0f;
    if (T >= 50) {
        for(int i = T-50; i < T; i++) {
            finalEnergy += S_av[i];
        }
        finalEnergy /= 50.0f;
    } else {
        finalEnergy = S_av.back();
    }

    std::cout << "Simulation complete.\n";
    std::cout << "Final mean spiking energy = " << finalEnergy << "\n";

    // Cleanup
    cudaFree(Q_d);
    cudaFree(vp_d);
    cudaFree(vn_d);
    cudaFree(spikesP_d);
    cudaFree(spikesN_d);

    return 0;
}