#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <random>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <cuda_runtime.h>
#include "KernelO3.cuh"
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
    std::vector<float> S_av(T, 0.0f);

    // Device memory for spike counting and history
    int *spikesP_d, *spikesN_d;
    int *spikeHistory_d, *currentIdx_d, *windowFull_d;
    float *runningSum_d;
    
    cudaMalloc(&spikesP_d, sizeof(int));
    cudaMalloc(&spikesN_d, sizeof(int));
    cudaMalloc(&spikeHistory_d, WINDOW_SIZE * sizeof(int));
    cudaMalloc(&currentIdx_d, sizeof(int));
    cudaMalloc(&windowFull_d, sizeof(int));
    cudaMalloc(&runningSum_d, sizeof(float));

    // Initialize device memory
    cudaMemset(spikeHistory_d, 0, WINDOW_SIZE * sizeof(int));
    cudaMemset(currentIdx_d, 0, sizeof(int));
    cudaMemset(windowFull_d, 0, sizeof(int));
    cudaMemset(runningSum_d, 0, sizeof(float));

    std::cout << "Starting simulation: " << nNeuron << " neurons, " << T << " steps.\n";

    // Initialize Q matrix
    int rows = nNeuron, cols = nNeuron;
    if (loadFromFile) {
        load_matrix_from_txt("TestQ.txt", Q_h, rows, cols);
    } else {
        initialize_random_matrix(Q_h, rows, cols);
    }

    // Allocate and initialize device memory
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
    size_t sharedMemSize = (TILE_SIZE * TILE_SIZE + nNeuron) * sizeof(float);

    float currentEnergy = 0.0f;

    // Run simulation loop
    for (int t = 0; t < T; ++t) {
        // Launch kernel - only update history every step now for accurate energy
        simulate_step<<<grid, block, sharedMemSize>>>(
            Q_d, vp_d, vn_d, 
            spikesP_d, spikesN_d,
            spikeHistory_d, currentIdx_d, runningSum_d, windowFull_d,
            nNeuron, dt, tau, vmax, vth, Lambda, C,
            true  // Always update history for accurate energy
        );

        if ((t + 1) % 100 == 0 || t == T-1) {
            cudaDeviceSynchronize();

            // Get current energy from device
            float runningSum;
            int windowFull;
            cudaMemcpy(&runningSum, runningSum_d, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&windowFull, windowFull_d, sizeof(int), cudaMemcpyDeviceToHost);
            
            // Only compute energy if we have enough samples
            if (windowFull || t >= WINDOW_SIZE-1) {
                currentEnergy = runningSum / (1.0f * WINDOW_SIZE * nNeuron);
            } else if (t > 0) {
                // If window not full yet, normalize by actual number of steps
                currentEnergy = runningSum / (2.0f*(t + 1) * nNeuron);
            }
            
            std::cout << "Step " << (t + 1) << "/" << T << ", Energy: " << currentEnergy << "\n";
        }
        
        // Store energy value (either updated or previous)
        S_av[t] = currentEnergy;
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
    cudaFree(spikeHistory_d);
    cudaFree(currentIdx_d);
    cudaFree(runningSum_d);
    cudaFree(windowFull_d);

    return 0;
}