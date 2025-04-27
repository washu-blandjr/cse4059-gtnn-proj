#include "kernel.cuh"
#include <cuda_runtime.h>

__global__ void simulate_step(
    const float* Q,
    float* vp,
    float* vn,
    int* spikesP,
    int* spikesN,
    int n,
    float dt,
    float tau,
    float vmax,
    float vth,
    float Lambda,
    float C
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Declare shared memory for caching vp and vn differences
    extern __shared__ float shared_mem[];
    float* shared_diff = shared_mem;  // Will store (vp[j] - vn[j])

    // Cache the voltage differences in shared memory
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        shared_diff[j] = vp[j] - vn[j];
    }
    __syncthreads();

    float vpi = vp[i];
    float vni = vn[i];
    bool spikedP = false;
    bool spikedN = false;

    // Basic DC input (only neuron 0)
    float netI = (i == 0) ? 0.09f : 0.0f;

    // Compute Q*(vp-vn) using shared memory
    float deltaV = 0.0f;
    for (int j = 0; j < n; ++j) {
        deltaV += Q[i*n + j] * shared_diff[j];
    }

    // Gradient terms
    float Gp = vpi - netI + deltaV;
    float Gn = vni + netI - deltaV;

    // Check for spikes and update
    if (vpi > vth) { 
        Gp += C; 
        vpi = vth;
        spikedP = true;
    }
    if (vni > vth) { 
        Gn += C; 
        vni = vth;
        spikedN = true;
    }

    // Prevent denominator from getting too close to zero
    float eps = 1e-6f;
    float denomP = -vpi * Gp + Lambda * vmax;
    float denomN = -vni * Gn + Lambda * vmax;
    
    // Add small epsilon to prevent division by zero
    denomP = (fabsf(denomP) < eps) ? (denomP < 0 ? -eps : eps) : denomP;
    denomN = (fabsf(denomN) < eps) ? (denomN < 0 ? -eps : eps) : denomN;
    
    float dtTau = dt / tau;
    float numP = vpi * vpi - vmax * vmax;
    float numN = vni * vni - vmax * vmax;
    
    float dvp = dtTau * ((numP * Gp) / denomP);
    float dvn = dtTau * ((numN * Gn) / denomN);
    
    // Update voltages with bounds checking
    vp[i] = fmaxf(fminf(vpi + dvp, vmax), -vmax);
    vn[i] = fmaxf(fminf(vni + dvn, vmax), -vmax);

    // Atomically increment spike counters if needed
    if (spikedP) atomicAdd(spikesP, 1);
    if (spikedN) atomicAdd(spikesN, 1);
}