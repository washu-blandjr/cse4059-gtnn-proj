#include "KernelO3.cuh"
#include <cuda_runtime.h>

__global__ void simulate_step(
    const float* Q,
    float* vp,
    float* vn,
    int* spikesP,
    int* spikesN,
    int* spikeHistory,
    int* currentIdx,
    float* runningSum,
    int* windowFull,
    int n,
    float dt,
    float tau,
    float vmax,
    float vth,
    float Lambda,
    float C,
    bool updateHistory
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Reset spike counters at start of kernel
    if (i == 0) {
        *spikesP = 0;
        *spikesN = 0;
    }
    __syncthreads();

    // Shared memory for voltage differences
    extern __shared__ float shared_mem[];
    float* shared_diff = shared_mem;  // n elements

    // Cache voltage differences in shared memory with coalesced access
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        shared_diff[j] = vp[j] - vn[j];
    }
    __syncthreads();

    float vpi = vp[i];
    float vni = vn[i];
    bool spikedP = false;
    bool spikedN = false;

    // Compute Q*(vp-vn) directly with loop unrolling
    float deltaV = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < n; j++) {
        deltaV += Q[i * n + j] * shared_diff[j];
    }
    __syncthreads();

    // Basic DC input (only neuron 0)
    float netI = (i == 0) ? 0.09f : 0.0f;

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

    // Atomically increment spike counters
    if (spikedP) atomicAdd(spikesP, 1);
    if (spikedN) atomicAdd(spikesN, 1);

    // Update spike history and running sum (only thread 0 does this)
    if (i == 0 && updateHistory) {
        __syncthreads();  // Wait for all spike counts
        int totalSpikes = *spikesP + *spikesN;
        int idx = *currentIdx;
        
        // Subtract old value if window is full
        if (*windowFull) {
            atomicAdd(runningSum, -float(spikeHistory[idx]));
        }
        
        // Add new value to sum and history
        atomicAdd(runningSum, float(totalSpikes));
        spikeHistory[idx] = totalSpikes;
        
        // Update circular buffer index
        *currentIdx = (idx + 1) % WINDOW_SIZE;
        
        // Mark window as full once we've gone through WINDOW_SIZE steps
        if (!(*windowFull) && *currentIdx == 0) {
            *windowFull = 1;
        }
    }
}