#include "kernel.cuh"

__global__ void simulate_step(
    const float* Q,
    float* vp,
    float* vn,
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

    float vpi = vp[i];
    float vni = vn[i];

    // Basic DC input (only neuron 0)
    float netI = (i == 0) ? 0.09f : 0.0f;

    // Compute Q*(vp-vn)
    float deltaV = 0.0f;
    for (int j = 0; j < n; ++j) {
        deltaV += Q[i*n + j] * (vp[j] - vn[j]);
    }

    // Gradient terms
    float Gp = vpi - netI + deltaV;
    float Gn = vni + netI - deltaV;

    // Spiking reset
    if (vpi > vth) { Gp += C; vpi = vth; }
    if (vni > vth) { Gn += C; vni = vth; }

    // Update dynamics
    vp[i] = vpi + (dt/tau) * (((vpi*vpi - vmax*vmax)*Gp) / (-vpi*Gp + Lambda*vmax));
    vn[i] = vni + (dt/tau) * (((vni*vni - vmax*vmax)*Gn) / (-vni*Gn + Lambda*vmax));
}