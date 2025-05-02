Base version (kernel.cu): Uses a naive implementation with direct global memory access for computing Q*(vp-vn)
O1 version (KernelO1.cu): Uses shared memory to cache the voltage differences (vp-vn) to reduce global memory accesses
O2 version (KernelO2): Tiled matrix multiplication for Q*(vp-vn) computation. 
Loop unrolling to increase instruction-level parallelism
Coalesced memory access patterns
O3 version (KernelO3): 

Original Implementation (Before):

Every timestep (t) required:
Two cudaMemset calls to reset spike counters
One kernel launch with simulate_step
One cudaDeviceSynchronize
Two cudaMemcpy calls (one each for P and N spikes)
Spike history and energy calculations on CPU
Writing to S_av array every step
Optimized Implementation (After):

For most timesteps (when t+1 is not divisible by 100):

Two cudaMemset calls (still needed to reset counters)
One kernel launch
No synchronization
No memory transfers
Simply copy the last known energy value to S_av
Only every 100 steps (and at final step):

Two cudaMemset calls
One kernel launch
One cudaDeviceSynchronize
Two cudaMemcpy calls
Full spike history and energy calculations
Logging to console

*FOR RUN SCRIPTING:* for i in {1..3}; do sudo nsys profile -o simO$i-JETSON-nNeuron-x000 --trace osrt,cuda --gpuctxsw true --gpu-metrics-devices all -s process-tree --sampling-period 375000 --export json ./simO$i; done
