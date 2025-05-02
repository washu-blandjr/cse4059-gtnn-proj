#!/usr/bin/env python3
"""
GTNN_Headless_Sim.py
"""
import json
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import cupy as cp
from cupyx.profiler import benchmark


# --- Load configuration ---
def load_config(config_path="Python_GTNN/configFile.json"):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg

def load_user_data(user_data_path, DTYPE):
    if os.path.exists(user_data_path):
        mat_data = loadmat(user_data_path)
        if "userdata" in mat_data:
            userdata = mat_data["userdata"]
            dataLen, dataDim = userdata.shape
            # Cast user data once to our global DTYPE.
            return userdata.astype(DTYPE), dataLen, dataDim
        else:
            print("Warning: 'userdata' not found in", user_data_path)
    return None, 0, 0

def GTNN_Headless_Sim():
    # 1) Load JSON config
    cfg = load_config("configFile.json")
    
    # --- Configuration parameters (constants) ---
    nNeuron        = int(cfg["nNeuron"])
    T              = int(cfg["T"])
    useGPU         = bool(cfg["useGPU"])
    plotMembrane   = bool(cfg["plotMembrane"])
    nSpeed         = int(cfg["nSpeed"])
    learnFlag      = bool(cfg["learnFlag"])
    dataflag       = bool(cfg["dataflag"])
    repeatdata     = int(cfg["repeatdata"])
    Tiled          = bool(cfg["Tiled"])
    TileDivisor    = int(cfg["TileDivisor"])
    feedForward    = bool(cfg["feedForward"])
    feedBack       = bool(cfg["feedBack"])
    floatbits      = int(cfg["floatbits"])

    
    # --- Choose our array module and set the global data type ---
    # On GPU xp will be cupy; on CPU, numpy.
    if useGPU:
            xp = cp
            device = xp.cuda.Device(0)
            device.use()
            handle = device.cublas_handle
            xp.cuda.cublas.setMathMode(handle, xp.cuda.cublas.CUBLAS_TENSOR_OP_MATH)
    else:
        xp = np

    # Set the global data type once.
    if floatbits == 64:
        DTYPE = xp.float64
    else:
        DTYPE = xp.float32
        
    # --- Cast simulation constants to DTYPE only once ---
    dt             = DTYPE(cfg["dt"])
    tau            = DTYPE(cfg["tau"])
    eta            = DTYPE(cfg["eta"])
    sparsityFactor = DTYPE(cfg["sparsityFactor"])
    Lambda         = DTYPE(cfg["Lambda"])
    vmax           = DTYPE(cfg["vmax"])
    vth            = DTYPE(cfg["vth"])
    C              = DTYPE(cfg["C"])

    # Additional parameters for input stimuli
    I_input_val = DTYPE(cfg.get("I_input", 0.09))
    ac_amp_val  = DTYPE(cfg.get("ac_amp", 0))
    freq_val    = DTYPE(cfg.get("freq", 5))
    a_val       = DTYPE(cfg.get("a", 1))
    
    # --- Initialize input stimuli ---
    I_input  = xp.zeros((nNeuron, 1), dtype=DTYPE)
    I_input[0] = I_input_val
    ac_amp   = xp.zeros((nNeuron, 1), dtype=DTYPE)
    freq     = xp.full((nNeuron, 1), freq_val, dtype=DTYPE)
    a        = xp.ones((nNeuron, 1), dtype=DTYPE)
    
    # --- Load user data if needed ---
    if dataflag:
        userDataPath = cfg.get("userDataPath", "userdata.mat")
        userdata, dataLen, dataDim = load_user_data(userDataPath, DTYPE)
    else:
        userdata, dataLen, dataDim = None, 0, 0

    # --- Initialize Q, M, I ---
    Q = ((0.5) * xp.random.randn(nNeuron, nNeuron)).astype(DTYPE)
    
    if Tiled:
        tileSize = nNeuron // TileDivisor
        Q_tiled = xp.zeros((nNeuron, nNeuron), dtype=DTYPE)
        for iBlock in range(TileDivisor):
            start = iBlock * tileSize
            end = start + tileSize
            Q_tiled[start:end, start:end] = Q[start:end, start:end]
        
        if feedForward:
            for iBlock in range(TileDivisor - 1):
                row_start = iBlock * tileSize
                row_end   = row_start + tileSize
                col_start = (iBlock + 1) * tileSize
                col_end   = col_start + tileSize
                mask = (xp.random.rand(tileSize, tileSize).astype(DTYPE) < sparsityFactor).astype(DTYPE)
                Q_tiled[row_start:row_end, col_start:col_end] = Q[row_start:row_end, col_start:col_end] * mask
        
        if feedBack:
            for iBlock in range(1, TileDivisor):
                row_start = iBlock * tileSize
                row_end   = row_start + tileSize
                col_start = (iBlock - 1) * tileSize
                col_end   = col_start + tileSize
                mask = (xp.random.rand(tileSize, tileSize).astype(DTYPE) < sparsityFactor).astype(DTYPE)
                Q_tiled[row_start:row_end, col_start:col_end] = Q[row_start:row_end, col_start:col_end] * mask
        
        Q = Q_tiled

    # Create binary mask M and identity I (both in DTYPE)
    M = (Q != 0).astype(DTYPE)
    I = xp.eye(nNeuron, dtype=DTYPE)
    
    # --- Optional: Plot initial connectivity matrix ---
    if plotMembrane:
        cmap = plt.cm.jet
        if useGPU:
            Q_dense = cp.asnumpy(Q)
        else:
            Q_dense = Q
        plt.figure()
        plt.imshow(Q_dense + np.eye(nNeuron, dtype=np.float16), cmap=cmap)
        plt.colorbar()
        plt.title("Initial Connectivity Matrix")
        plt.xlabel("Pre-synaptic")
        plt.ylabel("Post-synaptic")
        plt.clim(-1, 1)
        plt.savefig("Initial_State_Plot.png")
    
    # --- Move data to GPU if desired (if not already on GPU) ---
    if useGPU:
        M = xp.asarray(M)
        I = xp.asarray(I)
        I_input = xp.asarray(I_input)
        a = xp.asarray(a)
        ac_amp = xp.asarray(ac_amp)
        freq = xp.asarray(freq)
    
    # --- Initialize state variables ---
    vp   = xp.full((nNeuron, 1), (-0.5), dtype=DTYPE)
    vn   = xp.full((nNeuron, 1), (-0.5), dtype=DTYPE)
    Psip = xp.zeros((nNeuron, 1), dtype=DTYPE)
    Psin = xp.zeros((nNeuron, 1), dtype=DTYPE)
    
    win = 900
    S_hist = xp.zeros(win)
    S_av   = xp.zeros(T)
    spikeEnergy = (0.0)
    
    currentIndex = 0
    currentCount = 0
    if dataflag and userdata is not None:
        output = xp.zeros((userdata.shape[0], 1), dtype=DTYPE)
    else:
        output = None
    
    ylog = xp.zeros((nNeuron, T),)
    
    # Pre-cast constant for sinusoid stimulus
    two_pi_over_1000 = (2 * np.pi / 1000)
    
    #print(f"Starting simulation with {nNeuron} neurons for {T} steps...")
    iter_counter = 1
    
    # --- MAIN SIMULATION LOOP ---
    for t in range(T):
        for subIter in range(nSpeed):
            if dataflag and userdata is not None:
                if currentCount > repeatdata:
                    if learnFlag:
                        currentIndex = np.random.randint(0, userdata.shape[0])
                    else:
                        currentIndex += 1
                        if currentIndex >= userdata.shape[0]:
                            currentIndex = 0
                    currentCount = 0
                currentCount += 1

                data_row = userdata[currentIndex, :]
                if data_row.size < nNeuron:
                    netI = np.concatenate([data_row, np.zeros(nNeuron - data_row.size, dtype=np.float16)])
                else:
                    netI = data_row[:nNeuron]
                netI = netI.reshape((nNeuron, 1))
                netI[-1, 0] = (0.1)
                if useGPU:
                    netI = xp.asarray(netI)
            else:
                netI = I_input + ac_amp * xp.sin(two_pi_over_1000 * freq * (iter_counter))
            
            spikedP = (vp > vth)
            spikedN = (vn > vth)
            Psip[spikedP] = C
            Psin[spikedN] = C
            vp[spikedP] = vth
            vn[spikedN] = vth
            
            ylog[:, t] = vp[:, 0]
            
            diff_v = vp - vn
            Q_diff = Q @ diff_v
            Gp = vp - netI + Q_diff + Psip
            Gn = vn + netI - Q_diff + Psin
            
            vp = a * vp + (dt / tau) * (((vp ** 2 - vmax ** 2) * Gp) / (-vp * Gp + Lambda * vmax))
            vn = a * vn + (dt / tau) * (((vn ** 2 - vmax ** 2) * Gn) / (-vn * Gn + Lambda * vmax))
            
            numSpikes = xp.sum(spikedP) + xp.sum(spikedN)
            S_hist = xp.concatenate([S_hist[1:], xp.array([numSpikes])])
            # Reorder the division to avoid overflow in the product:
            spikeEnergy = (xp.sum(S_hist) / (win)) / ((2) * (nNeuron))
            S_av[t] = spikeEnergy
            
            if learnFlag:
                outer_update = (Psip - Psin) @ (diff_v.T)
                update_term = M * outer_update
                Q = Q + (0.5) * eta * update_term
            
            Psip[:] = 0
            Psin[:] = 0
            
            if dataflag and (not learnFlag) and (output is not None):
                output[currentIndex] = spikeEnergy
            
            iter_counter += 1
        
    if useGPU:
        Q = cp.asnumpy(Q)
        ylog = cp.asnumpy(ylog)
        S_av = cp.asnumpy(S_av)
    
    print("Simulation complete.")
    print("Final mean spiking energy = {:.9f}".format(np.mean(S_av[-50:])))
    
    if plotMembrane:
        cmap = plt.cm.jet
        if useGPU:
            Q_dense = cp.asnumpy(Q)
        else:
            Q_dense = Q
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(Q_dense + np.eye(nNeuron, dtype=np.float16), cmap=cmap)
        plt.colorbar()
        plt.xlabel("Pre-synaptic")
        plt.ylabel("Post-synaptic")
        plt.title("Connectivity Matrix")
        plt.clim(-1, 1)

        plt.subplot(1, 2, 2)
        plt.plot(S_av, linewidth=2)
        plt.xlabel("Time step")
        plt.ylabel("Spiking Energy")
        plt.grid(True)
        plt.title("Spiking Energy")
        plt.savefig("Final_State_Plot.png")

if __name__ == "__main__":
    start_time = time.time()
    GTNN_Headless_Sim()
    print("Total simulation time: {:.2f} seconds".format(time.time() - start_time))
    #print(benchmark(lambda: GTNN_Headless_Sim(), n_repeat=1))
