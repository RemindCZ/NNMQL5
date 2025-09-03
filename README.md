<p align="center">
  <img src="Logo.png" alt="NNMQL5 Logo" width="200"/>
</p>
# NNMQL5 — Lightweight Neural Network DLL for MetaTrader 5 (x64, MSVC)
[![Watch demo]([path/to/thumbnail.png)](https://github.dev/RemindCZ/NNMQL5/blob/master/delete%20BPNNN.png)]()


A lightweight, dependency-free C++ DLL that brings **dense multilayer perceptrons (MLP)** and **batch training wrappers** directly into **MQL5**.  
Ideal for time-series analysis and custom indicator prototyping — no Python runtime or external frameworks required.

> **Build:** Visual Studio 2022 • x64 • MSVC  
> **License:** MIT-like spirit — use freely, please credit the author.  
> **Author:** Tomáš Bělák — Remind  
> **Downloads & Documentation:** [remind.cz/neural-networks-in-mql5](https://remind.cz/neural-networks-in-mql5/)

---

## Features
- C ABI for painless `import` in MQL5
- Dense MLP with SGD, He/Xavier initialization, gradient clipping
- Activations: **SIGMOID**, **TANH**, **RELU**, **LINEAR**, **SYM_SIG**
- Batch training & inference: `NN_ForwardBatch`, `NN_TrainBatch`
- Weights access: `NN_GetWeights`, `NN_SetWeights`
- Internal tensor utilities (sliding windows, tensor→matrix)
- Conv1DLayer (forward only, internal prototype)

---

## Quick Start (MQL5)

### 1. Install the DLL
Copy `NNMQL5.dll` to:
```
%APPDATA%\MetaQuotes\Terminal\<your-terminal-id>\MQL5\Libraries\
```
In MT5 enable: **Tools → Options → Expert Advisors → Allow DLL imports**.

---

### 2. Import functions in your `.mq5`
```mql5
#import "NNMQL5.dll"
int  NN_Create();
void NN_Free(int h);
bool NN_AddDense(int h, int inSz, int outSz, int act);
int  NN_InputSize(int h);
int  NN_OutputSize(int h);
bool NN_Forward(int h, const double& in[], int in_len, double& out[], int out_len);
bool NN_TrainOne(int h, const double& in[], int in_len,
                 const double& tgt[], int tgt_len, double lr, double& mse);
bool NN_ForwardBatch(int h, const double& in[], int batch, int in_len,
                     double& out[], int out_len);
bool NN_TrainBatch(int h, const double& in[], int batch, int in_len,
                   const double& tgt[], int tgt_len, double lr, double& mean_mse);
bool NN_GetWeights(int h, int i, double& W[], int Wlen, double& b[], int blen);
bool NN_SetWeights(int h, int i, const double& W[], int Wlen, const double& b[], int blen);
#import
```

---

### 3. Create a network
```mql5
int h = NN_Create();
NN_AddDense(h, 240, 64, 1);   // ReLU
NN_AddDense(h, 64, 15, 3);    // Linear

double x[240], y[15]; 
ArrayInitialize(x, 0.0);

NN_Forward(h, x, 240, y, 15);
NN_Free(h);
```

---

### 4. Train one sample
```mql5
double mse;
NN_TrainOne(h, x, 240, t, 15, 0.001, mse);
```

---

### 5. Train batch
```mql5
const int BATCH=32, IN=240, OUT=15;
double X[BATCH*IN], T[BATCH*OUT], mean_mse;

NN_TrainBatch(h, X, BATCH, IN, T, OUT, 0.001, mean_mse);
```

---

## API Reference

**Lifecycle**
- `NN_Create`, `NN_Free`

**Topology**
- `NN_AddDense`  
  - Activation codes: `0=SIGMOID`, `1=RELU`, `2=TANH`, `3=LINEAR`, `4=SYM_SIG`

**Introspection**
- `NN_InputSize`, `NN_OutputSize`

**Inference**
- `NN_Forward`, `NN_ForwardBatch`

**Training**
- `NN_TrainOne`, `NN_TrainBatch`

**Weights**
- `NN_GetWeights`, `NN_SetWeights`

---

## Example: Sliding Window Forecast
```mql5
int h = NN_Create();
NN_AddDense(h, 240, 128, 1);  // ReLU
NN_AddDense(h, 128, 15, 3);   // Linear

double x[240], t[15], y[15], mse;

// Train 10 epochs
for(int e=0; e<10; e++)
   NN_TrainOne(h, x, 240, t, 15, 0.001, mse);

NN_Forward(h, x, 240, y, 15);
NN_Free(h);
```

---

## Build From Source
- **Toolchain:** Visual Studio 2022, MSVC, C++17, x64  
- **Configuration:** `Release | x64`  
- **Entry:** `dllmain.cpp` (precompiled header `pch.cpp/h`)  
- **Outputs:** `NNMQL5.dll` (+ optional `.lib`, not required for MT5)  
- **Note:** Compiler warning `D9007 /C` can be safely ignored.

---

## Data Conventions
- **Type:** `double`  
- **Shape:** flat vectors, batch = `[batch × in_len]`  
- **Loss:** MSE  
- **Optimizer:** SGD with gradient clipping ±5  
- **Init:** He (ReLU) / Xavier (others)  
- **Normalization:** scale inputs/targets (e.g., `[-1,1]`)  
- **Determinism:** initial weights random (`std::rand()`); set via `NN_SetWeights` for reproducibility  

---

## Roadmap
- Conv1D backpropagation  
- Momentum / Adam optimizers  
- FP32/FP64 switch  
- Save / Load weights  

---

## License
MIT-like spirit — free for commercial and non-commercial use.  
Please attribute:

**Author:** Tomáš Bělák — Remind
