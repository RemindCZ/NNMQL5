NNMQL5 — Lightweight DLL for MLP in MQL5 (MetaTrader 5)

A minimalist C/C++ DLL with a clean C ABI providing a multi-layer perceptron (dense layers) for MetaTrader 5 (MQL5).
Supports forward pass, backpropagation with MSE loss and SGD, He/Xavier initialization, and gradient clipping.
Callable directly from MQL5 scripts, EAs, or indicators.

No magic, just straightforward math.

Disclaimer: Educational tool only. This is not financial trading product.

Contents

Features

Requirements

Build (Visual Studio)

Installation into MetaTrader

C API (exports)

Activation function codes

Example MQL5 import

Example usage in MQL5

Determinism, thread safety, performance

Limitations and roadmap

Windows icon and metadata

License

Author & contact

Features

Dense layers with selectable activations: SIGMOID, RELU, TANH, LINEAR, SYM_SIG

Inference (NN_Forward) and online training per sample (NN_TrainOne)

Mini-batch wrappers: NN_ForwardBatch, NN_TrainBatch (internally loops over samples)

Multiple networks managed via integer handles (no global singletons)

Weight initialization: He (for ReLU) / Xavier-like (others)

Gradient clipping (±5 per neuron)

Direct weight access: NN_GetWeights / NN_SetWeights (debugging, migration)

Plain C ABI, x64 (MSVC), suitable for #import in MQL5

Requirements

Windows 10/11 x64

Visual Studio 2019/2022 (MSVC), /std:c++17

MetaTrader 5 (for MQL5 integration)

Build (Visual Studio)

Recommended configuration:

Configuration: Release | x64

C++: /std:c++17 /O2 /EHsc

Runtime: /MD (shared CRT)

Disable C++ exceptions across the C ABI boundary (API returns bool/int only).

Suggested folder structure:

/src/dllmain.cpp
/src/pch.h        (optional)
/res/NNMQL5.rc    (icon, version info)
/res/resource.h
/build/           (output)

Installation into MetaTrader

Compile NNMQL5.dll.

Copy it into your terminal’s:

MQL5\Libraries\


In your EA/indicator use:

#import "NNMQL5.dll"
...
#import


In MetaTrader: Options → Expert Advisors → Allow DLL imports.

C API (exports)
int   NN_Create(void);
void  NN_Free(int h);

bool  NN_AddDense(int h,int inSz,int outSz,int act);

int   NN_InputSize(int h);
int   NN_OutputSize(int h);

bool  NN_Forward(int h,const double* in,int in_len,double* out,int out_len);
bool  NN_TrainOne(int h,const double* in,int in_len,
                  const double* tgt,int tgt_len,double lr,double* mse);

// Batch (samples stored row-major)
bool  NN_ForwardBatch(int h,const double* in,int batch,int in_len,
                      double* out,int out_len);
bool  NN_TrainBatch (int h,const double* in,int batch,int in_len,
                      const double* tgt,int tgt_len,double lr,double* mean_mse);

// Weights of layer i (0-based)
bool  NN_GetWeights(int h,int i,double* W,int Wlen,double* b,int blen);
bool  NN_SetWeights(int h,int i,const double* W,int Wlen,const double* b,int blen);


Conventions:

NN_AddDense: layer dimensions must match (out(prev) == in(next)).

NN_Forward/TrainOne: in_len == NN_InputSize(h), out_len == NN_OutputSize(h).

Batch arrays are [batch × length] row-major.

Activation function codes
Code	Activation	Formula
0	SIGMOID	1 / (1 + e^-x)
1	RELU	max(0, x) (He init)
2	TANH	tanh(x)
3	LINEAR	identity
4	SYM_SIG	2σ(x) − 1 ∈ (−1,1)
Example MQL5 import
#import "NNMQL5.dll"
int  NN_Create();  void NN_Free(int h);
bool NN_AddDense(int h,int inSz,int outSz,int act);
bool NN_Forward(int h,const double &in[],int in_len,double &out[],int out_len);
bool NN_TrainOne(int h,const double &in[],int in_len,
                 const double &tgt[],int tgt_len,double lr,double &mse);
int  NN_InputSize(int h); int NN_OutputSize(int h);
bool NN_ForwardBatch(int h,const double &in[],int batch,int in_len,
                     double &out[],int out_len);
bool NN_TrainBatch (int h,const double &in[],int batch,int in_len,
                     const double &tgt[],int tgt_len,double lr,double &mean_mse);
bool NN_GetWeights(int h,int i,double &W[],int Wlen,double &b[],int blen);
bool NN_SetWeights(int h,int i,const double &W[],int Wlen,const double &b[],int blen);
#import

Example usage in MQL5

1) Create network + inference

int h = NN_Create();
NN_AddDense(h, 32, 64, 1);     // RELU
NN_AddDense(h, 64,  1, 3);     // LINEAR

double x[32];   // normalized input
double y[1];
NN_Forward(h, x, 32, y, 1);


2) Online training (per sample)

double mse=0.0;
double x[32], t[1];
NN_TrainOne(h, x, 32, t, 1, 0.01, mse);


3) Mini-batch training

int B=64, inLen=32, outLen=1;
double xin[];  ArrayResize(xin, B*inLen);
double tgt[];  ArrayResize(tgt, B*outLen);

double mean_mse=0.0;
NN_TrainBatch(h, xin, B, inLen, tgt, outLen, 0.01, mean_mse);


4) Access weights

int layer = 0;
int inSz=32, outSz=64;
double W[], b[];
ArrayResize(W, outSz*inSz);
ArrayResize(b, outSz);

NN_GetWeights(h, layer, W, ArraySize(W), b, ArraySize(b));
// modify or save
NN_SetWeights(h, layer, W, ArraySize(W), b, ArraySize(b));

Determinism, thread safety, performance

Seed RNG: call std::srand(fixed_seed) in host before first NN_Create.

Thread safety: instance map is guarded by std::mutex. One network = one thread client (don’t call into the same handle concurrently).

Performance: train in batches, reuse buffers, normalize inputs.

Limitations & roadmap

Current limitations

Only SGD optimizer (no momentum, Adam, …)

TrainBatch = sequential calls to TrainOne (no vectorization)

No built-in model serialization (use Get/SetWeights)

Planned

Save/load model directly in DLL

Adam / RMSProp, true mini-batch gradient accumulation

Optional layers (Conv1D, pooling), topology query API

Windows icon & metadata

Add /res/NNMQL5.rc and /res/resource.h

Include an .ico file and VERSIONINFO block (product, version, description).

Windows Explorer can display DLL icon & version.

License

MIT-like spirit — use freely, but please keep attribution.
For GitHub, include a LICENSE file (MIT template) with explicit attribution note.

Author & contact

Author: Tomáš Bělák

Website: https://remind.cz

Article: https://remind.cz/neural-networks-in-mql5/
