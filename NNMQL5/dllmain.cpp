// dllmain.cpp — DLL pro MQL5 (x64, MSVC)
#include "pch.h"
#include <windows.h>

#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <limits>
#include <cstdlib>
#include <algorithm>

#define DLL_EXPORT extern "C" __declspec(dllexport)

// ---------- Tensor (nepoužito teď, ponecháno pro budoucno) ----------
struct Tensor {
    size_t d{ 1 }, r{ 1 }, c{ 1 };
    std::vector<double> v;
    Tensor() = default;
    Tensor(size_t D, size_t R, size_t C, double init = 0.0) :d(D), r(R), c(C), v(D* R* C, init) {}
    inline size_t size() const { return v.size(); }
    inline double& at(size_t di, size_t ri, size_t ci) { return v[(di * r + ri) * c + ci]; }
    inline const double& at(size_t di, size_t ri, size_t ci) const { return v[(di * r + ri) * c + ci]; }
};

// ---------- Matrix ----------
struct Matrix {
    size_t rows{ 0 }, cols{ 0 };
    std::vector<double> a; // row-major
    Matrix() = default;
    Matrix(size_t R, size_t C) : rows(R), cols(C), a(R* C) {}
    inline double& at(size_t r, size_t c) { return a[r * cols + c]; }
    inline const double& at(size_t r, size_t c) const { return a[r * cols + c]; }
};

// ---------- Aktivace ----------
enum class ActKind : int { SIGMOID = 0, RELU = 1, TANH = 2, LINEAR = 3, SYM_SIG = 4 };

struct Activation {
    static double f(ActKind k, double x) {
        switch (k) {
        case ActKind::SIGMOID: return 1.0 / (1.0 + std::exp(-x));
        case ActKind::RELU:    return x > 0.0 ? x : 0.0;
        case ActKind::TANH:    return std::tanh(x);
        case ActKind::SYM_SIG: return 2.0 / (1.0 + std::exp(-x)) - 1.0; // −1..1
        case ActKind::LINEAR:  default: return x;
        }
    }
    // y = f(x) je už spočítané (posíláme i x kvůli ReLU)
    static double df(ActKind k, double y, double x) {
        switch (k) {
        case ActKind::SIGMOID: return y * (1.0 - y);
        case ActKind::RELU:    return x > 0.0 ? 1.0 : 0.0;
        case ActKind::TANH:    return 1.0 - y * y;
        case ActKind::SYM_SIG: return 0.5 * (1.0 - y * y); // odvozeno z 2*sigmoid(x)-1
        case ActKind::LINEAR:  default: return 1.0;
        }
    }
};

// ---------- Dense vrstva ----------
struct DenseLayer {
    size_t in_sz{ 0 }, out_sz{ 0 };
    Matrix W;
    std::vector<double> b;
    ActKind act;

    // cache
    std::vector<double> last_in, last_z, last_out;

    DenseLayer(size_t inSize, size_t outSize, ActKind k)
        : in_sz(inSize), out_sz(outSize), W(outSize, inSize), b(outSize, 0.0), act(k)
    {
        // Inicializace vah: He pro ReLU, jinak Xavier pro "měkké" aktivace
        const double scale = (k == ActKind::RELU)
            ? std::sqrt(2.0 / std::max<size_t>(1, in_sz))
            : std::sqrt(1.0 / std::max<size_t>(1, in_sz));

        for (double& w : W.a) {
            double u = (std::rand() / (double)RAND_MAX) * 2.0 - 1.0; // U[-1,1]
            w = u * scale;
        }
    }

    std::vector<double> forward(const std::vector<double>& x) {
        if (x.size() != in_sz) throw std::runtime_error("Dense forward: bad input size");
        last_in = x; last_z.assign(out_sz, 0.0); last_out.assign(out_sz, 0.0);
        for (size_t o = 0; o < out_sz; ++o) {
            double z = b[o];
            const double* wrow = &W.a[o * in_sz];
            for (size_t i = 0; i < in_sz; ++i) z += wrow[i] * x[i];
            last_z[o] = z;
            last_out[o] = Activation::f(act, z);
        }
        return last_out;
    }

    std::vector<double> backward(const std::vector<double>& dL_dy, double lr) {
        // dL/dz
        std::vector<double> dL_dz(out_sz);
        for (size_t o = 0; o < out_sz; ++o) {
            double y = last_out[o], z = last_z[o];
            dL_dz[o] = dL_dy[o] * Activation::df(act, y, z);
        }

        // gradient clipping (per-neuron)
        const double gclip = 5.0;
        for (double& g : dL_dz) {
            if (g > gclip) g = gclip;
            if (g < -gclip) g = -gclip;
        }

        // update b, W
        for (size_t o = 0; o < out_sz; ++o) {
            b[o] -= lr * dL_dz[o];
            double* wrow = &W.a[o * in_sz];
            for (size_t i = 0; i < in_sz; ++i) wrow[i] -= lr * dL_dz[o] * last_in[i];
        }

        // dL/dx = W^T * dL/dz
        std::vector<double> dL_dx(in_sz, 0.0);
        for (size_t o = 0; o < out_sz; ++o) {
            const double* wrow = &W.a[o * in_sz];
            for (size_t i = 0; i < in_sz; ++i) dL_dx[i] += wrow[i] * dL_dz[o];
        }
        return dL_dx;
    }
};

// ---------- MSE ----------
struct MSELoss {
    static double loss(const std::vector<double>& y, const std::vector<double>& t) {
        if (y.size() != t.size()) throw std::runtime_error("MSE: size mismatch");
        double s = 0.0; for (size_t i = 0; i < y.size(); ++i) { double e = y[i] - t[i]; s += e * e; }
        return s / (double)y.size();
    }
    static std::vector<double> dloss(const std::vector<double>& y, const std::vector<double>& t) {
        if (y.size() != t.size()) throw std::runtime_error("MSE: size mismatch");
        std::vector<double> g(y.size()); double n = (double)y.size();
        for (size_t i = 0; i < y.size(); ++i) g[i] = 2.0 * (y[i] - t[i]) / n; return g;
    }
};

// ---------- Síť ----------
class NeuralNetwork {
    std::vector<std::unique_ptr<DenseLayer>> layers;
    size_t input_size{ 0 }, output_size{ 0 };
public:
    bool add_dense(size_t in_sz, size_t out_sz, ActKind k) {
        if (layers.empty()) input_size = in_sz;
        else if (layers.back()->out_sz != in_sz) return false;
        layers.emplace_back(std::make_unique<DenseLayer>(in_sz, out_sz, k));
        output_size = out_sz; return true;
    }
    size_t in_size()  const { return input_size; }
    size_t out_size() const { return output_size; }

    bool forward(const double* in, int in_len, double* out, int out_len) {
        if ((int)input_size != in_len || (int)output_size != out_len || layers.empty()) return false;
        std::vector<double> x(in, in + in_len);
        for (auto& L : layers) x = L->forward(x);
        for (int i = 0; i < out_len; ++i) out[i] = x[i];
        return true;
    }
    bool train_one(const double* in, int in_len, const double* tgt, int tgt_len, double lr, double* mse = nullptr) {
        if ((int)input_size != in_len || (int)output_size != tgt_len || layers.empty()) return false;
        std::vector<double> x(in, in + in_len);
        for (auto& L : layers) x = L->forward(x);
        std::vector<double> t(tgt, tgt + tgt_len);
        if (mse) *mse = MSELoss::loss(x, t);
        std::vector<double> g = MSELoss::dloss(x, t);
        for (int li = (int)layers.size() - 1; li >= 0; --li) g = layers[li]->backward(g, lr);
        return true;
    }
};

// ---------- Správa instancí ----------
static std::unordered_map<int, std::unique_ptr<NeuralNetwork>> g_nets;
static std::mutex g_mtx; static int g_next_handle = 1;

static int alloc_handle() { std::lock_guard<std::mutex> lk(g_mtx); int h = g_next_handle++; g_nets.emplace(h, std::make_unique<NeuralNetwork>()); return h; }
static NeuralNetwork* get_net(int h) { std::lock_guard<std::mutex> lk(g_mtx); auto it = g_nets.find(h); return it == g_nets.end() ? nullptr : it->second.get(); }
static void free_handle(int h) { std::lock_guard<std::mutex> lk(g_mtx); g_nets.erase(h); }

// ---------- Exportované C API ----------
DLL_EXPORT int  NN_Create() { return alloc_handle(); }
DLL_EXPORT void NN_Free(int h) { free_handle(h); }
DLL_EXPORT bool NN_AddDense(int h, int inSz, int outSz, int act) {
    NeuralNetwork* net = get_net(h); if (!net) return false;
    ActKind k = (act == 0 ? ActKind::SIGMOID : act == 1 ? ActKind::RELU : act == 2 ? ActKind::TANH : act == 4 ? ActKind::SYM_SIG : ActKind::LINEAR);
    return net->add_dense(inSz, outSz, k);
}
DLL_EXPORT int  NN_InputSize(int h) { auto* n = get_net(h); return n ? (int)n->in_size() : 0; }
DLL_EXPORT int  NN_OutputSize(int h) { auto* n = get_net(h); return n ? (int)n->out_size() : 0; }
DLL_EXPORT bool NN_Forward(int h, const double* in, int in_len, double* out, int out_len) {
    auto* n = get_net(h); return n ? n->forward(in, in_len, out, out_len) : false;
}
DLL_EXPORT bool NN_TrainOne(int h, const double* in, int in_len, const double* tgt, int tgt_len, double lr, double* mse) {
    auto* n = get_net(h); return n ? n->train_one(in, in_len, tgt, tgt_len, lr, mse) : false;
}

// ---------- DllMain ----------
BOOL APIENTRY DllMain(HMODULE, DWORD, LPVOID) { return TRUE; }
