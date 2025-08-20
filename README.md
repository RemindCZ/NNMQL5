NNMQL5 — Lehká DLL pro MLP v MQL5 (MetaTrader 5)

Minimalistická C/C++ DLL s C ABI pro multi-layer perceptron (dense vrstvy) – dopředný průchod, backprop s MSE a SGD, He/Xavier inicializace, gradient clipping. Volatelná přímo z MQL5 (EA/indikátory). Bez magie, jen matika SŠ.

Upozornění: edukativní nástroj. Nejedná se o investiční doporučení.

Obsah

Vlastnosti

Požadavky

Build (Visual Studio)

Instalace do MetaTraderu

C API (exporty)

Aktivační funkce (kódy)

Ukázka MQL5 importu

Příklady použití v MQL5

Determinismus, thread-safety, výkon

Limity a roadmapa

Ikona a metadata ve Windows

Licence

Autor a kontakt

Vlastnosti

Dense vrstvy s volitelnou aktivací: SIGMOID, RELU, TANH, LINEAR, SYM_SIG.

Inference (NN_Forward) a online trénink po jednom vzorku (NN_TrainOne).

Mini-batch wrappery: NN_ForwardBatch, NN_TrainBatch (sekvenční loop přes TrainOne).

Správa více sítí přes integer handle (žádné globální singletony).

Inicializace vah: He (pro ReLU) / Xavier-like (ostatní).

Gradient clipping (±5 per neuron).

Přístup k vahám: NN_GetWeights / NN_SetWeights (ladění, migrace).

C ABI, x64 (MSVC), vhodné pro MQL5 #import.

Požadavky

Windows 10/11 x64

Visual Studio 2019/2022 (MSVC), /std:c++17

MetaTrader 5 (pro použití z MQL5)

Build (Visual Studio)

Doporučená konfigurace:

Configuration: Release | x64

C++: /std:c++17 /O2 /EHsc

Runtime: /MD (sdílená CRT)

Vypnout propagaci výjimek přes C hranici (API vrací bool/int).

Složky (doporučení):

/src/dllmain.cpp
/src/pch.h (volitelné)
/res/NNMQL5.rc (ikonka, verze)
/res/resource.h
/build/ (out)

Instalace do MetaTraderu

Zkompilovanou NNMQL5.dll zkopíruj do:

MQL5\Libraries\


V EA/indikátoru použij #import "NNMQL5.dll" a deklarace funkcí (viz níže).

Povol v MT5: Options → Expert Advisors → Allow DLL imports.

C API (exporty)
int   NN_Create(void);
void  NN_Free(int h);
bool  NN_AddDense(int h,int inSz,int outSz,int act);
int   NN_InputSize(int h);
int   NN_OutputSize(int h);

bool  NN_Forward(int h,const double* in,int in_len,double* out,int out_len);
bool  NN_TrainOne(int h,const double* in,int in_len,
                  const double* tgt,int tgt_len,double lr,double* mse);

// Batch (samples po řádcích, row-major)
bool  NN_ForwardBatch(int h,const double* in,int batch,int in_len,
                      double* out,int out_len);
bool  NN_TrainBatch (int h,const double* in,int batch,int in_len,
                      const double* tgt,int tgt_len,double lr,double* mean_mse);

// Váhy vrstvy i (0-based)
bool  NN_GetWeights(int h,int i,double* W,int Wlen,double* b,int blen);
bool  NN_SetWeights(int h,int i,const double* W,int Wlen,const double* b,int blen);

Konvence vstupů/výstupů

NN_AddDense: rozměry musí navazovat: out(prev) == in(next).

NN_Forward/TrainOne: in_len == NN_InputSize(h), out_len/tgt_len == NN_OutputSize(h).

Batch: in je [batch × in_len], out/tgt je [batch × out_len].

Aktivační funkce (kódy)
Kód	Aktivace	Poznámka
0	SIGMOID	1/(1+e^-x)
1	RELU	max(0,x) (He init)
2	TANH	tanh(x)
3	LINEAR	identita
4	SYM_SIG	2σ(x)-1 ∈ (-1,1)
Ukázka MQL5 importu
#import "NNMQL5.dll"
int  NN_Create();  void NN_Free(int h);
bool NN_AddDense(int h,int inSz,int outSz,int act);
bool NN_Forward(int h,const double &in[],int in_len,double &out[],int out_len);
bool NN_TrainOne(int h,const double &in[],int in_len,const double &tgt[],int tgt_len,double lr,double &mse);
int  NN_InputSize(int h); int NN_OutputSize(int h);
bool NN_ForwardBatch(int h,const double &in[],int batch,int in_len,double &out[],int out_len);
bool NN_TrainBatch (int h,const double &in[],int batch,int in_len,const double &tgt[],int tgt_len,double lr,double &mean_mse);
bool NN_GetWeights(int h,int i,double &W[],int Wlen,double &b[],int blen);
bool NN_SetWeights(int h,int i,const double &W[],int Wlen,const double &b[],int blen);
#import

Příklady použití v MQL5
1) Vytvoření sítě a inference
int h = NN_Create();
NN_AddDense(h, 32, 64, 1);     // RELU
NN_AddDense(h, 64,  1, 3);     // LINEAR

double x[32];   // připrav vstup (normalizovaný)
double y[1];
bool ok = NN_Forward(h, x, 32, y, 1);

2) Online trénink (per-sample)
double mse=0.0, lr=0.01;
double x[32], t[1];
bool ok = NN_TrainOne(h, x, 32, t, 1, lr, mse);

3) Mini-batch trénink
int B=64, inLen=32, outLen=1;
double xin[];  ArrayResize(xin, B*inLen);
double tgt[];  ArrayResize(tgt, B*outLen);

// naplň xin/tgt po řádcích...
double mean_mse=0.0;
bool ok = NN_TrainBatch(h, xin, B, inLen, tgt, outLen, 0.01, mean_mse);

4) Čtení / nastavení vah (např. uložení do CSV)
int layer = 0; int inSz=32, outSz=64;
int Wlen = outSz*inSz, blen = outSz;
double W[]; ArrayResize(W,Wlen);
double b[]; ArrayResize(b,blen);

NN_GetWeights(h, layer, W, Wlen, b, blen);
// ... uložit / modifikovat ...
NN_SetWeights(h, layer, W, Wlen, b, blen);

5) Normalizace dat (doporučeno)

Standard score: z = (x - μ)/σ počítané jen z tréninkového řezu.

Pro regresi používej LINEAR na výstupu (ne sigmoid).

Determinismus, thread-safety, výkon

Seed RNG: před prvním NN_Create() dej std::srand(fixed_seed) (v hostiteli). Při /MD může mít host a DLL sdílenou CRT (determinismus obvykle OK).

Thread-safety: tabulka instancí je chráněna std::mutex. Jedna síť = jeden vláknový klient (nevolej paralelně do stejného handle).

Výkon: trénuj v dávkách (batch wrappery), znovupoužívej buffery, normalizuj vstupy. Pro větší rychlost můžeš později přidat AVX/OpenMP (mimo scope této DLL).

Limity a roadmapa

Aktuální limity

Optimalizátor: pouze SGD (bez momentum, Adam…).

TrainBatch je sekvenční volání TrainOne (bez skutečné vektorové akcelerace).

Chybí vestavěná serializace modelu (řeš přes Get/SetWeights).

Plán

Serializace modelu (save/load) přímo v DLL.

Adam / RMSProp a mini-batch s akumulovaným gradientem.

Volitelné vrstvy (Conv1D, pooling) a info API o topologii.

Ikona a metadata ve Windows

Chceš, aby DLL měla vlastní ikonu a informace ve vlastnostech souboru:

Přidej res/NNMQL5.rc a res/resource.h, importuj BPNDLL.ico.

NNMQL5.rc může obsahovat i VERSIONINFO blok (produkt, verze, popis).

Zástupce ve Windows může používat ikonu přímo z NNMQL5.dll (Vlastnosti → Změnit ikonu…).

Licence

MIT-like spirit – používejte svobodně, prosíme o uvedení autorství.

Doporučení pro GitHub: přidejte soubor LICENSE s MIT licencí a do něj větu o požadované atribuci (attribution).
Příklad: „Permission is hereby granted… Attribution is required: please keep the author’s name in derived works.“

Autor a kontakt

Autor: Tomáš Bělák
Web: https://remind.cz

Článek: https://remind.cz/neural-networks-in-mql5/
