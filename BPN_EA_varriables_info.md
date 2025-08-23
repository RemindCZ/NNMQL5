## Input Parameters — Precise Definitions for RemindBPN_EA.mq5

### Data & Topology

**Lookback** (`int`, default `32`)  
Number of past bars fed to the network as a single input vector per sample.  
- Direct effect: sets input layer size = `Lookback`.  
- Larger values capture longer context but increase model complexity and training time.

**TrainBars** (`int`, default `600`)  
Length of the training segment (number of bars used to form samples and targets).  
- Direct effect: determines how many training samples are available: `N = TrainBars`.  
- More samples can improve generalization but take longer per epoch.

**FuturePtsInp** (`int`, default `100`)  
Number of steps to draw and compute the autoregressive forecast into the future.  
- Direct effect: controls length of forward recursion when producing the future path.  
- Does not change training; affects only drawing/forecast length.

**AnchorShift** (`int`, default `0`)  
Index (shift) of the last bar considered “now”. `0` = current bar, `1` = one bar ago, etc.  
- Direct effect: moves the training/forecast window backward in time.  
- Must be `>= 0`; insufficient history will abort dataset construction.

**Hidden** (`int`, default `64`)  
Width (neurons per layer) of hidden layers.  
- Direct effect: determines size of `Dense(Lookback→Hidden)` and subsequent hidden layers.  
- Larger values increase capacity and computation time.

**Depth** (`int`, default `2`)  
Number of hidden layers. The network topology is:
`Lookback → Hidden × Depth → 1`.  
- Direct effect: increases representational depth and parameter count.

**Act** (`enum`, default `ACT_TANH`)  
Activation used for hidden layers:  
`0=SIGMOID`, `1=RELU`, `2=TANH`, `3=LINEAR`, `4=SYM_SIG (≈tanh scaled)`.  
- Direct effect: nonlinearity of hidden layers.  
- Output layer is always LINEAR (regression).

---

### Optimization & Training

**LR** (`double`, default `0.01`)  
Initial learning rate.  
- Direct effect: step size for SGD updates.  
- Clamped to `[LR_Min, LR_Max]` on init; may be adapted by scheduler.

**TargetMSE** (`double`, default `0.001`)  
Training stops when the *normalized-space* mean squared error per sample ≤ `TargetMSE`.  
- Direct effect: early stopping threshold.  
- Lower values require more epochs and may not be reachable.

**Shuffle** (`bool`, default `true`)  
Shuffle order of samples each epoch.  
- Direct effect: reduces bias from temporal ordering; improves SGD stability.

**TimerMs** (`int`, default `25`)  
Interval in milliseconds for the `OnTimer()` training tick.  
- Direct effect: how often one epoch is executed and charts are redrawn.

**ShowFuture** (`bool`, default `true`)  
Whether to draw the autoregressive future path.  
- Direct effect: toggles drawing and computation of future recursion.  
- No effect on training.

**debug** (`bool`, default `false`)  
Verbose logging to the Experts tab.  
- Direct effect: prints per-epoch diagnostics (MSE, LR, batch info).

---

### Mini-Batch Training

**UseBatch** (`bool`, default `true`)  
Use mini-batch training (`NN_TrainBatch`) instead of per-sample SGD (`NN_TrainOne`).  
- Direct effect: changes gradient estimation mode and speed.  
- Typically faster and more stable for larger datasets.

**BatchSize** (`int`, default `64`)  
Number of samples per batch when `UseBatch=true`.  
- Direct effect: memory and variance of gradient estimates.  
- Larger batch → smoother updates but more GPU/CPU and RAM.

---

### Adaptive Learning Rate (Scheduler)

**LR_Schedule** (`enum`, default `LRM_PLATEAU`)  
Learning-rate policy:  
- `LRM_CONST` — fixed LR within `[LR_Min, LR_Max]`.  
- `LRM_PLATEAU` — decreases LR when smoothed MSE (EMA) stops improving.  
- `LRM_COSINE` — cosine annealing between `LR_Min` and `LR_Max` (with warm-up).  
- `LRM_CYCLIC` — triangular cyclic LR in `[LR_Min, LR_Max]`.  
- `LRM_PLATEAU_COS` — plateau policy modulated by a cosine factor.

**LR_Min** (`double`, default `1e-5`)  
Lower bound for LR used by all schedulers.  
- Direct effect: floor for `g_lr`.

**LR_Max** (`double`, default `0.05`)  
Upper bound for LR used by all schedulers.  
- Direct effect: cap for `g_lr` and warm-up target.

**LR_WarmupEpochs** (`int`, default `10`)  
Number of warm-up epochs for `LRM_COSINE` and `LRM_PLATEAU_COS`.  
- Direct effect: linearly ramps LR from `LR_Min` to `LR_Max` over this period.

**LR_Patience** (`int`, default `20`)  
For `LRM_PLATEAU`/`LRM_PLATEAU_COS`: number of epochs with no significant EMA-MSE improvement before reducing LR.  
- Direct effect: aggressiveness of LR reductions on plateaus.

**LR_Cooldown** (`int`, default `10`)  
For plateau policies: epochs to wait after an LR reduction before considering another reduction.  
- Direct effect: prevents rapid successive LR drops.

**LR_FactorDown** (`double`, default `0.5`)  
Multiplicative factor applied to LR when a plateau is detected.  
- Direct effect: `LR ← max(LR * LR_FactorDown, LR_Min)`.

**LR_FactorUp** (`double`, default `1.05`)  
Multiplicative factor applied when EMA-MSE improves.  
- Direct effect: mild LR increase, capped at `LR_Max`.

**LR_Cycle** (`int`, default `200`)  
Cycle length (epochs) for `LRM_COSINE`, `LRM_CYCLIC`, and the cosine component of `LRM_PLATEAU_COS`.  
- Direct effect: period of LR oscillation/annealing.  
- Non-positive disables cosine/cycle behavior where applicable.

**LR_Eps** (`double`, default `1e-4`)  
Minimum *relative* EMA-MSE improvement to be considered a true improvement in plateau logic.  
- Direct effect: `improved = (EMA_MSE < best_MSE * (1 - LR_Eps))`.

**LR_EMA_Alpha** (`double`, default `0.2`)  
Smoothing factor for the MSE exponential moving average used by schedulers.  
- Direct effect: higher values react faster to changes; lower values smooth more.

---

## Notes on Data Flow

- Samples are formed as:  
  `input[t] = { mid[t-Lookback], …, mid[t-1] }`, `target[t] = mid[t]`,  
  taken from the last `TrainBars` points anchored at `AnchorShift`.  
- Normalization uses mean and std computed over the train slice only.  
- Training minimizes MSE in normalized space; plotting de-normalizes predictions.  
- Future forecasting is an autoregressive recursion of length `FuturePtsInp` starting at `AnchorShift`.
