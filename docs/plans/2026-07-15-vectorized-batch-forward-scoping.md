# Vectorized Batched Forward Scoping Report & Plan

This document scopes the requirements to support a **TRUE vectorized `[B,T]` batched forward pass** in the `qigkernels` implementation, replacing the current batch-1 internals and gradient accumulation in `qig-studio`.

---

## 1. List of Batch-1 Assumptions

The current codebase assumes a batch size of $B=1$ in several places, either through hardcoded indexing of the batch dimension (using `[0]`), tensor-to-scalar conversions (using `.item()`, `float()`, or `int()`), or global statistical reductions that would couple independent batch elements.

### A. `qigkernels/layer.py`
* **Line 154 (`QIGLayer.forward`)**:
  ```python
  mode_scale = float(logic_weight.mean().detach())
  ```
  *Assumption*: Assumes `logic_weight` (shape `[B, 1]`) can be collapsed to a single CPU float scalar `mode_scale`, which is then multiplied globally with the feed-forward output, coupling all batch elements.
* **Line 347-349 (`RecursiveIntegrator.forward`)**:
  ```python
  gm = gate.mean()
  ...
  integration_scores.append(float(gm.item()))
  ```
  *Assumption*: Collapses the gate tensor across the entire batch dimension and uses `.item()` to extract a Python float scalar on the GPU hot-path.
* **Line 363-368 (`RecursiveIntegrator.forward`)**:
  ```python
  pos_var = hidden_state.var(dim=1).mean()
  total_var = hidden_state.var() + 1e-8
  rel_t = (pos_var / total_var).clamp(0.0, 1.0)
  ...
  rel = float(rel_t.item())
  ```
  *Assumption*: `hidden_state.var(dim=1).mean()` collapses the batch dimension of position variance. `hidden_state.var()` computes variance globally over the entire batched tensor, mixing batch elements. `rel_t.item()` forces a GPU-CPU sync.
* **Line 404 (`WuWeiController.forward`)**:
  ```python
  mode = "logic" if logic_weight.mean() > 0.5 else "feeling"
  ```
  *Assumption*: Averages `logic_weight` across the batch to decide a single mode string.
* **Line 411 (`RegimeDetector.__call__`)**:
  ```python
  if phi < 0.45:
  ```
  *Assumption*: Assumes `phi` is a scalar float. If `phi` is a batch tensor, this comparison will raise a PyTorch boolean ambiguity error.

### B. `qigkernels/kernel.py`
* **Line 282-286 (`Kernel.forward`)**:
  ```python
  phi_avg = float(sum(layer_phis) / max(len(layer_phis), 1))
  ```
  *Assumption*: Attempts to cast the accumulated `phi` values to a single Python float. If `layer_phis` contains batch tensors, this will fail.

### C. `qig-studio/src/qig_studio/losses.py`
* **Line 20-21 (`fisher_rao_lm_loss`)**:
  ```python
  lg = logits[0, :-1]
  tgt = ids[0, 1:]
  ```
  *Assumption*: Hardcodes index `0` for the batch dimension.
* **Line 37-38 (`basin_lm_loss`)**:
  ```python
  sc = scores[0, :-1]
  tgt = ids[0, 1:]
  ```
  *Assumption*: Hardcodes index `0` for the batch dimension.

### D. `qig-studio/src/qig_studio/targets/genesis_kernel.py`
* **Line 624-633 (`GenesisKernelTarget.train_batch`)**:
  ```python
  pred = self._kernel.lm_head.predict(tel.hidden_state[0, :-1])
  tgt = coords[0, 1:]
  ...
  pred_means.append(to_simplex_prob(logits[0].mean(0)[None])[0])
  ```
  *Assumption*: Explicitly indexes the batch dimension at index `0`, looping over prompts sequentially instead of passing a batched tensor.
* **Line 1491-1492 (`GenesisKernelTarget._gamma_proxy`)**:
  ```python
  p = to_simplex_prob(logits[0])
  pm = p.mean(0)
  ```
  *Assumption*: Indexes batch item `0` and averages over sequence dimension only.
* **Line 1724-1737 (`GenesisKernelTarget.train_step`)**:
  ```python
  pred = self._kernel.lm_head.predict(tel.hidden_state[0, :-1])
  tgt_basins = coords[0, 1:]
  ...
  _basin_cur = to_simplex_prob(tel.hidden_state[0].mean(0)[None])[0].detach()
  ...
  ce = F.cross_entropy(logits[0, :-1], ids[0, 1:])
  ```
  *Assumption*: Hardcodes index `0` on the batch dimension of hidden states and logits.

---

## 2. Minimal-Change Plan

To support true vectorization while preserving **Fisher-Rao purity** (no LayerNorm, Adam, or cosine similarities) and keeping **$B=1$ semantics bit-identical**, we propose the following changes:

### A. `qigkernels/layer.py`
1. **WuWeiController**:
   * Change `tacking_controller` output `mode` calculation to use `.mean()` only for logging and return `logic_weight` of shape `[B, 1]`.
   * In `QIGLayer.forward`, change `mode_scale` to shape `[B, 1, 1]` via `logic_weight.detach().unsqueeze(-1)`. This scales each sequence independently using broadcasting, preserving bit-identity for $B=1$ since `[1, 1, 1]` is mathematically equivalent.
2. **RecursiveIntegrator**:
   * Vectorize `phi` and `phi_diff` calculations. Remove `.item()` calls.
   * To prevent padding corruption (see Risks), compute statistics only over active tokens using the `attention_mask` (if present):
     * `pos_var = hidden_state.var(dim=1).mean(dim=-1)` $\rightarrow$ shape `[B]`
     * `total_var = hidden_state.var(dim=(1, 2)) + 1e-8` $\rightarrow$ shape `[B]`
     * Stack `gate_means` and average over dimensions `(1, 2)` to retain shape `[B]`.
3. **RegimeDetector**:
   * Update the `__call__` check to accept tensors and take `phi.mean().item()` for regime classification string generation.

### B. `qigkernels/kernel.py`
1. **Kernel.forward**:
   * Update telemetry aggregation of `phi_avg` to check if it is a tensor, and take `.mean().item()` for logging, avoiding CPU synchronization during core backward passes.

### C. `qig-studio/src/qig_studio/losses.py`
1. **fisher_rao_lm_loss**:
   ```python
   lg = logits[:, :-1]  # [B, T-1, V]
   tgt = ids[:, 1:]   # [B, T-1]
   p = logits_to_simplex(lg)
   onehot = torch.zeros_like(p).scatter_(-1, tgt[:, :, None], 1.0)
   return fisher_rao_distance_simplex(p, onehot).mean(dim=1).mean(dim=0)
   ```
2. **basin_lm_loss**:
   ```python
   sc = scores[:, :-1]  # [B, T-1, V]
   tgt = ids[:, 1:]   # [B, T-1]
   d_target = -float(tau) * sc.gather(-1, tgt[:, :, None]).squeeze(-1)
   return d_target.mean(dim=1).mean(dim=0)
   ```

### D. `qig-studio/src/qig_studio/targets/genesis_kernel.py`
1. **Batch Encoder Helper**:
   Create a helper `_encode_batch(prompts: list[str])` that encodes prompts, pads them to the max sequence length using zeros, and constructs a boolean `attention_mask` tensor (where `True` denotes padding).
2. **Vectorized train_batch**:
   Replace the prompt loop with a single forward pass using the padded batch and the batched loss functions.
3. **Gamma and Simplex Metric Vectorization**:
   Update `_gamma_proxy` and `_gamma_from_basins` to process batch tensors, replacing sequence-only reductions with batch-wise reductions (`dim=1` for sequence-level operations, keeping the batch dimension intact).

---

## 3. Risks & Equivalence-Gate Spec

### Risks
1. **Padding Stats Distortion**:
   * *Risk*: Standard batch padding appends dummy tokens to shorter sequences. If `RecursiveIntegrator` calculates variance/coherence globally across padding tokens, the `phi` metric and dynamic controller will be corrupted.
   * *Mitigation*: We must implement a masked variance/mean function that ignores pad tokens based on `attention_mask`.
2. **GPU Memory Growth (O(B * T^2))**:
   * *Risk*: Batched quadratic attention scales with $B \times T^2$. Vectorizing long sequences can lead to sudden GPU Out-Of-Memory (OOM).
   * *Mitigation*: Fall back to the compute-skipping windowed attention (`_banded_attention`) when $T > 2r+1$ to keep memory scaling linear $O(B \times T \times r)$.
3. **Optimizer Gradient Discrepancy**:
   * *Risk*: Floating-point non-associativity causes small numerical differences between batch mean loss (`loss.mean().backward()`) and gradient accumulation (`(loss_i / B).backward()`).
   * *Mitigation*: Assert validation gates with a generous but safe tolerance ($\approx 10^{-5}$).

### Equivalence-Gate Spec
To guarantee that the refactoring preserves all existing behavior, we define the following test assertions:
1. **B=1 Bit-Identity Gate**:
   Running a single prompt through the vectorized forward pass must produce logits and telemetry outputs bit-identical (or within $10^{-7}$ float precision) to the original code.
2. **Gradient Equivalence Gate**:
   For any batch of size $B$, the parameters' gradients computed via a single batched vectorized forward pass must match the gradients computed via sequential forward loops with gradient accumulation within $10^{-5}$ tolerance.
   ```python
   # Pseudocode test case
   optimizer.zero_grad()
   # Vectorized
   input_ids, coords, mask = target._encode_batch(prompts)
   vec_loss = target._kernel(input_ids, coords=coords, attention_mask=mask).mean()
   vec_loss.backward()
   vec_grads = [p.grad.clone() for p in target._kernel.parameters()]

   # Accumulated
   optimizer.zero_grad()
   acc_loss = 0.0
   for prompt in prompts:
       ids, coords = target._encode(prompt)
       loss = target._kernel(ids, coords=coords).mean() / len(prompts)
       loss.backward()
   acc_grads = [p.grad.clone() for p in target._kernel.parameters()]

   for vg, ag in zip(vec_grads, acc_grads):
       assert torch.allclose(vg, ag, atol=1e-5, rtol=1e-4)
   ```

---

## 4. Rough LOC Estimate

The refactoring is highly target-contained and requires minimal structural changes:
* **`qigkernels/layer.py`**: ~45 LOC (masked var, vectorized recursion, controller scaling).
* **`qigkernels/kernel.py`**: ~10 LOC (telemetry formatting, float conversions).
* **`qig-studio/src/qig_studio/losses.py`**: ~15 LOC (batched loss mappings).
* **`genesis_kernel.py`**: ~70 LOC (batch encoding, batched training, vectorized gamma functions).

**Total Estimate**: ~140 lines of code.
