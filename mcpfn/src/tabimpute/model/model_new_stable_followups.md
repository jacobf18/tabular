# Stable Model Follow-Ups

Keep this list for the round after initial RMSNorm/dropout/DropPath results.

- Optimizer param groups: set zero weight decay on RMSNorm weights, embeddings, mask token, and residual-scale parameters.
- EMA or checkpoint averaging: your earlier runs peak mid/late and then drift, so average the best checkpoint window instead of trusting the final step.
- Component ablations: run RMSNorm-only, dropout-only, DropPath-only, and residual-scale-only variants so we know which change is actually helping.
- Inter-axis FFN variant: test `attention -> FFN -> attention -> FFN` against the current `attention -> attention -> FFN` block once the current stable run has results.
- Depth push: if the 8-layer stable model holds up, try 10 and 12 layers before widening the model again.
- Residual-scale sweep: extend `residual_scale_init` beyond `0.05/0.10/0.20` only if deeper variants look underpowered.
- QK normalization: only add this if deeper runs are still noisy after the current changes.
- Weight decay retune: keep `0.03` as the anchor, but retest `0.02` and `0.04` after the new architecture settles.
- RoPE fraction: keep `0.75` as the default anchor; revisit `1.0` only if the new regularized stack is clearly better.
