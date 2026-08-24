# Post-publication evidence

This index records evidence, counterexamples, and stronger baselines published after the frozen `v1.0.0` edition of *Deadline-Typed Inference*.

These notes do not retroactively modify the paper, establish priority over earlier work, or convert a concept architecture into a measured result. Each note separates:

1. what the external source measured or claimed;
2. what that evidence supports;
3. what it does not establish; and
4. the smallest useful experiment it adds to the validation program.

The canonical `v1.0.0` publication remains the version archived under DOI [10.5281/zenodo.21442841](https://doi.org/10.5281/zenodo.21442841). Its release-artifact hashes remain unchanged in `SHA256SUMS`.

## Evidence log

| Date added | External result | Relevance | Note |
|---|---|---|---|
| 2026-08-24 | AirLLM v3.1.0: Kimi K3 expert streaming | Separates model-capacity feasibility from bounded token service and supplies a cold-weight-streaming baseline | [AirLLM Kimi K3: capacity is not service](evidence/2026-08-24-airllm-kimi-k3.md) |
