# AirLLM Kimi K3: capacity is not service

- **Evidence date:** 29 July 2026
- **Added:** 24 August 2026
- **Status:** Post-publication empirical baseline
- **Relation to v1.0.0:** Supports the problem framing and strengthens the validation program; it does not validate the proposed architecture.

## External result

AirLLM v3.1.0 reported end-to-end generation from the full Kimi K3 checkpoint by streaming selected mixture-of-experts weights through a single RTX 6000 Ada. The maintainer reported:

- a 2.8-trillion-parameter sparse MoE checkpoint occupying 1.56 TB;
- 3.72 GB peak GPU memory during generation;
- 0.83 GB peak GPU memory after initialization;
- 900 seconds of one-time process initialization; and
- 292 seconds per generated token, described as disk-bound.

The release states that Kimi K3 has 896 experts per layer and routes each token to 16. AirLLM loads only the routed experts rather than materializing the expanded expert layer. The checkpoint's MXFP4 weights cross PCIe in packed form and expand on the GPU. AirLLM did not need to apply a new quantization pass, but the released checkpoint is not an uncompressed full-precision artifact.

These are maintainer-reported measurements, not independent measurements made by this project.

## What the result establishes

The result cleanly separates two questions that conventional “model fits” language often collapses:

1. **Capacity feasibility:** Can an execution engine address and execute a model whose total checkpoint is much larger than accelerator memory?
2. **Service feasibility:** Can the required weights and state be delivered with a finite upper bound that satisfies the artifact's declared token deadline?

AirLLM answers the first question affirmatively for this configuration. Its reported generation time answers the second negatively for interactive service on the measured system.

This is a concrete example of the lower-bound term described in the v1.0.0 paper:

```text
T_step >= B_weight / BW_weight
```

Reducing peak residency does not remove the bytes that must cross the storage and host-device hierarchy. It changes where those bytes wait.

## Classification under Deadline-Typed Inference

The reported Kimi K3 artifact belongs in the paper's **batch/research** service class. It has a synchronous cold-weight dependency on every decode step and therefore cannot be relabeled as interactive merely because its peak GPU-memory footprint is small.

For the paper's illustrative 20 ms interactive token deadline, the measured 292-second token time is 14,600 times the entire deadline before any tighter decomposition of queueing, compute, synchronization, interference, or guard terms. A conforming admission checker would reject that image for the interactive lane and identify weight movement as a binding resource. The same checker could still admit it to an explicitly non-real-time batch lane.

This comparison is illustrative, not a claim that the RTX 6000 Ada system was designed for the paper's target or that AirLLM promised interactive latency.

## Architectural implication

AirLLM and Deadline-Typed Inference pursue different objectives:

- AirLLM minimizes accelerator residency by moving layers or routed experts through the device as needed.
- Deadline-Typed Inference proposes stable weight or expert ownership for an admitted image, moving activations and compact routing data while making every required transfer part of a checked resource-time contract.

The AirLLM result therefore does not demonstrate the proposed weight-stationary plane. It supplies an empirical baseline for the failure mode that plane is intended to avoid: repeatedly placing cold weight movement on the token-critical path.

For sparse MoE systems, the result also sharpens the real problem. Sparse activation reduces `W_active`, but a strict system must still prove bounded router fan-out, route concentration, expert readiness, and local service for every legal route. Sparse addressing alone is not bounded service.

## What this does not establish

- It does not show that Deadline-Typed Inference can meet its illustrative deadline.
- It does not prove that stationary expert ownership is economical at Kimi K3 scale.
- It does not provide a worst-case bound; the reported 292 seconds is a measurement.
- It does not measure multi-tenant interference, route concentration, tail behavior, power, or fault transitions.
- It does not show that every model can be transformed into a certifiable interactive image.
- It does not create novelty or priority for layer-wise inference, offloading, expert streaming, or tiered-memory execution. AirLLM predates the v1.0.0 paper.

## Validation-program addition

Add AirLLM-style layer or expert streaming as a named throughput-oriented baseline in Stages 1 and 3. Begin with a smaller sparse MoE model that permits controlled repetition before attempting a checkpoint at Kimi K3 scale.

For the same model, route trace, numerical format, prompt, and generated-token count, compare:

- cold initialization time;
- time to first token and decode time per token;
- bytes read from storage, host memory, and PCIe per token;
- peak VRAM and host-memory residency;
- route concentration and expert reuse;
- the distribution and conservative upper bound of transfer service;
- energy per accepted token; and
- an emulated stationary-expert placement with the same active experts.

The experiment should fail the interactive image whenever a required cold transfer cannot be proven ready before consumer release. That rejection is a result, not a benchmark failure.

## Primary sources

- G. Li. “AirLLM v3.1.0 — Kimi K3 (2.8T) on a single card.” GitHub release, 29 July 2026. <https://github.com/lyogavin/airllm/releases/tag/v3.1.0>
- G. Li. *AirLLM*. Project repository. <https://github.com/lyogavin/airllm>
- Moonshot AI. “Kimi K3: Open Frontier Intelligence.” 2026. <https://arxiv.org/abs/2607.24653>
- Moonshot AI. *Kimi K3* model repository. <https://huggingface.co/moonshotai/Kimi-K3>

## Canonical paper

- K. Ferrell. *Deadline-Typed Inference: A Certifying Real-Time Dataflow Computer for Bounded-Latency Neural Inference*, v1.0.0, 19 July 2026. <https://doi.org/10.5281/zenodo.21442841>
