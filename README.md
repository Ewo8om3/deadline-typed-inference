# Deadline-Typed Inference

> A concept architecture for a certifying real-time dataflow computer in which time, bandwidth, and residency are correctness dimensions—not post-deployment statistics.

[Read the web edition](https://ewo8om3.github.io/deadline-typed-inference/) · [Download the PDF](paper/deadline-typed-inference-design.pdf) · [Read the source](paper/deadline-typed-inference-design.md) · [Open a technical critique](https://github.com/Ewo8om3/deadline-typed-inference/issues/new?template=technical-critique.md)

**Status:** Concept architecture v1.0. This repository contains a falsifiable systems proposal, not measured silicon results or a production latency guarantee.

![Demand and capability certificates are checked before work enters certified, elastic, or best-effort execution lanes.](paper/assets/proof-carrying-inference.png)

## The thesis

Current inference stacks generally maximize throughput and measure the resulting latency distribution. This proposal reverses that relationship:

1. The model declares a bounded **Attention Contract**.
2. A contract-aware MLIR compiler emits an execution schedule and **demand certificate**.
3. The machine publishes a **capability certificate** with guaranteed service curves.
4. A small independent checker admits work only when aggregate demand fits guaranteed supply in every relevant resource-time window.
5. Hardware enforces the resulting lease; telemetry audits whether the physical assumptions remain true.

The compact admission invariant is:

```text
Σ Dᵢ(resource, time, window)
  ≤ S(resource, time, window) − fault_margin(resource, time, window)
```

The equation is an invariant, not the entire scheduler. The checker must also validate precedence, release times, deadlines, bursts, queues, memory live ranges, conditional branches, and fault transitions.

## Architecture at a glance

- **Weight-stationary compute plane:** model weights keep stable tile ownership; activations and compact queries move through the fabric.
- **Explicit state plane:** immediate state lives in SRAM/eDRAM, active-session state in high-bandwidth local memory, and cold capacity in a CXL-backed pool.
- **No synchronous cold access in the certified lane:** prefetch, use a trained fallback, defer through an isolated continuation, or fail within a bounded time.
- **Deadline-typed MLIR:** values carry residency, maximum bytes, readiness, lifetime, access class, and service class in addition to shape and dtype.
- **Proof-carrying deployment:** the optimizing compiler is untrusted; its certificate is checked by a deliberately small deployment verifier.
- **Deterministic multiplexing:** certified demand is composed before admission; elastic and best-effort work can use bounded deferral or revocable slack.

## What is actually new

The ingredients have serious prior art: spatial dataflow, Roofline analysis, I/O-aware attention, paged KV management, bounded and recurrent attention, CXL pooling, worst-case execution-time analysis, real-time/network calculus, deterministic NoCs, slack stealing, proof-carrying code, and MLIR.

The differentiated claim is their composition into one closed loop:

```text
bounded model semantics
        ↓
contract-aware compiler and schedule proof
        ↓
proof-carrying model image
        ↓
two-sided demand/capability admission
        ↓
hardware-enforced bounded execution
        ↓
telemetry audit and capability revocation
```

The paper includes 26 primary and technical references and separates established evidence from proposed mechanisms.

## What this does not claim

- It does not prove that a model's answer is correct or useful.
- It does not guarantee successful completion through arbitrary hardware failure.
- It does not assume CXL or commodity memory controllers already expose adequate worst-case bounds.
- It does not claim every Transformer checkpoint can be certified without retraining or architectural changes.
- It does not justify custom silicon before the software contract, checker, simulator, and bounded-state models survive validation.

## Feedback wanted

The useful response is the one that breaks the proposal cleanly. In particular:

1. Where is the proof boundary incomplete or circular?
2. Which hardware service assumptions are impossible, unmeasurable, or so conservative that capacity collapses?
3. Which model behaviors cannot tolerate the proposed bounds or trained fallbacks?
4. What is the smallest certificate language that remains expressive enough for real workloads?
5. Which existing accelerator or real-time system is the strongest counterexample or baseline?
6. What experiment would falsify the architecture fastest?

Please [open a technical critique](https://github.com/Ewo8om3/deadline-typed-inference/issues/new?template=technical-critique.md) with a concrete claim, failure case, or source. Agreement is welcome; disconfirmation is more valuable.

## Repository map

```text
paper/
  deadline-typed-inference-design.md   Source document
  deadline-typed-inference-design.pdf  58-page rendered paper
  design.css                           Print/web stylesheet
  assets/                              Architecture figures
docs/
  index.html                           Self-contained web edition
.github/ISSUE_TEMPLATE/                Structured critique templates
```

## Build

See [BUILDING.md](BUILDING.md). The checked-in PDF and web edition are generated from the Markdown source.

## License

The paper, figures, and repository documentation are licensed under [Creative Commons Attribution 4.0 International](LICENSE). Attribution does not imply endorsement.
