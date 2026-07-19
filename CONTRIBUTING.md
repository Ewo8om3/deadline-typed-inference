# Contributing

This is an architecture-research repository. The goal is to make the proposal more precise, more testable, or easier to falsify—not merely more persuasive.

## Good contributions

- identify an unsound proof step or missing resource;
- supply a concrete counterexample workload;
- replace an assumption with measured hardware evidence;
- tighten a service-curve or certificate formulation;
- show where model quality fails under bounded information flow;
- add a stronger baseline or primary source;
- propose a smaller experiment that resolves an important uncertainty;
- correct a factual, mathematical, or bibliographic error.

## Opening an issue

Use the technical-critique template when possible. Separate:

1. the claim being challenged;
2. the failure mechanism or contrary evidence;
3. the consequence for the architecture;
4. the smallest repair or experiment that would resolve it.

Please link primary sources for empirical claims. A strong intuition is useful when labeled as such.

## Pull requests

Edit the Markdown source in `paper/`. Generated HTML and PDF should be updated in the same pull request when the source changes materially. Keep proposed mechanisms clearly distinguished from measured results and established prior art.

Before submitting:

- verify all new links;
- check that figures remain legible in the PDF;
- confirm the web edition is self-contained;
- avoid claims stronger than the cited evidence;
- explain whether the change affects the Attention Contract, certificate semantics, hardware requirements, or validation plan.

## Conduct

Direct technical criticism is encouraged. Personal attacks, credential games, engagement farming, and vague dismissal are not useful. Attack the invariant, the assumption, the model, or the experiment.
