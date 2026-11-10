# Public snapshot and reproducibility scope

## Versioned components

- Evaluator and frozen scientific criteria: 0.7.0, unchanged by this snapshot.
- Author-defined task-complexity annotations: 1.0, approved 2026-11-10.
- Public metadata and documentation snapshot: 2026-11-10.

The [manifest](../benchmark/manifest.json) records file hashes and separates
task-tier metadata from scoring configurations. A paper can cite a specific
GitHub commit for this snapshot; a persistent DOI archive is not claimed.

## What other researchers can do

Read a task's `protein.fasta` together with `task_input.json` and the linked
frozen case definition. The FASTA alone does not define ligand occupancy,
metals, covalent modifications or target state. Generate compatible predictions
with the chosen model, provide any required atom-name mapping, and run the
independent mechanism and general stages. These are model-neutral task inputs,
not automatically interchangeable native model configuration files.

The public baseline gives M/G/J passing counts out of five predictions for each
of 20 tasks and four models. J is the same-sample intersection. These are not
Top1 counts. The baseline belongs to the 20-task frozen protocol, not a direct
name-based extract of the broader 200-case evaluation.

## Validation and its boundary

Before this snapshot, all 400 fixed structures were evaluated in a clean
Python 3.12 environment and their full scientific outputs matched the frozen
baseline with zero differences. This is evaluation replay, not new cofolding
inference or independent validation of catalytic activity. Test and replay
receipts are summarized in the public manifest.

The 400 coordinates and the original model-native inference files are not
included here. Complete public reproduction of those original predictions is
therefore not claimed. No model weights, private server settings, manuscript
drafts, full publication PDFs or unrelated research data are released.

## Suggested manuscript wording

We provide 20 mechanism-informed evaluation tasks with fixed input definitions,
case-specific geometric criteria and an executable evaluator. The tasks are
annotated with author-defined qualitative complexity tiers (2 Easy, 11 Medium,
7 Hard), based on the declared site and chemical-organization requirements;
sequence/template support and chemical-state references are documented
separately. These retrospective annotations do not represent independently
validated prediction-difficulty levels. Mechanistic and general structural
checks are reported independently, together with their same-structure
intersection. Code, task inputs, tier annotations and aggregate baseline results
are available at https://github.com/langlijun/mcsa-enzyme-benchmark.

Only use that availability sentence for a snapshot actually present in the
public repository. It does not state that the coordinate archive is public.
