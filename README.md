# M-CSA Enzyme Benchmark

Mechanism-informed evaluation of predicted enzyme complexes, with independent
mechanism and general structural checks. This repository packages the reviewed
20-case evaluator, version **0.7.0**. It does not run cofolding models.

[中文说明](README_zh.md) | [Methods and scope](docs/methods.md) |
[Case catalog](docs/cases.md) | [Reproduction](docs/reproduction.md)

## Public benchmark snapshot

The 2026-11-10 snapshot provides the evaluator under the [MIT license](LICENSE),
20 [model-neutral task inputs](benchmark/tasks), frozen case definitions,
[task-complexity tiers](docs/task_complexity.md), and
[five-sample baseline counts](benchmark/baseline_counts.csv).
The scoring version remains **0.7.0**; no cutoff, input state, or stage verdict
was changed by this documentation and metadata release.

Task-complexity version **1.0** assigns 2 Easy, 11 Medium and 7 Hard tasks.
These are author-defined qualitative task categories, not validated prediction
difficulty or training-novelty labels. MSA/template support is reported separately.
Read the [release scope](docs/publication.md) and
[third-party notices](THIRD_PARTY_NOTICES.md) before reusing baseline data.

## What is evaluated?

Given a predicted mmCIF and a registered case, the evaluator checks the declared
assembly, chemical components, atom correspondence and selected M-CSA event.
The input contract is case specific. Arbitrary enzymes or a different occupancy
state are not silently treated as one of the registered tasks.

| Stage | Question | Examples |
| --- | --- | --- |
| Mechanism | Does the selected step meet its declared geometric criteria? | Reaction distance, attack direction, selected residue contacts and metal organization |
| General | Are the implemented chemical and physical checks satisfied? | Input stereochemistry, existing bond geometry, planarity, nonmetal short contacts and volume overlap |
| Joint | Does the same sample pass both stages? | Optional intersection, never a replacement for the two separate outcomes |

**Metal-distance lower bounds do not enter general checks.** Previously approved
case-specific mechanism metal windows remain unchanged. 1L7N has no Mg distance
lower bound. The general checks are adapted for this benchmark and must not be
reported as an unmodified PoseBusters score. Energy is not calculated.

## Install

Python 3.12 is the validated baseline. Create an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
mcsa-benchmark list-cases
mcsa-benchmark criteria --case 1L7N
```

On Windows, activate with `.venv\Scripts\activate` instead. Later Python versions
may work when all pinned dependencies provide compatible builds; they are not
claimed to be validated by the baseline replay.

## Evaluate a new prediction

```bash
mcsa-benchmark evaluate prediction.cif --case 1L7N --stage mechanism --json mechanism.json
mcsa-benchmark evaluate prediction.cif --case 1L7N --stage general --json general.json
mcsa-benchmark evaluate prediction.cif --case 1L7N --stage all --json result.json --csv result.csv
```

```python
from mcsa_benchmark import evaluate

result = evaluate("prediction.cif", "1L7N")
print(result["mechanism_status"], result["general_status"])
print(result["mechanism"].get("failure_tags", []))
```

The API returns one result. The CLI JSON wraps results in `evaluations` and
adds a summary. Both preserve measurements, atom identities and failure tags.
`PASS`, `FAIL`, `ERROR` and `NOT_RUN` are distinct outcomes. A scientific `FAIL`
does not indicate a software crash. CLI exit code 2 indicates a technical error;
ordinary geometric failure returns code 0 so complete panels can be analyzed.

2GSA requires an explicit naming profile or complete atom map:

```bash
mcsa-benchmark evaluate prediction.cif --case 2GSA --atom-profile Protenix --json result.json
```

Unknown atom names require an explicit map, not a guessed nearest-atom match.
See [input and mapping](docs/inputs.md). 1LDM additionally supports an explicit
`--site-map`; its primary task is tetramer + 4 NAD + 4 L-lactate.

## Evaluate a panel

The panel command uses a portable JSON manifest and model-native ranks:

```bash
mcsa-benchmark-panel --manifest panel.json --data-root ./data_private --output ./results/run1
```

Each case/model must have five distinct samples with native ranks 1 through 5.
Top1 is rank 1. Best of 5 asks whether any one complete sample passes. Joint Best
of 5 never combines mechanism success from one sample with general success from
another. Missing files, invalid ranks and hash mismatches are errors, not
silently omitted cases. See [reproduction](docs/reproduction.md) for schema,
regression comparison and technical-error handling.

## Tests and data availability

```bash
python tools/run_tests.py
python tools/run_tests.py --fixtures /path/to/reviewed-data-archive
```

The first command runs fixture-independent tests and explicitly reports skipped
structure-dependent tests. The second requires the private reviewed fixtures
and must finish without skips. Neither command downloads structures.

Prediction CIFs, historical Excel files, server configuration and article PDFs
are **not included**. Aggregate baseline counts are public; replaying the original
400 structures still requires the separately maintained coordinate archive.
New predictions can be generated from the published task definitions and scored
with this evaluator. The original model-native inputs, MSA/template files and
inference environments are not a bundled inference pipeline.
See [data and licensing status](docs/data_policy.md).

## Scientific scope

The registered criteria are literature-informed operational screens. Many
numerical tolerances are author-defined, not experimentally calibrated activity
thresholds. Selection and some revisions were retrospective; this work is not
described as preregistered or training-set independent. Passing one selected
event does not prove catalytic activity, a complete reaction cycle, exact
protonation, or a validated transition state. Distance-only tasks do not certify
a hydride face or hydrogen trajectory. Boundary cases retain continuous values.

The family label organizes the net reaction; the evaluator follows the selected
mechanism step, which may be a preparatory event. References, target state,
atom roles, exclusions and numerical rules are stored with each case definition.

## Repository layout

```text
src/mcsa_benchmark/       API, CLI, common checks and case evaluators
src/mcsa_benchmark/data/  Frozen case definitions, evidence and criteria hashes
tests/                   Configuration, synthetic and optional fixture tests
tools/                   Test runner and release checks
examples/                Synthetic example; not a biological validation case
docs/                    Input contracts, scope, reproduction and case catalog
```

Keep code changes separate from scientific rule changes. See
[contributing](CONTRIBUTING.md) before changing a cutoff, atom mapping or input
contract. The MIT license covers original software and documentation, not
third-party coordinates, model outputs, weights, or articles. AF3-derived
baseline information retains its applicable output terms.

