# Author defined task complexity

Version **1.0**, approved and frozen on **2026-11-10**, for evaluator **0.7.0**.
This annotation contains **2 Easy, 11 Medium and 7 Hard** tasks. It is separate
from the frozen scientific scoring rules and never changes a PASS/FAIL result.

## Meaning of the tiers

| Tier | Qualitative task organization |
| --- | --- |
| Easy | A focused single-site task without repeated-site or pre-existing covalent-cofactor organization. Selected local metal contacts can still be required. |
| Medium | Directed approach, internal near-attack conformation, a limited number of repeated sites, or coupled contacts within one site. |
| Hard | More extensive repeated occupancy, coupled multimetal/cross-subunit roles, or placement around an existing covalent cofactor. |

Assignments are retrospective, case-specific author judgments, not a learned
score or an outcome-blinded study. Actual pass rates do not define the tiers.
The three-site/four-site boundary is a qualitative convention, not a validated
physical cutoff. Protein chain count alone is insufficient: 1XVA has four protein
chains but only one declared occupied reaction site.

MSA depth, supplied templates and chemical-state references are reported in
[information_support20.json](../benchmark/information_support20.json), not folded
into a model-independent difficulty score. Those MSA/template metrics describe
the observed AF3 preparation inputs, not all models' final retained features.
Raw MSA counts are not Neff. Old PDB dates indicate possible exposure, not actual
training membership; missing exact reactant-state reference does not prove novelty.

## Twenty frozen assignments

| Case | Tier | Occupied sites | Rationale |
| --- | --- | ---: | --- |
| [1ZIO](../src/mcsa_benchmark/data/1ZIO.json) | Easy | 1 | One occupied site; a focused phosphoryl axis and intracomplex Mg-ATP contacts, without repeated-site or pre-existing covalent-cofactor organization. |
| [1NDP](../src/mcsa_benchmark/data/1NDP.json) | Hard | 6 | Six occupied sites each couple a His-ATP axis to the same Mg contacting three ATP phosphate groups; all declared sites must pass. |
| [1OTH](../src/mcsa_benchmark/data/1OTH.json) | Medium | 3 | Three repeated two-reactant axes require colocalization and orientation, without coupled multimetal or covalent-cofactor constraints. |
| [1B6B](../src/mcsa_benchmark/data/1B6B.json) | Medium | 1 | One site must bring two separate reactants together in a directed carbonyl-attack arrangement. |
| [1XVA](../src/mcsa_benchmark/data/1XVA.json) | Medium | 1 | A single declared SAM/glycine pair must share a site and meet the N-C-S axis; four protein chains do not imply four scored occupied sites. |
| [1GSA](../src/mcsa_benchmark/data/1GSA.json) | Hard | 2 | Two occupied sites couple reaction axes to two distinct Mg roles, shared donors and the same ATP within each site. |
| [1KQP](../src/mcsa_benchmark/data/1KQP.json) | Hard | 2 | Two occupied sites require distinct Mg roles, multiple phosphate donors and equivalent-oxygen handling; ammonium is present but its later attack is not scored. |
| [1DHP](../src/mcsa_benchmark/data/1DHP.json) | Hard | 4 | Four occupied sites must each recover a directed catalytic-Lys-to-pyruvate approach; this is a four-site task rather than four independent enzymes. |
| [1IGS](../src/mcsa_benchmark/data/1IGS.json) | Medium | 1 | A single substrate must adopt the declared intramolecular near-attack distance and angle, not merely occupy the binding site. |
| [5ICD](../src/mcsa_benchmark/data/5ICD.json) | Hard | 2 | Two sites couple hydride proximity, reaction-face selection, a partner-subunit Lys and cross-subunit Mg donor organization. |
| [1LDM](../src/mcsa_benchmark/data/1LDM.json) | Hard | 4 | Four independently occupied NAD/lactate sites each require donor-acceptor proximity and the catalytic His contact. |
| [1IDT](../src/mcsa_benchmark/data/1IDT.json) | Medium | 2 | Two NADH/FMN pairs require independent site assignment and proximity; no hydride-face or hydrogen-trajectory hard gate is imposed. |
| [1XEL](../src/mcsa_benchmark/data/1XEL.json) | Easy | 1 | One occupied site and one primary heavy-atom hydride-proximity check; later sugar rotation and the complete epimerization cycle are outside the task. |
| [1I8T](../src/mcsa_benchmark/data/1I8T.json) | Medium | 2 | Two sites must meet the reduced-FAD N5-to-sugar C1 axis and leaving-bond direction; superseded auxiliary contact networks are not included. |
| [1QFE](../src/mcsa_benchmark/data/1QFE.json) | Medium | 2 | Two sites must meet directed Lys-to-substrate carbonyl approach; no pre-existing Lys-substrate bond or metal requirement is imposed. |
| [1ET0](../src/mcsa_benchmark/data/1ET0.json) | Hard | 2 | Two sites combine incoming substrate placement with an existing polymer-bound Lys-PLP cofactor; its chemical integrity remains a separate general check. |
| [1QQ5](../src/mcsa_benchmark/data/1QQ5.json) | Medium | 1 | One occupied site in a dimer must meet an Asp-O/C/Cl backside-attack axis; the unoccupied second site and later hydrolysis are not scored. |
| [1L7N](../src/mcsa_benchmark/data/1L7N.json) | Medium | 1 | One site couples a reaction axis, an acid contact and four specified donors to the same Mg; atom roles must hold in the same assignment. |
| [1IM5](../src/mcsa_benchmark/data/1IM5.json) | Medium | 1 | One site couples a Cys attack axis, a catalytic-base contact and the same Zn with its declared protein/substrate donors. |
| [2GSA](../src/mcsa_benchmark/data/2GSA.json) | Medium | 2 | Two free PMP/substrate pairs must be independently assigned and meet directed carbonyl attack; the full catalytic cycle is outside scope. |

## Use and revision

Use the phrase **author-defined task-complexity tiers** in the manuscript.
Do not say the levels were independently validated or that Easy tasks must pass.
Stage-specific pass counts remain separate from these annotations. Small groups,
especially the two Easy tasks, do not establish a general difficulty hierarchy.
Any reassignment requires a new annotation version, explicit rationale and
preservation of this file's hash. Changes to cutoffs or target states require a
separate scientific-criteria revision, even if a tier label remains the same.
