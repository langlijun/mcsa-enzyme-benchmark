# Evaluation contract

## From mechanism to geometry

For each registered case, the configuration identifies a selected M-CSA event,
target chemical state, required assembly and components, atom roles, geometric
checks, evidence and exclusions. The net-reaction family is not itself an
evaluator. Preparatory events, first half-reactions and later chemical states
must remain distinguishable.

The runtime reads the supplied structure, verifies the declared input and atom
correspondence, assigns required sites without optimizing the score, then runs
the requested stages independently. Equivalent oxygens may be enumerated only
within the declared chemical group. The same atom tuple must satisfy its joint
distance and angle constraints. Successful candidates cannot be assembled from
different sites or structures.

## Mechanism

Case-specific checks can include reaction distance, an attack angle, a defined
reaction face, named catalytic-residue contacts or named metal donors. Their
actual combination, bounds and occupied-site requirements are in the case JSON.
There is no universal distance cutoff across all families.

Water absence does not automatically invalidate every metric: a measurable
first-half-reaction can be evaluated without claiming a complete water network.
Likewise, heavy-atom proximity does not certify protonation or a hydrogen path.
Reactant predictions must not be judged against transition-state bond lengths.

## General

The implemented shared checks include input graph consistency when declared,
specified stereochemistry, existing bond and 1-3 geometry, internal nonbonded
distances, planarity, nonmetal intermolecular short contacts, ligand/protein
volume overlap and proximity. Distance-inferred connectivity is diagnostic,
not a binary gate. Some cases have explicit SAM, LLP or stereochemical handling.

The fixed shared numerical policy is in `general_policy.json`; selected values
include 0.75/1.25 bounds factors, 0.75 intermolecular van der Waals ratio, 7.5%
protein volume overlap and 5 A nearest-protein proximity. These are adapted
structural checks, not an unmodified external benchmark implementation.

Metal distance lower bounds are excluded from general. Existing mechanism metal
windows remain case specific; they must not be counted a second time as organic
clashes. 1L7N additionally excludes Mg from general composition handling, as
documented in its approved contract. Other composition rules are not silently
relaxed. 2GSA retains its declared forming-pair and prospective 1-3 exclusions;
its strict unexempted result is diagnostic.

## Status and failure semantics

`PASS` means all checks in the declared scope pass. `FAIL` means a criterion or
declared composition requirement was not met. `ERROR` means the requested stage
could not be evaluated. `NOT_RUN` means it was not requested. General failure
does not prevent mechanism measurement, and vice versa when the input remains
interpretable. A shared parse/mapping error can affect both stages.

Failures preserve raw tags and continuous evidence. Count only top-level stage
failures as sample outcomes; unselected candidate failures are not new sample
failures. Failure categories are nonexclusive. Aggregation and diagnostic labels
are not independent physical causes. Stereochemical inversion and failure to
recover an assignable 3D stereocenter are different descriptions.

## Evidence and limitations

Each case records which statements are supported by literature and which
numerical tolerances are author operational choices. Exact bounds generally are
not experimentally calibrated activity limits. The panel and rule revision
history are retrospective. Preserve continuous measurements, including values
near a boundary; report sensitivity analyses separately without retuning the
primary rule to maximize prediction success.

No result proves a full catalytic cycle, enzyme activity, reaction energy or
training-set independence. No aggregate score replaces the independent stages.

