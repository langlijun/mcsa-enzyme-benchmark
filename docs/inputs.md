# Inputs and atom correspondence

Use a registered case code and an mmCIF matching its declared protein sequence,
assembly, ligand graph, copy counts and target state. Use `mcsa-benchmark criteria`
to inspect the scoring contract and the case JSON for the complete input graph.

The primary panel includes approved assembly and occupancy revisions. For
example, 2GSA is a dimer with two PMP and two GSA molecules; 1GSA and 1KQP use
two component sets and four Mg ions; 1LDM uses four NAD/lactate pairs; 1XEL uses
one protein and one pair. Explicit controls `1KQP_ONE_SET` and `5ICD_ONE_SET`
are not members of the primary 20-case denominator.

Known atom naming templates are registered. Unknown names require an explicit
atom map, not inferred correspondence chosen to improve the mechanism score:

```bash
mcsa-benchmark evaluate known.cif --case 1L7N --export-map atom_map.json
mcsa-benchmark evaluate renamed.cif --case 1L7N --atom-map atom_map.json --json result.json
```

An exported map describes the original file. Update the chain/residue/name
selectors when the new file changes them; do not assume a map is transferable
without inspection. Atom indices in configurations are zero-based input-graph
indices, not PDB residue or atom numbers.

2GSA requires an explicit registered profile (`AF3`, `Boltz2`, `Chai1`, or
`Protenix`, as defined in its JSON) or a complete map:

```bash
mcsa-benchmark evaluate prediction.cif --case 2GSA --atom-profile Protenix
```

1LDM supports `--site-map` for explicit independently justified site assignments.
Other cases reject that option rather than silently ignoring it. Never map by
the shortest reactive distance when a distinct declared site assignment is
required. The evaluated output records the actual mapping and source hash.

