# Third party material and baseline notices

The root MIT license applies to original evaluator software and documentation.
It does not grant a new license to third-party model weights, coordinates,
MSA/template databases, publications, or model outputs and their derivatives.
Dependencies retain their respective licenses and are not vendored here.

## Mechanism and reference sources

M-CSA entries and literature sources are cited in the frozen case definitions
in `src/mcsa_benchmark/data/`. PDB identifiers identify reference material, not
complete target-state truth. Small reference-derived assignment anchors are
part of the recorded scientific evidence; original sources retain their terms.
No article PDFs or experimental coordinate files are included in this snapshot.

## AlphaFold 3 derived baseline information

The AF3 entries in `benchmark/baseline_counts.csv` and `baseline_summary.json`
are analyses derived from AlphaFold 3 predictions. The modifications consist of
computing the benchmark checks and aggregating five samples per task. They are
not experimental measurements and are not endorsed by Google or Isomorphic Labs.
AF3-derived information is provided under and subject to the
[AlphaFold 3 Output Terms of Use](benchmark/notices/AF3_OUTPUT_TERMS_OF_USE.md),
not the repository MIT license. Other entries in these mixed-model tables do not
become AF3 outputs merely by being in the same file.

See the accompanying [legally binding notice](benchmark/notices/Legally_Binding_Terms_of_Use.txt).
The source terms are dated 2024-11-09 and were retrieved from the official
Google DeepMind repository on 2026-11-10. The upstream version applicable to the
original output remains controlling; this repository does not rewrite it.

Cite: Abramson, J. et al. Accurate structure prediction of biomolecular
interactions with AlphaFold 3. Nature 630, 493-500 (2024).
https://doi.org/10.1038/s41586-024-07487-w

## Other prediction models

Baseline rows explicitly identify Boltz2, Chai1 and Protenix separately.
This snapshot distributes numerical benchmark analyses, not their model weights
or predicted coordinate files. No new rights in those models or outputs are
granted by the evaluator code license. Model and dependency attribution must be
retained in any subsequent coordinate release.
