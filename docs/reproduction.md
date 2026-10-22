# Reproduction and native-rank aggregation

## New predictions

Supply compatible mmCIF files and a JSON array. Each row requires:

```json
{
  "sample_key": "example_1L7N_AF3_s0",
  "code": "1L7N",
  "model": "AF3",
  "sample": 0,
  "native_rank": 1,
  "input_file": "1L7N/af3/sample_0.cif",
  "sha256": "REPLACE_WITH_ACTUAL_64_CHARACTER_SHA256"
}
```

This is a schema illustration, not an included measured result. Supply all five
rows per case/model, with distinct samples and native ranks 1 through 5. The
`input_file` is relative to `--data-root`; parent traversal outside that root,
missing files, duplicate keys, invalid ranks or hash mismatches are rejected.
2GSA rows also require `atom_profile`, using the profile matching actual atom
names. The manifest format does not infer or select atom/site maps; inputs
requiring an explicit map should first use the single-structure CLI/API.

```bash
mcsa-benchmark-panel --manifest panel.json --data-root ./data_private --output ./results/run1
```

The output directory must not already exist. Results include one JSON per
sample, `samples.json`, `full_results.json`, `case_models.json`, `summary.json`
and a minimal manifest without original machine paths or historical labels.
The full per-sample output records the actual structure path used locally;
review generated reports before publishing them.

Top1 uses the supplied native rank, not sample number or geometry. Best of 5
uses the existence of a passing complete sample. General, mechanism and joint
are aggregated separately. Joint requires both stages in that same sample.
Technical errors remain in denominators and status counts; if no sample passes
and a stage has ERROR, its Best5 status is ERROR, not a claim that every geometry
failed. This is descriptive aggregation, not independent-sample inference.

## Replay of the reviewed private archive

The separately maintained archive has `validation/primary400.json`, coordinate
fixtures and `results/full_results_400.json`. These files are not shipped here.

```bash
python tools/run_tests.py --fixtures /path/to/archive --json results_tests.json
mcsa-benchmark-panel \
  --manifest /path/to/archive/validation/primary400.json \
  --data-root /path/to/archive \
  --baseline /path/to/archive/results/full_results_400.json \
  --output ./results/replay
```

Historical result labels are used only after fresh evaluation to compare the
scientific output. They are never inputs to the evaluator. The comparator checks
the complete mechanism and general payloads, atom mapping, input status and
configuration/input hashes after JSON normalization. It excludes path and
software metadata, not scientific numbers, tags or conditions. No rounding or
numeric tolerance hides changed measurements. Exit code 2 signals a technical
evaluation error or any baseline difference.

The private test runner resolves the external archive's relative fixture map,
checks every entry exists, and treats any skipped test as failure. Some fixtures
are controls or perturbation parents; they are not primary benchmark samples.

## What CI proves

Code-only CI installs a wheel, runs tests without private fixtures and checks
configuration locks. Skipped structural tests are explicitly listed. This proves
neither a fresh 400-structure replay nor external biological calibration.
Full replay is a separate maintainer check before tagging a scientific release.

