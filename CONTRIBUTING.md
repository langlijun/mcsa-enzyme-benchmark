# Contributing

Separate engineering changes from scientific changes.

1. Preserve the frozen evaluator and case definitions during packaging or
   refactoring. Run fixture-independent tests and, when data are available, the
   complete structure tests and panel replay.
2. For a scientific revision, identify the old/new criterion IDs, target state,
   reaction atoms, literature evidence, author-defined tolerance and expected
   scope change. Do not tune a rule to observed model pass rates.
3. Retain raw measurements, failure tags, atom mappings and configuration hashes.
   Compare full mechanism/general results, not only aggregate counts.
4. Missing structures, ambiguous mapping and technical errors must not disappear
   from denominators. Do not silently convert them into ordinary geometric FAIL.
5. Do not commit coordinates, private results, PDFs, credentials or local paths.
   Review the staged files even when `.gitignore` is present.

Tests use the standard library `unittest`. The complete structure suite requires
an explicitly supplied reviewed data archive. CI without that archive must
report its skipped fixture tests honestly; it is not a full scientific replay.

