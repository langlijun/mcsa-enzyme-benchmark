"""Generate navigation from packaged definitions, without copying result tables."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = root/'src/mcsa_benchmark/data'
catalog = json.loads((data/'catalog.json').read_text())
lines = ['# Reviewed case catalog', '',
         'These are 20 case-specific tasks, not 20 universal family-wide cutoffs. '
         'The linked JSON is authoritative for target state, atom roles, numerical '
         'bounds, literature support, author-defined tolerances and exclusions.', '',
         'Protein copy count does not by itself specify ligand occupancy. Read the '
         'input contract; for example, a one-pair task in a multimer is not a '
         'requirement that all potential sites are occupied.', '',
         '| Case | M-CSA | Net-reaction family | Protein copies | Definition | Evaluator |',
         '| --- | --- | --- | ---: | --- | --- |']
for case in catalog:
    filename = case['config_file']
    definition = json.loads((data/filename).read_text())
    module = definition['mechanism_type']
    from_name = module
    source = root/f'src/mcsa_benchmark/mechanisms/{from_name}.py'
    assert source.exists(), source
    lines.append(f"| {case['code']} | [{case['case_id'].split('_')[0]}]({case['mcsa_url']}) | "
                 f"`{case['family_id']}` | {case['protein_copies']} | "
                 f"[JSON](../src/mcsa_benchmark/data/{filename}) | "
                 f"[source](../src/mcsa_benchmark/mechanisms/{from_name}.py) |")
lines += ['', '## Frozen configuration', '',
          '`criteria_lock.json` records the reviewed configuration hashes. '
          'The test suite verifies each primary configuration and the shared '
          'general policy against those hashes. Changing scientific content '
          'requires an explicitly documented criterion revision, not a silent '
          'lock update.', '',
          'The legacy private primary-manifest hash is provenance, not a promise '
          'that the coordinate archive is distributed in this repository. '
          'See [data policy](data_policy.md).', '']
(root/'docs/cases.md').write_text('\n'.join(lines), encoding='utf-8')
print(f'Generated {len(catalog)} case entries.')

