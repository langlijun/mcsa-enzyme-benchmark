"""Public task annotations do not alter frozen scientific scoring contracts."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class PublicMetadataTests(unittest.TestCase):
    def setUp(self):
        self.manifest = read(ROOT/'benchmark/manifest.json')
        self.tiers = read(ROOT/'benchmark/task_complexity_v1.json')
        self.catalog = read(ROOT/'src/mcsa_benchmark/data/catalog.json')

    def test_twenty_same_case_definitions(self):
        expected = {c['code']:c['config_sha256'] for c in self.catalog}
        actual = {r['case']:r['definition_sha256'] for r in self.tiers['entries']}
        self.assertEqual(len(actual),20)
        self.assertEqual(expected,actual)
        for row in self.tiers['entries']:
            self.assertEqual(sha(ROOT/row['definition']),row['definition_sha256'])

    def test_inputs_and_fasta_match(self):
        for row in self.tiers['entries']:
            task = read(ROOT/row['task_input'])
            definition = read(ROOT/row['definition'])
            self.assertEqual(task['frozen_definition_sha256'], row['definition_sha256'])
            self.assertEqual(task['protein_copies'], definition['protein_copies'])
            self.assertEqual(task['protein_sequence'], definition['protein_sequence'])
            fasta=(ROOT/row['protein_fasta']).read_text()
            sequence=''.join(s for s in fasta.splitlines() if not s.startswith('>'))
            self.assertEqual(sequence,definition['protein_sequence'])

    def test_tier_scope_and_counts(self):
        self.assertEqual(Counter(r['tier'] for r in self.tiers['entries']), {'Easy':2,'Medium':11,'Hard':7})
        self.assertEqual(self.tiers['status'],'AUTHOR_APPROVED_FROZEN')
        self.assertFalse(self.tiers['changes_scoring'])
        self.assertFalse(self.tiers['observed_pass_rates_used_for_assignment'])
        self.assertFalse(self.tiers['msa_template_reference_support_used_as_tier_thresholds'])

    def test_information_support_is_separate(self):
        support=read(ROOT/'benchmark/information_support20.json')
        self.assertEqual({r['case'] for r in support['entries']},{c['code'] for c in self.catalog})
        self.assertFalse(support['training_membership_confirmed'])
        self.assertTrue(support['raw_msa_rows_are_not_Neff'])

    def test_baseline_counts(self):
        with (ROOT/'benchmark/baseline_counts.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),80)
        self.assertEqual(len({(r['case'],r['model']) for r in rows}),80)
        totals=[0,0,0]
        for r in rows:
            m,g,j=[int(r[k]) for k in ['mechanism_pass','general_pass','joint_pass']]
            self.assertTrue(0 <= j <= min(m,g) <= 5)
            self.assertEqual(int(r['samples']),5)
            totals=[x+y for x,y in zip(totals,[m,g,j])]
        self.assertEqual(totals,[165,164,87])

    def test_manifest_hashes(self):
        for relative, expected in self.manifest['files'].items():
            self.assertEqual(sha(ROOT/relative),expected,relative)
        self.assertTrue(self.manifest['coordinate_files_included'])

    def test_prediction_data_are_repository_local(self):
        from mcsa_benchmark.panel import validate_manifest
        path=ROOT/self.manifest['coordinate_manifest']
        rows=read(path)
        self.assertEqual(len(validate_manifest(rows,ROOT)),400)
        self.assertEqual(len({r['code'] for r in rows}),20)
        self.assertFalse(self.manifest['external_dataset_required'])
        for row in rows:
            self.assertTrue(row['input_file'].startswith('benchmark/predictions/'))
            self.assertTrue(row['task_input_file'].startswith('benchmark/tasks/'))
            self.assertEqual(sha(ROOT/row['task_input_file']),row['task_input_sha256'])

    def test_no_scoring_module_imports_tiers(self):
        for path in (ROOT/'src/mcsa_benchmark').rglob('*.py'):
            self.assertNotIn('task_complexity_v1',path.read_text())

    def test_public_files_do_not_include_local_paths(self):
        for relative in self.manifest['files']:
            text=(ROOT/relative).read_text()
            self.assertNotIn('/Users/',text)
            self.assertNotIn('/data/node',text)
            self.assertNotIn('/home/lijun',text)

if __name__ == '__main__':
    unittest.main()
