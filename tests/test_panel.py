import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mcsa_benchmark.panel import baseline_by_key, scientific_payload, summarize, validate_manifest


class PanelTests(unittest.TestCase):
    def rows(self, root):
        rows = []
        for i in range(5):
            path = root/f's{i}.txt'
            path.write_text(f'synthetic input {i}')
            rows.append({'sample_key':f'test_s{i}','code':'1L7N','model':'test',
                         'sample':i,'native_rank':5-i,'input_file':path.name,
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        return rows

    def test_manifest_hashes_and_native_rank(self):
        with tempfile.TemporaryDirectory() as d:
            rows = self.rows(Path(d))
            self.assertEqual(len(validate_manifest(rows, d)), 5)
            rows[0]['sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                validate_manifest(rows, d)

    def test_missing_duplicate_and_rank_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows = self.rows(Path(d))
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                validate_manifest(rows[:-1], d)
            rows[0]['native_rank'] = rows[1]['native_rank']
            with self.assertRaisesRegex(ValueError, 'Native ranks'):
                validate_manifest(rows, d)
            rows[0]['sample_key'] = '../escape'
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                validate_manifest(rows, d)

    def test_path_outside_root_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows = self.rows(Path(d))
            rows[0]['input_file'] = str(Path(d)/'s0.txt')
            with self.assertRaisesRegex(ValueError, 'out-of-root'):
                validate_manifest(rows, d)

    def test_top1_not_sample_zero_and_joint_not_mixed(self):
        members = []
        for i in range(5):
            members.append({'code':'1L7N','model':'test','sample':i,'native_rank':5-i,
                            'mechanism_status':'PASS' if i == 0 else 'FAIL',
                            'general_status':'PASS' if i == 1 else 'FAIL','joint_status':'FAIL'})
        row = summarize(members)['case_models'][0]
        self.assertEqual(row['top1_sample'], 4)
        self.assertEqual(row['top1_mechanism_pass'], 0)
        self.assertEqual(row['best5_mechanism_pass'], 1)
        self.assertEqual(row['best5_general_pass'], 1)
        self.assertEqual(row['best5_joint_pass'], 0)

    def test_error_not_hidden_as_all_geometry_failed(self):
        rows = [{'code':'1L7N','model':'test','sample':i,'native_rank':i+1,
                 'mechanism_status':'ERROR' if i == 0 else 'FAIL',
                 'general_status':'FAIL','joint_status':'ERROR' if i == 0 else 'FAIL'} for i in range(5)]
        result = summarize(rows)
        self.assertEqual(result['samples'], 5)
        self.assertEqual(result['totals']['mechanism'], {'ERROR':1,'FAIL':4})
        self.assertEqual(result['case_models'][0]['best5_mechanism_status'], 'ERROR')

    def test_inconsistent_joint_label_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Joint PASS'):
            summarize([{'code':'1L7N','model':'test','sample':0,'native_rank':1,
                        'mechanism_status':'FAIL','general_status':'PASS','joint_status':'PASS'}])

    def test_baseline_identity_required(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'baseline.json'
            p.write_text(json.dumps([{'sample_key':'one','result':{}}]))
            with self.assertRaisesRegex(ValueError, 'cohort'):
                baseline_by_key(p, ['two'])

    def test_payload_keeps_numbers_and_tags_but_not_path(self):
        a = {'structure':'old.cif','mechanism':{'value':3.0,'failure_tags':[]},'general_status':'PASS'}
        b = dict(a, structure='new.cif')
        self.assertEqual(scientific_payload(a), scientific_payload(b))
        b['mechanism'] = {'value':3.00000001,'failure_tags':[]}
        self.assertNotEqual(scientific_payload(a), scientific_payload(b))

