"""Tests requiring only installed package resources, not prediction data."""
import hashlib
from importlib.resources import files
import json
import tempfile
from pathlib import Path
import unittest

from mcsa_benchmark import evaluate, list_cases, load_case


class DistributionTests(unittest.TestCase):
    def test_twenty_configs_match_frozen_hashes(self):
        data = files('mcsa_benchmark').joinpath('data')
        lock = json.loads(data.joinpath('criteria_lock.json').read_text())
        registry = json.loads(data.joinpath('registry.json').read_text())
        self.assertEqual(set(list_cases()), set(lock['cases']))
        self.assertEqual(len(list_cases()), 20)
        for code, expected in lock['cases'].items():
            self.assertEqual(hashlib.sha256(data.joinpath(registry[code]).read_bytes()).hexdigest(), expected)
            self.assertEqual(load_case(code)['configuration_sha256'], expected)
        self.assertEqual(hashlib.sha256(data.joinpath('general_policy.json').read_bytes()).hexdigest(), lock['general_policy_sha256'])

    def test_missing_structure_is_error_not_pass(self):
        with tempfile.TemporaryDirectory() as d:
            result = evaluate(Path(d)/'missing.cif', '1L7N')
        self.assertEqual(result['input_status'], 'ERROR')
        self.assertEqual(result['mechanism_status'], 'ERROR')
        self.assertEqual(result['general_status'], 'ERROR')

    def test_packaged_data_not_external_paths(self):
        data = files('mcsa_benchmark').joinpath('data')
        for path in data.iterdir():
            if path.name.endswith('.json'):
                text = path.read_text()
                self.assertNotIn('/Users/', text)
                self.assertNotIn('/home/', text)

