"""Unified registry, frozen primary cohort and stage policy contracts."""
import json
import os
from pathlib import Path
import unittest

from mcsa_benchmark import list_cases,load_case
from mcsa_benchmark.evaluate import MECHANISMS
from mcsa_benchmark.failure_labels import describe

PANEL=os.environ.get('MCSA_PANEL_MANIFEST')


class UnifiedTests(unittest.TestCase):
    @unittest.skipUnless(PANEL, 'External primary400 manifest not supplied')
    def test_twenty_explicit_primary_cases(self):
        rows=json.loads(Path(PANEL).read_text())
        self.assertEqual(len(rows),400)
        self.assertEqual(len({r['sample_key'] for r in rows}),400)
        self.assertEqual(set(list_cases()),{r['code'] for r in rows})
        for code in list_cases():
            for model in ['AF3','Boltz2','Chai1','Protenix']:
                group=[r for r in rows if r['code']==code and r['model']==model]
                self.assertEqual(sorted(r['sample'] for r in group),list(range(5)))
                self.assertEqual(sorted(r['native_rank'] for r in group),list(range(1,6)))

    def test_no_general_metal_floor_in_any_primary_definition(self):
        for code in list_cases():
            c=load_case(code)
            self.assertNotIn('general_metal_policy',c)
            self.assertFalse(c['general_policy']['metal_distance_lower_bounds_affect_pass'])
            self.assertIn(c['mechanism_type'],MECHANISMS)
        self.assertEqual(load_case('1GSA')['metal_distance_diagnostic_policy']['minimum_distance_A'],1.7)

    def test_ndp_latest_Mg_rule_not_old_axis_only(self):
        c=load_case('1NDP')
        self.assertEqual(c['mechanism_criteria']['criteria_id'],'M0150_1NDP_HIS_ATP_MG_ABG_SIXSITE_V2_20261021')
        self.assertEqual(set(c['mechanism_criteria']['magnesium']['groups']),{'alpha','beta','gamma'})

    def test_required_input_variants(self):
        self.assertEqual(load_case('1LDM')['protein_copies'],4)
        self.assertEqual(load_case('1XEL')['protein_copies'],1)
        self.assertEqual(load_case('5ICD')['variant'],'dimer_two_sets')
        self.assertEqual(load_case('1KQP')['metals']['Mg'],4)
        self.assertEqual(load_case('1GSA')['metals']['MG'],4)
        self.assertTrue(load_case('2GSA')['atom_profile_required'])

    def test_forbidden_general_metal_failure_is_not_classified_as_ordinary_clash(self):
        with self.assertRaises(ValueError):describe('F05_METAL_COORDINATION_TOO_SHORT','general')
        self.assertEqual(describe('ZN_ASP52_CONTACT_TOO_SHORT','mechanism')['category'],'METAL_DISTANCE')

    def test_unknown_failure_tag_must_be_explained(self):
        with self.assertRaises(ValueError):describe('UNKNOWN_FAILURE','mechanism')
