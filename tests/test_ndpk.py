"""Multi-copy and independent-stage regression; perturbations are synthetic."""

import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gemmi
import numpy as np

from mcsa_benchmark import evaluate, evaluate_mechanism
from mcsa_benchmark.evaluate import load_case
from mcsa_benchmark.general import short_pairs
from mcsa_benchmark import _kernels as k

FIXTURE = os.environ.get('MCSA_1NDP_TEST_CIF')
api = importlib.import_module('mcsa_benchmark.evaluate')


class ContactKernelTests(unittest.TestCase):
    def test_tree_matches_exact_naive_contacts(self):
        rng = np.random.default_rng(0)
        first = [{'xyz': p, 'element': 'O', 'atom_name': str(i)} for i, p in enumerate(rng.normal(size=(10,3)))]
        second = [{'xyz': p, 'element': 'N', 'atom_name': str(i)} for i, p in enumerate(rng.normal(size=(30,3))*3)]
        expected = []
        for a in first:
            for b in second:
                d = float(np.linalg.norm(a['xyz']-b['xyz']))
                if d <= 4.5 and d < 0.75*(k.vdw_radius('O')+k.vdw_radius('N')):
                    expected.append((a['atom_name'], b['atom_name'], d))
        actual = [(r['first']['atom_name'],r['second']['atom_name'],r['distance_A'])
                  for r in short_pairs(first,second,'test')]
        self.assertEqual(actual, expected)


@unittest.skipUnless(FIXTURE, 'Set MCSA_1NDP_TEST_CIF to AF3 sample 0')
class NDPKTests(unittest.TestCase):
    def variant(self, change, stage='mechanism', mapping=None):
        s = gemmi.read_structure(FIXTURE)
        change(s)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'new.cif'
            s.make_mmcif_document().write_file(str(path))
            result = evaluate(path, '1NDP', stage, mapping)
            json.dumps(result, allow_nan=False)
            return result

    def test_complete_baseline(self):
        r = evaluate(FIXTURE, '1NDP')
        self.assertEqual(r['mechanism_status'], 'PASS')
        self.assertEqual(r['general_status'], 'PASS')
        self.assertEqual(r['mechanism']['passing_site_count'], 6)
        self.assertEqual(len(r['mechanism']['all_candidate_axes']), 36)
        self.assertEqual(len(r['mechanism']['metal_diagnostics']['phosphate_oxygen_distances']), 360)

    def test_mapping_round_trip(self):
        r = evaluate_mechanism(FIXTURE, '1NDP')
        mapping = {key: {k:v for k,v in row.items() if k != 'mapping_method'} for key,row in r['atom_mapping'].items()}
        other = evaluate_mechanism(FIXTURE,'1NDP',mapping)
        self.assertEqual(r['mechanism'], other['mechanism'])

    def test_partial_identical_copy_map_rejected(self):
        r = evaluate_mechanism(FIXTURE, '1NDP', {'ATP_1': {'chain':'G','atom_names':[]}})
        self.assertEqual(r['mechanism_status'], 'ERROR')
        self.assertIn('INVALID_ATOM_MAP', r['input_problem'])

    def test_labels_order_and_author_numbering_not_site_identity(self):
        def change(s):
            for chain in s[0]:
                for i, residue in enumerate(chain):
                    residue.seqid = gemmi.SeqId(i+300, ' ')
                    atoms = list(residue.clone())
                    for j in range(len(residue)-1,-1,-1):
                        del residue[j]
                    for atom in reversed(atoms):
                        residue.add_atom(atom)
                chain.name = 'X'+str(90-ord(chain.name))
        r = self.variant(change)
        self.assertEqual(r['mechanism_status'], 'PASS')
        self.assertEqual(r['mechanism']['passing_site_count'], 6)

    def test_one_failed_site_cannot_hide_behind_five_successes(self):
        def change(s):
            s[0]['A'][121].find_atom('ND1','\x00').pos.x += 30
        r = self.variant(change)
        self.assertEqual(r['mechanism_status'],'FAIL')
        self.assertEqual(r['mechanism']['passing_site_count'],5)
        self.assertIn('REACTION_DISTANCE_TOO_LONG',r['mechanism']['failure_tags'])
        assigned = next(a for a in r['mechanism']['site_assignments'] if a['ligand_chain']=='G')
        self.assertEqual(assigned['protein_chain'],'A')

    def test_direction_and_short_distance_fail_separately(self):
        for amount, expected in [(3.,'ATTACK_DIRECTION_MISMATCH'),(-1.8,'REACTION_DISTANCE_TOO_SHORT_FOR_REACTANT_TASK')]:
            def change(s):
                p = np.array(s[0]['G'][0].find_atom('P3','\x00').pos.tolist())
                o = np.array(s[0]['G'][0].find_atom('O8','\x00').pos.tolist())
                u = (o-p)/np.linalg.norm(o-p)
                s[0]['A'][121].find_atom('ND1','\x00').pos = gemmi.Position(*(p+amount*u))
            r = self.variant(change)
            self.assertEqual(r['mechanism_status'],'FAIL')
            self.assertIn(expected,r['mechanism']['failure_tags'])

    def test_duplicate_occupancy_is_not_force_matched_to_empty_site(self):
        def change(s):
            by_name = {a.name:a.pos for a in s[0]['G'][0]}
            for a in s[0]['H'][0]:
                a.pos = by_name[a.name]
        r = self.variant(change)
        self.assertEqual(r['mechanism']['axis_status'],'FAIL')
        self.assertIn('MULTIPLE_ATP_ASSIGNED_TO_ONE_SITE',r['mechanism']['failure_tags'])
        self.assertIn('CATALYTIC_SITE_UNOCCUPIED',r['mechanism']['failure_tags'])

    def test_missing_copy_not_silently_dropped(self):
        r = self.variant(lambda s:s[0].remove_chain('G'))
        self.assertEqual(r['input_status'],'FAIL')
        self.assertEqual(r['mechanism_status'],'ERROR')
        self.assertIn('LIGAND_COMPOSITION_OR_COPY_MISMATCH',r['input_problem'])

    def test_missing_metal_does_not_erase_axis(self):
        r = self.variant(lambda s:s[0].remove_chain('M'),stage='all')
        self.assertEqual(r['mechanism']['axis_status'],'PASS')
        self.assertEqual(r['mechanism']['mg_status'],'FAIL')
        self.assertEqual(r['mechanism_status'],'FAIL')
        self.assertEqual(r['general_status'],'FAIL')
        self.assertIn('METAL_COMPOSITION_OR_COPY_MISMATCH',r['general']['failure_tags'])
        self.assertEqual(r['mechanism']['metal_diagnostics']['observed_Mg'],5)

    def test_missing_nd1_general_still_runs(self):
        r = self.variant(lambda s:s[0]['A'][121].remove_atom('ND1','\x00'),stage='all')
        self.assertEqual(r['mechanism_status'],'ERROR')
        self.assertEqual(r['general_status'],'FAIL')
        self.assertTrue(r['general']['recomputed'])

    def test_mechanism_only_does_not_call_general(self):
        with patch.object(api,'run_general',side_effect=AssertionError('not requested')):
            r = evaluate_mechanism(FIXTURE,'1NDP')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['general_status'],'NOT_RUN')

    def test_no_old_results_in_criterion(self):
        c = load_case('1NDP')
        self.assertTrue(c['mechanism_criteria']['retrospective'])
        self.assertNotIn('samples',c)
        self.assertIn('Author operational',c['mechanism_criteria']['main_axis']['distance_upper_basis'])


if __name__ == '__main__':
    unittest.main()
