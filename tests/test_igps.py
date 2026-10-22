"""Software fixtures for the defined NAC task; not validated reactive structures."""

import copy
import importlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gemmi
import numpy as np
from rdkit import Chem

from mcsa_benchmark import evaluate
from mcsa_benchmark.evaluate import load_case
from mcsa_benchmark.mechanisms.igps_nac import evaluate_mechanism
from mcsa_benchmark.structure import read_case

FIXTURE = os.environ.get('MCSA_1IGS_TEST_CIF')
API = importlib.import_module('mcsa_benchmark.evaluate')


class IGPSDefinitionTests(unittest.TestCase):
    def test_atom_roles(self):
        case = load_case('1IGS')
        mol = Chem.MolFromSmiles(case['ligands']['substrate']['smiles'])
        roles = case['reaction_atoms']
        c1, c6, c2, ketone, oxygen = [roles[k] for k in
            ('C1_index', 'C6_index', 'N_substituted_aromatic_C2_index', 'C2prime_index', 'O2prime_index')]
        self.assertTrue(mol.GetAtomWithIdx(c1).GetIsAromatic())
        self.assertIsNotNone(mol.GetBondBetweenAtoms(c1, c6))
        self.assertIsNotNone(mol.GetBondBetweenAtoms(c1, c2))
        self.assertFalse(any(n.GetSymbol() == 'N' for n in mol.GetAtomWithIdx(c6).GetNeighbors()))
        self.assertTrue(any(n.GetSymbol() == 'N' for n in mol.GetAtomWithIdx(c2).GetNeighbors()))
        self.assertEqual(mol.GetBondBetweenAtoms(ketone, oxygen).GetBondType(), Chem.BondType.DOUBLE)

    def test_policy_distinguishes_evidence(self):
        rules = load_case('1IGS')['mechanism_criteria']
        self.assertEqual(rules['distance_A']['max'], 3.5)
        self.assertNotIn('min', rules['distance_A'])
        self.assertFalse(rules['lys110_support']['hard_gate'])
        self.assertTrue(rules['retrospective'])
        self.assertFalse(rules['water_required'])


@unittest.skipUnless(FIXTURE, 'Set MCSA_1IGS_TEST_CIF to AF3 s0')
class IGPSStructureTests(unittest.TestCase):
    def setUp(self):
        self.case = load_case('1IGS')
        self.context = read_case(FIXTURE, self.case)

    def geometry(self, distance=3.2, degrees=120):
        context = copy.deepcopy(self.context)
        substrate = context['ligands']['substrate']
        roles = self.case['reaction_atoms']
        substrate[roles['C1_index']]['xyz'] = np.array([0., 0., 0.])
        substrate[roles['C6_index']]['xyz'] = np.array([1.4, 0., 0.])
        a = math.radians(degrees)
        substrate[roles['C2prime_index']]['xyz'] = np.array([distance*math.cos(a), distance*math.sin(a), 0.])
        return context

    def test_measured_baseline(self):
        r = evaluate(FIXTURE, '1IGS')
        self.assertEqual((r['mechanism_status'], r['general_status']), ('FAIL', 'PASS'))
        m = r['mechanism']['measurements']
        self.assertAlmostEqual(m['C1_C2prime_distance_A'], 4.445345993283313)
        self.assertAlmostEqual(m['C2prime_C1_C6_angle_deg'], 137.93461267057324)
        self.assertAlmostEqual(r['mechanism']['lys110_support']['distance_A'], 2.5231189032623886)

    def test_positive_and_boundaries(self):
        for distance, theta in [(3.2, 120), (3.5, 100), (3.5, 140)]:
            with self.subTest(distance=distance, theta=theta):
                r = evaluate_mechanism(self.geometry(distance, theta), self.case)
                self.assertEqual(r['status'], 'PASS')
        for distance, theta, tag in [(3.501, 120, 'RING_CLOSURE_DISTANCE_TOO_LONG'),
                                     (3.2, 99.99, 'RING_CLOSURE_DIRECTION_MISMATCH'),
                                     (3.2, 140.01, 'RING_CLOSURE_DIRECTION_MISMATCH')]:
            with self.subTest(distance=distance, theta=theta):
                self.assertIn(tag, evaluate_mechanism(self.geometry(distance, theta), self.case)['failure_tags'])

    def test_lys_support_is_not_silently_a_gate(self):
        context = self.geometry()
        residue = next(iter(context['protein_chains'].values()))[109]
        nz = next(a for a in residue['atoms'] if a['atom_name'] == 'NZ')
        nz['xyz'] = np.array([200., 200., 200.])
        r = evaluate_mechanism(context, self.case)
        self.assertEqual(r['status'], 'PASS')
        self.assertEqual(r['lys110_support']['operational_window_status'], 'FAIL')
        self.assertEqual(r['legacy_three_gate_sensitivity']['status'], 'FAIL')

    def test_missing_support_atom_does_not_erase_nac(self):
        context = self.geometry()
        residue = next(iter(context['protein_chains'].values()))[109]
        residue['atoms'] = [a for a in residue['atoms'] if a['atom_name'] != 'NZ']
        r = evaluate_mechanism(context, self.case)
        self.assertEqual(r['status'], 'PASS')
        self.assertEqual(r['lys110_support']['status'], 'UNAVAILABLE')

    def test_coincident_atoms_are_error_not_nan(self):
        with self.assertRaisesRegex(ValueError, 'coincide'):
            evaluate_mechanism(self.geometry(0), self.case)

    def test_stage_isolation(self):
        with patch.object(API, 'run_general', side_effect=AssertionError('general not requested')):
            r = evaluate(FIXTURE, '1IGS', 'mechanism')
            self.assertEqual(r['general_status'], 'NOT_RUN')
        with patch.dict(API.MECHANISMS, {'igps_nac': lambda *_: (_ for _ in ()).throw(AssertionError('mechanism not requested'))}):
            r = evaluate(FIXTURE, '1IGS', 'general')
            self.assertEqual(r['mechanism_status'], 'NOT_RUN')

    def test_new_cif_positive_nac_and_general_failure_are_separate(self):
        context = self.geometry()
        structure = gemmi.read_structure(FIXTURE)
        mapping = self.context['atom_mapping']['substrate']
        names = mapping['atom_names']
        atoms = {a.name: a for a in structure[0][mapping['chain']][0]}
        for index in [3, 4, 11]:
            atoms[names[index]].pos = gemmi.Position(*context['ligands']['substrate'][index]['xyz'])
        for chain in structure[0]:
            chain.name = 'NEW_' + chain.name
            for i, residue in enumerate(chain):
                residue.seqid = gemmi.SeqId(1000+i, ' ')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'independent_new.cif'
            structure.make_mmcif_document().write_file(str(path))
            result = evaluate(path, '1IGS')
        json.dumps(result, allow_nan=False)
        self.assertEqual(result['mechanism_status'], 'PASS')
        self.assertEqual(result['general_status'], 'FAIL')
        self.assertFalse(result['historical_results_used'])


if __name__ == '__main__':
    unittest.main()
