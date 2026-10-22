"""Monomer GalE tests; synthetic coordinates test code, not chemical plausibility."""
import copy
import importlib
import json
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
from mcsa_benchmark.gale_general import review_stereochemistry
from mcsa_benchmark.mechanisms.gale_hydride import evaluate_mechanism
from mcsa_benchmark.structure import InputProblem, read_case

API = importlib.import_module('mcsa_benchmark.evaluate')
FIXTURE = os.environ.get('MCSA_1XEL_TEST_CIF')


class GalEDefinitionTests(unittest.TestCase):
    def test_input_contract(self):
        case = load_case('1XEL')
        self.assertEqual(case['protein_copies'], 1)
        self.assertEqual(len(case['protein_sequence']), 338)
        self.assertEqual(case['metals'], {})
        self.assertEqual(set(case['ligands']), {'NAD_1', 'sugar_1'})
        self.assertTrue(all(d['group_copies'] == 1 for d in case['ligands'].values()))

    def test_hydride_atoms(self):
        c = load_case('1XEL')
        nad = Chem.MolFromSmiles(c['ligands']['NAD_1']['smiles'])
        sugar = Chem.MolFromSmiles(c['ligands']['sugar_1']['smiles'])
        n = next(a.GetIdx() for a in nad.GetAtoms() if a.GetSymbol() == 'N' and a.GetFormalCharge() == 1)
        self.assertEqual(len(Chem.GetShortestPath(nad, 1, n)), 4)
        self.assertEqual(sugar.GetAtomWithIdx(23).GetSymbol(), 'C')
        self.assertIsNotNone(sugar.GetBondBetweenAtoms(23, 24))
        self.assertEqual(sugar.GetAtomWithIdx(24).GetSymbol(), 'O')
        self.assertEqual(dict(Chem.FindMolChiralCenters(sugar))[23], 'R')

    def test_criteria_are_honest_about_scope(self):
        c = load_case('1XEL')['mechanism_criteria']
        self.assertEqual(c['main_axis']['distance_A'], {'min': 3.0, 'max': 4.0, 'inclusive': True})
        self.assertIn('author operational', c['basis'])
        self.assertTrue(c['registered_before_monomer_inference'])
        self.assertFalse(c['whole_study_preregistered'])
        self.assertFalse(c['angles_scored'])
        self.assertFalse(c['residue_contacts_scored'])


@unittest.skipUnless(FIXTURE, 'Set MCSA_1XEL_TEST_CIF to monomer AF3 sample 0')
class GalETests(unittest.TestCase):
    def setUp(self):
        self.case = load_case('1XEL')
        self.context = read_case(FIXTURE, self.case)

    def variant(self, change, stage='all', mapping=None):
        s = gemmi.read_structure(FIXTURE)
        change(s)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'unseen_input.cif'
            s.make_mmcif_document().write_file(str(path))
            r = evaluate(path, '1XEL', stage, mapping)
            json.dumps(r, allow_nan=False)
            return r

    def synthetic(self, d):
        c = copy.deepcopy(self.context)
        c['ligands']['sugar_1'][23]['xyz'] = np.array([0., 0., 0.])
        c['ligands']['NAD_1'][1]['xyz'] = np.array([d, 0., 0.])
        return c

    def place_axis(self, s):
        sugar = np.array(s[0]['D'][0].find_atom('C10', '\x00').pos.tolist())
        nad = s[0]['C'][0].find_atom('C2', '\x00')
        nad.pos = gemmi.Position(*(sugar + np.array([3.5, 0., 0.])))

    def test_baseline(self):
        r = evaluate(FIXTURE, '1XEL')
        self.assertEqual((r['mechanism_status'], r['general_status']), ('FAIL', 'PASS'))
        self.assertAlmostEqual(r['mechanism']['measurements']['C4_C4_distance_A'], 4.5085823714333975)
        self.assertFalse(r['historical_results_used'])

    def test_boundaries_and_failure_tags(self):
        for d in (3.0, 3.5, 4.0):
            self.assertEqual(evaluate_mechanism(self.synthetic(d), self.case)['status'], 'PASS')
        for d, tag in [(2.999, 'REACTION_C4_C4_TOO_SHORT'), (4.001, 'REACTION_C4_C4_TOO_LONG'), (0., 'REACTION_C4_C4_TOO_SHORT')]:
            r = evaluate_mechanism(self.synthetic(d), self.case)
            self.assertEqual(r['status'], 'FAIL')
            self.assertIn(tag, r['failure_tags'])

    def test_no_residue_or_water_hard_gate(self):
        c = self.synthetic(3.5)
        c['protein_chains'] = {'A': []}
        r = evaluate_mechanism(c, self.case)
        self.assertEqual(r['status'], 'PASS')
        self.assertFalse(r['water_required'])
        self.assertFalse(r['residue_contacts_scored'])

    def test_new_cif_mechanism_pass_and_general_fail(self):
        r = self.variant(self.place_axis)
        self.assertEqual(r['mechanism_status'], 'PASS')
        self.assertEqual(r['general_status'], 'FAIL')

    def test_sugar_C4_flip_not_misread_as_reaction_success(self):
        def change(s):
            self.place_axis(s)
            atoms = {a.name: a for a in s[0]['D'][0]}
            p = np.array(atoms['C10'].pos.tolist())
            v1 = np.array(atoms['C8'].pos.tolist()) - p
            v2 = np.array(atoms['C11'].pos.tolist()) - p
            normal = np.cross(v1, v2)
            normal /= np.linalg.norm(normal)
            q = np.array(atoms['O12'].pos.tolist())
            atoms['O12'].pos = gemmi.Position(*(q - 2*np.dot(q-p, normal)*normal))
        r = self.variant(change)
        self.assertEqual(r['mechanism_status'], 'PASS')
        self.assertIn('INPUT_STEREOCHEMISTRY_MISMATCH', r['general']['failure_tags'])
        centers = r['general']['components']['sugar_1']['gale_stereochemistry']['centers']
        center = next(c for c in centers if c['index'] == 23)
        self.assertEqual((center['expected_CIP'], center['observed_CIP']), ('R', 'S'))

    def test_registered_phosphate_only_exception(self):
        r = evaluate(FIXTURE, '1XEL', 'general')
        g = r['general']['components']['NAD_1']
        self.assertFalse(g['raw_tetrahedral_stereochemistry_valid'])
        self.assertTrue(g['tetrahedral_stereochemistry_valid'])
        self.assertEqual(g['gale_stereochemistry']['excluded_indices'], [13, 17])
        self.assertEqual(g['gale_stereochemistry']['meaningful_mismatches'], [])

    def test_invalid_carbon_exception_is_rejected(self):
        c = copy.deepcopy(self.case)
        c['case_general_requirements']['phosphate_formal_stereo_exclusions']['sugar_1'] = [23]
        d = c['ligands']['sugar_1']
        with self.assertRaisesRegex(InputProblem, 'INVALID_STEREOCHEMISTRY_EXCEPTION'):
            review_stereochemistry('sugar_1', self.context['ligands']['sugar_1'], Chem.MolFromSmiles(d['smiles']),
                                   {'tetrahedral_stereochemistry_valid': False}, c)

    def test_phosphate_exception_does_not_hide_bad_bond(self):
        def change(s):
            a = s[0]['C'][0].find_atom('O3', '\x00')
            a.pos.x += 10.
        r = self.variant(change, 'general')
        self.assertEqual(r['general_status'], 'FAIL')
        self.assertIn('BOND_LENGTH_OUT_OF_RANGE', r['general']['failure_tags'])

    def test_chain_and_residue_labels_atom_order_are_not_scoring_inputs(self):
        def change(s):
            for ch in s[0]:
                ch.name = 'X' + ch.name
                for i, res in enumerate(ch):
                    res.seqid = gemmi.SeqId(i + 901, ' ')
                    atoms = list(res.clone())
                    for j in range(len(res)-1, -1, -1):
                        del res[j]
                    for a in reversed(atoms):
                        res.add_atom(a)
        r = self.variant(change)
        self.assertEqual((r['mechanism_status'], r['general_status']), ('FAIL', 'PASS'))
        self.assertAlmostEqual(r['mechanism']['measurements']['C4_C4_distance_A'], 4.5085823714333975)

    def test_custom_atom_names_need_verified_map(self):
        mapping = {role: {k: v for k, v in data.items() if k != 'mapping_method'} for role, data in self.context['atom_mapping'].items()}
        for entry in mapping.values():
            entry['atom_names'] = ['X_' + n for n in entry['atom_names']]
        def change(s):
            for ch in ('C', 'D'):
                for a in s[0][ch][0]:
                    a.name = 'X_' + a.name
        r = self.variant(change, 'mechanism')
        self.assertIn('ATOM_MAPPING_REQUIRED', r['input_problem'])
        r = self.variant(change, 'mechanism', mapping)
        self.assertEqual(r['input_status'], 'PASS')
        self.assertAlmostEqual(r['mechanism']['measurements']['C4_C4_distance_A'], 4.5085823714333975)

    def test_missing_nad_is_explicit_failure(self):
        r = self.variant(lambda s: s[0].remove_chain('C'))
        self.assertEqual(r['input_status'], 'FAIL')
        self.assertIn('LIGAND_COMPOSITION_OR_COPY_MISMATCH', r['input_problem'])

    def test_dimer_is_not_silently_truncated(self):
        def change(s):
            c = s[0]['A'].clone()
            c.name = 'B'
            s[0].add_chain(c)
        r = self.variant(change)
        self.assertIn('PROTEIN_COPY_COUNT_MISMATCH', r['input_problem'])

    def test_extra_ligand_is_not_cherry_picked(self):
        def change(s):
            c = s[0]['D'].clone()
            c.name = 'EXTRA'
            s[0].add_chain(c)
        r = self.variant(change)
        self.assertIn('LIGAND_COMPOSITION_OR_COPY_MISMATCH', r['input_problem'])

    def test_explicit_input_stereo_metadata_mismatch_is_general_only(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'changed_metadata.cif'
            doc = gemmi.cif.read(FIXTURE)
            values = doc.sole_block().find_values('_chem_comp.pdbx_smiles')
            self.assertTrue(len(values))
            for i, value in enumerate(values):
                smiles = gemmi.cif.as_string(value)
                if smiles == self.case['ligands']['sugar_1']['smiles']:
                    mol = Chem.MolFromSmiles(smiles)
                    mol.GetAtomWithIdx(23).InvertChirality()
                    values[i] = gemmi.cif.quote(Chem.MolToSmiles(mol))
            doc.write_file(str(p))
            r = evaluate(p, '1XEL')
        self.assertEqual(r['mechanism_status'], 'FAIL')
        self.assertAlmostEqual(r['mechanism']['measurements']['C4_C4_distance_A'], 4.5085823714333975)
        self.assertIn('DECLARED_LIGAND_CHEMISTRY_MISMATCH', r['general']['failure_tags'])

    def test_stage_isolation(self):
        with patch.object(API, 'run_general', side_effect=AssertionError('not requested')):
            r = evaluate(FIXTURE, '1XEL', 'mechanism')
            self.assertEqual(r['general_status'], 'NOT_RUN')
        with patch.dict(API.MECHANISMS, {'gale_hydride': lambda *_: (_ for _ in ()).throw(AssertionError('not requested'))}):
            r = evaluate(FIXTURE, '1XEL', 'general')
            self.assertEqual(r['mechanism_status'], 'NOT_RUN')

    def test_general_error_preserves_mechanism_measurement(self):
        with patch.object(API, 'run_general', side_effect=ValueError('synthetic error')):
            r = evaluate(FIXTURE, '1XEL')
            self.assertEqual(r['general_status'], 'ERROR')
            self.assertAlmostEqual(r['mechanism']['measurements']['C4_C4_distance_A'], 4.5085823714333975)


if __name__ == '__main__':
    unittest.main()
