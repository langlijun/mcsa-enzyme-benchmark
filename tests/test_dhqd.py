"""Synthetic tests check software rules, not positive biochemical controls."""

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
from mcsa_benchmark.structure import read_case
from mcsa_benchmark.mechanisms.dhqd_lys_carbonyl import evaluate_mechanism

FIXTURE = os.environ.get('MCSA_1QFE_TEST_CIF')
API = importlib.import_module('mcsa_benchmark.evaluate')


class DefinitionTests(unittest.TestCase):
    def test_unique_ketone_not_carboxyl(self):
        c = load_case('1QFE')
        mol = Chem.MolFromSmiles(c['ligands']['DHQ_1']['smiles'])
        ids = c['reaction_atoms']
        carbon = mol.GetAtomWithIdx(ids['carbonyl_C_index'])
        self.assertEqual(sorted(n.GetSymbol() for n in carbon.GetNeighbors()), ['C', 'C', 'O'])
        self.assertEqual(mol.GetBondBetweenAtoms(ids['carbonyl_C_index'], ids['carbonyl_O_index']).GetBondType(), Chem.BondType.DOUBLE)
        self.assertEqual(sum(a.GetFormalCharge() for a in mol.GetAtoms()), -1)
        self.assertEqual(c['protein_sequence'][169], 'K')
        self.assertEqual(c['protein_copies'], 2)

    def test_scope_and_threshold_authorship(self):
        c = load_case('1QFE')['mechanism_criteria']
        self.assertIsNone(c['main_axis']['distance_A']['min'])
        self.assertEqual(c['main_axis']['distance_A']['max'], 4.0)
        self.assertEqual(c['main_axis']['angle_deg']['min'], 90)
        self.assertEqual(c['main_axis']['angle_deg']['max'], 120)
        self.assertFalse(c['legacy_relay_is_hard_gate'])
        self.assertTrue(c['retrospective'])
        self.assertIn('Author operational', c['main_axis']['basis'])


@unittest.skipUnless(FIXTURE, 'Set MCSA_1QFE_TEST_CIF to AF3 sample 0')
class DHQDTests(unittest.TestCase):
    def setUp(self):
        self.case = load_case('1QFE')
        self.context = read_case(FIXTURE, self.case)

    def axis_geometry(self, context, chain, instance, d=3., theta=105.):
        ligand = context['ligands'][instance]
        center = ligand[1]['xyz']
        u = ligand[0]['xyz'] - center
        u = u / np.linalg.norm(u)
        v = np.cross(u, [1., 0., 0.])
        if np.linalg.norm(v) < 1e-5:
            v = np.cross(u, [0., 1., 0.])
        v /= np.linalg.norm(v)
        xyz = center + d * (math.cos(math.radians(theta))*u + math.sin(math.radians(theta))*v)
        next(a for a in context['protein_chains'][chain][169]['atoms'] if a['atom_name'] == 'NZ')['xyz'] = xyz

    def positive(self):
        c = copy.deepcopy(self.context)
        for a in evaluate_mechanism(c, self.case)['selected_axes']:
            self.axis_geometry(c, a['protein_chain'], a['ligand_instance'])
        self.assertEqual(evaluate_mechanism(c, self.case)['status'], 'PASS')
        return c

    def variant(self, change, stage='all', mapping=None):
        structure = gemmi.read_structure(FIXTURE)
        change(structure)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'new.cif'
            structure.make_mmcif_document().write_file(str(path))
            r = evaluate(path, '1QFE', stage, mapping)
            json.dumps(r, allow_nan=False)
            return r

    def test_archived_axis_recomputed(self):
        r = evaluate(FIXTURE, '1QFE')
        self.assertEqual(r['mechanism_status'], 'FAIL')
        self.assertEqual(len(r['mechanism']['all_candidate_axes']), 4)
        a = next(a for a in r['mechanism']['selected_axes'] if a['ligand_chain'] == 'C')
        self.assertEqual(a['protein_chain'], 'B')
        self.assertAlmostEqual(a['distance_A'], 3.064918106573159)
        self.assertAlmostEqual(a['angle_deg'], 48.56416574226478)
        self.assertFalse(r['historical_results_used'])

    def test_positive_and_inclusive_boundaries(self):
        c = self.positive()
        a = evaluate_mechanism(c, self.case)['selected_axes'][0]
        for d, theta in [(4., 105.), (3., 90.), (3., 120.), (2., 105.)]:
            with self.subTest(d=d, theta=theta):
                self.axis_geometry(c, a['protein_chain'], a['ligand_instance'], d, theta)
                self.assertEqual(evaluate_mechanism(c, self.case)['status'], 'PASS')

    def test_failures_and_partial_recovery(self):
        c = self.positive()
        a = evaluate_mechanism(c, self.case)['selected_axes'][0]
        for d, theta, tag in [(4.001, 105., 'REACTION_DISTANCE_TOO_LONG'),
                               (3., 89.99, 'ATTACK_DIRECTION_MISMATCH'),
                               (3., 120.01, 'ATTACK_DIRECTION_MISMATCH')]:
            self.axis_geometry(c, a['protein_chain'], a['ligand_instance'], d, theta)
            r = evaluate_mechanism(c, self.case)
            self.assertEqual(r['status'], 'FAIL')
            self.assertEqual(r['passing_site_count'], 1)
            self.assertIn(tag, r['failure_tags'])
            self.assertIn('PARTIAL_SITE_RECOVERY', r['failure_tags'])

    def test_distance_and_angle_cannot_be_mixed_across_sites(self):
        c = self.positive()
        a, b = evaluate_mechanism(c, self.case)['selected_axes']
        self.axis_geometry(c, a['protein_chain'], a['ligand_instance'], 3., 45.)
        self.axis_geometry(c, b['protein_chain'], b['ligand_instance'], 5., 105.)
        r = evaluate_mechanism(c, self.case)
        self.assertEqual(r['passing_site_count'], 0)
        self.assertEqual(r['status'], 'FAIL')

    def test_general_stereo_does_not_block_mechanism(self):
        c = self.positive()
        def change(s):
            for chain, residues in c['protein_chains'].items():
                target = next(a['xyz'] for a in residues[169]['atoms'] if a['atom_name'] == 'NZ')
                s[0][chain][169].find_atom('NZ', '\x00').pos = gemmi.Position(*target)
        r = self.variant(change)
        self.assertEqual(r['mechanism_status'], 'PASS')
        self.assertEqual(r['general_status'], 'FAIL')
        self.assertIn('INPUT_STEREOCHEMISTRY_MISMATCH', r['general']['failure_tags'])

    def test_relay_removed_and_both_histidine_nitrogens_reported(self):
        c = self.positive()
        for residues in c['protein_chains'].values():
            for index in [85, 142]:
                for a in residues[index]['atoms']:
                    a['xyz'] += 100.
        r = evaluate_mechanism(c, self.case)
        self.assertEqual(r['status'], 'PASS')
        self.assertFalse(r['relay_gates_applied'])
        self.assertEqual(len(r['relay_diagnostics']), 12)
        self.assertTrue(all(not d['affects_pass'] for d in r['relay_diagnostics']))

    def test_missing_diagnostic_atom_does_not_make_axis_unmeasurable(self):
        r = self.variant(lambda s: s[0]['A'][142].remove_atom('NE2', '\x00'))
        self.assertEqual(r['mechanism_status'], 'FAIL')
        self.assertTrue(any(d['status'] == 'ATOM_MISSING' for d in r['mechanism']['relay_diagnostics']))
        self.assertEqual(r['general_status'], 'FAIL')

    def test_pairing_does_not_optimize_reaction_coordinates(self):
        c = self.positive()
        old = evaluate_mechanism(c, self.case)['site_assignment']['selected_assignment']
        for chain, residues in c['protein_chains'].items():
            next(a for a in residues[169]['atoms'] if a['atom_name']=='NZ')['xyz'] += 100.
        new = evaluate_mechanism(c, self.case)
        self.assertEqual(new['site_assignment']['selected_assignment'], old)
        self.assertEqual(new['passing_site_count'], 0)

    def test_numeric_pairing_tie_is_explicit(self):
        c = self.positive()
        for residues in c['protein_chains'].values():
            for a in residues[169]['atoms']:
                if a['atom_name'] in ['CA','CB']:
                    a['xyz'] = np.zeros(3)
        r = evaluate_mechanism(c, self.case)
        self.assertEqual(r['status'], 'ERROR')
        self.assertIn('SITE_ASSIGNMENT_NUMERIC_TIE', r['failure_tags'])

    def test_degenerate_axis_reports_failure_not_nan(self):
        c = self.positive()
        a = evaluate_mechanism(c, self.case)['selected_axes'][0]
        self.axis_geometry(c, a['protein_chain'], a['ligand_instance'], 0.)
        r = evaluate_mechanism(c, self.case)
        self.assertIn('DEGENERATE_REACTION_AXIS', r['failure_tags'])
        json.dumps(r, allow_nan=False)

    def test_missing_required_atom_keeps_general_independent(self):
        r = self.variant(lambda s: s[0]['A'][169].remove_atom('NZ', '\x00'))
        self.assertEqual(r['mechanism_status'], 'ERROR')
        self.assertIn('MECHANISM_REQUIRED_ATOM_MISSING', r['mechanism']['failure_tags'])
        self.assertEqual(r['general_status'], 'FAIL')

    def test_missing_ligand_is_explicit_composition_failure(self):
        r = self.variant(lambda s: s[0].remove_chain('C'))
        self.assertEqual(r['input_status'], 'FAIL')
        self.assertIn('LIGAND_COMPOSITION_OR_COPY_MISMATCH', r['input_problem'])

    def test_chain_renaming_and_atom_reordering(self):
        def change(s):
            for chain in s[0]:
                chain.name = 'NEW_' + chain.name
                for i, residue in enumerate(chain):
                    residue.seqid = gemmi.SeqId(1000+i, ' ')
                    atoms = list(residue.clone())
                    for j in range(len(residue)-1, -1, -1):
                        del residue[j]
                    for atom in reversed(atoms):
                        residue.add_atom(atom)
        r = self.variant(change, 'mechanism')
        self.assertEqual(r['mechanism_status'], 'FAIL')
        a = next(a for a in r['mechanism']['selected_axes'] if a['ligand_chain']=='NEW_C')
        self.assertEqual(a['protein_chain'], 'NEW_B')
        self.assertAlmostEqual(a['distance_A'], 3.064918106573159)

    def test_custom_mapping_for_new_atom_names(self):
        mapping = {k: {n:v for n,v in r.items() if n!='mapping_method'} for k,r in evaluate(FIXTURE,'1QFE','mechanism')['atom_mapping'].items()}
        for r in mapping.values():
            r['atom_names'] = ['NEW_'+name for name in r['atom_names']]
        def change(s):
            for name in ['C','D']:
                for a in s[0][name][0]:
                    a.name = 'NEW_'+a.name
        unknown = self.variant(change, 'mechanism')
        self.assertEqual(unknown['mechanism_status'], 'ERROR')
        mapped = self.variant(change, 'mechanism', mapping)
        self.assertEqual(mapped['mechanism_status'], 'FAIL')
        self.assertAlmostEqual(mapped['mechanism']['selected_axes'][0]['distance_A'], 3.064918106573159)

    def test_separate_stages(self):
        with patch.object(API, 'run_general', side_effect=AssertionError('not requested')):
            self.assertEqual(evaluate(FIXTURE,'1QFE','mechanism')['general_status'], 'NOT_RUN')
        with patch.dict(API.MECHANISMS, {'dhqd_lys_carbonyl': lambda *_: (_ for _ in ()).throw(AssertionError('not requested'))}):
            self.assertEqual(evaluate(FIXTURE,'1QFE','general')['mechanism_status'], 'NOT_RUN')

    def test_general_error_does_not_erase_mechanism(self):
        with patch.object(API, 'run_general', side_effect=ValueError('synthetic')):
            r = evaluate(FIXTURE,'1QFE')
        self.assertEqual(r['mechanism_status'], 'FAIL')
        self.assertEqual(r['general_status'], 'ERROR')

    def test_declared_graph_mismatch_is_general_only(self):
        doc = gemmi.cif.read_file(FIXTURE)
        block = doc.sole_block()
        ids = list(block.find_values('_chem_comp.id'))
        block.find_values('_chem_comp.pdbx_smiles')[ids.index('LIG_C')] = gemmi.cif.quote('CC')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'different.cif'
            doc.write_file(str(path))
            r = evaluate(path,'1QFE')
        self.assertEqual(r['mechanism_status'], 'FAIL')
        self.assertIn('DECLARED_LIGAND_CHEMISTRY_MISMATCH', r['general']['failure_tags'])
