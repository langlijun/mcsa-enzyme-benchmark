"""Software invariants and synthetic perturbations; not new biological samples."""

from copy import deepcopy
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

from mcsa_benchmark import evaluate, evaluate_general, evaluate_mechanism
from mcsa_benchmark.evaluate import load_case
from mcsa_benchmark.mechanisms.nicotinamidase import evaluate_mechanism as mechanism
from mcsa_benchmark.structure import read_case

FIXTURE = os.environ.get('MCSA_1IM5_TEST_CIF')
api_module = importlib.import_module('mcsa_benchmark.evaluate')


class CriteriaTests(unittest.TestCase):
    def test_accepted_operational_limits(self):
        case = load_case('M0716_1IM5')
        cuts = case['mechanism_criteria']['cutoffs']
        self.assertEqual((cuts['reaction_distance_A']['min'], cuts['reaction_distance_A']['max']), (2.5, 4.5))
        self.assertEqual((cuts['attack_angle_deg']['min'], cuts['attack_angle_deg']['max']), (90, 120))
        self.assertEqual((cuts['Zn_contact_A']['min'], cuts['Zn_contact_A']['max']), (1.8, 2.6))
        self.assertEqual(cuts['base_contact_A']['max'], 4)
        self.assertTrue(case['mechanism_criteria']['retrospective'])
        self.assertEqual(case['reaction_atoms']['substrate_ring_N'], 7)
        self.assertNotEqual(case['reaction_atoms']['substrate_ring_N'], case['reaction_atoms']['substrate_amide_N'])


@unittest.skipUnless(FIXTURE, 'Set MCSA_1IM5_TEST_CIF to the Boltz2 s0 CIF')
class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.case = load_case('1IM5')
        self.context = read_case(FIXTURE, self.case)

    def protein_atom(self, position, name):
        residues = next(iter(self.context['protein_chains'].values()))
        return next(a for a in residues[position - 1]['atoms'] if a['atom_name'] == name)

    def run_geometry(self):
        result = mechanism(self.context, self.case)
        json.dumps(result, allow_nan=False)
        return result

    def test_baseline_seven_checks_and_auxiliary_only(self):
        result = self.run_geometry()
        self.assertEqual(len(result['checks']), 7)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(len(result['diagnostics']['oxyanion_hole']), 2)
        self.assertTrue(all(not r['affects_pass'] for r in result['diagnostics']['oxyanion_hole']))

    def test_distance_boundaries(self):
        s, c = self.protein_atom(133, 'SG'), self.context['ligands']['substrate'][1]
        unit = s['xyz'] - c['xyz']
        unit /= np.linalg.norm(unit)
        for value, expected in ((2.5, 'PASS'), (4.5, 'PASS'), (2.499, 'FAIL'), (4.501, 'FAIL')):
            with self.subTest(value=value):
                s['xyz'] = c['xyz'] + unit * value
                self.assertEqual(self.run_geometry()['checks']['reaction_distance'], expected)

    def test_angle_boundaries(self):
        s = self.protein_atom(133, 'SG')
        c, o = self.context['ligands']['substrate'][1:3]
        unit = o['xyz'] - c['xyz']
        unit /= np.linalg.norm(unit)
        normal = np.cross(unit, [1., 0., 0.])
        if np.linalg.norm(normal) < 1e-6:
            normal = np.cross(unit, [0., 1., 0.])
        normal /= np.linalg.norm(normal)
        for value, expected in ((90., 'PASS'), (120., 'PASS'), (89.999, 'FAIL'), (120.001, 'FAIL')):
            with self.subTest(value=value):
                rad = math.radians(value)
                s['xyz'] = c['xyz'] + 3 * (math.cos(rad)*unit + math.sin(rad)*normal)
                result = self.run_geometry()
                self.assertEqual(result['checks']['attack_angle'], expected)
                if expected == 'FAIL':
                    self.assertIn('ATTACK_ANGLE_OUTSIDE_OPERATIONAL_WINDOW', result['failure_tags'])

    def test_base_contact_boundary(self):
        s = self.protein_atom(133, 'SG')
        self.protein_atom(10, 'OD2')['xyz'] = s['xyz'] + [6, 0, 0]
        for value, expected in ((4., 'PASS'), (4.001, 'FAIL')):
            self.protein_atom(10, 'OD1')['xyz'] = s['xyz'] + [value, 0, 0]
            self.assertEqual(self.run_geometry()['checks']['asp10_activation_contact'], expected)

    def test_zinc_boundaries(self):
        zn, ring_n = self.context['metals']['Zn'], self.context['ligands']['substrate'][7]
        for value, expected in ((1.8, 'PASS'), (2.6, 'PASS'), (1.799, 'FAIL'), (2.601, 'FAIL')):
            with self.subTest(value=value):
                ring_n['xyz'] = zn['xyz'] + [value, 0, 0]
                self.assertEqual(self.run_geometry()['checks']['Zn_substrate_ring_N'], expected)

    def test_amide_n_cannot_substitute_for_ring_n(self):
        zn, substrate = self.context['metals']['Zn'], self.context['ligands']['substrate']
        substrate[0]['xyz'] = zn['xyz'] + [2.1, 0, 0]
        substrate[7]['xyz'] = zn['xyz'] + [20, 0, 0]
        self.assertEqual(self.run_geometry()['checks']['Zn_substrate_ring_N'], 'FAIL')

    def test_wrong_histidine_n_cannot_substitute(self):
        zn = self.context['metals']['Zn']
        self.protein_atom(54, 'ND1')['xyz'] = zn['xyz'] + [2.1, 0, 0]
        self.protein_atom(54, 'NE2')['xyz'] = zn['xyz'] + [12, 0, 0]
        self.assertEqual(self.run_geometry()['checks']['Zn_His54_NE2'], 'FAIL')

    def test_short_carboxylate_contact_not_hidden_by_second_oxygen(self):
        zn = self.context['metals']['Zn']
        self.protein_atom(52, 'OD1')['xyz'] = zn['xyz'] + [1.0, 0, 0]
        self.protein_atom(52, 'OD2')['xyz'] = zn['xyz'] + [2.1, 0, 0]
        result = self.run_geometry()
        self.assertEqual(result['checks']['Zn_Asp52_O'], 'FAIL')
        self.assertIn('ZN_ASP52_CONTACT_TOO_SHORT', result['failure_tags'])

    def test_equivalent_oxygen_names(self):
        before = self.run_geometry()
        for position in (10, 52):
            first, second = self.protein_atom(position, 'OD1'), self.protein_atom(position, 'OD2')
            first['xyz'], second['xyz'] = second['xyz'].copy(), first['xyz'].copy()
        after = self.run_geometry()
        self.assertEqual(before['checks'], after['checks'])

    def test_optional_oxyanion_contact_not_a_hidden_gate(self):
        baseline = self.run_geometry()
        self.protein_atom(129, 'N')['xyz'] += [100, 0, 0]
        result = self.run_geometry()
        self.assertEqual(result['checks'], baseline['checks'])
        self.assertGreater(result['diagnostics']['oxyanion_hole'][0]['value'], 50)

    def test_missing_zinc_preserves_axis_measurements(self):
        del self.context['metals']['Zn']
        result = self.run_geometry()
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(result['legacy_axis_status'], 'PASS')
        self.assertEqual(result['checks']['Zn_substrate_ring_N'], 'ERROR')
        self.assertEqual(len(result['errors']), 4)

    def test_degenerate_angle_is_explicit(self):
        self.protein_atom(133, 'SG')['xyz'] = self.context['ligands']['substrate'][1]['xyz'].copy()
        result = self.run_geometry()
        self.assertEqual(result['checks']['attack_angle'], 'ERROR')
        self.assertIn('DEGENERATE_REACTION_ANGLE', [r['tag'] for r in result['errors']])

    def test_missing_auxiliary_atom_not_core_error(self):
        residues = next(iter(self.context['protein_chains'].values()))
        residues[128]['atoms'] = [a for a in residues[128]['atoms'] if a['atom_name'] != 'N']
        result = self.run_geometry()
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['diagnostics']['oxyanion_hole'][0]['status'], 'NOT_MEASURED')


@unittest.skipUnless(FIXTURE, 'Set MCSA_1IM5_TEST_CIF to the Boltz2 s0 CIF')
class IntegrationTests(unittest.TestCase):
    def variant(self, modify, mapping=None):
        structure = gemmi.read_structure(FIXTURE)
        modify(structure)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'new.cif'
            structure.make_mmcif_document().write_file(str(path))
            result = evaluate(path, '1IM5', atom_map=mapping)
            json.dumps(result, allow_nan=False)
            return result

    def test_independent_mechanism_stage(self):
        with patch.object(api_module, 'run_general', side_effect=AssertionError('not called')):
            result = evaluate_mechanism(FIXTURE, '1IM5')
        self.assertEqual(result['mechanism_status'], 'PASS')
        self.assertEqual(result['general_status'], 'NOT_RUN')

    def test_independent_general_stage(self):
        with patch.dict(api_module.MECHANISMS, {'nicotinamidase': lambda *_: (_ for _ in ()).throw(AssertionError('not called'))}):
            result = evaluate_general(FIXTURE, '1IM5')
        self.assertEqual(result['general_status'], 'PASS')
        self.assertEqual(result['mechanism_status'], 'NOT_RUN')

    def test_general_failure_preserves_mechanism(self):
        def change(structure):
            ligand = structure[0]['B'][0]
            amide_n = next(a for a in ligand if a.name == 'N15')
            amide_n.pos.x += 12
        result = self.variant(change)
        self.assertEqual(result['mechanism_status'], 'PASS')
        self.assertEqual(result['general_status'], 'FAIL')
        self.assertIn('BOND_LENGTH_OUT_OF_RANGE', result['general']['failure_tags'])

    def test_far_zinc_fails_mechanism_not_general(self):
        def change(structure):
            structure[0]['C'][0][0].pos.x += 100
        result = self.variant(change)
        self.assertEqual(result['mechanism_status'], 'FAIL')
        self.assertEqual(result['general_status'], 'PASS')
        self.assertEqual(result['mechanism']['legacy_axis_status'], 'PASS')

    def test_missing_zinc_independent_general_failure(self):
        result = self.variant(lambda s: s[0].remove_chain('C'))
        self.assertEqual(result['mechanism_status'], 'ERROR')
        self.assertEqual(result['general_status'], 'FAIL')
        self.assertEqual(result['mechanism']['legacy_axis_status'], 'PASS')

    def test_required_atom_missing_preserves_other_measurements(self):
        result = self.variant(lambda s: s[0]['A'][53].remove_atom('NE2', '\x00'))
        self.assertEqual(result['mechanism_status'], 'ERROR')
        self.assertEqual(result['general_status'], 'FAIL')
        self.assertEqual(result['mechanism']['checks']['Zn_His54_NE2'], 'ERROR')
        self.assertEqual(result['mechanism']['checks']['reaction_distance'], 'PASS')

    def test_reorder_rename_and_renumber(self):
        def change(structure):
            for chain in structure[0]:
                for i, residue in enumerate(chain):
                    atoms = [a.clone() for a in residue]
                    for j in range(len(residue)-1, -1, -1):
                        del residue[j]
                    for atom in reversed(atoms):
                        residue.add_atom(atom)
                    residue.seqid = gemmi.SeqId(1000+i, ' ')
                chain.name = 'RENAMED_' + chain.name
        result = self.variant(change)
        self.assertEqual(result['joint_status'], 'PASS')
        self.assertFalse(result['historical_results_used'])

    def test_added_water_not_required_or_scored(self):
        def change(structure):
            chain, residue, atom = gemmi.Chain('W'), gemmi.Residue(), gemmi.Atom()
            residue.name, residue.seqid, residue.het_flag = 'HOH', gemmi.SeqId(1, ' '), 'H'
            atom.name, atom.element, atom.pos = 'O', gemmi.Element('O'), gemmi.Position(500, 500, 500)
            residue.add_atom(atom)
            chain.add_residue(residue)
            structure[0].add_chain(chain)
        result = self.variant(change)
        self.assertEqual(result['joint_status'], 'PASS')
        self.assertEqual(result['ignored_water_count'], 1)

    def test_unknown_atom_names_need_explicit_mapping(self):
        names = load_case('1IM5')['ligands']['substrate']['atom_name_templates'][1]
        renamed = {name: f'X{i}' for i, name in enumerate(names)}
        def change(structure):
            for atom in structure[0]['B'][0]:
                atom.name = renamed[atom.name]
        self.assertEqual(self.variant(change)['input_status'], 'ERROR')
        result = self.variant(change, {'substrate': {'atom_names': [renamed[n] for n in names]}})
        self.assertEqual(result['joint_status'], 'PASS')

    def test_nan_is_explicit_error(self):
        result = self.variant(lambda s: setattr(s[0]['B'][0][0].pos, 'x', float('nan')))
        self.assertEqual(result['mechanism_status'], 'ERROR')
        self.assertIn('NONFINITE_COORDINATE', result['input_problem'])


if __name__ == '__main__':
    unittest.main()
