"""Single-occupancy SAM tests; synthetic geometries are software fixtures only."""

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
from mcsa_benchmark.mechanisms.sam_n_methyl import evaluate_mechanism
from mcsa_benchmark.sam_general import evaluate_sam_checks
from mcsa_benchmark.structure import read_case

FIXTURE = os.environ.get('MCSA_1XVA_TEST_CIF')
API = importlib.import_module('mcsa_benchmark.evaluate')


class SAMDefinitionTests(unittest.TestCase):
    def test_atom_roles_and_occupancy(self):
        c = load_case('1XVA')
        sam = Chem.MolFromSmiles(c['ligands']['SAM']['smiles'])
        gly = Chem.MolFromSmiles(c['ligands']['glycine']['smiles'])
        self.assertEqual(c['protein_copies'], 4)
        self.assertEqual(len(c['ligands']), 2)
        self.assertEqual(sam.GetAtomWithIdx(0).GetDegree(), 1)
        self.assertEqual(sam.GetAtomWithIdx(0).GetTotalNumHs(), 3)
        self.assertEqual(sam.GetAtomWithIdx(1).GetSymbol(), 'S')
        self.assertEqual(sam.GetAtomWithIdx(1).GetFormalCharge(), 1)
        self.assertEqual(gly.GetAtomWithIdx(0).GetFormalCharge(), 0)
        self.assertEqual(Chem.GetFormalCharge(gly), -1)
        self.assertTrue(c['mechanism_criteria']['retrospective'])
        self.assertIn('author operational', c['mechanism_criteria']['basis'])


@unittest.skipUnless(FIXTURE, 'Set MCSA_1XVA_TEST_CIF to AF3 sample 2')
class SAMTests(unittest.TestCase):
    def setUp(self):
        self.case = load_case('1XVA')
        self.context = read_case(FIXTURE, self.case)

    def synthetic(self, d=3., theta=160., cs=1.8):
        context = copy.deepcopy(self.context)
        context['ligands']['SAM'][0]['xyz'] = np.array([0., 0., 0.])
        context['ligands']['SAM'][1]['xyz'] = np.array([cs, 0., 0.])
        t = math.radians(theta)
        context['ligands']['glycine'][0]['xyz'] = np.array([d*math.cos(t), d*math.sin(t), 0.])
        return context

    def variant(self, change, stage='all', mapping=None):
        s = gemmi.read_structure(FIXTURE)
        change(s)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'new_prediction.cif'
            s.make_mmcif_document().write_file(str(path))
            r = evaluate(path, '1XVA', stage, mapping)
            json.dumps(r, allow_nan=False)
            return r

    def orient_glycine(self, s):
        c = np.array(s[0]['E'][0].find_atom('C1', '\x00').pos.tolist())
        sulfur = np.array(s[0]['E'][0].find_atom('S1', '\x00').pos.tolist())
        n = np.array(s[0]['F'][0].find_atom('N1', '\x00').pos.tolist())
        target = c + 3.*(c-sulfur)/np.linalg.norm(c-sulfur)
        for atom in s[0]['F'][0]:
            atom.pos = gemmi.Position(*(np.array(atom.pos.tolist()) + target-n))

    def test_baseline(self):
        r = evaluate(FIXTURE, '1XVA')
        self.assertEqual(r['mechanism_status'], 'FAIL')
        self.assertEqual(r['general_status'], 'FAIL')
        self.assertEqual(r['mechanism']['declared_reaction_pairs'], 1)
        self.assertAlmostEqual(r['mechanism']['measurements']['N_C_distance_A'], 3.0784640650818065)
        self.assertAlmostEqual(r['mechanism']['measurements']['N_C_S_angle_deg'], 97.3257941736836)

    def test_axis_boundaries(self):
        for d, theta in [(2.5, 140.), (3.5, 180.), (3., 160.)]:
            self.assertEqual(evaluate_mechanism(self.synthetic(d, theta), self.case)['status'], 'PASS')
        for d, theta in [(2.499, 160.), (3.501, 160.), (3., 139.99)]:
            self.assertEqual(evaluate_mechanism(self.synthetic(d, theta), self.case)['status'], 'FAIL')

    def test_cs_bond_does_not_overwrite_axis(self):
        for cs in [1.399, 2.201]:
            c = self.synthetic(cs=cs)
            m, g = evaluate_mechanism(c, self.case), evaluate_sam_checks(c, self.case)
            self.assertEqual(m['status'], 'PASS')
            self.assertEqual(m['legacy_three_condition_status'], 'FAIL')
            self.assertEqual(g['checks']['SAM.intact_methyl_C_S'], 'FAIL')
        for cs in [1.4, 2.2]:
            self.assertEqual(evaluate_sam_checks(self.synthetic(cs=cs), self.case)['checks']['SAM.intact_methyl_C_S'], 'PASS')

    def test_degenerate_axis_is_serializable(self):
        r = evaluate_mechanism(self.synthetic(d=0.), self.case)
        self.assertEqual(r['status'], 'FAIL')
        self.assertIn('DEGENERATE_REACTION_AXIS', r['failure_tags'])
        json.dumps(r, allow_nan=False)

    def test_far_partners_not_named_as_a_specific_pocket(self):
        r = evaluate_mechanism(self.synthetic(d=25.), self.case)
        self.assertIn('REACTION_PARTNERS_SEPARATED', r['failure_tags'])
        self.assertNotIn('protein_chain_assignment', r)

    def test_corrected_axis_new_cif_general_stays_independent(self):
        r = self.variant(self.orient_glycine)
        self.assertEqual(r['mechanism_status'], 'PASS')
        self.assertEqual(r['general_status'], 'FAIL')
        self.assertFalse(r['historical_results_used'])

    def test_sulfur_chirality_flip_preserves_reaction_axis(self):
        def change(s):
            self.orient_glycine(s)
            atoms = {a.name: a for a in s[0]['E'][0]}
            sulfur = np.array(atoms['S1'].pos.tolist())
            c0 = np.array(atoms['C1'].pos.tolist())
            c9 = np.array(atoms['C6'].pos.tolist())
            normal = np.cross(c0-sulfur, c9-sulfur)
            normal /= np.linalg.norm(normal)
            point = np.array(atoms['C2'].pos.tolist())
            atoms['C2'].pos = gemmi.Position(*(point - 2*np.dot(point-sulfur, normal)*normal))
        r = self.variant(change)
        self.assertEqual(r['mechanism_status'], 'PASS')
        self.assertIn('INPUT_STEREOCHEMISTRY_MISMATCH', r['general']['failure_tags'])
        center = r['general']['case_specific_checks']['stereochemistry']['centers'][0]
        self.assertEqual((center['input_CIP'], center['observed_CIP']), ('S', 'R'))

    def test_sam_CIP_name_correction_does_not_invert_input(self):
        r = evaluate_sam_checks(self.context, self.case)
        center = next(c for c in r['stereochemistry']['centers'] if c['atom']['element']=='S')
        self.assertEqual((center['legacy_input_label'], center['input_CIP']), ('R', 'S'))
        self.assertTrue(center['matches'])

    def test_nonpolymer_GLY_is_free_ligand(self):
        def change(s):
            s[0]['F'][0].name = 'GLY'
            s[0]['F'][0].entity_type = gemmi.EntityType.NonPolymer
        r = self.variant(change, 'all')
        self.assertEqual(r['input_status'], 'PASS')
        self.assertEqual(r['mechanism_status'], 'FAIL')
        self.assertEqual(r['atom_mapping']['glycine']['residue_name'], 'GLY')
        self.assertEqual(r['general_status'], 'FAIL')
        self.assertEqual(len(r['general']['components']), 2)

    def test_GLY_within_protein_not_removed(self):
        expected_gly = self.case['protein_sequence'].count('G') * 4
        observed_gly = sum(r['name']=='GLY' for rr in self.context['protein_chains'].values() for r in rr)
        self.assertEqual(observed_gly, expected_gly)

    def test_unknown_names_need_explicit_map(self):
        mapping = {role: {k: v for k, v in data.items() if k != 'mapping_method'}
                   for role, data in self.context['atom_mapping'].items()}
        for role in mapping:
            mapping[role]['atom_names'] = ['X_'+name for name in mapping[role]['atom_names']]
        def change(s):
            for ch in ['E', 'F']:
                for a in s[0][ch][0]:
                    a.name = 'X_'+a.name
        missing = self.variant(change, 'mechanism')
        self.assertIn('ATOM_MAPPING_REQUIRED', missing['input_problem'])
        mapped = self.variant(change, 'mechanism', mapping)
        self.assertEqual(mapped['input_status'], 'PASS')
        self.assertAlmostEqual(mapped['mechanism']['measurements']['N_C_distance_A'], 3.0784640650818065)

    def test_chain_labels_atom_order_and_numbers_not_pose_selection(self):
        def change(s):
            for chain in s[0]:
                chain.name = 'X'+str(100-ord(chain.name))
                for i, res in enumerate(chain):
                    res.seqid = gemmi.SeqId(i+500, ' ')
                    atoms = list(res.clone())
                    for j in range(len(res)-1, -1, -1): del res[j]
                    for a in reversed(atoms): res.add_atom(a)
        r = self.variant(change, 'mechanism')
        self.assertEqual(r['input_status'], 'PASS')
        self.assertAlmostEqual(r['mechanism']['measurements']['N_C_S_angle_deg'], 97.3257941736836)

    def test_no_missing_water_requirement(self):
        self.assertEqual(self.context['ignored_water_count'], 0)
        self.assertFalse(evaluate_mechanism(self.context, self.case)['water_required'])

    def test_missing_partner_not_silently_evaluated(self):
        r = self.variant(lambda s: s[0].remove_chain('F'))
        self.assertIn('LIGAND_COMPOSITION_OR_COPY_MISMATCH', r['input_problem'])
        self.assertEqual(r['mechanism_status'], 'ERROR')

    def test_extra_partner_cannot_be_cherry_picked(self):
        def change(s):
            other = s[0]['F'].clone()
            other.name = 'EXTRA'
            s[0].add_chain(other)
        r = self.variant(change)
        self.assertIn('LIGAND_COMPOSITION_OR_COPY_MISMATCH', r['input_problem'])

    def test_stage_isolation(self):
        with patch.object(API, 'run_general', side_effect=AssertionError('not requested')):
            r = evaluate(FIXTURE, '1XVA', 'mechanism')
            self.assertEqual(r['general_status'], 'NOT_RUN')
        with patch.dict(API.MECHANISMS, {'sam_n_methyl': lambda *_: (_ for _ in ()).throw(AssertionError('not requested'))}):
            r = evaluate(FIXTURE, '1XVA', 'general')
            self.assertEqual(r['mechanism_status'], 'NOT_RUN')


if __name__ == '__main__':
    unittest.main()
