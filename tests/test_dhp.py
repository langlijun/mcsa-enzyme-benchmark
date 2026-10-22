"""Four-copy tests. Constructed positives test code, not biochemical plausibility."""

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
from mcsa_benchmark.mechanisms.lys_carbonyl_multisite import evaluate_mechanism
from mcsa_benchmark.structure import read_case

FIXTURE = os.environ.get('MCSA_1DHP_TEST_CIF')
API = importlib.import_module('mcsa_benchmark.evaluate')


class DHPDefinitionTests(unittest.TestCase):
    def test_correct_ketone_not_carboxylate(self):
        case = load_case('1DHP')
        role = case['reaction_atoms']
        mol = Chem.MolFromSmiles(case['ligands']['PYR_1']['smiles'])
        c, o = role['carbonyl_C_index'], role['carbonyl_O_index']
        self.assertEqual(sum(n.GetSymbol() == 'C' for n in mol.GetAtomWithIdx(c).GetNeighbors()), 2)
        self.assertEqual(mol.GetBondBetweenAtoms(c, o).GetBondType(), Chem.BondType.DOUBLE)
        self.assertNotIn(c, role['nonreactive_indices'])
        self.assertNotIn(o, role['nonreactive_indices'])
        self.assertTrue(case['mechanism_criteria']['retrospective'])
        self.assertIn('author operational', case['mechanism_criteria']['main_axis']['basis'])


@unittest.skipUnless(FIXTURE, 'Set MCSA_1DHP_TEST_CIF to AF3 s0')
class DHPTests(unittest.TestCase):
    def setUp(self):
        self.case = load_case('1DHP')
        self.context = read_case(FIXTURE, self.case)

    def positive(self, d=3., theta=105.):
        context = copy.deepcopy(self.context)
        for i, (chain, residues) in enumerate(sorted(context['protein_chains'].items())):
            base = np.array([100.*i, 0., 0.])
            lys = {a['atom_name']: a for a in residues[160]['atoms']}
            for name in ('CA', 'CB'):
                lys[name]['xyz'] = base.copy()
            rad = math.radians(theta)
            lys['NZ']['xyz'] = base + [d*math.cos(rad), d*math.sin(rad), 0.]
            ligand = context['ligands'][f'PYR_{i+1}']
            for atom in ligand:
                atom['xyz'] = base.copy()
            for index, offset in [(0,[-1.5,0.,0.]),(3,[1.5,0.,0.]),(4,[2.,1.,0.]),(5,[-2.,-1.,0.])]:
                ligand[index]['xyz'] = base + offset
            ligand[2]['xyz'] = base + [1.2, 0., 0.]
        return context

    def variant(self, change, stage='all', mapping=None):
        s = gemmi.read_structure(FIXTURE)
        change(s)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'new.cif'
            s.make_mmcif_document().write_file(str(path))
            result = evaluate(path, '1DHP', stage, mapping)
            json.dumps(result, allow_nan=False)
            return result

    def test_baseline_and_complete_enumeration(self):
        r = evaluate(FIXTURE, '1DHP')
        self.assertEqual((r['mechanism_status'], r['general_status']), ('FAIL', 'FAIL'))
        self.assertEqual(len(r['mechanism']['selected_axes']), 4)
        self.assertEqual(len(r['mechanism']['all_candidate_axes']), 16)
        self.assertEqual(len(r['mechanism']['site_assignment']['options']), 24)
        self.assertEqual(len(r['general']['components']), 4)

    def test_positive_and_inclusive_boundaries(self):
        for d, theta in [(3.,105.),(2.5,90.),(4.,120.)]:
            r = evaluate_mechanism(self.positive(d,theta),self.case)
            self.assertEqual(r['status'],'PASS')
            self.assertEqual(r['passing_site_count'],4)

    def test_one_failed_site_fails_sample(self):
        context = self.positive()
        nz = next(a for a in context['protein_chains']['A'][160]['atoms'] if a['atom_name']=='NZ')
        nz['xyz'] *= 3
        r = evaluate_mechanism(context,self.case)
        self.assertEqual(r['status'],'FAIL')
        self.assertEqual(r['passing_site_count'],3)
        self.assertEqual(r['site_assignment']['selected_assignment']['PYR_1'],'A')

    def test_short_distance_and_direction_reasons(self):
        for d, theta, tag in [(2.499,105.,'REACTION_DISTANCE_TOO_SHORT_FOR_REACTANT_TASK'),
                               (3.,89.9,'ATTACK_DIRECTION_MISMATCH'),(4.001,105.,'REACTION_DISTANCE_TOO_LONG')]:
            r=evaluate_mechanism(self.positive(d,theta),self.case)
            self.assertIn(tag,r['failure_tags'])

    def test_equal_cost_different_outcomes_not_picked_by_score(self):
        context=self.positive()
        for a in context['protein_chains']['B'][160]['atoms']:
            if a['atom_name'] in ('CA','CB'):
                a['xyz']=np.zeros(3)
        r=evaluate_mechanism(context,self.case)
        self.assertEqual(r['status'],'ERROR')
        self.assertIn('SITE_ASSIGNMENT_CHANGES_VERDICT',r['failure_tags'])

    def test_missing_ligand_not_dropped(self):
        r=self.variant(lambda s:s[0].remove_chain('E'))
        self.assertEqual(r['input_status'],'FAIL')
        self.assertIn('LIGAND_COMPOSITION_OR_COPY_MISMATCH',r['input_problem'])

    def test_missing_nz_general_still_runs(self):
        r=self.variant(lambda s:s[0]['A'][160].remove_atom('NZ','\x00'))
        self.assertEqual(r['mechanism_status'],'ERROR')
        self.assertTrue(r['general']['recomputed'])

    def test_renamed_reordered_atoms_and_numbering(self):
        def change(s):
            for chain in s[0]:
                chain.name='NEW_'+str(100-ord(chain.name))
                for i,res in enumerate(chain):
                    res.seqid=gemmi.SeqId(i+500,' ')
                    atoms=list(res.clone())
                    for j in range(len(res)-1,-1,-1):del res[j]
                    for a in reversed(atoms):res.add_atom(a)
        r=self.variant(change,'mechanism')
        self.assertEqual(r['mechanism_status'],'FAIL')
        expected=sorted(x['distance_A'] for x in evaluate_mechanism(self.context,self.case)['selected_axes'])
        actual=sorted(x['distance_A'] for x in r['mechanism']['selected_axes'])
        np.testing.assert_allclose(actual,expected,atol=1e-10)

    def test_all_copy_explicit_mapping_roundtrip(self):
        r=evaluate(FIXTURE,'1DHP','mechanism')
        mapping={role:{k:v for k,v in d.items() if k!='mapping_method'} for role,d in r['atom_mapping'].items()}
        other=evaluate(FIXTURE,'1DHP','mechanism',mapping)
        self.assertEqual(r['mechanism'],other['mechanism'])
        mapping.pop('PYR_4')
        bad=evaluate(FIXTURE,'1DHP','mechanism',mapping)
        self.assertIn('INVALID_ATOM_MAP',bad['input_problem'])

    def test_independent_stages(self):
        with patch.object(API,'run_general',side_effect=AssertionError('not requested')):
            r=evaluate(FIXTURE,'1DHP','mechanism')
            self.assertEqual(r['general_status'],'NOT_RUN')
        with patch.dict(API.MECHANISMS,{'lys_carbonyl_multisite':lambda *_:(_ for _ in ()).throw(AssertionError('not requested'))}):
            r=evaluate(FIXTURE,'1DHP','general')
            self.assertEqual(r['mechanism_status'],'NOT_RUN')

    def test_nonfinite_general_output_isolated(self):
        with patch.object(API,'run_general',return_value={'status':'FAIL','measurement':float('inf')}):
            r=evaluate(FIXTURE,'1DHP')
        self.assertEqual(r['mechanism_status'],'FAIL')
        self.assertEqual(r['general_status'],'ERROR')
        self.assertIn('NONFINITE_MEASUREMENT',r['general']['failure_tags'])
        json.dumps(r,allow_nan=False)

    def test_new_cif_can_pass_mechanism_but_fail_general(self):
        context=self.positive()
        def change(s):
            for chain,residues in context['protein_chains'].items():
                for a in residues[160]['atoms']:
                    if a['atom_name'] in ('CA','CB','NZ'):
                        s[0][chain][160].find_atom(a['atom_name'],'\x00').pos=gemmi.Position(*a['xyz'])
            for atoms in context['ligands'].values():
                for a in atoms:
                    s[0][a['chain']][0].find_atom(a['atom_name'],'\x00').pos=gemmi.Position(*a['xyz'])
        r=self.variant(change)
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['general_status'],'FAIL')
        self.assertFalse(r['historical_results_used'])


if __name__=='__main__':
    unittest.main()
