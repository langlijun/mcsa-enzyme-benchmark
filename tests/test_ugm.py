"""UGM task isolation and correspondence tests; perturbations are not experiments."""

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
from mcsa_benchmark.mechanisms.ugm_n5 import evaluate_mechanism
from mcsa_benchmark.structure import read_case
from mcsa_benchmark.ugm_general import evaluate_ugm_checks

FIXTURE=os.environ.get('MCSA_1I8T_TEST_CIF')
API=importlib.import_module('mcsa_benchmark.evaluate')


class UGMDefinitionTests(unittest.TestCase):
    def test_reactive_atoms_and_declared_microstate(self):
        case=load_case('1I8T')
        fad=Chem.MolFromSmiles(case['ligands']['FAD_1']['smiles'])
        udp=Chem.MolFromSmiles(case['ligands']['UDP_Galp_1']['smiles'])
        self.assertEqual((fad.GetAtomWithIdx(52).GetSymbol(),fad.GetAtomWithIdx(52).GetTotalNumHs()),('N',1))
        self.assertEqual((fad.GetAtomWithIdx(45).GetFormalCharge(),fad.GetAtomWithIdx(45).GetTotalNumHs()),(0,1))
        self.assertEqual(Chem.GetFormalCharge(fad),-2)
        self.assertIsNotNone(udp.GetBondBetweenAtoms(18,17))
        self.assertTrue(any(a.GetSymbol()=='P' for a in udp.GetAtomWithIdx(17).GetNeighbors()))
        self.assertFalse(case['mechanism_criteria']['auxiliary']['affects_mechanism_status'])
        self.assertTrue(case['mechanism_criteria']['retrospective'])


@unittest.skipUnless(FIXTURE,'Set MCSA_1I8T_TEST_CIF to AF3 s2')
class UGMTests(unittest.TestCase):
    def setUp(self):
        self.case=load_case('1I8T')
        self.context=read_case(FIXTURE,self.case)

    def variant(self,change,stage='all',mapping=None):
        s=gemmi.read_structure(FIXTURE)
        change(s)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'new.cif'
            s.make_mmcif_document().write_file(str(p))
            r=evaluate(p,'1I8T',stage,mapping)
            json.dumps(r,allow_nan=False)
            return r

    def geometry(self,d=3.5,degrees=155.):
        context=copy.deepcopy(self.context)
        for fr,ur in [('FAD_1','UDP_Galp_1'),('FAD_2','UDP_Galp_2')]:
            c=context['ligands'][ur][18]['xyz']
            o=context['ligands'][ur][17]['xyz']
            u=(o-c)/np.linalg.norm(o-c)
            trial=np.array([1.,0.,0.]) if abs(u[0])<0.9 else np.array([0.,1.,0.])
            v=np.cross(u,trial);v/=np.linalg.norm(v)
            theta=math.radians(degrees)
            context['ligands'][fr][52]['xyz']=c+d*(math.cos(theta)*u+math.sin(theta)*v)
        return context

    def test_baseline_axis_pass_auxiliary_fail(self):
        r=evaluate(FIXTURE,'1I8T')
        self.assertEqual((r['mechanism_status'],r['general_status']),('PASS','PASS'))
        self.assertEqual(r['mechanism']['passing_sites'],2)
        self.assertEqual(r['mechanism']['legacy_six_check_status'],'FAIL')
        self.assertTrue(all(a['Arg_support']['status']=='FAIL' for a in r['mechanism']['auxiliary_contacts']))
        self.assertEqual(r['mechanism']['failure_tags'],[])

    def test_axis_boundaries(self):
        for d,a in [(2.5,135.),(4.,180.),(3.5,155.)]:
            self.assertEqual(evaluate_mechanism(self.geometry(d,a),self.case)['status'],'PASS')
        for d,a in [(2.499,155.),(4.001,155.),(3.5,134.99)]:
            self.assertEqual(evaluate_mechanism(self.geometry(d,a),self.case)['status'],'FAIL')

    def test_one_failed_site_cannot_hide(self):
        context=self.geometry()
        n=context['ligands']['FAD_1'][52]
        c=context['ligands']['UDP_Galp_1'][18]['xyz']
        n['xyz']=c+2*(n['xyz']-c)
        r=evaluate_mechanism(context,self.case)
        self.assertEqual((r['status'],r['passing_sites']),('FAIL',1))
        self.assertIn('REACTION_DISTANCE_TOO_LONG',r['failure_tags'])

    def test_moving_scored_N5_does_not_select_new_pairing(self):
        original=evaluate_mechanism(self.context,self.case)
        context=self.geometry(d=15.)
        moved=evaluate_mechanism(context,self.case)
        self.assertEqual(original['site_assignment'],moved['site_assignment'])

    def test_auxiliary_residue_missing_does_not_erase_axis(self):
        r=self.variant(lambda s:s[0]['A'][169].remove_atom('NH1','\x00'))
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['general_status'],'FAIL')
        a=next(a for a in r['mechanism']['auxiliary_contacts'] if a['protein_chain']=='A')
        self.assertEqual(a['Arg_support']['status'],'UNAVAILABLE')
        self.assertIn('A:170:NH1',a['Arg_support']['missing'])

    def test_missing_site_anchor_general_still_runs(self):
        r=self.variant(lambda s:s[0]['A'][173].remove_atom('CA','\x00'))
        self.assertEqual(r['mechanism_status'],'ERROR')
        self.assertIn('MECHANISM_SITE_ANCHOR_MISSING',r['mechanism']['failure_tags'])
        self.assertTrue(r['general']['recomputed'])

    def test_bond_guard_is_general_only(self):
        def change(s):
            c=np.array(s[0]['D'][0].find_atom('C7','\x00').pos.tolist())
            o=np.array(s[0]['D'][0].find_atom('O9','\x00').pos.tolist())
            s[0]['D'][0].find_atom('O9','\x00').pos=gemmi.Position(*(c+2.*(o-c)/np.linalg.norm(o-c)))
        r=self.variant(change)
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['general_status'],'FAIL')
        self.assertIn('UDP_GLYCOSIDIC_BOND_OUT_OF_RANGE',r['general']['failure_tags'])

    def test_bond_bounds(self):
        for d,expected in [(1.2,'PASS'),(1.7,'PASS'),(1.199,'FAIL'),(1.701,'FAIL')]:
            context=copy.deepcopy(self.context)
            c=context['ligands']['UDP_Galp_1'][18]['xyz']
            context['ligands']['UDP_Galp_1'][17]['xyz']=c+np.array([d,0.,0.])
            g=evaluate_ugm_checks(context,self.case)
            self.assertEqual(g['checks']['UDP_Galp_1.intact_C1_OUDP'],expected)

    def test_duplicate_occupancy_is_failure_not_forced_pairing(self):
        context=copy.deepcopy(self.context)
        for a,b in zip(context['ligands']['FAD_2'],context['ligands']['FAD_1']):a['xyz']=b['xyz'].copy()
        r=evaluate_mechanism(context,self.case)
        self.assertEqual(r['status'],'FAIL')
        self.assertIn('REQUIRED_POCKET_UNOCCUPIED',r['failure_tags'])
        self.assertIsNone(r['site_assignment']['FAD']['selected_assignment'])

    def test_symmetric_anchor_tie_is_explicit(self):
        context=copy.deepcopy(self.context)
        poses=self.case['reaction_atoms']['pocket_sequence_positions']
        for pos in poses:
            ca_a=next(a for a in context['protein_chains']['A'][pos-1]['atoms'] if a['atom_name']=='CA')
            ca_b=next(a for a in context['protein_chains']['B'][pos-1]['atoms'] if a['atom_name']=='CA')
            ca_b['xyz']=ca_a['xyz'].copy()
        r=evaluate_mechanism(context,self.case)
        self.assertEqual(r['status'],'ERROR')
        self.assertIn('LIGAND_POCKET_ASSIGNMENT_TIED',r['failure_tags'])

    def test_new_CIF_axis_pass_general_can_fail(self):
        context=self.geometry(d=3.)
        def change(s):
            for role in ['FAD_1','FAD_2']:
                a=context['ligands'][role][52]
                s[0][a['chain']][0].find_atom(a['atom_name'],'\x00').pos=gemmi.Position(*a['xyz'])
        r=self.variant(change)
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['general_status'],'FAIL')
        self.assertFalse(r['historical_results_used'])

    def test_chain_renaming_and_author_numbering(self):
        def change(s):
            for chain in s[0]:
                chain.name='X'+str(100-ord(chain.name))
                for i,res in enumerate(chain):
                    res.seqid=gemmi.SeqId(500+i,' ')
                    atoms=list(res.clone())
                    for j in range(len(res)-1,-1,-1):del res[j]
                    for atom in reversed(atoms):res.add_atom(atom)
        r=self.variant(change,'mechanism')
        self.assertEqual(r['mechanism_status'],'PASS')
        a=sorted(s['N5_C1_A'] for s in r['mechanism']['sites'])
        b=sorted(s['N5_C1_A'] for s in evaluate_mechanism(self.context,self.case)['sites'])
        np.testing.assert_allclose(a,b,atol=1e-10)

    def test_unknown_names_require_crosswalk(self):
        mapping={role:{k:v for k,v in row.items() if k!='mapping_method'} for role,row in self.context['atom_mapping'].items()}
        for row in mapping.values():row['atom_names']=['NEW_'+n for n in row['atom_names']]
        def change(s):
            for ch in ['C','D','E','F']:
                for a in s[0][ch][0]:a.name='NEW_'+a.name
        missing=self.variant(change,'mechanism')
        self.assertIn('ATOM_MAPPING_REQUIRED',missing['input_problem'])
        good=self.variant(change,'mechanism',mapping)
        self.assertEqual(good['mechanism_status'],'PASS')

    def test_generic_N5_name_not_used_as_reactive_identity(self):
        self.assertEqual(self.context['ligands']['FAD_1'][52]['atom_name'],'N9')
        self.assertNotEqual(self.context['ligands']['FAD_1'][52]['atom_name'],'N5')

    def test_missing_component_not_silently_dropped(self):
        r=self.variant(lambda s:s[0].remove_chain('E'))
        self.assertEqual(r['input_status'],'FAIL')
        self.assertIn('LIGAND_COMPOSITION_OR_COPY_MISMATCH',r['input_problem'])

    def test_stage_isolation(self):
        with patch.object(API,'run_general',side_effect=AssertionError('not requested')):
            r=evaluate(FIXTURE,'1I8T','mechanism')
            self.assertEqual(r['general_status'],'NOT_RUN')
        with patch.dict(API.MECHANISMS,{'ugm_n5':lambda *_:(_ for _ in ()).throw(AssertionError('not requested'))}):
            r=evaluate(FIXTURE,'1I8T','general')
            self.assertEqual(r['mechanism_status'],'NOT_RUN')

    def test_microstate_disclosed_not_inferred_from_CIF(self):
        r=evaluate(FIXTURE,'1I8T','mechanism')
        self.assertFalse(r['input_state_contract']['observed_electronic_or_protonation_state_verified'])
        self.assertFalse(r['input_state_contract']['charged_state_difference_is_model_failure'])
        self.assertEqual(r['input_state_contract']['FAD_N1_formal_charge'],0)
        self.assertFalse(r['mechanism']['water_required'])


if __name__=='__main__':
    unittest.main()
