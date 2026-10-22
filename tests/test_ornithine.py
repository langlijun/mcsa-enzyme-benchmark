"""1OTH software tests; synthetic perturbations are not biochemical evidence."""

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
from mcsa_benchmark.mechanisms.ornithine_carbamoyl import evaluate_mechanism

FIXTURE=os.environ.get('MCSA_1OTH_TEST_CIF')
API=importlib.import_module('mcsa_benchmark.evaluate')


class DefinitionTests(unittest.TestCase):
    def test_fixed_atom_roles_and_charge(self):
        case=load_case('1OTH')
        orn=Chem.MolFromSmiles(case['ligands']['ORN_1']['smiles'])
        cp=Chem.MolFromSmiles(case['ligands']['CP_1']['smiles'])
        self.assertEqual(orn.GetAtomWithIdx(0).GetFormalCharge(),1)
        self.assertEqual(orn.GetAtomWithIdx(0).GetDegree(),1)
        self.assertEqual(orn.GetAtomWithIdx(0).GetNeighbors()[0].GetIdx(),1)
        self.assertEqual(cp.GetAtomWithIdx(1).GetSymbol(),'C')
        self.assertEqual(cp.GetBondBetweenAtoms(1,2).GetBondType(),Chem.BondType.DOUBLE)
        self.assertEqual(cp.GetBondBetweenAtoms(1,3).GetBondType(),Chem.BondType.SINGLE)
        self.assertEqual(cp.GetAtomWithIdx(4).GetSymbol(),'P')

    def test_approved_scope_no_extra_contact_gate(self):
        c=load_case('1OTH')['mechanism_criteria']
        self.assertTrue(c['retrospective'])
        self.assertEqual(c['main_axis']['distance_A']['max'],4.0)
        self.assertEqual(c['main_axis']['angle_deg']['min'],90.0)
        self.assertFalse(c['water_required'])
        self.assertFalse(c['metal_required'])
        self.assertFalse(c['new_residue_contact_gates'])


@unittest.skipUnless(FIXTURE,'Set MCSA_1OTH_TEST_CIF to AF3 sample 0')
class OrnithineTests(unittest.TestCase):
    def setUp(self):
        self.case=load_case('1OTH')
        self.context=read_case(FIXTURE,self.case)
        first=evaluate_mechanism(self.context,self.case)['selected_axes'][0]
        self.orn_name,self.cp_name=first['ORN_instance'],first['CP_instance']

    def geometry(self,d,theta):
        c=copy.deepcopy(self.context)
        orn,cp=c['ligands'][self.orn_name],c['ligands'][self.cp_name]
        origin=cp[1]['xyz'];u=cp[2]['xyz']-origin;u/=np.linalg.norm(u)
        v=orn[0]['xyz']-origin;v-=np.dot(v,u)*u;v/=np.linalg.norm(v)
        angle=math.radians(theta)
        orn[0]['xyz']=origin+d*(math.cos(angle)*u+math.sin(angle)*v)
        return c

    def variant(self,change,stage='mechanism',mapping=None):
        s=gemmi.read_structure(FIXTURE);change(s)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'new.cif';s.make_mmcif_document().write_file(str(path))
            result=evaluate(path,'1OTH',stage,mapping)
            json.dumps(result,allow_nan=False)
            return result

    def test_baseline_stages_are_not_conflated(self):
        r=evaluate(FIXTURE,'1OTH')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['general_status'],'FAIL')
        self.assertEqual(r['mechanism']['passing_site_count'],3)
        self.assertEqual(len(r['mechanism']['all_candidate_axes']),9)
        self.assertIn('PROTEIN_LIGAND_VOLUME_OVERLAP',r['general']['failure_tags'])

    def test_distance_and_angle_boundaries(self):
        for d,theta in [(2.5,100),(4.0,100),(3.,90),(3.,120)]:
            with self.subTest(d=d,theta=theta):
                r=evaluate_mechanism(self.geometry(d,theta),self.case)
                self.assertEqual(r['status'],'PASS')
        for d,theta,tag in [(2.499,100,'REACTION_DISTANCE_TOO_SHORT_FOR_REACTANT_TASK'),
                            (4.001,100,'REACTION_DISTANCE_TOO_LONG'),
                            (3.,89.99,'ATTACK_DIRECTION_MISMATCH'),(3.,120.01,'ATTACK_DIRECTION_MISMATCH')]:
            with self.subTest(d=d,theta=theta):
                r=evaluate_mechanism(self.geometry(d,theta),self.case)
                self.assertEqual(r['status'],'FAIL')
                self.assertIn(tag,r['failure_tags'])

    def test_one_failed_site_is_not_hidden_by_two_passes(self):
        r=evaluate_mechanism(self.geometry(5.,100),self.case)
        self.assertEqual(r['status'],'FAIL')
        self.assertEqual(r['passing_site_count'],2)

    def test_carbonyl_oxygen_is_not_bridge_oxygen(self):
        r=evaluate_mechanism(self.context,self.case)
        for axis in r['selected_axes']:
            atoms=self.context['ligands'][axis['CP_instance']]
            self.assertEqual(axis['atoms'][2]['atom_name'],atoms[2]['atom_name'])
            self.assertNotEqual(axis['atoms'][2]['atom_name'],atoms[3]['atom_name'])

    def test_duplicate_ornithine_site_fails_coverage(self):
        def change(s):
            source={a.name:a.pos for a in s[0]['D'][0]}
            for atom in s[0]['E'][0]:atom.pos=source[atom.name]
        r=self.variant(change)
        self.assertEqual(r['mechanism_status'],'FAIL')
        self.assertIn('MULTIPLE_ORN_AT_SITE',r['mechanism']['failure_tags'])
        self.assertIn('ORN_SITE_UNOCCUPIED',r['mechanism']['failure_tags'])

    def test_missing_ligand_is_explicit_input_failure(self):
        r=self.variant(lambda s:s[0].remove_chain('D'))
        self.assertEqual(r['input_status'],'FAIL')
        self.assertIn('LIGAND_COMPOSITION_OR_COPY_MISMATCH',r['input_problem'])

    def test_rename_reorder_and_renumber(self):
        def change(s):
            for chain in s[0]:
                chain.name='NEW_'+chain.name
                for i,residue in enumerate(chain):
                    residue.seqid=gemmi.SeqId(900+i,' ')
                    atoms=list(residue.clone())
                    for j in range(len(residue)-1,-1,-1):del residue[j]
                    for atom in reversed(atoms):residue.add_atom(atom)
        r=self.variant(change)
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['mechanism']['passing_site_count'],3)

    def test_mapping_round_trip(self):
        r=evaluate(FIXTURE,'1OTH','mechanism')
        mapping={k:{n:v for n,v in row.items() if n!='mapping_method'} for k,row in r['atom_mapping'].items()}
        again=evaluate(FIXTURE,'1OTH','mechanism',mapping)
        self.assertEqual(r['mechanism'],again['mechanism'])

    def test_unknown_atom_names_require_explicit_mapping(self):
        r=self.variant(lambda s:setattr(s[0]['D'][0][0],'name','UNKNOWN'))
        self.assertEqual(r['mechanism_status'],'ERROR')
        self.assertIn('ATOM_MAPPING_REQUIRED',r['input_problem'])

    def test_no_cys_activation_gate(self):
        def change(s):
            for chain in ['A','B','C']:s[0][chain][269].remove_atom('SG','\x00')
        r=self.variant(change,stage='all')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['general_status'],'FAIL')
        self.assertIn('PROTEIN_HEAVY_ATOM_MISMATCH',r['general']['failure_tags'])

    def test_stereochemistry_is_general_not_axis(self):
        def change(s):
            atoms={a.name:a for a in s[0]['D'][0]}
            names=self.context['atom_mapping']['ORN_1']['atom_names']
            center=np.array(atoms[names[4]].pos.tolist())
            u=np.array(atoms[names[3]].pos.tolist())-center
            v=np.array(atoms[names[6]].pos.tolist())-center
            n=np.cross(u,v);n/=np.linalg.norm(n)
            x=np.array(atoms[names[5]].pos.tolist())
            atoms[names[5]].pos=gemmi.Position(*(x-2*np.dot(x-center,n)*n))
        r=self.variant(change,stage='all')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertIn('INPUT_STEREOCHEMISTRY_MISMATCH',r['general']['failure_tags'])

    def test_separate_stage_calls(self):
        with patch.object(API,'run_general',side_effect=AssertionError('not requested')):
            r=evaluate(FIXTURE,'1OTH','mechanism');self.assertEqual(r['general_status'],'NOT_RUN')
        with patch.dict(API.MECHANISMS,{'ornithine_carbamoyl':lambda *_:(_ for _ in ()).throw(AssertionError('not requested'))}):
            r=evaluate(FIXTURE,'1OTH','general');self.assertEqual(r['mechanism_status'],'NOT_RUN')

    def test_general_error_does_not_erase_axis(self):
        with patch.object(API,'run_general',side_effect=ValueError('test error')):
            r=evaluate(FIXTURE,'1OTH')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['general_status'],'ERROR')

    def test_zero_length_axis_is_flagged_not_nan(self):
        c=copy.deepcopy(self.context)
        c['ligands'][self.orn_name][0]['xyz']=c['ligands'][self.cp_name][1]['xyz'].copy()
        r=evaluate_mechanism(c,self.case)
        json.dumps(r,allow_nan=False)
        self.assertEqual(r['status'],'FAIL')
        self.assertIn('DEGENERATE_REACTION_AXIS',r['failure_tags'])


if __name__=='__main__':unittest.main()
