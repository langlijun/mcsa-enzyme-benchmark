"""PSP task logic and real-CIF replay; synthetic geometries are software tests."""
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
from mcsa_benchmark.structure import read_case
from mcsa_benchmark.mechanisms.phosphoserine_phosphatase import evaluate_mechanism, bound

FIXTURE=os.environ.get('MCSA_1L7N_TEST_CIF')
API=importlib.import_module('mcsa_benchmark.evaluate')


class DefinitionTests(unittest.TestCase):
    def test_graph_and_residue_roles(self):
        c=load_case('1L7N');m=Chem.MolFromSmiles(c['ligands']['PLS']['smiles'])
        self.assertEqual(len(c['protein_sequence']),211)
        self.assertEqual([c['protein_sequence'][i-1] for i in [11,13,167]],['D']*3)
        self.assertEqual(m.GetAtomWithIdx(1).GetProp('_CIPCode'),'S')
        self.assertEqual(m.GetAtomWithIdx(4).GetSymbol(),'P')
        self.assertEqual(sorted(a.GetIdx() for a in m.GetAtomWithIdx(3).GetNeighbors()),[2,4])
        self.assertEqual(c['metals'],{'MG':1})

    def test_scope_and_threshold_provenance(self):
        c=load_case('1L7N');k=c['mechanism_criteria']
        self.assertEqual(k['attack_distance_max_A'],3.5)
        self.assertEqual(k['attack_angle_min_deg'],150)
        self.assertEqual(k['acid_contact_max_A'],3.5)
        self.assertEqual(k['Mg_contact_max_A'],2.8)
        self.assertFalse(k['distance_lower_bounds_in_mechanism'])
        self.assertFalse(k['water_required'])
        self.assertFalse(k['Asp171_direct_coordination_required'])
        self.assertTrue(k['retrospective'])
        self.assertNotIn('general_metal_policy',c)
        self.assertEqual(c['general_excluded_metals'],['MG'])
        self.assertTrue(all('Author' in k['threshold_provenance'][n] for n in ['attack_distance','attack_angle','acid_contact','Mg_contact']))

    def test_inclusive_numerical_boundaries(self):
        for maximum in [3.5,2.8]:
            self.assertTrue(bound(maximum,maximum));self.assertFalse(bound(maximum+1e-5,maximum))
        self.assertTrue(bound(150,150,'min'));self.assertFalse(bound(149.999,150,'min'))
        self.assertFalse(bound(None,150,'min'))


@unittest.skipUnless(FIXTURE,'Set MCSA_1L7N_TEST_CIF to AF3 sample0')
class PSPTests(unittest.TestCase):
    def setUp(self):
        self.case=load_case('1L7N');self.context=read_case(FIXTURE,self.case)

    def atom(self,ctx,position,name):
        return next(a for a in ctx['protein_chains']['A'][position-1]['atoms'] if a['atom_name']==name)

    def variant(self,change,stage='all',mapping=None):
        s=gemmi.read_structure(FIXTURE);change(s)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'new.cif';s.make_mmcif_document().write_file(str(path))
            r=evaluate(path,'1L7N',stage,atom_map=mapping)
            json.dumps(r,allow_nan=False)
            return r

    def test_current_sample_has_coupled_geometry(self):
        r=evaluate(FIXTURE,'1L7N');m=r['mechanism']
        self.assertEqual(m['status'],'PASS')
        self.assertEqual(len(m['variants']),2)
        self.assertAlmostEqual(m['selected_variant']['attack_contact']['distance_A'],2.9793276087063667)
        self.assertEqual(m['selected_variant']['attack_oxygen']['atom_name'],'OD1')
        self.assertEqual(m['selected_variant']['coupled_Asp11_Mg_oxygen']['atom_name'],'OD2')
        self.assertFalse(r['historical_results_used'])

    def test_oxygen_name_permutations(self):
        def change(s):
            for n in [11,13,167]:
                for a in s[0]['A'][n-1]:
                    if a.name in ['OD1','OD2']:a.name={'OD1':'OD2','OD2':'OD1'}[a.name]
        r=self.variant(change,'mechanism')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['mechanism']['selected_variant']['attack_oxygen']['atom_name'],'OD2')

    def test_cannot_mix_two_Asp11_variants(self):
        c=copy.deepcopy(self.context);l=c['ligands']['PLS'];mg=c['metal_copies']['MG'][0]
        l[4]['xyz']=np.array([0.,0,0]);l[3]['xyz']=np.array([1.6,0,0]);mg['xyz']=np.array([-2.,1,0])
        self.atom(c,11,'OD1')['xyz']=np.array([-3.,0,0])
        self.atom(c,11,'OD2')['xyz']=np.array([-2.,5,0])
        self.atom(c,13,'O')['xyz']=mg['xyz']+[0,0,2.1]
        self.atom(c,13,'OD1')['xyz']=l[3]['xyz']+[0,2.8,0]
        self.atom(c,167,'OD1')['xyz']=mg['xyz']+[0,-2,0]
        l[5]['xyz']=mg['xyz']+[2,0,0]
        r=evaluate_mechanism(c,self.case)
        self.assertTrue(any(v['axis_status']=='PASS' for v in r['variants']))
        self.assertTrue(any(v['Mg_organization_status']=='PASS' for v in r['variants']))
        self.assertEqual(r['status'],'FAIL')
        self.assertIn('NO_COHERENT_FIRST_STEP_VARIANT',r['failure_tags'])

    def test_acid_contact_is_separate(self):
        c=copy.deepcopy(self.context)
        for n in ['OD1','OD2']:self.atom(c,13,n)['xyz']+=100
        r=evaluate_mechanism(c,self.case)
        self.assertEqual(r['axis_any_variant_status'],'PASS')
        self.assertEqual(r['status'],'FAIL')
        self.assertIn('ASP13_LEAVING_GROUP_CONTACT_TOO_LONG',r['failure_tags'])

    def test_backbone_not_sidechain_is_Mg_donor(self):
        c=copy.deepcopy(self.context);self.atom(c,13,'O')['xyz']+=100
        r=evaluate_mechanism(c,self.case)
        self.assertEqual(r['acid_contact']['status'],'PASS')
        self.assertEqual(r['status'],'FAIL')
        self.assertIn('MG_ASP13_BACKBONE_O_TOO_LONG',r['failure_tags'])

    def test_Asp167_not_Asp171_is_required(self):
        c=copy.deepcopy(self.context)
        for n in ['OD1','OD2']:self.atom(c,171,n)['xyz']+=100
        self.assertEqual(evaluate_mechanism(c,self.case)['status'],'PASS')
        for n in ['OD1','OD2']:self.atom(c,167,n)['xyz']+=100
        self.assertEqual(evaluate_mechanism(c,self.case)['status'],'FAIL')

    def test_missing_diagnostic_Asp171_oxygen(self):
        r=self.variant(lambda s:s[0]['A'][170].remove_atom('OD1','\x00'))
        self.assertEqual(r['mechanism_status'],'PASS');self.assertEqual(r['general_status'],'FAIL')

    def test_Mg_position_does_not_change_general(self):
        # Compare identical CIF serialization, which omits original SMILES metadata.
        baseline=self.variant(lambda s:None)
        def change(s):
            oxygen=s[0]['A'][19].find_atom('OE1','\x00').pos
            s[0]['C'][0][0].pos=gemmi.Position(oxygen.x+1.6,oxygen.y,oxygen.z)
        r=self.variant(change)
        self.assertEqual(r['general'],baseline['general'])
        self.assertNotIn('F05_METAL_COORDINATION_TOO_SHORT',r['general']['failure_tags'])
        self.assertLessEqual(r['mechanism']['Mg_minimum_nonwater_O_contacts_diagnostic'][0]['distance_A'],1.600001)

    def test_general_has_no_active_Mg_lower_bound(self):
        r=evaluate(FIXTURE,'1L7N')
        self.assertEqual(r['general']['metal_coordination_check']['status'],'NOT_APPLICABLE')
        self.assertEqual(r['general']['excluded_metals'],['MG'])
        self.assertNotIn('F05_METAL_COORDINATION_TOO_SHORT',r['general']['failure_tags'])

    def test_missing_Mg_preserves_axis_and_is_failure(self):
        r=self.variant(lambda s:s[0].remove_chain('C'))
        self.assertEqual(r['mechanism_status'],'FAIL')
        self.assertEqual(r['mechanism']['axis_any_variant_status'],'PASS')
        self.assertIn('REQUIRED_MG_COUNT_MISMATCH',r['mechanism']['failure_tags'])
        self.assertEqual(r['general_status'],evaluate(FIXTURE,'1L7N')['general_status'])
        self.assertIn('METAL_COMPOSITION_OR_COPY_MISMATCH',[i['tag'] for i in r['general']['excluded_metal_input_diagnostics']])

    def test_extra_Mg_is_not_chosen_for_best_score(self):
        def change(s):
            ch=s[0]['C'].clone();ch.name='EXTRA';s[0].add_chain(ch)
        r=self.variant(change)
        self.assertIn('REQUIRED_MG_COUNT_MISMATCH',r['mechanism']['failure_tags'])
        self.assertEqual(r['general_status'],evaluate(FIXTURE,'1L7N')['general_status'])

    def test_missing_required_sidechain_atom(self):
        r=self.variant(lambda s:s[0]['A'][10].remove_atom('OD1','\x00'))
        self.assertEqual(r['mechanism_status'],'ERROR');self.assertEqual(r['general_status'],'FAIL')

    def test_missing_substrate(self):
        r=self.variant(lambda s:s[0].remove_chain('B'))
        self.assertEqual(r['input_status'],'FAIL')

    def test_degenerate_axis_is_not_nan(self):
        c=copy.deepcopy(self.context)
        for n in ['OD1','OD2']:self.atom(c,11,n)['xyz']=c['ligands']['PLS'][4]['xyz'].copy()
        r=evaluate_mechanism(c,self.case)
        self.assertEqual(r['status'],'FAIL');self.assertIn('DEGENERATE_REACTION_AXIS',r['failure_tags'])
        json.dumps(r,allow_nan=False)

    def test_renumber_rename_and_reorder(self):
        def change(s):
            for ch in s[0]:
                ch.name='NEW_'+ch.name
                for i,r in enumerate(ch):
                    r.seqid=gemmi.SeqId(1000+i,' ');aa=list(r.clone())
                    for j in range(len(r)-1,-1,-1):del r[j]
                    for a in reversed(aa):r.add_atom(a)
        self.assertEqual(self.variant(change,'mechanism')['mechanism_status'],'PASS')

    def test_custom_atom_map(self):
        mapping={k:{n:v for n,v in x.items() if n!='mapping_method'} for k,x in evaluate(FIXTURE,'1L7N','mechanism')['atom_mapping'].items()}
        for x in mapping.values():x['atom_names']=['NEW_'+n for n in x['atom_names']]
        def change(s):
            for a in s[0]['B'][0]:a.name='NEW_'+a.name
        self.assertEqual(self.variant(change,'mechanism')['mechanism_status'],'ERROR')
        self.assertEqual(self.variant(change,'mechanism',mapping)['mechanism_status'],'PASS')

    def test_stages_independent(self):
        with patch.object(API,'run_general',side_effect=ValueError('synthetic')):
            r=evaluate(FIXTURE,'1L7N')
        self.assertEqual(r['mechanism_status'],'PASS');self.assertEqual(r['general_status'],'ERROR')
        with patch.object(API,'run_general',side_effect=AssertionError('not requested')):
            self.assertEqual(evaluate(FIXTURE,'1L7N','mechanism')['general_status'],'NOT_RUN')
        with patch.dict(API.MECHANISMS,{'phosphoserine_phosphatase':lambda *_: (_ for _ in ()).throw(AssertionError('not requested'))}):
            self.assertEqual(evaluate(FIXTURE,'1L7N','general')['mechanism_status'],'NOT_RUN')

    def test_stereo_failure_does_not_block_mechanism(self):
        path=Path(FIXTURE).with_name('M0727_1L7N__AF3__s1.cif')
        r=evaluate(path,'1L7N')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertIn('INPUT_STEREOCHEMISTRY_MISMATCH',r['general']['failure_tags'])
