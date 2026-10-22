"""AANAT axis identity, auxiliary isolation, chemistry checks and new-CIF tests."""

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
from mcsa_benchmark.mechanisms.aanat_acyl import evaluate_mechanism
from mcsa_benchmark.aanat_general import evaluate_aanat_checks

FIXTURE=os.environ.get('MCSA_1B6B_TEST_CIF')
API=importlib.import_module('mcsa_benchmark.evaluate')


class AANATDefinitionTests(unittest.TestCase):
    def test_fixed_roles(self):
        case=load_case('1B6B')
        coa=Chem.MolFromSmiles(case['ligands']['acetyl_CoA']['smiles'])
        ser=Chem.MolFromSmiles(case['ligands']['serotonin']['smiles'])
        self.assertEqual((ser.GetAtomWithIdx(0).GetSymbol(),ser.GetAtomWithIdx(0).GetDegree()),('N',1))
        self.assertEqual(coa.GetSubstructMatches(Chem.MolFromSmarts('[C;X3](=[O;X1])([S])[C]')),((1,2,3,0),))
        self.assertEqual(Chem.GetFormalCharge(coa),-4)
        self.assertEqual(Chem.GetFormalCharge(ser),1)

    def test_accepted_scope(self):
        c=load_case('1B6B')['mechanism_criteria']
        self.assertEqual(c['distance_upper_A'],4.)
        self.assertIsNone(c['distance_lower_A'])
        self.assertEqual(c['angle_deg'],{'min':90.,'max':120.,'inclusive':True})
        self.assertFalse(c['auxiliary_affects_status'])
        self.assertFalse(c['His122_direct_contact_required'])
        self.assertTrue(c['retrospective'])


@unittest.skipUnless(FIXTURE,'Set MCSA_1B6B_TEST_CIF to Boltz2 sample2')
class AANATTests(unittest.TestCase):
    def setUp(self):
        self.case=load_case('1B6B')
        self.context=read_case(FIXTURE,self.case)

    def variant(self,change,stage='all',mapping=None):
        s=gemmi.read_structure(FIXTURE)
        change(s)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'new.cif';s.make_mmcif_document().write_file(str(p))
            r=evaluate(p,'1B6B',stage,mapping)
            json.dumps(r,allow_nan=False)
            return r

    def geometry(self,d=3.5,theta=105.):
        context=copy.deepcopy(self.context)
        coa=context['ligands']['acetyl_CoA']
        c,o=coa[1]['xyz'],coa[2]['xyz']
        u=(o-c)/np.linalg.norm(o-c)
        trial=np.array([1.,0.,0.]) if abs(u[0])<.9 else np.array([0.,1.,0.])
        v=np.cross(u,trial);v/=np.linalg.norm(v)
        context['ligands']['serotonin'][0]['xyz']=c+d*(math.cos(math.radians(theta))*u+math.sin(math.radians(theta))*v)
        return context

    def test_real_joint_pass_despite_long_His_contact(self):
        r=evaluate(FIXTURE,'1B6B')
        self.assertEqual((r['mechanism_status'],r['general_status'],r['joint_status']),('PASS','PASS','PASS'))
        his=next(a for a in r['mechanism']['auxiliary_contacts'] if a['id'].startswith('His122'))
        self.assertGreater(his['distance_A'],3.5)
        self.assertFalse(his['affects_mechanism_status'])

    def test_distance_and_angle_inclusive_bounds(self):
        for d,a in [(4.,90.),(4.,120.),(3.,105.),(2.4,105.)]:
            self.assertEqual(evaluate_mechanism(self.geometry(d,a),self.case)['status'],'PASS')
        for d,a in [(4.00001,105.),(3.,89.999),(3.,120.001)]:
            self.assertEqual(evaluate_mechanism(self.geometry(d,a),self.case)['status'],'FAIL')

    def test_same_axis_must_meet_both_conditions(self):
        r=evaluate_mechanism(self.geometry(3.,70.),self.case)
        self.assertEqual((r['status'],r['axis']['distance_status'],r['axis']['angle_status']),('FAIL','PASS','FAIL'))

    def test_His_movement_is_auxiliary(self):
        def change(s):s[0]['A'][94].find_atom('ND1','\x00').pos.x+=100
        r=self.variant(change)
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertTrue(r['general']['recomputed'])

    def test_missing_His_does_not_erase_axis(self):
        r=self.variant(lambda s:s[0]['A'][94].remove_atom('ND1','\x00'))
        self.assertEqual((r['mechanism_status'],r['general_status']),('PASS','FAIL'))
        his=next(a for a in r['mechanism']['auxiliary_contacts'] if a['id'].startswith('His122'))
        self.assertEqual(his['status'],'UNAVAILABLE_DIAGNOSTIC_ONLY')

    def test_missing_Tyr_is_auxiliary(self):
        r=self.variant(lambda s:s[0]['A'][140].remove_atom('OH','\x00'))
        self.assertEqual((r['mechanism_status'],r['general_status']),('PASS','FAIL'))

    def test_nearer_indole_N_not_selected(self):
        c=copy.deepcopy(self.context)
        c['ligands']['serotonin'][5]['xyz']=c['ligands']['acetyl_CoA'][1]['xyz']+[1.,0.,0.]
        a,b=evaluate_mechanism(self.context,self.case),evaluate_mechanism(c,self.case)
        self.assertEqual(a['axis'],b['axis'])

    def test_nearer_other_carbonyl_not_selected(self):
        c=copy.deepcopy(self.context)
        c['ligands']['acetyl_CoA'][7]['xyz']=c['ligands']['serotonin'][0]['xyz']+[1.,0.,0.]
        a,b=evaluate_mechanism(self.context,self.case),evaluate_mechanism(c,self.case)
        self.assertEqual(a['axis'],b['axis'])

    def test_S_not_used_as_angle_oxygen(self):
        c=copy.deepcopy(self.context)
        c['ligands']['acetyl_CoA'][3]['xyz']+=[50.,-30.,20.]
        a,b=evaluate_mechanism(self.context,self.case),evaluate_mechanism(c,self.case)
        self.assertEqual(a['axis'],b['axis'])
        self.assertNotEqual(a['diagnostics']['N_C_S_deg'],b['diagnostics']['N_C_S_deg'])

    def test_existing_C_S_bond_is_general_only(self):
        name=self.context['ligands']['acetyl_CoA'][3]['atom_name']
        def change(s):s[0]['B'][0].find_atom(name,'\x00').pos.x+=10
        r=self.variant(change)
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertIn('ACYL_EXISTING_C_S_BOND_OUT_OF_RANGE',r['general']['failure_tags'])

    def test_C_O_length_independent_of_attack_direction(self):
        coa=self.context['ligands']['acetyl_CoA']
        c,o=coa[1]['xyz'],coa[2]['xyz']
        def change(s):s[0]['B'][0].find_atom(coa[2]['atom_name'],'\x00').pos=gemmi.Position(*(c+3*(o-c)/np.linalg.norm(o-c)))
        r=self.variant(change)
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertIn('ACYL_EXISTING_C_O_BOND_OUT_OF_RANGE',r['general']['failure_tags'])

    def test_general_named_bond_has_explicit_limits(self):
        g=evaluate_aanat_checks(self.context,self.case,FIXTURE)
        self.assertEqual(len(g['existing_bonds']),2)
        for row in g['existing_bonds']:
            self.assertEqual(row['status'],'PASS')
            self.assertLess(row['lower_A'],row['distance_A'])
            self.assertGreater(row['upper_A'],row['distance_A'])

    def test_short_contact_not_exempted(self):
        c=self.geometry(1.5,105.)
        n=c['ligands']['serotonin'][0]
        def change(s):s[0]['C'][0].find_atom(n['atom_name'],'\x00').pos=gemmi.Position(*n['xyz'])
        r=self.variant(change)
        self.assertEqual((r['mechanism_status'],r['general_status']),('PASS','FAIL'))
        self.assertIn('F05_NONCOVALENT_SHORT_CONTACT',r['general']['failure_tags'])

    def test_degenerate_axis_is_explicit_failure(self):
        c=copy.deepcopy(self.context)
        c['ligands']['acetyl_CoA'][2]['xyz']=c['ligands']['acetyl_CoA'][1]['xyz'].copy()
        r=evaluate_mechanism(c,self.case)
        self.assertEqual(r['status'],'FAIL')
        self.assertIn('DEGENERATE_REACTION_AXIS',r['failure_tags'])
        json.dumps(r,allow_nan=False)

    def test_CoA_stereo_flip_does_not_change_axis(self):
        coa=self.context['ligands']['acetyl_CoA']
        center=coa[14]['xyz'];u=coa[12]['xyz']-center;v=coa[16]['xyz']-center
        normal=np.cross(u,v);normal/=np.linalg.norm(normal)
        xyz=coa[15]['xyz'];new=xyz-2*np.dot(xyz-center,normal)*normal
        def change(s):s[0]['B'][0].find_atom(coa[15]['atom_name'],'\x00').pos=gemmi.Position(*new)
        r=self.variant(change)
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertIn('INPUT_STEREOCHEMISTRY_MISMATCH',r['general']['failure_tags'])

    def test_chain_renaming_reordering_and_numbering(self):
        def change(s):
            for chain in s[0]:
                chain.name='NEW_'+chain.name
                for i,res in enumerate(chain):
                    res.seqid=gemmi.SeqId(700+i,' ')
                    atoms=list(res.clone())
                    for j in range(len(res)-1,-1,-1):del res[j]
                    for atom in reversed(atoms):res.add_atom(atom)
        r=self.variant(change)
        self.assertEqual(r['joint_status'],'PASS')

    def test_rigid_transform(self):
        def change(s):
            for chain in s[0]:
                for res in chain:
                    for atom in res:atom.pos=gemmi.Position(-atom.pos.y+3,atom.pos.x-17,atom.pos.z+12)
        r=self.variant(change,'mechanism')
        original=evaluate_mechanism(self.context,self.case)
        np.testing.assert_allclose([r['mechanism']['axis']['N_C_A'],r['mechanism']['axis']['N_C_O_deg']],
                                   [original['axis']['N_C_A'],original['axis']['N_C_O_deg']],atol=1e-10)

    def test_unknown_names_require_map(self):
        mapping={role:{k:v for k,v in row.items() if k!='mapping_method'} for role,row in self.context['atom_mapping'].items()}
        for row in mapping.values():row['atom_names']=['X'+n for n in row['atom_names']]
        def change(s):
            for chain in ['B','C']:
                for a in s[0][chain][0]:a.name='X'+a.name
        self.assertIn('ATOM_MAPPING_REQUIRED',self.variant(change,'mechanism')['input_problem'])
        self.assertEqual(self.variant(change,'all',mapping)['joint_status'],'PASS')

    def test_missing_substrate(self):
        r=self.variant(lambda s:s[0].remove_chain('C'))
        self.assertEqual(r['input_status'],'FAIL')
        self.assertIn('LIGAND_COMPOSITION_OR_COPY_MISMATCH',r['input_problem'])

    def test_stage_isolation(self):
        with patch.object(API,'run_general',side_effect=AssertionError('not requested')):
            self.assertEqual(evaluate(FIXTURE,'1B6B','mechanism')['general_status'],'NOT_RUN')
        with patch.dict(API.MECHANISMS,{'aanat_acyl':lambda *_:(_ for _ in ()).throw(AssertionError('not requested'))}):
            self.assertEqual(evaluate(FIXTURE,'1B6B','general')['mechanism_status'],'NOT_RUN')

    def test_general_error_does_not_erase_axis(self):
        with patch.object(API,'run_general',side_effect=ValueError('synthetic failure')):
            r=evaluate(FIXTURE,'1B6B')
            self.assertEqual((r['mechanism_status'],r['general_status']),('PASS','ERROR'))

    def test_new_covalent_link_is_general_only(self):
        doc=gemmi.cif.read_file(FIXTURE)
        loop=doc.sole_block().init_loop('_struct_conn.',['id','conn_type_id','ptnr1_auth_asym_id','ptnr1_auth_seq_id','ptnr1_label_atom_id','ptnr2_auth_asym_id','ptnr2_auth_seq_id','ptnr2_label_atom_id'])
        loop.add_row(['new','covale','B','1',self.context['ligands']['acetyl_CoA'][1]['atom_name'],'C','1',self.context['ligands']['serotonin'][0]['atom_name']])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'adduct.cif';doc.write_file(str(p));r=evaluate(p,'1B6B')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertIn('UNDECLARED_REACTANT_COVALENT_LINK',r['general']['failure_tags'])

    def test_explicit_wrong_carbonyl_order(self):
        atoms=self.context['ligands']['acetyl_CoA'];mol=Chem.MolFromSmiles(self.case['ligands']['acetyl_CoA']['smiles'])
        doc=gemmi.cif.read_file(FIXTURE)
        loop=doc.sole_block().init_loop('_chem_comp_bond.',['comp_id','atom_id_1','atom_id_2','value_order'])
        for bond in mol.GetBonds():
            i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
            order='SING' if {i,j}=={1,2} else {'SINGLE':'SING','DOUBLE':'DOUB','AROMATIC':'AROM'}[str(bond.GetBondType())]
            loop.add_row([atoms[0]['residue_name'],atoms[i]['atom_name'],atoms[j]['atom_name'],order])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'wrong_order.cif';doc.write_file(str(p));r=evaluate(p,'1B6B')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertIn('DECLARED_ACYL_BOND_ORDER_MISMATCH',r['general']['failure_tags'])

    def test_microstate_not_claimed(self):
        r=evaluate(FIXTURE,'1B6B','mechanism')
        self.assertFalse(r['mechanism']['proton_wire_evaluated'])
        self.assertFalse(r['input_state_contract']['observed_electronic_or_protonation_state_verified'])


if __name__=='__main__':unittest.main()
