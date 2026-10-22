"""1KQP scoring invariants. Synthetic passing tuples are not biochemical controls."""
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

from mcsa_benchmark import evaluate,evaluate_general,evaluate_mechanism
from mcsa_benchmark.evaluate import load_case
from mcsa_benchmark.structure import read_case,atom_record,AA,InputProblem
from mcsa_benchmark.mechanisms import nad_synthetase as m

FIXTURE=os.environ.get('MCSA_1KQP_TEST_CIF')
DOUBLE=os.environ.get('MCSA_1KQP_DOUBLE_TEST_CIF')
REFERENCE=os.environ.get('MCSA_1KQP_REFERENCE_CIF')
API=importlib.import_module('mcsa_benchmark.evaluate')


class ContractTests(unittest.TestCase):
    def test_default_double_set(self):
        c=load_case('1KQP');self.assertEqual(c['metals'],{'Mg':4});self.assertEqual(len(c['ligands']),6)
        self.assertEqual(load_case('1KQP_ONE_SET')['metals'],{'Mg':2})

    def test_accepted_limits_and_scope(self):
        c=load_case('1KQP');cut=c['mechanism_criteria']['cutoffs']
        self.assertEqual(cut['reaction_distance_A'],{'max':3.5})
        self.assertEqual(cut['attack_angle_deg'],{'min':150.,'max':180.})
        self.assertEqual(cut['Mg_O_A'],{'min':1.8,'max':2.6})
        self.assertFalse(c['ligands']['ammonium_1']['mechanism_required'])

    def test_graph_roles(self):
        r=load_case('1KQP')['reaction_atoms']
        self.assertEqual(r['ATP_alpha_P'],15);self.assertEqual(r['ATP_alpha_beta_O'],18)
        self.assertEqual(r['NaAD_oxygen_indices'],[33,34])
        self.assertEqual(r['ATP_nonbridging_O'],{'alpha':[16,17],'beta':[20,21],'gamma':[24,25,26]})


@unittest.skipUnless(FIXTURE,'Set 1KQP single-set fixture')
class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.case=load_case('1KQP_ONE_SET');self.context=read_case(FIXTURE,self.case)
        self.sites,_=m.pair_components(self.context,self.case)
        m.assign_proteins(self.context,self.case,self.sites);m.assign_metals(self.context,self.case,self.sites)
        self.site=self.sites[0]

    def synthetic(self):
        s=self.site;a=s['atp'];n=s['naad'];s['Mg1']['xyz']=np.array([0.,0.,0.]);s['Mg2']['xyz']=np.array([4.,0.,0.])
        def setpos(atom,p):atom['xyz']=np.array(p,dtype=float)
        for index,p in {15:[4,0,4],18:[4,0,5.6],16:[4,2,0],17:[4,5,0],20:[0,2,0],21:[0,5,0],24:[0,0,2],25:[4,0,2],26:[2,4,0]}.items():setpos(a[index],p)
        setpos(n[33],[4,0,1]);setpos(n[34],[6,0,4])
        for pos,res,names,coords in [(50,'ASP',['OD1','OD2'],[[0,-2,0],[0,-5,0]]),(162,'GLU',['OE1','OE2'],[[0,0,-2],[0,0,-5]])]:
            for name,xyz in zip(names,coords):setpos(m.protein_atom(self.context,s['protein_chain'],pos,res,name),xyz)

    def score(self):
        r=m.evaluate_site(self.context,self.case,self.site);json.dumps(r,allow_nan=False);return r

    def test_synthetic_all_eight_measurements_pass(self):
        self.synthetic();r=self.score();self.assertEqual(r['status'],'PASS')
        self.assertEqual(len(r['measurements']),6);self.assertEqual(r['axis_status'],'PASS')

    def test_distance_upper_and_no_lower(self):
        self.synthetic();a=self.site['atp'];n=self.site['naad']
        for d,expected in [(2.,'PASS'),(3.5,'PASS'),(3.5001,'FAIL')]:
            n[33]['xyz']=a[15]['xyz']+[0,0,-d];n[34]['xyz']=a[15]['xyz']+[0,0,-10]
            self.assertEqual(self.score()['axis_status'],expected)

    def test_angle_boundary(self):
        self.synthetic();a=self.site['atp'];n=self.site['naad']
        n[34]['xyz']=a[15]['xyz']+[10,0,0]
        for angle,expected in [(150.,'PASS'),(180.,'PASS'),(149.999,'FAIL')]:
            t=math.radians(angle);n[33]['xyz']=a[15]['xyz']+3*np.array([math.sin(t),0,math.cos(t)])
            self.assertEqual(self.score()['axis_status'],expected)

    def test_same_oxygen_tuple_required(self):
        self.synthetic();a=self.site['atp'];n=self.site['naad']
        n[33]['xyz']=a[15]['xyz']+[3,0,0];n[34]['xyz']=a[15]['xyz']+[0,0,-4]
        r=self.score();self.assertEqual(r['checks']['reaction_distance'],'PASS');self.assertEqual(r['checks']['attack_angle'],'PASS')
        self.assertEqual(r['axis_status'],'FAIL')

    def test_oxygen_label_permutation(self):
        self.synthetic();before=self.score();n=self.site['naad'];a=self.site['atp']
        for group,indices in [(n,[33,34]),(a,[16,17]),(a,[20,21]),(a,[24,25,26])]:
            coords=[group[i]['xyz'].copy() for i in indices]
            for i,p in zip(indices,reversed(coords)):group[i]['xyz']=p
        self.assertEqual(self.score()['checks'],before['checks'])

    def test_alpha_not_gamma(self):
        self.synthetic();a=self.site['atp'];n=self.site['naad'];a[23]['xyz']=n[33]['xyz']+[0,0,3]
        a[15]['xyz']+=[50,0,0]
        self.assertEqual(self.score()['axis_status'],'FAIL')

    def test_mg_window(self):
        self.synthetic();a=self.site['atp'];a[21]['xyz']=np.array([0,10,0.])
        for d,status in [(1.8,'PASS'),(2.6,'PASS'),(1.7999,'FAIL'),(2.6001,'FAIL')]:
            a[20]['xyz']=self.site['Mg1']['xyz']+[0,d,0]
            self.assertEqual(self.score()['checks']['Mg1_ATP_beta_O'],status)

    def test_short_contact_cannot_be_hidden(self):
        self.synthetic();a=self.site['atp'];a[20]['xyz']=np.array([0,1,0.]);a[21]['xyz']=np.array([0,2,0.])
        self.assertIn('MG1_ATP_BETA_O_TOO_SHORT',self.score()['failure_tags'])

    def test_distinct_gamma_oxygens(self):
        self.synthetic();a=self.site['atp']
        a[24]['xyz']=np.array([2,0,0.]);a[25]['xyz']=np.array([20,20,20.]);a[26]['xyz']=np.array([30,30,30.])
        r=self.score();self.assertFalse(r['gamma_oxygen_mapping']['joint_pass']);self.assertEqual(r['status'],'FAIL')

    def test_gamma_short_group_guard(self):
        self.synthetic();a=self.site['atp'];a[26]['xyz']=np.array([0,0,1.])
        self.assertIn('MG1_ATP_GAMMA_O_TOO_SHORT',self.score()['failure_tags'])

    def test_no_metal_role_reuse(self):
        self.synthetic();self.site['Mg2']['xyz']=self.site['Mg1']['xyz'].copy()
        self.assertEqual(self.score()['checks']['Mg2_ATP_alpha_O'],'FAIL')

    def test_metal_order_invariance(self):
        before=m.evaluate_mechanism(self.context,self.case)
        self.context['metal_copies']['Mg'].reverse()
        after=m.evaluate_mechanism(self.context,self.case)
        self.assertEqual(before['sites'],after['sites'])

    def test_metal_assignment_excludes_scored_O(self):
        before=m.evaluate_mechanism(self.context,self.case)['site_assignment']['metals']
        for ch in self.context['protein_chains']:
            for pos,res,names in [(50,'ASP',['OD1','OD2']),(162,'GLU',['OE1','OE2'])]:
                for name in names:m.protein_atom(self.context,ch,pos,res,name)['xyz']+=[50,0,0]
        for i in [16,17,20,21,24,25,26]:self.site['atp'][i]['xyz']+=[10,0,0]
        after=m.evaluate_mechanism(self.context,self.case)['site_assignment']['metals']
        self.assertEqual(before,after)

    def test_coincident_metals_are_ambiguous(self):
        self.context['metal_copies']['Mg'][1]['xyz']=self.context['metal_copies']['Mg'][0]['xyz'].copy()
        r=m.evaluate_mechanism(self.context,self.case)
        self.assertEqual(r['status'],'ERROR');self.assertIn('SITE_ASSIGNMENT_AMBIGUOUS',r['failure_tags'])

    def test_degenerate_axis(self):
        self.synthetic();self.site['atp'][18]['xyz']=self.site['atp'][15]['xyz'].copy()
        self.assertEqual(self.score()['axis_status'],'ERROR')


@unittest.skipUnless(FIXTURE and DOUBLE,'Set single and double 1KQP fixtures')
class IntegrationTests(unittest.TestCase):
    def variant(self,change,path=FIXTURE,case='1KQP_ONE_SET',mapping=None):
        s=gemmi.read_structure(path);change(s)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'new.cif';s.make_mmcif_document().write_file(str(p));r=evaluate(p,case,atom_map=mapping)
            json.dumps(r,allow_nan=False);return r

    def test_baseline_runs(self):
        r=evaluate(FIXTURE,'1KQP_ONE_SET');self.assertEqual(r['input_status'],'PASS');self.assertEqual(r['general_status'],'PASS')
        self.assertEqual(len(r['mechanism']['sites']),1)
        r=evaluate(DOUBLE,'1KQP');self.assertEqual(r['input_status'],'PASS');self.assertEqual(len(r['mechanism']['sites']),2)

    def test_stage_independence(self):
        with patch.object(API,'run_general',side_effect=AssertionError('not called')):
            r=evaluate_mechanism(DOUBLE,'1KQP');self.assertEqual(r['general_status'],'NOT_RUN')
        with patch.dict(API.MECHANISMS,{'nad_synthetase':lambda *_:(_ for _ in ()).throw(AssertionError('not called'))}):
            r=evaluate_general(DOUBLE,'1KQP');self.assertEqual(r['mechanism_status'],'NOT_RUN')

    def test_missing_ammonium_does_not_hide_axis(self):
        r=self.variant(lambda s:s[0].remove_chain('E'))
        self.assertEqual(r['general_status'],'FAIL');self.assertNotEqual(r['mechanism_status'],'ERROR')
        self.assertIn('LIGAND_COMPOSITION_OR_COPY_MISMATCH',r['general']['failure_tags'])

    def test_far_ammonium_is_only_general(self):
        before=evaluate_mechanism(FIXTURE,'1KQP_ONE_SET')['mechanism']
        r=self.variant(lambda s:setattr(s[0]['E'][0][0].pos,'x',500))
        self.assertEqual(r['mechanism'],before);self.assertIn('LIGAND_TOO_FAR_FROM_PROTEIN',r['general']['failure_tags'])

    def test_missing_metal_preserves_axis(self):
        r=self.variant(lambda s:s[0].remove_chain('F'))
        self.assertEqual(r['mechanism_status'],'ERROR');self.assertEqual(r['general_status'],'FAIL')
        self.assertEqual(len(r['mechanism']['sites'][0]['axes']),2)

    def test_scored_sidechain_missing(self):
        def change(s):s[0]['A'][49].remove_atom('OD2','\x00')
        r=self.variant(change)
        self.assertEqual(r['mechanism']['sites'][0]['checks']['Mg1_Asp50_O'],'ERROR')
        self.assertEqual(len(r['mechanism']['sites'][0]['axes']),2)

    def test_general_catches_short_reaction_contact(self):
        def change(s):
            a=next(a for a in s[0]['C'][0] if a.name=='P1');o=next(a for a in s[0]['D'][0] if a.name=='O10')
            o.pos=gemmi.Position(a.pos.x+2,a.pos.y,a.pos.z)
        r=self.variant(change);self.assertEqual(r['mechanism']['sites'][0]['checks']['reaction_distance'],'PASS')
        self.assertIn('F05_NONCOVALENT_SHORT_CONTACT',r['general']['failure_tags'])

    def test_chain_renaming_reordering_transform(self):
        before=evaluate(DOUBLE,'1KQP')
        def change(s):
            copies=[c.clone() for c in s[0]]
            for name in [c.name for c in s[0]]:s[0].remove_chain(name)
            for i,ch in enumerate(reversed(copies)):
                ch.name=f'new{i}'
                for j,r in enumerate(ch):
                    atoms=[a.clone() for a in r]
                    for k in range(len(r)-1,-1,-1):del r[k]
                    for a in reversed(atoms):
                        x,y,z=a.pos;a.pos=gemmi.Position(-y+10,x-20,z+30);r.add_atom(a)
                    r.seqid=gemmi.SeqId(1000+j,' ')
                s[0].add_chain(ch)
        after=self.variant(change,DOUBLE,'1KQP')
        self.assertEqual(before['mechanism_status'],after['mechanism_status']);self.assertEqual(before['general_status'],after['general_status'])
        self.assertEqual(before['mechanism']['failure_tags'],after['mechanism']['failure_tags'])
        old=sorted(v['value'] for s in before['mechanism']['sites'] for v in s['measurements'].values())
        new=sorted(v['value'] for s in after['mechanism']['sites'] for v in s['measurements'].values())
        np.testing.assert_allclose(old,new,atol=1e-8)

    def test_unknown_names_require_map(self):
        c=read_case(FIXTURE,load_case('1KQP_ONE_SET'));names=c['atom_mapping']['naad']['atom_names']
        rename={name:f'X{i}' for i,name in enumerate(names)}
        def change(s):
            for a in s[0]['D'][0]:a.name=rename[a.name]
        self.assertEqual(self.variant(change)['input_status'],'ERROR')
        r=self.variant(change,mapping={'naad':{'chain':'D','atom_names':[rename[n] for n in names]}})
        self.assertEqual(r['input_status'],'PASS')

    def test_wrong_stoichiometry_explicit(self):
        self.assertEqual(evaluate_mechanism(DOUBLE,'1KQP_ONE_SET')['input_status'],'FAIL')

    def test_nan_rejected(self):
        r=self.variant(lambda s:setattr(s[0]['D'][0][0].pos,'x',float('nan')))
        self.assertIn('NONFINITE_COORDINATE',r['input_problem'])


@unittest.skipUnless(REFERENCE,'Set reference intermediate CIF')
class ReferenceTests(unittest.TestCase):
    def test_backbone_role_assignment_not_intermediate_bond_scoring(self):
        s=gemmi.read_structure(REFERENCE)[0];c=load_case('1KQP');proteins={}
        metals=[]
        for ch in s:
            proteins[ch.name]=[{'name':r.name,'atoms':[atom_record(ch,r,a) for a in r if not a.element.is_hydrogen]} for r in ch if r.name in AA]
            for r in ch:
                if r.name=='MG' and r.seqid.num!=5005:metals.append(atom_record(ch,r,r[0]))
        context={'protein_chains':proteins,'metal_copies':{'Mg':list(reversed(metals))}}
        sites=[{'protein_chain':'A'},{'protein_chain':'B'}];m.assign_metals(context,c,sites)
        self.assertEqual([sites[0]['Mg1']['residue_number'],sites[0]['Mg2']['residue_number']],[5001,5002])
        self.assertEqual([sites[1]['Mg1']['residue_number'],sites[1]['Mg2']['residue_number']],[5004,5003])


if __name__=='__main__':unittest.main()
