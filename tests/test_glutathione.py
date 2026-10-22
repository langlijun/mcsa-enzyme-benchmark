"""1GSA numerical and invariance tests; not validation of enzyme activity."""
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
from mcsa_benchmark.general import metal_coordination_check
from mcsa_benchmark.mechanisms.glutathione_synthesis import evaluate_mechanism, protein_core, phosphate_coupling, within
from mcsa_benchmark.structure import read_case

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=Path(os.environ.get('MCSA_1GSA_TEST_CIF',ROOT/'validation/1GSA/cifs/M0199_1GSA__AF3__s0.cif'))
API=importlib.import_module('mcsa_benchmark.evaluate')


def atom(xyz,name='O',chain='X',residue='LIG',number=1,element='O'):
    return {'xyz':np.asarray(xyz,dtype=float),'atom_name':name,'chain':chain,'residue_name':residue,
            'residue_number':number,'insertion_code':'','element':element}


class Definitions(unittest.TestCase):
    def setUp(self):self.case=load_case('1GSA')

    def test_reaction_atom_chemistry(self):
        r=self.case['reaction_atoms'];m=Chem.MolFromSmiles(self.case['ligands']['ATP_1']['smiles'])
        p=r['Pgamma_index'];o=r['leaving_oxygen_index']
        self.assertEqual(m.GetAtomWithIdx(p).GetSymbol(),'P')
        self.assertEqual([n.GetIdx() for n in m.GetAtomWithIdx(o).GetNeighbors()],[19,23])
        self.assertEqual(r['phosphate_oxygen_indices'],{'alpha':[16,17],'beta':[20,21],'gamma':[24,25,26]})
        g=Chem.MolFromSmiles(self.case['ligands']['GGC_1']['smiles'])
        self.assertTrue(all(g.GetBondBetweenAtoms(10,i) for i in r['GGC_nucleophile_indices']))
        self.assertFalse(self.case['mechanism_criteria']['matched_reactant_calibration'])
        self.assertFalse(self.case['mechanism_criteria']['metal_atp']['analog_topology_diagnostic']['affects_primary_pass'])

    def test_window_boundaries(self):
        for d in [1.7,2.8]:self.assertTrue(within(d,[1.7,2.8]))
        for d in [1.699,2.801]:self.assertFalse(within(d,[1.7,2.8]))

    def reference_context(self):
        ref=self.case['site_reference']
        residues=[{'name':'UNK','atoms':[]} for _ in range(316)]
        for i,names,res in [(273,['OD1','OD2'],'ASP'),(281,['OE1','OE2'],'GLU'),(283,['OD1'],'ASN')]:
            residues[i-1]={'name':res,'atoms':[atom(ref['named_donor_coordinates'][f'{i}:{n}'],n,'A',res,i) for n in names]}
        return atom(ref['Mg1'],'MG','M1',element='MG'),atom(ref['Mg2'],'MG','M2',element='MG'),residues

    def test_reference_core_and_equivalent_oxygen_swap(self):
        a,b,res=self.reference_context()
        ref=self.case['site_reference'];c=self.case['mechanism_criteria']['metal_core']
        r=protein_core(a,b,res,ref,c)
        self.assertEqual(r['status'],'PASS')
        self.assertLess(max(abs(x['delta_A']) for x in r['edges']),1e-10)
        res[280]['atoms'][0]['atom_name'],res[280]['atoms'][1]['atom_name']='OE2','OE1'
        r=protein_core(a,b,res,ref,c)
        self.assertEqual(r['status'],'PASS')

    def test_two_separate_glu_oxygens_cannot_fake_shared_bridge(self):
        a,b,res=self.reference_context()
        direction=b['xyz']-a['xyz'];v=np.cross(direction,[0.,0.,1.]);v/=np.linalg.norm(v)
        res[280]['atoms'][0]['xyz']=a['xyz']+1.958*v
        res[280]['atoms'][1]['xyz']=b['xyz']+2.198*v
        r=protein_core(a,b,res,self.case['site_reference'],self.case['mechanism_criteria']['metal_core'])
        self.assertEqual(r['status'],'FAIL')
        e=[r for r in r['edges'] if 'Glu' in r['edge']]
        self.assertEqual(e[0]['donor'],e[1]['donor'])

    def test_each_mg_requires_same_atp_phosphate_and_general_owns_short(self):
        a,b=atom([0,0,0],'MG','M1',element='MG'),atom([4,0,0],'MG','M2',element='MG')
        atp=[atom([50+i,30,0],f'O{i}') for i in range(31)]
        atp[16]['xyz']=np.array([2.,0.,0.])
        indices=self.case['reaction_atoms']['phosphate_oxygen_indices'];c=self.case['mechanism_criteria']['metal_atp']
        r=phosphate_coupling(a,b,atp,indices,c)
        self.assertEqual(r['status'],'PASS')
        self.assertEqual(r['ADP_topology_diagnostic']['status'],'FAIL')
        atp[16]['xyz']=np.array([0.,2.,0.])
        self.assertIn('MG2_ATP_PHOSPHATE_CONTACT_MISSING',phosphate_coupling(a,b,atp,indices,c)['failure_tags'])
        atp[20]['xyz']=np.array([2.,0.,0.]);atp[17]['xyz']=np.array([1.,0.,0.])
        self.assertEqual(phosphate_coupling(a,b,atp,indices,c)['status'],'PASS')
        context={'metal_copies':{'MG':[a,b]},'protein_atoms':[],'ligands':{'ATP_1':atp}}
        physical=metal_coordination_check(context,self.case)
        self.assertEqual(physical['status'],'NOT_SCORED')
        self.assertFalse(physical['affects_general_pass'])
        self.assertTrue(all(x['tag']=='F05_METAL_COORDINATION_TOO_SHORT' for x in physical['pairs']))
        atp[17]['xyz']=np.array([50.,50.,50.]);atp[16]['xyz']=np.array([50.,50.,50.]);atp[20]['xyz']=np.array([50.,50.,50.])
        atp[11]['xyz']=np.array([2.,0.,0.])
        self.assertEqual(phosphate_coupling(a,b,atp,indices,c)['status'],'FAIL')

    def test_normal_coordination_is_not_vdw_clash(self):
        context={'metal_copies':{'MG':[atom([0,0,0],'MG',element='MG')]},'protein_atoms':[],
                 'ligands':{'ATP_1':[atom([2,0,0])]}}
        self.assertEqual(metal_coordination_check(context,self.case)['status'],'NOT_SCORED')
        for d,expected_pairs in [(1.7,0),(1.699,1)]:
            context['ligands']['ATP_1'][0]['xyz']=np.array([d,0,0])
            diagnostic=metal_coordination_check(context,self.case)
            self.assertEqual(diagnostic['status'],'NOT_SCORED')
            self.assertEqual(len(diagnostic['pairs']),expected_pairs)

    def test_reference_compression_is_not_duplicate_mechanism_failure(self):
        a,b,res=self.reference_context()
        oxygen=res[272]['atoms'][1]
        direction=oxygen['xyz']-a['xyz'];direction/=np.linalg.norm(direction)
        oxygen['xyz']=a['xyz']+1.65*direction
        r=protein_core(a,b,res,self.case['site_reference'],self.case['mechanism_criteria']['metal_core'])
        self.assertEqual(r['status'],'PASS')
        self.assertEqual(r['two_sided_reference_recovery_diagnostic']['status'],'FAIL')
        context={'metal_copies':{'MG':[a,b]},'protein_atoms':[o for rr in res for o in rr['atoms']],'ligands':{}}
        self.assertEqual(metal_coordination_check(context,self.case)['status'],'NOT_SCORED')
        self.assertTrue(metal_coordination_check(context,self.case)['pairs'])


@unittest.skipUnless(FIXTURE.exists(),'1GSA fixture not found')
class Structures(unittest.TestCase):
    def setUp(self):
        self.case=load_case('1GSA');self.context=read_case(FIXTURE,self.case)

    def test_baseline_separate_subchecks(self):
        r=evaluate(FIXTURE,'1GSA')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['general_status'],'PASS')
        self.assertEqual(len(r['mechanism']['sites']),2)
        self.assertTrue(all(v=='PASS' for v in r['mechanism']['subchecks'].values()))

    def test_no_reuse_of_one_mg_and_missing_metal_not_silently_passed(self):
        c=copy.deepcopy(self.context);c['metal_copies']['MG'].pop()
        r=evaluate_mechanism(c,self.case)
        self.assertEqual(r['status'],'FAIL')
        self.assertEqual(r['subchecks']['reaction_axis'],'PASS')
        self.assertIn('DECLARED_TWO_MG_SITE_OCCUPANCY_MISMATCH',r['failure_tags'])

    def test_moving_one_site_metals_cannot_be_rescued_by_other_site(self):
        c=copy.deepcopy(self.context)
        baseline=evaluate_mechanism(c,self.case)
        assignments=baseline['metal_site_assignments']
        for item in assignments:
            if item['site']=='A':c['metal_copies']['MG'][item['index']]['xyz']+=np.array([100.,100.,100.])
        r=evaluate_mechanism(c,self.case)
        self.assertNotEqual(r['status'],'PASS')
        self.assertEqual(r['subchecks']['reaction_axis'],'PASS')

    def test_site_assignment_does_not_use_reactive_oxygen(self):
        original=evaluate_mechanism(self.context,self.case)
        c=copy.deepcopy(self.context)
        for role in ['GGC_1','GGC_2']:
            for i in [11,12]:c['ligands'][role][i]['xyz']+=np.array([25.,40.,30.])
        r=evaluate_mechanism(c,self.case)
        self.assertEqual(original['ligand_site_assignments'],r['ligand_site_assignments'])
        self.assertEqual(r['subchecks']['reaction_axis'],'FAIL')

    def test_general_does_not_veto_or_block_mechanism(self):
        with patch.object(API,'run_general',side_effect=ValueError('test')):
            r=evaluate(FIXTURE,'1GSA')
        self.assertEqual((r['mechanism_status'],r['general_status']),('PASS','ERROR'))

    def test_protenix_short_mg_contact_is_general_only(self):
        p=FIXTURE.with_name('M0199_1GSA__Protenix__s0.cif')
        r=evaluate(p,'1GSA')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['general_status'],'FAIL')
        self.assertNotIn('F05_METAL_COORDINATION_TOO_SHORT',r['general']['failure_tags'])
        self.assertFalse(r['general']['metal_coordination_check']['affects_general_pass'])
        self.assertFalse(any('SHORT' in t for t in r['mechanism']['failure_tags']))

    def test_mechanism_only_does_not_run_general(self):
        with patch.object(API,'run_general',side_effect=AssertionError('not requested')):
            r=evaluate(FIXTURE,'1GSA','mechanism')
        self.assertEqual(r['general_status'],'NOT_RUN')

    def test_explicit_map_duplicate_rejected(self):
        r=evaluate(FIXTURE,'1GSA','mechanism')
        mapping={k:{kk:vv for kk,vv in v.items() if kk!='mapping_method'} for k,v in r['atom_mapping'].items()}
        mapping['ATP_2']=mapping['ATP_1']
        r=evaluate(FIXTURE,'1GSA','mechanism',atom_map=mapping)
        self.assertEqual(r['mechanism_status'],'ERROR')

    def test_chain_and_residue_renumbering_rigid_transform(self):
        baseline=evaluate(FIXTURE,'1GSA','mechanism')
        s=gemmi.read_structure(str(FIXTURE))
        for ch in s[0]:
            ch.name='new_'+ch.name
            for i,r in enumerate(ch):
                r.seqid=gemmi.SeqId(1000+i,' ')
                for a in r:
                    x,y,z=a.pos.x,a.pos.y,a.pos.z;a.pos=gemmi.Position(-y+20,x-15,z+7)
                aa=[a.clone() for a in r]
                for j in reversed(range(len(r))):del r[j]
                for a in reversed(aa):r.add_atom(a)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'new.cif';s.make_mmcif_document().write_file(str(p));r=evaluate(p,'1GSA','mechanism')
        self.assertEqual(r['mechanism_status'],baseline['mechanism_status'])
        self.assertFalse(r['historical_results_used'])
        for x,y in zip(r['mechanism']['sites'],baseline['mechanism']['sites']):
            self.assertAlmostEqual(x['reaction_axis']['witness']['distance_A'],y['reaction_axis']['witness']['distance_A'],places=6)
        json.dumps(r,allow_nan=False)


if __name__=='__main__':unittest.main()
