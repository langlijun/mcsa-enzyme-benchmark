"""5ICD contract, numerical boundaries, copy invariance and independent stages."""

from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gemmi
import numpy as np

from mcsa_benchmark import evaluate, evaluate_general, evaluate_mechanism
from mcsa_benchmark.evaluate import load_case
from mcsa_benchmark.mechanisms.isocitrate import evaluate_mechanism as mechanism, assignment, face_measurement
from mcsa_benchmark.structure import read_case, InputProblem

FIXTURE=os.environ.get('MCSA_5ICD_TEST_CIF')
DOUBLE=os.environ.get('MCSA_5ICD_DOUBLE_TEST_CIF')
API=importlib.import_module('mcsa_benchmark.evaluate')
REFERENCES=os.environ.get('MCSA_5ICD_REFERENCE_DIR')


class ContractTests(unittest.TestCase):
    def test_default_is_double_set(self):
        self.assertEqual(load_case('5ICD')['variant'],'dimer_two_sets')
        self.assertEqual(load_case('M0007_5ICD')['metals']['Mg'],2)

    def test_approved_rules(self):
        c=load_case('5ICD_ONE_SET')
        self.assertEqual(c['mechanism_criteria']['cutoffs']['hydride_distance_A'],{'max':4.0})
        self.assertEqual(c['mechanism_criteria']['cutoffs']['lys_OH_distance_A'],{'max':3.5})
        self.assertEqual(c['mechanism_criteria']['cutoffs']['Mg_O_distance_A'],{'min':1.8,'max':2.6})
        self.assertEqual(c['case_general_requirements']['excluded_stereochemistry_elements'],['P'])

    def test_two_set_contract_not_auto_selected(self):
        self.assertEqual(load_case('5ICD_ONE_SET')['metals']['Mg'],1)
        self.assertEqual(load_case('5ICD_TWO_SETS')['metals']['Mg'],2)
        self.assertEqual(len(load_case('5ICD_TWO_SETS')['ligands']),4)

    def test_assignment_tie_is_explicit(self):
        with self.assertRaises(InputProblem): assignment([[1.,1.],[1.,1.]])


@unittest.skipUnless(REFERENCES,'Set experimental module reference folder')
class ReferenceModuleTests(unittest.TestCase):
    def test_same_enzyme_ca_re_face_and_axis(self):
        model=gemmi.read_structure(str(Path(REFERENCES)/'4AJ3-assembly1.cif.gz'))[0]
        ict=next(r for r in model[0] if r.name=='ICT');nap=next(r for r in model[0] if r.name=='NAP')
        def a(res,name):
            raw=next(x for x in res if x.name==name)
            return {'xyz':np.array(list(raw.pos)),'atom_name':name}
        d,c,c3,c5=a(ict,'C2'),a(nap,'C4N'),a(nap,'C3N'),a(nap,'C5N')
        r=face_measurement(d,c,c3,c5)
        self.assertAlmostEqual(r['value'],2.8059634289418414)
        distance=float(np.linalg.norm(d['xyz']-c['xyz']))
        self.assertAlmostEqual(distance,2.9688632504714665)
        self.assertLess(distance,load_case('5ICD_ONE_SET')['mechanism_criteria']['cutoffs']['hydride_distance_A']['max'])
        other_lys=next(r for r in model[1] if r.seqid.num==230)
        base=np.linalg.norm(a(other_lys,'NZ')['xyz']-a(ict,'O7')['xyz'])
        self.assertAlmostEqual(base,3.1632908497322854)

    def test_same_enzyme_mg_four_named_donors(self):
        model=gemmi.read_structure(str(Path(REFERENCES)/'5ICD-assembly1.cif.gz'))[0]
        mg=next(a for r in model[0] for a in r if a.element.name=='Mg')
        ict=next(r for r in model[0] if r.name=='ICT')
        asp307=next(r for r in model[0] if r.seqid.num==307)
        asp283=next(r for r in model[1] if r.seqid.num==283)
        values=[min(mg.pos.dist(a.pos) for a in r if a.name in names)
                for r,names in [(ict,['O7']),(ict,['O1','O2']),(asp307,['OD1','OD2']),(asp283,['OD1','OD2'])]]
        for value,expected in zip(values,[1.914244237290537,2.2364547838040463,2.0967462888961963,2.3431973881856387]):
            self.assertAlmostEqual(value,expected)
            self.assertTrue(1.8<=value<=2.6)


@unittest.skipUnless(FIXTURE and DOUBLE,'Set 5ICD one-set and double-set fixtures')
class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.case=load_case('5ICD_ONE_SET');self.context=read_case(FIXTURE,self.case)

    def run_geometry(self):
        r=mechanism(self.context,self.case)
        json.dumps(r,allow_nan=False)
        return r

    def aa(self,chain,pos,name):
        return next(a for a in self.context['protein_chains'][chain][pos-1]['atoms'] if a['atom_name']==name)

    def test_baseline(self):
        r=self.run_geometry();self.assertEqual(r['status'],'PASS')
        self.assertEqual(len(r['sites'][0]['checks']),7)

    def test_no_carbon_lower_bound_and_upper_boundary(self):
        d,a=self.context['ligands']['isocitrate'][8],self.context['ligands']['nadp'][1]
        v=d['xyz']-a['xyz'];v/=np.linalg.norm(v)
        for distance,expected in [(2.4,'PASS'),(2.999,'PASS'),(4.,'PASS'),(4.0001,'FAIL')]:
            d['xyz']=a['xyz']+v*distance
            self.assertEqual(self.run_geometry()['sites'][0]['checks']['hydride_distance'],expected)

    def test_lys_upper_bound(self):
        baseline=self.run_geometry()['sites'][0]
        n=self.aa(baseline['opposite_chain'],228,'NZ');o=self.context['ligands']['isocitrate'][9]
        for distance,expected in [(2.,'PASS'),(3.5,'PASS'),(3.5001,'FAIL')]:
            n['xyz']=o['xyz']+[distance,0,0]
            self.assertEqual(self.run_geometry()['sites'][0]['checks']['lys230_prime_OH'],expected)

    def test_mg_bounds(self):
        mg=self.context['metal_copies']['Mg'][0];o=self.context['ligands']['isocitrate'][9]
        for distance,expected in [(1.8,'PASS'),(2.6,'PASS'),(1.7999,'FAIL'),(2.6001,'FAIL')]:
            o['xyz']=mg['xyz']+[distance,0,0]
            self.assertEqual(self.run_geometry()['sites'][0]['checks']['Mg_substrate_OH'],expected)

    def test_short_oxygen_not_hidden_by_other_oxygen(self):
        mg=self.context['metal_copies']['Mg'][0]
        for index,distance in [(11,1.),(12,2.1)]:
            self.context['ligands']['isocitrate'][index]['xyz']=mg['xyz']+[distance,0,0]
        r=self.run_geometry()
        self.assertIn('MG_SUBSTRATE_CARBOXYLATE_TOO_SHORT',r['failure_tags'])

    def test_re_si_reflection(self):
        d=self.context['ligands']['isocitrate'][8]
        n=self.context['ligands']['nadp'];c=n[1]['xyz']
        norm=-np.cross(n[2]['xyz']-c,n[0]['xyz']-c);norm/=np.linalg.norm(norm)
        d['xyz']-=2*np.dot(d['xyz']-c,norm)*norm
        r=self.run_geometry()['sites'][0]
        self.assertEqual(r['checks']['acceptor_re_face'],'FAIL')
        self.assertEqual(r['checks']['hydride_distance'],'PASS')

    def test_face_zero_and_degenerate(self):
        n=self.context['ligands']['nadp'];d=self.context['ligands']['isocitrate'][8]
        d['xyz']=n[1]['xyz']+(n[2]['xyz']-n[1]['xyz'])*2
        self.assertEqual(self.run_geometry()['sites'][0]['checks']['acceptor_re_face'],'FAIL')
        n[2]['xyz']=n[1]['xyz'].copy()
        self.assertEqual(self.run_geometry()['sites'][0]['checks']['acceptor_re_face'],'ERROR')

    def test_equivalent_oxygens(self):
        before=self.run_geometry()['sites'][0]['checks']
        ligand=self.context['ligands']['isocitrate']
        ligand[11]['xyz'],ligand[12]['xyz']=ligand[12]['xyz'].copy(),ligand[11]['xyz'].copy()
        for chain in self.context['protein_chains']:
            for pos in (281,305):
                a,b=self.aa(chain,pos,'OD1'),self.aa(chain,pos,'OD2')
                a['xyz'],b['xyz']=b['xyz'].copy(),a['xyz'].copy()
        self.assertEqual(self.run_geometry()['sites'][0]['checks'],before)

    def test_cross_subunit_lys_is_required(self):
        baseline=self.run_geometry()['sites'][0]
        o=self.context['ligands']['isocitrate'][9]
        self.aa(baseline['local_chain'],228,'NZ')['xyz']=o['xyz']+[3.,0,0]
        self.aa(baseline['opposite_chain'],228,'NZ')['xyz']=o['xyz']+[20.,0,0]
        self.assertEqual(self.run_geometry()['sites'][0]['checks']['lys230_prime_OH'],'FAIL')

    def test_scored_atoms_do_not_drive_assignment(self):
        before=self.run_geometry()['site_assignment']
        for ch in self.context['protein_chains']:
            self.aa(ch,228,'NZ')['xyz']+=[100,0,0]
            for pos in (281,305):
                for name in ('OD1','OD2'):self.aa(ch,pos,name)['xyz']+=[100,0,0]
        self.context['ligands']['isocitrate'][8]['xyz']+=[20,0,0]
        self.context['ligands']['nadp'][1]['xyz']+=[-20,0,0]
        self.assertEqual(self.run_geometry()['site_assignment'],before)

    def test_two_declared_sites_required(self):
        c=load_case('5ICD_TWO_SETS');context=read_case(DOUBLE,c)
        r=mechanism(context,c);self.assertEqual(r['status'],'PASS');self.assertEqual(len(r['sites']),2)
        site=r['sites'][0];role=site['donor_role']
        context['ligands'][role][8]['xyz']+=[100,0,0]
        r=mechanism(context,c)
        self.assertEqual(r['status'],'FAIL')
        self.assertEqual(sum(s['checks']['hydride_distance']=='PASS' for s in r['sites']),1)


@unittest.skipUnless(FIXTURE and DOUBLE,'Set 5ICD one-set and double-set fixtures')
class IntegrationTests(unittest.TestCase):
    def variant(self,change,path=FIXTURE,case='5ICD_ONE_SET',mapping=None):
        s=gemmi.read_structure(path);change(s)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'new.cif';s.make_mmcif_document().write_file(str(p))
            r=evaluate(p,case,atom_map=mapping);json.dumps(r,allow_nan=False)
            return r

    def test_stage_separation(self):
        with patch.object(API,'run_general',side_effect=AssertionError('must not run')):
            r=evaluate_mechanism(FIXTURE,'5ICD_ONE_SET')
            self.assertEqual(r['general_status'],'NOT_RUN')
        with patch.dict(API.MECHANISMS,{'isocitrate':lambda *_:(_ for _ in ()).throw(AssertionError('must not run'))}):
            r=evaluate_general(FIXTURE,'5ICD_ONE_SET');self.assertEqual(r['mechanism_status'],'NOT_RUN')

    def test_far_mg_independent_of_general(self):
        r=self.variant(lambda s:setattr(s[0]['E'][0][0].pos,'x',500))
        self.assertEqual(r['general_status'],'PASS');self.assertEqual(r['mechanism_status'],'FAIL')

    def test_missing_mg_preserves_hydride(self):
        r=self.variant(lambda s:s[0].remove_chain('E'))
        self.assertEqual(r['general_status'],'FAIL');self.assertEqual(r['mechanism_status'],'ERROR')
        self.assertEqual(r['mechanism']['sites'][0]['checks']['hydride_distance'],'PASS')

    def test_general_catches_too_short_reaction_carbon(self):
        def change(s):
            d=next(a for a in s[0]['D'][0] if a.name=='C5');a=next(a for a in s[0]['C'][0] if a.name=='C2')
            v=np.array(list(d.pos))-np.array(list(a.pos));v/=np.linalg.norm(v)
            d.pos=gemmi.Position(*(np.array(list(a.pos))+2.4*v))
        r=self.variant(change)
        self.assertEqual(r['mechanism']['sites'][0]['checks']['hydride_distance'],'PASS')
        self.assertEqual(r['general_status'],'FAIL')
        self.assertTrue(any({a['atom_name'] for a in (p['first'],p['second'])}=={'C5','C2'} and p['category']=='nadp_isocitrate'
                            for p in r['general']['short_contact_atom_pairs']))

    def test_wrong_carbon_stereo_fails_general_not_P(self):
        def change(s):
            atoms={a.name:a for a in s[0]['D'][0]}
            # Swap two substituent coordinates at the non-reactive stereocenter.
            first,second=tuple(atoms['C2'].pos),tuple(atoms['C3'].pos)
            atoms['C2'].pos,atoms['C3'].pos=gemmi.Position(*second),gemmi.Position(*first)
        r=self.variant(change)
        self.assertIn('INPUT_STEREOCHEMISTRY_MISMATCH',r['general']['failure_tags'])

    def test_P_mismatch_excluded_with_explicit_records(self):
        r=evaluate_general(FIXTURE,'5ICD_ONE_SET')
        records=r['general']['components']['nadp']['tetrahedral_center_records']
        self.assertTrue(any(x['element']=='P' and not x['match'] and not x['counted'] for x in records))
        self.assertEqual(r['general_status'],'PASS')

    def test_rename_reorder_renumber_and_rigid_transform(self):
        def change(s):
            for ch in s[0]:
                for i,res in enumerate(ch):
                    atoms=[a.clone() for a in res]
                    for j in range(len(res)-1,-1,-1):del res[j]
                    for a in reversed(atoms):
                        x,y,z=a.pos;a.pos=gemmi.Position(-y+10,x-20,z+30);res.add_atom(a)
                    res.seqid=gemmi.SeqId(1000+i,' ')
                ch.name='renamed_'+ch.name
        r=self.variant(change)
        self.assertEqual(r['joint_status'],'PASS')

    def test_double_set_order_invariance(self):
        def change(s):
            chains=[c.clone() for c in s[0]]
            for name in [c.name for c in s[0]]:s[0].remove_chain(name)
            for i,c in enumerate(reversed(chains)):
                c.name=f'copy{i}';s[0].add_chain(c)
        r=self.variant(change,DOUBLE,'5ICD_TWO_SETS')
        self.assertEqual(r['mechanism_status'],'PASS');self.assertEqual(len(r['mechanism']['sites']),2)

    def test_mismatched_stoichiometry_not_silently_accepted(self):
        r=evaluate_mechanism(DOUBLE,'5ICD_ONE_SET')
        self.assertEqual(r['input_status'],'FAIL')

    def test_unknown_atom_names_and_user_map(self):
        original=read_case(FIXTURE,load_case('5ICD_ONE_SET'))['atom_mapping']['isocitrate']
        names=original['atom_names'];renamed={n:f'X{i}' for i,n in enumerate(names)}
        def change(s):
            for a in s[0]['D'][0]:a.name=renamed[a.name]
        self.assertEqual(self.variant(change)['input_status'],'ERROR')
        mapping={'isocitrate':{'chain':'D','atom_names':[renamed[n] for n in names]}}
        self.assertEqual(self.variant(change,mapping=mapping)['joint_status'],'PASS')

    def test_missing_required_sidechain_only_affects_relevant_metrics(self):
        def change(s):s[0]['B'][227].remove_atom('NZ','\x00')
        r=self.variant(change)
        self.assertEqual(r['mechanism']['sites'][0]['checks']['lys230_prime_OH'],'ERROR')
        self.assertEqual(r['mechanism']['sites'][0]['checks']['hydride_distance'],'PASS')
        self.assertEqual(r['general_status'],'FAIL')

    def test_nan_is_rejected(self):
        r=self.variant(lambda s:setattr(s[0]['D'][0][0].pos,'x',float('nan')))
        self.assertIn('NONFINITE_COORDINATE',r['input_problem'])

    def test_coincident_bond_reports_failure_without_invalid_json(self):
        def change(s):
            atoms=list(s[0]['D'][0]);atoms[1].pos=gemmi.Position(*tuple(atoms[0].pos))
        r=self.variant(change)
        self.assertEqual(r['general_status'],'FAIL')
        self.assertIn('BOND_LENGTH_OUT_OF_RANGE',r['general']['failure_tags'])


if __name__=='__main__':unittest.main()
