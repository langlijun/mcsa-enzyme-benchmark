"""2GSA software tests; synthetic geometries are not experimental validation."""
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
from mcsa_benchmark.general import reaction_center_exemptions, atom_key
from mcsa_benchmark.mechanisms.pmp_carbonyl import evaluate_mechanism, judge_axis, geometry
from mcsa_benchmark.structure import read_case

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(os.environ.get('MCSA_2GSA_TEST_CIF', ROOT/'validation/2GSA/cifs/M0195_2GSA__DIMER_2PMP_2GSA__AF3__s0.cif'))
API = importlib.import_module('mcsa_benchmark.evaluate')


class Rules(unittest.TestCase):
    def test_reactive_roles_and_evidence(self):
        case = load_case('2GSA')
        pmp, gsa = [Chem.MolFromSmiles(case['ligands'][role]['smiles']) for role in ['PMP_1','GSA_1']]
        atoms = case['reaction_atoms']
        n = pmp.GetAtomWithIdx(atoms['PMP_N_index'])
        self.assertEqual((n.GetSymbol(), n.GetDegree(), n.GetIsAromatic()),('N',1,False))
        c, o = atoms['GSA_C_index'], atoms['GSA_O_index']
        self.assertEqual(gsa.GetAtomWithIdx(c).GetTotalNumHs(),1)
        self.assertEqual(gsa.GetBondBetweenAtoms(c,o).GetBondType(),Chem.BondType.DOUBLE)
        self.assertFalse(case['mechanism_criteria']['exact_cutoffs_literature_validated'])
        self.assertIsNone(case['mechanism_criteria']['cutoffs']['distance_A']['min'])

    def test_boundaries_and_short_distance_not_veto(self):
        rules = load_case('2GSA')['mechanism_criteria']
        for d,a in [(4,90),(4,120),(2.2,105),(1.7,110)]:
            self.assertEqual(judge_axis(d,a,rules),[])
        self.assertIn('REACTION_DISTANCE_TOO_LONG',judge_axis(4.001,105,rules))
        self.assertIn('ATTACK_ANGLE_BELOW_WINDOW',judge_axis(3,89.999,rules))
        self.assertIn('ATTACK_ANGLE_ABOVE_WINDOW',judge_axis(3,120.001,rules))
        for d,a in [(0,100),(math.nan,100),(3,math.inf),(3,190)]:
            with self.assertRaises(ValueError):
                judge_axis(d,a,rules)

    def test_coincident_axis_is_error(self):
        with self.assertRaisesRegex(ValueError,'coincide'):
            geometry(np.zeros(3),np.zeros(3),np.ones(3))


@unittest.skipUnless(FIXTURE.exists(),'2GSA fixture not found')
class Structures(unittest.TestCase):
    def setUp(self):
        self.case=load_case('2GSA')
        for d in self.case['ligands'].values():
            d['atom_name_templates']=[self.case['atom_profiles']['AF3'][d['copy_group']]]
        self.context=read_case(FIXTURE,self.case)

    def test_explicit_profile_required(self):
        r=evaluate(FIXTURE,'2GSA','mechanism')
        self.assertEqual(r['mechanism_status'],'ERROR')
        self.assertIn('ATOM_PROFILE_REQUIRED',r['mechanism']['failure_tags'])
        r=evaluate(FIXTURE,'2GSA','mechanism',atom_profile='unknown')
        self.assertIn('UNKNOWN_ATOM_PROFILE',r['mechanism']['failure_tags'])

    def test_explicit_map_roundtrip(self):
        r=evaluate(FIXTURE,'2GSA','mechanism',atom_profile='AF3')
        mapping={k:{f:v for f,v in spec.items() if f!='mapping_method'} for k,spec in r['atom_mapping'].items()}
        other=evaluate(FIXTURE,'2GSA','mechanism',atom_map=mapping)
        self.assertEqual(r['mechanism']['sites'],other['mechanism']['sites'])
        mapping['PMP_2']=mapping['PMP_1']
        self.assertEqual(evaluate(FIXTURE,'2GSA','mechanism',atom_map=mapping)['mechanism_status'],'ERROR')

    def test_partial_copy_map_rejected(self):
        mapping={'PMP_1':{'atom_names':self.case['atom_profiles']['AF3']['PMP']}}
        r=evaluate(FIXTURE,'2GSA','mechanism',atom_profile='AF3',atom_map=mapping)
        self.assertEqual(r['mechanism_status'],'ERROR')

    def test_baseline_and_assignment_independent_of_reactive_atoms(self):
        baseline=evaluate_mechanism(self.context,self.case)
        self.assertEqual(baseline['status'],'FAIL')
        self.assertAlmostEqual(baseline['sites'][0]['distance_A'],2.769133619022383)
        moved=copy.deepcopy(self.context)
        for role in ('PMP_1','PMP_2'):
            moved['ligands'][role][self.case['reaction_atoms']['PMP_N_index']]['xyz'] += np.array([20.,30.,40.])
        changed=evaluate_mechanism(moved,self.case)
        self.assertEqual(baseline['site_assignments'],changed['site_assignments'])

    def test_reactive_exemptions_do_not_include_protein(self):
        exemptions=reaction_center_exemptions(self.context,self.case)
        n=self.context['ligands']['PMP_1'][self.case['reaction_atoms']['PMP_N_index']]
        c=self.context['ligands']['GSA_1'][self.case['reaction_atoms']['GSA_C_index']]
        self.assertEqual(exemptions[frozenset([atom_key(n),atom_key(c)])],'FORMING_PAIR')
        lys=next(a for a in self.context['protein_chains']['A'][266]['atoms'] if a['atom_name']=='NZ')
        self.assertNotIn(frozenset([atom_key(n),atom_key(lys)]),exemptions)

    def test_both_sites_required_for_sample_pass(self):
        context=copy.deepcopy(self.context)
        baseline=evaluate_mechanism(context,self.case)
        indices=self.case['reaction_atoms']
        for site in baseline['sites']:
            p,g=site['PMP_copy'],site['GSA_copy']
            n=context['ligands'][p][indices['PMP_N_index']]['xyz']
            c=n+np.array([3.,0.,0.])
            context['ligands'][g][indices['GSA_C_index']]['xyz']=c
            context['ligands'][g][indices['GSA_O_index']]['xyz']=c+1.2*np.array([-math.cos(math.radians(105)),math.sin(math.radians(105)),0.])
        result=evaluate_mechanism(context,self.case)
        self.assertEqual(result['status'],'PASS')
        g=result['sites'][1]['GSA_copy']
        c=context['ligands'][g][indices['GSA_C_index']]['xyz']
        context['ligands'][g][indices['GSA_O_index']]['xyz']=c+1.2*np.array([-.5,math.sqrt(3)/2,0.])
        result=evaluate_mechanism(context,self.case)
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual([s['status'] for s in result['sites']],['PASS','FAIL'])

    def test_missing_site_is_not_filled_with_closest_reaction_pair(self):
        context=copy.deepcopy(self.context)
        for original,other in [('GSA_1','GSA_2'),('PMP_1','PMP_2')]:
            for a,b in zip(context['ligands'][other],context['ligands'][original]):
                a['xyz']=b['xyz'].copy()
        r=evaluate_mechanism(context,self.case)
        self.assertEqual(r['status'],'FAIL')
        self.assertIn('DECLARED_SITE_COMPONENT_COUNT_MISMATCH',r['failure_tags'])

    def test_general_failure_or_error_does_not_erase_mechanism(self):
        with patch.object(API,'run_general',side_effect=ValueError('synthetic general error')):
            r=evaluate(FIXTURE,'2GSA',atom_profile='AF3')
        self.assertEqual((r['mechanism_status'],r['general_status']),('FAIL','ERROR'))
        self.assertEqual(len(r['mechanism']['sites']),2)

    def test_stage_isolation(self):
        with patch.object(API,'run_general',side_effect=AssertionError('not requested')):
            r=evaluate(FIXTURE,'2GSA','mechanism',atom_profile='AF3')
            self.assertEqual(r['general_status'],'NOT_RUN')
        with patch.dict(API.MECHANISMS,{'pmp_carbonyl':lambda *_: (_ for _ in ()).throw(AssertionError('not requested'))}):
            r=evaluate(FIXTURE,'2GSA','general',atom_profile='AF3')
            self.assertEqual(r['mechanism_status'],'NOT_RUN')

    def test_renumbered_rigid_transform_and_atom_reordering(self):
        baseline=evaluate(FIXTURE,'2GSA','mechanism',atom_profile='AF3')
        structure=gemmi.read_structure(str(FIXTURE))
        angle=.7
        rot=np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
        for chain in structure[0]:
            chain.name='NEW_'+chain.name
            for index,residue in enumerate(chain):
                residue.seqid=gemmi.SeqId(1000+index,' ')
                for atom in residue:
                    xyz=np.array([atom.pos.x,atom.pos.y,atom.pos.z])@rot + np.array([19.,-13.,8.])
                    atom.pos=gemmi.Position(*xyz)
                atoms=[a.clone() for a in residue]
                for i in reversed(range(len(residue))):
                    del residue[i]
                for atom in reversed(atoms):
                    residue.add_atom(atom)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'new.cif'
            structure.make_mmcif_document().write_file(str(p))
            r=evaluate(p,'2GSA','mechanism',atom_profile='AF3')
        self.assertFalse(r['historical_results_used'])
        self.assertEqual(r['mechanism_status'],baseline['mechanism_status'])
        for a,b in zip(r['mechanism']['sites'],baseline['mechanism']['sites']):
            self.assertAlmostEqual(a['distance_A'],b['distance_A'],places=5)
            self.assertAlmostEqual(a['angle_deg'],b['angle_deg'],places=4)
        json.dumps(r,allow_nan=False)

    def test_missing_component_and_nan_are_not_success(self):
        for defect in ('delete','nan'):
            structure=gemmi.read_structure(str(FIXTURE))
            if defect=='delete':
                del structure[0]['F']
            else:
                structure[0]['E'][0][0].pos.x=math.nan
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/'broken.cif'
                structure.make_mmcif_document().write_file(str(p))
                r=evaluate(p,'2GSA',atom_profile='AF3')
            self.assertEqual(r['mechanism_status'],'ERROR')
            self.assertEqual(r['general_status'],'ERROR')


if __name__=='__main__':
    unittest.main()
