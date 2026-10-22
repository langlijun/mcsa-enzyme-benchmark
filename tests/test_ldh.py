"""Tetramer assignment, independent stages and synthetic LDH geometry tests."""
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
from mcsa_benchmark.mechanisms.ldh_tetramer import assign_group, assign_sites, evaluate_mechanism, fit_frame
from mcsa_benchmark.structure import InputProblem, read_case

API = importlib.import_module('mcsa_benchmark.evaluate')
FIXTURE = os.environ.get('MCSA_1LDM_TEST_CIF')


class LDHDefinitionTests(unittest.TestCase):
    def test_four_site_input_and_atoms(self):
        c = load_case('1LDM')
        self.assertEqual(c['protein_copies'], 4)
        self.assertEqual(len(c['protein_sequence']), 329)
        self.assertEqual(c['protein_sequence'][192], 'H')
        self.assertEqual(c['reaction_atoms']['His_atom_name'], 'NE2')
        self.assertEqual(len(c['ligands']), 8)
        self.assertEqual(c['metals'], {})
        self.assertTrue(all(x['group_copies'] == 4 for x in c['ligands'].values()))
        self.assertEqual(dict(Chem.FindMolChiralCenters(Chem.MolFromSmiles(c['ligands']['lactate_1']['smiles']))), {1: 'S'})

    def test_accepted_cutoffs_and_scope(self):
        c = load_case('1LDM')['mechanism_criteria']
        self.assertEqual((c['hydride_distance']['min_A'], c['hydride_distance']['max_A']), (3., 4.))
        self.assertEqual(c['catalytic_base_contact']['max_A'], 3.5)
        self.assertIsNone(c['catalytic_base_contact']['min_A'])
        self.assertFalse(c['angles_or_hydride_faces_scored'])
        self.assertFalse(c['whole_study_preregistered'])

    def test_solver_ambiguity_is_explicit(self):
        with self.assertRaisesRegex(InputProblem, 'SITE_MAPPING_AMBIGUOUS'):
            assign_group(np.zeros((4, 4)), ['A','B','C','D'], ['a','b','c','d'], 1e-6)

    def test_solver_unique_one_to_one(self):
        costs = np.full((4,4), 10.)
        for i in range(4): costs[i,3-i] = i
        mapping, info = assign_group(costs, ['A','B','C','D'], ['a','b','c','d'], 1e-6)
        self.assertEqual(mapping, {'A':'d','B':'c','C':'b','D':'a'})
        self.assertGreater(info['margin_A'], 0)

    def test_rigid_fit_and_degeneracy(self):
        x = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        y = x @ np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]]) + [2.,3.,4.]
        r, t, rmsd = fit_frame(x, y)
        self.assertTrue(np.allclose(x @ r+t, y))
        self.assertLess(rmsd, 1e-10)
        with self.assertRaisesRegex(InputProblem, 'DEGENERATE_PROTEIN_FRAME'):
            fit_frame(np.zeros((4,3)), np.zeros((4,3)))


@unittest.skipUnless(FIXTURE, 'Set MCSA_1LDM_TEST_CIF to four-pair Chai1 sample 3')
class LDHTests(unittest.TestCase):
    def setUp(self):
        self.case = load_case('1LDM')
        self.context = read_case(FIXTURE, self.case)

    def variant(self, change, stage='all', atom_map=None, site_map=None):
        s = gemmi.read_structure(FIXTURE)
        change(s)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'new_structure.cif'
            s.make_mmcif_document().write_file(str(p))
            result = evaluate(p, '1LDM', stage, atom_map, site_map)
            json.dumps(result, allow_nan=False)
            return result

    def synthetic(self, cc=3.5, no=3.0):
        c = copy.deepcopy(self.context)
        mapping, _ = assign_sites(c, self.case)
        for chain, slot in mapping.items():
            his = next(a for a in c['protein_chains'][chain][192]['atoms'] if a['atom_name']=='NE2')
            nad, lac = c['ligands'][slot['NAD']], c['ligands'][slot['lactate']]
            nad[1]['xyz'] = lac[1]['xyz'] + [cc,0.,0.]
            lac[2]['xyz'] = his['xyz'] + [0.,no,0.]
        return c

    def test_positive_baseline(self):
        r = evaluate(FIXTURE, '1LDM')
        self.assertEqual((r['mechanism_status'],r['general_status'],r['joint_status']), ('PASS','PASS','PASS'))
        self.assertEqual(r['mechanism']['site_counts'], {'hydride_sites_passed':4,'His_contact_sites_passed':4,'both_sites_passed':4})

    def test_cutoff_boundaries(self):
        for cc, no in [(3.,3.5),(4.,3.5),(3.5,2.7)]:
            self.assertEqual(evaluate_mechanism(self.synthetic(cc,no),self.case)['status'],'PASS')
        for cc, no, tag in [(2.999,3.,'HYDRIDE_DISTANCE_TOO_SHORT'),(4.001,3.,'HYDRIDE_DISTANCE_TOO_LONG'),(3.5,3.501,'HIS193_CONTACT_TOO_LONG')]:
            r = evaluate_mechanism(self.synthetic(cc,no),self.case)
            self.assertEqual(r['status'],'FAIL')
            self.assertIn(tag,r['failure_tags'])

    def test_one_failed_site_not_hidden_by_three_passes(self):
        c = self.synthetic()
        mapping, _ = assign_sites(c,self.case)
        first = mapping[sorted(mapping)[0]]
        c['ligands'][first['NAD']][1]['xyz'] = c['ligands'][first['lactate']][1]['xyz']+[5.,0.,0.]
        r = evaluate_mechanism(c,self.case)
        self.assertEqual(r['site_counts']['both_sites_passed'],3)
        self.assertEqual(r['status'],'FAIL')

    def test_scored_atoms_do_not_choose_assignment(self):
        c = self.synthetic()
        before, _ = assign_sites(c,self.case)
        for role in c['ligands']:
            c['ligands'][role][1]['xyz'] += np.array([100.,50.,30.])
            if role.startswith('lactate'):
                c['ligands'][role][2]['xyz'] += np.array([10.,20.,30.])
        after, _ = assign_sites(c,self.case)
        self.assertEqual(before,after)

    def test_global_rigid_motion_preserves_sites_and_distances(self):
        c = copy.deepcopy(self.context)
        r = np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
        for atom in c['protein_atoms'] + [a for aa in c['ligands'].values() for a in aa]:
            atom['xyz'] = atom['xyz'] @ r + [300.,-80.,12.]
        old, new = evaluate_mechanism(self.context,self.case), evaluate_mechanism(c,self.case)
        self.assertEqual(old['assignment']['mapping'],new['assignment']['mapping'])
        for x,y in zip(old['sites'],new['sites']):
            self.assertAlmostEqual(x['hydride_distance_A'],y['hydride_distance_A'])
            self.assertAlmostEqual(x['His_contact_distance_A'],y['His_contact_distance_A'])

    def test_ND1_cannot_replace_NE2(self):
        c = self.synthetic()
        mapping, _ = assign_sites(c,self.case)
        chain = sorted(mapping)[0]
        oxygen = c['ligands'][mapping[chain]['lactate']][2]['xyz']
        res = c['protein_chains'][chain][192]
        for a in res['atoms']:
            if a['atom_name']=='NE2':a['xyz']=oxygen+[8.,0.,0.]
            if a['atom_name']=='ND1':a['xyz']=oxygen+[2.7,0.,0.]
        r = evaluate_mechanism(c,self.case)
        self.assertEqual(r['site_counts']['His_contact_sites_passed'],3)

    def test_missing_NE2_keeps_other_sites_and_general(self):
        def change(s):
            r = s[0]['A'][192]
            for i in range(len(r)-1,-1,-1):
                if r[i].name=='NE2':del r[i]
        r = self.variant(change)
        self.assertEqual(r['mechanism_status'],'FAIL')
        self.assertIn('CATALYTIC_HIS193_NE2_MISSING',r['mechanism']['failure_tags'])
        self.assertEqual(r['mechanism']['site_counts']['His_contact_sites_passed'],3)
        self.assertTrue(r['general']['recomputed'])
        self.assertIn('PROTEIN_HEAVY_ATOM_MISMATCH',r['general']['failure_tags'])

    def test_wrong_NE2_element_cannot_pass_contact(self):
        def change(s):
            for a in s[0]['A'][192]:
                if a.name=='NE2':a.element=gemmi.Element('C')
        r=self.variant(change)
        self.assertEqual(r['mechanism_status'],'FAIL')
        self.assertIn('CATALYTIC_HIS193_NE2_ELEMENT_MISMATCH',r['mechanism']['failure_tags'])
        self.assertIn('PROTEIN_ATOM_ELEMENT_MISMATCH',r['general']['failure_tags'])

    def test_explicit_site_mapping_is_one_to_one(self):
        mapping, _ = assign_sites(self.context,self.case)
        r = evaluate(FIXTURE,'1LDM',site_map=mapping)
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['mechanism']['assignment']['method'],'USER_VERIFIED_EXPLICIT_SITE_MAP')
        chains=sorted(mapping);mapping[chains[1]]=dict(mapping[chains[0]])
        r=evaluate(FIXTURE,'1LDM',site_map=mapping)
        self.assertEqual(r['mechanism_status'],'ERROR')
        self.assertIn('INVALID_SITE_MAP',r['mechanism']['failure_tags'])
        self.assertEqual(r['general_status'],'PASS')

    def test_real_ambiguity_requires_map(self):
        c = copy.deepcopy(self.context)
        chains=sorted(c['protein_chains'])
        for ch in chains[1:]:
            c['protein_chains'][ch]=copy.deepcopy(c['protein_chains'][chains[0]])
        with self.assertRaisesRegex(InputProblem,'SITE_MAPPING_AMBIGUOUS'):
            assign_sites(c,self.case)

    def test_short_His_contact_is_general_not_extra_mechanism_lower_bound(self):
        r=evaluate_mechanism(self.synthetic(cc=3.5,no=1.),self.case)
        self.assertEqual(r['status'],'PASS')

    def test_actual_too_short_His_contact_fails_general(self):
        mapping,_=assign_sites(self.context,self.case)
        oxygen=self.context['ligands'][mapping['A']['lactate']][2]
        def change(s):
            for a in s[0]['A'][192]:
                if a.name=='NE2':a.pos=gemmi.Position(*(oxygen['xyz']+[1.,0.,0.]))
        r=self.variant(change)
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertIn('F05_NONCOVALENT_SHORT_CONTACT',r['general']['failure_tags'])

    def test_bond_failure_has_atomic_evidence(self):
        p=Path(FIXTURE).with_name('M0584_1LDM__assembly_complete__Protenix__s0.cif')
        r=evaluate(p,'1LDM','general')
        bad=[b for c in r['general']['components'].values() for b in c['ldh_bond_length_outliers']]
        self.assertEqual(len(bad),1)
        self.assertEqual({a['atom_name'] for a in bad[0]['atoms']},{'C9','O1'})
        self.assertGreater(bad[0]['distance_A'],bad[0]['upper_bound_A'])

    def test_lactate_stereo_failure_does_not_erase_mechanism(self):
        mapping,_=assign_sites(self.context,self.case)
        role=mapping[sorted(mapping)[0]]['lactate']
        chain=self.context['ligands'][role][0]['chain']
        names=[a['atom_name'] for a in self.context['ligands'][role]]
        def change(s):
            atoms={a.name:a for a in s[0][chain][0]}
            center=np.array(atoms[names[1]].pos.tolist())
            v1=np.array(atoms[names[0]].pos.tolist())-center
            v2=np.array(atoms[names[3]].pos.tolist())-center
            normal=np.cross(v1,v2);normal/=np.linalg.norm(normal)
            p=np.array(atoms[names[2]].pos.tolist())
            atoms[names[2]].pos=gemmi.Position(*(p-2*np.dot(p-center,normal)*normal))
        r=self.variant(change)
        self.assertTrue(r['mechanism']['recomputed'])
        self.assertEqual(r['mechanism']['hydride_status'],'PASS')
        self.assertIn('INPUT_STEREOCHEMISTRY_MISMATCH',r['general']['failure_tags'])
        bad=r['general']['components'][role]['ldh_stereochemistry']['meaningful_mismatches']
        self.assertEqual((bad[0]['expected_CIP'],bad[0]['observed_CIP']),('S','R'))

    def test_missing_and_extra_components_rejected(self):
        chain=self.context['ligands']['lactate_1'][0]['chain']
        r=self.variant(lambda s:s[0].remove_chain(chain))
        self.assertIn('LIGAND_COMPOSITION_OR_COPY_MISMATCH',r['input_problem'])
        def add(s):
            c=s[0][chain].clone();c.name='EXTRA';s[0].add_chain(c)
        r=self.variant(add)
        self.assertIn('LIGAND_COMPOSITION_OR_COPY_MISMATCH',r['input_problem'])

    def test_monomer_rejected(self):
        def change(s):
            for ch in ('B','C','D'):s[0].remove_chain(ch)
        r=self.variant(change)
        self.assertIn('PROTEIN_COPY_COUNT_MISMATCH',r['input_problem'])

    def test_chain_names_numbers_and_atom_order_not_assumed(self):
        def change(s):
            for ch in s[0]:
                ch.name='Z'+ch.name
                for i,res in enumerate(ch):
                    res.seqid=gemmi.SeqId(i+1001,' ')
                    aa=list(res.clone())
                    for j in range(len(res)-1,-1,-1):del res[j]
                    for a in reversed(aa):res.add_atom(a)
        r=self.variant(change)
        self.assertEqual((r['mechanism_status'],r['general_status']),('PASS','PASS'))

    def test_custom_atom_names(self):
        mapping={r:{k:v for k,v in m.items() if k!='mapping_method'} for r,m in self.context['atom_mapping'].items()}
        chains={a[0]['chain'] for a in self.context['ligands'].values()}
        for m in mapping.values():m['atom_names']=['Q_'+n for n in m['atom_names']]
        def change(s):
            for ch in s[0]:
                if ch.name in chains:
                    for res in ch:
                        for a in res:a.name='Q_'+a.name
        missing=self.variant(change,'mechanism')
        self.assertIn('ATOM_MAPPING_REQUIRED',missing['input_problem'])
        mapped=self.variant(change,'mechanism',mapping)
        self.assertEqual(mapped['mechanism_status'],'PASS')

    def test_stage_isolation(self):
        with patch.object(API,'run_general',side_effect=AssertionError('not requested')):
            r=evaluate(FIXTURE,'1LDM','mechanism')
            self.assertEqual(r['general_status'],'NOT_RUN')
        with patch.dict(API.MECHANISMS,{'ldh_tetramer':lambda *_:(_ for _ in ()).throw(AssertionError('not requested'))}):
            r=evaluate(FIXTURE,'1LDM','general')
            self.assertEqual(r['mechanism_status'],'NOT_RUN')

    def test_general_error_preserves_four_site_results(self):
        with patch.object(API,'run_general',side_effect=ValueError('test')):
            r=evaluate(FIXTURE,'1LDM')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(len(r['mechanism']['sites']),4)
        self.assertEqual(r['general_status'],'ERROR')


if __name__=='__main__':unittest.main()
