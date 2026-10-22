"""Synthetic geometry validates software, not enzyme activity."""
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
from mcsa_benchmark.mechanisms.nitroreductase_hydride import evaluate_mechanism, assign_pockets, kabsch, registration

FIXTURE = os.environ.get('MCSA_1IDT_TEST_CIF')
API = importlib.import_module('mcsa_benchmark.evaluate')


class DefinitionTests(unittest.TestCase):
    def test_exact_reaction_graph_roles(self):
        c = load_case('1IDT')
        d = Chem.MolFromSmiles(c['ligands']['NADH_1']['smiles'])
        f = Chem.MolFromSmiles(c['ligands']['FMN_1']['smiles'])
        c4, n5 = d.GetAtomWithIdx(43), f.GetAtomWithIdx(17)
        self.assertEqual(c4.GetSymbol(), 'C')
        self.assertEqual(c4.GetTotalNumHs(), 2)
        self.assertTrue(c4.IsInRingSize(6))
        self.assertEqual(n5.GetSymbol(), 'N')
        self.assertEqual(n5.GetDegree(), 2)
        self.assertEqual(f.GetAtomWithIdx(8).GetDegree(), 3)  # N10, not N5.
        self.assertEqual(len(c['protein_sequence']), 217)
        self.assertEqual(c['protein_copies'], 2)

    def test_literature_operational_scope(self):
        c = load_case('1IDT')['mechanism_criteria']
        self.assertEqual(c['distance_A']['max'], 3.8)
        self.assertIsNone(c['distance_A']['min'])
        self.assertTrue(c['retrospective'])
        self.assertIn('not a case-calibrated', c['basis'])
        self.assertIn('proximity only', c['scope'])
        self.assertTrue(any(r.get('doi') == '10.1016/j.str.2017.05.002' for r in c['references']))

    def test_assignment_tie_explicit(self):
        r = assign_pockets({'a':np.zeros(3),'b':np.zeros(3)},
                           {'x':np.array([1,0,0]),'y':np.array([-1,0,0])},1e-6)
        self.assertTrue(r['numeric_tie'])

    def test_alignment_rigid_transform(self):
        x = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        target = x @ np.array([[0,-1,0],[1,0,0],[0,0,1]]) + [10,20,30]
        r,t,v = kabsch(x,target)
        self.assertLess(v,1e-12)
        np.testing.assert_allclose(x@r+t,target,atol=1e-12)


@unittest.skipUnless(FIXTURE, 'Set MCSA_1IDT_TEST_CIF to Protenix sample 0')
class NitroreductaseTests(unittest.TestCase):
    def setUp(self):
        self.case = load_case('1IDT')
        self.context = read_case(FIXTURE,self.case)

    def positive(self, d=3.5):
        ctx = copy.deepcopy(self.context)
        for s in evaluate_mechanism(ctx,self.case)['selected_axes']:
            ctx['ligands'][s['donor_instance']][43]['xyz'] = ctx['ligands'][s['flavin_instance']][17]['xyz']+np.array([d,0,0])
        return ctx

    def variant(self, change, stage='all', mapping=None):
        st = gemmi.read_structure(FIXTURE)
        change(st)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'test.cif'
            st.make_mmcif_document().write_file(str(p))
            r = evaluate(p,'1IDT',stage,mapping)
            json.dumps(r,allow_nan=False)
            return r

    def test_existing_sample(self):
        r = evaluate(FIXTURE,'1IDT')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['general_status'],'FAIL')
        self.assertFalse(r['historical_results_used'])
        values = sorted(a['distance_A'] for a in r['mechanism']['selected_axes'])
        np.testing.assert_allclose(values,[3.338844776,3.345247507],atol=1e-8)
        self.assertEqual(len(r['mechanism']['all_candidate_pairs']),4)
        self.assertTrue(any(abs(p['distance_A']-1.6055665154763916)<1e-10 for p in r['general']['short_contact_atom_pairs']))

    def test_upper_inclusive_and_no_lower_gate(self):
        for d in [3.8,3.0,2.0]:
            with self.subTest(distance=d):
                r = evaluate_mechanism(self.positive(d),self.case)
                self.assertEqual(r['status'],'PASS')
                self.assertFalse(r['distance_lower_bound_applied'])
        self.assertEqual(evaluate_mechanism(self.positive(3.801),self.case)['status'],'FAIL')

    def test_partial_sites_fail(self):
        c = self.positive()
        a = evaluate_mechanism(c,self.case)['selected_axes'][0]
        c['ligands'][a['donor_instance']][43]['xyz'] += [10,0,0]
        r = evaluate_mechanism(c,self.case)
        self.assertEqual(r['passing_site_count'],1)
        self.assertIn('PARTIAL_SITE_RECOVERY',r['failure_tags'])

    def test_pairing_not_optimized_by_reaction_distance(self):
        c = self.positive()
        before = evaluate_mechanism(c,self.case)['site_assignment']
        for role in ['NADH_1','NADH_2']:
            c['ligands'][role][43]['xyz'] += [100,20,30]
        for role in ['FMN_1','FMN_2']:
            c['ligands'][role][17]['xyz'] -= [100,20,30]
        after = evaluate_mechanism(c,self.case)['site_assignment']
        self.assertEqual(before['NADH'],after['NADH'])
        self.assertEqual(before['FMN'],after['FMN'])

    def test_two_pairs_in_one_pocket_do_not_pass(self):
        c = copy.deepcopy(self.context)
        rotation, translation, _ = registration(c,self.case)
        pocket = np.asarray(self.case['reference_sites']['A']['pocket_centroid_excluding_N5'])
        for i,s in enumerate(evaluate_mechanism(c,self.case)['selected_axes']):
            f = c['ligands'][s['flavin_instance']]
            center = np.mean([f[j]['xyz'] for j in self.case['reaction_atoms']['FMN_ring_indices'] if j!=17],axis=0)
            target = (pocket+np.array([0,i,0])-translation) @ rotation.T
            displacement = target-center
            for role in [s['donor_instance'],s['flavin_instance']]:
                for a in c['ligands'][role]: a['xyz'] += displacement
        r = evaluate_mechanism(c,self.case)
        self.assertEqual(r['proximity_passing_pair_count'],2)
        self.assertEqual(r['status'],'FAIL')
        self.assertLess(r['passing_site_count'],2)
        self.assertIn('DISTINCT_POCKET_OCCUPANCY_NOT_RECOVERED',r['failure_tags'])

    def test_orientation_not_false_re_si(self):
        r = evaluate_mechanism(self.context,self.case)
        for a in r['selected_axes']:
            self.assertFalse(a['orientation_diagnostics']['affects_pass'])
            self.assertEqual(a['orientation_diagnostics']['face_status'],'NOT_CERTIFIED')

    def test_zero_axis_remains_finite(self):
        r = evaluate_mechanism(self.positive(0),self.case)
        self.assertEqual(r['status'],'PASS')  # Compression belongs to general.
        self.assertTrue(all(a['orientation_diagnostics']['donor_vector_to_FMN_plane_normal_deg'] is None for a in r['selected_axes']))
        json.dumps(r,allow_nan=False)

    def test_rigid_transform_invariance(self):
        def change(s):
            for chain in s[0]:
                for residue in chain:
                    for a in residue:
                        a.pos = gemmi.Position(-a.pos.y+10,a.pos.x+20,a.pos.z+30)
        r = self.variant(change,'mechanism')
        self.assertEqual(r['mechanism_status'],'PASS')
        np.testing.assert_allclose(sorted(a['distance_A'] for a in r['mechanism']['selected_axes']),
                                   [3.338844776,3.345247507],atol=1e-8)

    def test_chain_and_residue_renaming_atom_order_invariance(self):
        def change(s):
            for ch in s[0]:
                ch.name = 'new_'+ch.name
                for i,r in enumerate(ch):
                    r.seqid = gemmi.SeqId(1000+i,' ')
                    aa = list(r.clone())
                    for j in range(len(r)-1,-1,-1): del r[j]
                    for a in reversed(aa): r.add_atom(a)
        r = self.variant(change,'mechanism')
        self.assertEqual(r['mechanism_status'],'PASS')

    def test_missing_ligand(self):
        r = self.variant(lambda s:s[0].remove_chain('C'))
        self.assertEqual(r['input_status'],'FAIL')

    def test_unknown_atom_names_and_explicit_map(self):
        mapping = {k:{n:v for n,v in value.items() if n!='mapping_method'} for k,value in evaluate(FIXTURE,'1IDT','mechanism')['atom_mapping'].items()}
        for v in mapping.values(): v['atom_names']=['new_'+n for n in v['atom_names']]
        def change(s):
            for ch in s[0]:
                if ch.name in ['C','D','E','F']:
                    for r in ch:
                        for a in r: a.name='new_'+a.name
        self.assertEqual(self.variant(change,'mechanism')['mechanism_status'],'ERROR')
        self.assertEqual(self.variant(change,'mechanism',mapping)['mechanism_status'],'PASS')

    def test_independent_stages_and_general_failure(self):
        with patch.object(API,'run_general',side_effect=ValueError('synthetic error')):
            r = evaluate(FIXTURE,'1IDT')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['general_status'],'ERROR')
        with patch.object(API,'run_general',side_effect=AssertionError('must not run')):
            self.assertEqual(evaluate(FIXTURE,'1IDT','mechanism')['general_status'],'NOT_RUN')
        with patch.dict(API.MECHANISMS,{'nitroreductase_hydride':lambda *_: (_ for _ in ()).throw(AssertionError('must not run'))}):
            self.assertEqual(evaluate(FIXTURE,'1IDT','general')['mechanism_status'],'NOT_RUN')

    def test_missing_CA_anchor(self):
        r = self.variant(lambda s:s[0]['A'][10].remove_atom('CA','\x00'))
        self.assertEqual(r['mechanism_status'],'ERROR')
        self.assertIn('PROTEIN_CA_ANCHOR_MISSING',r['mechanism']['failure_tags'])
        self.assertEqual(r['general_status'],'FAIL')

