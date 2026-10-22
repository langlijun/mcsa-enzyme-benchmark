"""Synthetic contact tests, not physical or experimentally validated complexes."""

import copy
import unittest

import numpy as np
from rdkit import Chem

from mcsa_benchmark.evaluate import load_case
from mcsa_benchmark.mechanisms.ndpk_histidine import evaluate_magnesium


class MagnesiumTests(unittest.TestCase):
    def setUp(self):
        self.case = load_case('1NDP')
        ligands, metals = {}, []
        for site in range(6):
            base = np.array([site*30.,0.,0.])
            atoms = [{'chain':f'L{site}','residue_name':'ATP','residue_number':1,
                      'insertion_code':'','atom_name':f'A{i}','element':'C','xyz':base.copy()} for i in range(31)]
            for i, delta in zip([15,19,23],[-1,0,1]):
                atoms[i]['element']='P'
                atoms[i]['xyz']=base+np.array([delta,0.,0.])
            for axis, indices in enumerate(self.case['reaction_atoms']['Mg_donor_indices'].values()):
                for j,index in enumerate(indices):
                    shift=np.zeros(3);shift[axis]=2.1 if j==0 else 5.+j
                    atoms[index]['xyz']=base+shift
                    atoms[index]['element']='O'
            ligands[f'ATP_{site+1}']=atoms
            metals.append({'chain':f'M{site}','residue_name':'MG','residue_number':1,
                           'insertion_code':'','atom_name':'MG','element':'MG','xyz':base.copy()})
        self.context={'ligands':ligands,'metal_copies':{'MG':metals}}

    def test_three_donor_groups_are_nonbridging_oxygens(self):
        mol=Chem.MolFromSmiles(self.case['ligands']['ATP_1']['smiles'])
        for group, p in [('alpha',15),('beta',19),('gamma',23)]:
            for i in self.case['reaction_atoms']['Mg_donor_indices'][group]:
                a=mol.GetAtomWithIdx(i)
                self.assertEqual(a.GetSymbol(),'O')
                self.assertEqual([n.GetIdx() for n in a.GetNeighbors()],[p])

    def test_three_contacts_all_six_sites(self):
        r=evaluate_magnesium(self.context,self.case)
        self.assertEqual(r['status'],'PASS')
        self.assertEqual(r['passing_ATP_count'],6)
        for site in r['site_results']:
            self.assertEqual(set(site['groups']),{'alpha','beta','gamma'})
            self.assertTrue(all(g['metal']==site['assigned_metals'][0] for g in site['groups'].values()))

    def test_boundaries_inclusive_and_failures_labeled(self):
        for d,tag in [(1.8,None),(2.6,None),(1.799,'MG_ATP_ALPHA_CONTACT_TOO_SHORT'),
                      (2.601,'MG_ATP_ALPHA_CONTACT_TOO_LONG')]:
            with self.subTest(distance=d):
                context=copy.deepcopy(self.context)
                context['ligands']['ATP_1'][16]['xyz']=np.array([d,0.,0.])
                r=evaluate_magnesium(context,self.case)
                self.assertEqual(r['status'],'FAIL' if tag else 'PASS')
                if tag:
                    self.assertIn(tag,r['failure_tags'])

    def test_short_oxygen_cannot_be_bypassed_by_other_passing_oxygen(self):
        self.context['ligands']['ATP_1'][16]['xyz']=np.array([1.7,0.,0.])
        self.context['ligands']['ATP_1'][17]['xyz']=np.array([2.1,0.,0.])
        r=evaluate_magnesium(self.context,self.case)
        self.assertEqual(r['status'],'FAIL')
        self.assertEqual(r['site_results'][0]['groups']['alpha']['nearest']['oxygen_index'],16)

    def test_ribose_or_bridge_oxygen_cannot_replace_phosphate_donor(self):
        atoms=self.context['ligands']['ATP_1']
        for i in [16,17]: atoms[i]['xyz']=np.array([8.,0.,0.])
        for i in [11,14,18,22]: atoms[i]['xyz']=np.array([2.1,0.,0.])
        r=evaluate_magnesium(self.context,self.case)
        self.assertIn('MG_ATP_ALPHA_CONTACT_TOO_LONG',r['failure_tags'])

    def test_beta_and_gamma_fail_independently(self):
        for group in ['beta','gamma']:
            with self.subTest(group=group):
                context=copy.deepcopy(self.context)
                for i in self.case['reaction_atoms']['Mg_donor_indices'][group]:
                    context['ligands']['ATP_1'][i]['xyz']=np.array([8.,0.,0.])
                r=evaluate_magnesium(context,self.case)
                self.assertIn(f'MG_ATP_{group.upper()}_CONTACT_TOO_LONG',r['failure_tags'])

    def test_duplicate_mg_not_force_matched_to_unoccupied_atp(self):
        self.context['metal_copies']['MG'][1]['xyz']=np.array([0.1,0.,0.])
        r=evaluate_magnesium(self.context,self.case)
        self.assertEqual(r['status'],'FAIL')
        self.assertIn('MG_NOT_MATCHED_TO_ATP_SITE',r['failure_tags'])
        self.assertIn('MG_ATP_SITE_UNRECOVERED',r['failure_tags'])
        self.assertEqual(r['passing_ATP_count'],5)
        self.assertEqual(r['site_results'][0]['status'],'PASS')

    def test_no_mg_is_failure_with_explicit_count(self):
        self.context['metal_copies']['MG']=[]
        r=evaluate_magnesium(self.context,self.case)
        self.assertEqual(r['status'],'FAIL')
        self.assertIn('MECHANISM_MG_COPY_COUNT_MISMATCH',r['failure_tags'])
        self.assertEqual(r['passing_ATP_count'],0)

    def test_equally_close_atps_are_not_selected_by_donor_score(self):
        self.context['metal_copies']['MG'][0]['xyz']=np.array([15.,0.,0.])
        r=evaluate_magnesium(self.context,self.case)
        self.assertEqual(r['status'],'ERROR')
        self.assertIn('MG_ATP_ASSIGNMENT_AMBIGUOUS',r['failure_tags'])
        self.assertIsNone(r['assignments'][0]['ligand_instance'])

    def test_assignment_uses_p_centroids_not_passing_oxygen(self):
        for indices in self.case['reaction_atoms']['Mg_donor_indices'].values():
            for i in indices:
                self.context['ligands']['ATP_1'][i]['xyz']+=np.array([30.,0.,0.])
        r=evaluate_magnesium(self.context,self.case)
        self.assertEqual(r['assignments'][0]['ligand_instance'],'ATP_1')
        self.assertEqual(r['site_results'][0]['status'],'FAIL')

    def test_metal_permutation_does_not_change_status(self):
        self.context['metal_copies']['MG'].reverse()
        for i,mg in enumerate(self.context['metal_copies']['MG']): mg['chain']=f'NEW_{i}'
        r=evaluate_magnesium(self.context,self.case)
        self.assertEqual(r['status'],'PASS')
        self.assertEqual(r['passing_ATP_count'],6)

    def test_water_and_angle_are_not_undeclared_gates(self):
        rules=self.case['mechanism_criteria']['magnesium']
        self.assertFalse(rules['water_shell_required'])
        self.assertFalse(rules['coordination_angle_gate'])


if __name__=='__main__':
    unittest.main()
