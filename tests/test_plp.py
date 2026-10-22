"""LLP polymer support and independent two-stage semantics on modified CIFs."""

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
from rdkit.Chem import AllChem

from mcsa_benchmark import evaluate
from mcsa_benchmark.evaluate import load_case
from mcsa_benchmark.structure import read_case
from mcsa_benchmark.mechanisms.plp_transaldimination import evaluate_mechanism
from mcsa_benchmark.plp_general import local_fragment, evaluate_plp_checks, stereo_details
from mcsa_benchmark import _kernels as k

FIXTURE = os.environ.get('MCSA_1ET0_TEST_CIF')
API = importlib.import_module('mcsa_benchmark.evaluate')


class PLPDefinitionTests(unittest.TestCase):
    def test_criteria_and_atom_roles(self):
        c = load_case('1ET0')
        self.assertEqual(c['mechanism_criteria']['distance_upper_A'],3.5)
        self.assertIsNone(c['mechanism_criteria']['distance_lower_A'])
        self.assertEqual(c['mechanism_criteria']['angle_N_C4prime_NZ'],'DIAGNOSTIC_ONLY')
        self.assertTrue(c['mechanism_criteria']['retrospective'])
        self.assertEqual(c['polymer_modifications'][0]['position'],140)
        self.assertEqual(len(c['polymer_modifications'][0]['atoms']),24)
        mol = Chem.MolFromSmiles(c['ligands']['ADC_1']['smiles'])
        self.assertEqual((mol.GetAtomWithIdx(12).GetSymbol(),mol.GetAtomWithIdx(12).GetFormalCharge()),('N',1))

    def test_electronic_state_scope_is_explicit(self):
        c = load_case('1ET0')
        self.assertFalse(c['input_state_contract']['observed_electronic_or_protonation_state_verified'])
        self.assertFalse(c['input_state_contract']['input_changed'])
        self.assertFalse(c['mechanism_criteria']['water_required'])


@unittest.skipUnless(FIXTURE,'Set MCSA_1ET0_TEST_CIF to AF3 sample 2')
class PLPTests(unittest.TestCase):
    def setUp(self):
        self.case = load_case('1ET0')
        self.context = read_case(FIXTURE,self.case)

    def variant(self,change,stage='all',mapping=None):
        structure = gemmi.read_structure(FIXTURE)
        change(structure)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'new_prediction.cif'
            structure.make_mmcif_document().write_file(str(p))
            result = evaluate(p,'1ET0',stage,mapping)
            json.dumps(result,allow_nan=False)
            return result

    def geometry(self,d):
        c = copy.deepcopy(self.context)
        original = evaluate_mechanism(c,self.case)
        for site in original['sites']:
            atoms = {a['atom_name']:a for a in c['protein_chains'][site['protein_chain']][139]['atoms']}
            n = c['ligands'][site['ADC_role']][12]
            v = n['xyz']-atoms["C4'"]['xyz']
            n['xyz'] = atoms["C4'"]['xyz']+d*v/np.linalg.norm(v)
        return c

    def test_baseline(self):
        r = evaluate(FIXTURE,'1ET0')
        self.assertEqual((r['input_status'],r['mechanism_status'],r['general_status']),('PASS','PASS','FAIL'))
        self.assertEqual(r['mechanism']['passing_sites'],2)
        self.assertFalse(r['historical_results_used'])

    def test_LLP_is_protein_not_third_ligand(self):
        self.assertEqual(set(self.context['ligands']),{'ADC_1','ADC_2'})
        for residues in self.context['protein_chains'].values():
            self.assertEqual(len(residues),269)
            self.assertEqual(residues[139]['name'],'LLP')
        self.assertEqual(sum(a['residue_name']=='LLP' for a in self.context['protein_atoms']),48)

    def test_upper_bound_and_no_lower_mechanism_gate(self):
        for d,status in [(3.5,'PASS'),(3.50001,'FAIL'),(2.399,'PASS'),(1.5,'PASS')]:
            self.assertEqual(evaluate_mechanism(self.geometry(d),self.case)['status'],status)

    def test_one_site_cannot_cover_other(self):
        c = self.geometry(3.)
        c['ligands']['ADC_1'][12]['xyz'] += [100,0,0]
        r = evaluate_mechanism(c,self.case)
        self.assertEqual((r['status'],r['passing_sites']),('FAIL',1))
        self.assertIn('PARTIAL_SITE_RECOVERY',r['failure_tags'])

    def test_pairing_ignores_reactive_atoms(self):
        original = evaluate_mechanism(self.context,self.case)
        c = self.geometry(100.)
        for chain in c['protein_chains'].values():
            for a in chain[139]['atoms']:
                if a['atom_name'] in {"C4'",'NZ'}:
                    a['xyz'] += [20,0,0]
        moved = evaluate_mechanism(c,self.case)
        self.assertEqual(original['site_assignment'],moved['site_assignment'])

    def test_duplicate_pocket_is_failure(self):
        c = copy.deepcopy(self.context)
        for a,b in zip(c['ligands']['ADC_2'],c['ligands']['ADC_1']):
            a['xyz'] = b['xyz'].copy()
        r = evaluate_mechanism(c,self.case)
        self.assertEqual(r['status'],'FAIL')
        self.assertIn('REQUIRED_POCKET_UNOCCUPIED',r['failure_tags'])

    def test_tied_sites_are_explicit_error(self):
        c = copy.deepcopy(self.context)
        left = {a['atom_name']:a for a in c['protein_chains']['A'][139]['atoms']}
        for a in c['protein_chains']['B'][139]['atoms']:
            if a['atom_name'] in self.case['reaction_atoms']['PLP_ring_names']:
                a['xyz'] = left[a['atom_name']]['xyz'].copy()
        r = evaluate_mechanism(c,self.case)
        self.assertEqual(r['status'],'ERROR')
        self.assertIn('LIGAND_POCKET_ASSIGNMENT_TIED',r['failure_tags'])

    def test_stretched_existing_bond_general_only(self):
        def change(s):
            c = s[0]['A'][139].find_atom("C4'",'\x00').pos
            s[0]['A'][139].find_atom('NZ','\x00').pos = gemmi.Position(c.x+2.2,c.y,c.z)
        r = self.variant(change)
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertIn('LLP_INTERNAL_ALDIMINE_BOND_OUT_OF_RANGE',r['general']['failure_tags'])

    def test_existing_bond_window_boundaries(self):
        for value,status in [(1.15,'PASS'),(1.70,'PASS'),(1.149,'FAIL'),(1.701,'FAIL')]:
            c = copy.deepcopy(self.context)
            a = {a['atom_name']:a for a in c['protein_chains']['A'][139]['atoms']}
            a['NZ']['xyz'] = a["C4'"]['xyz']+[value,0,0]
            g = evaluate_plp_checks(c,self.case,FIXTURE)
            self.assertEqual(g['checks']['A.LLP140.existing_C4prime_NZ'],status)

    def test_missing_NZ_does_not_remove_distance_verdict(self):
        r = self.variant(lambda s:s[0]['A'][139].remove_atom('NZ','\x00'))
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['general_status'],'FAIL')
        site = next(x for x in r['mechanism']['sites'] if x['protein_chain']=='A')
        self.assertIsNone(site['N_C4prime_NZ_deg'])

    def test_degenerate_auxiliary_angle_not_primary_error(self):
        c = copy.deepcopy(self.context)
        a = {a['atom_name']:a for a in c['protein_chains']['A'][139]['atoms']}
        a['NZ']['xyz'] = a["C4'"]['xyz'].copy()
        r = evaluate_mechanism(c,self.case)
        self.assertEqual(r['status'],'PASS')
        json.dumps(r,allow_nan=False)

    def test_missing_remote_LLP_atom_general_only(self):
        r = self.variant(lambda s:s[0]['A'][139].remove_atom('OP2','\x00'))
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertEqual(r['general_status'],'FAIL')
        self.assertIn('LLP_LOCAL_CHEMISTRY_INCOMPLETE',r['general']['failure_tags'])

    def test_missing_site_anchor_has_concrete_failure(self):
        r = self.variant(lambda s:s[0]['A'][139].remove_atom('N1','\x00'))
        self.assertEqual(r['mechanism_status'],'FAIL')
        self.assertIn('REQUIRED_PLP_REACTION_SITE_MISSING',r['mechanism']['failure_tags'])
        self.assertTrue(r['general']['recomputed'])

    def test_missing_peptide_cap_general_only(self):
        r = self.variant(lambda s:s[0]['A'][138].remove_atom('C','\x00'))
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertIn('LLP_LOCAL_CHEMISTRY_INCOMPLETE',r['general']['failure_tags'])

    def test_peptide_disconnection_is_measured(self):
        def change(s):
            s[0]['A'][138].find_atom('C','\x00').pos.x += 8
        r = self.variant(change)
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertIn('LLP_PEPTIDE_LINK_OUT_OF_RANGE',r['general']['failure_tags'])

    def test_missing_modification_not_silently_lysine(self):
        def change(s):
            res = s[0]['A'][139]
            res.name = 'LYS'
            keep = {'N','CA','C','O','CB','CG','CD','CE','NZ'}
            for i in range(len(res)-1,-1,-1):
                if res[i].name not in keep:
                    del res[i]
        r = self.variant(change)
        self.assertEqual(r['mechanism_status'],'FAIL')
        self.assertIn('REQUIRED_POLYMER_MODIFICATION_MISSING',r['general']['failure_tags'])

    def test_covalent_neighbors_not_reported_as_external_clashes(self):
        g = evaluate_plp_checks(self.context,self.case,FIXTURE)
        self.assertTrue(all(r['existing_bond']['status']=='PASS' for r in g['LLP_components']))
        self.assertNotIn('LLP_NONLOCAL_PROTEIN_SHORT_CONTACT',[f['tag'] for f in g['failures']])
        self.assertTrue(all(r['status']=='PASS' for c in g['LLP_components'] for r in c['bond_evidence'] if r['peptide_link']))

    def test_capped_graph_preserves_L_lysine(self):
        mol,atoms,missing = local_fragment(self.context,'A',self.case['polymer_modifications'][0])
        self.assertFalse(missing)
        self.assertEqual((mol.GetNumAtoms(),len(atoms)),(29,29))
        self.assertEqual(stereo_details(mol,atoms)[0]['expected_CIP'],'S')
        h = Chem.AddHs(mol)
        self.assertEqual(AllChem.EmbedMolecule(h,randomSeed=123),0)
        h = Chem.RemoveHs(h)
        synthetic = copy.deepcopy(atoms)
        for i,a in enumerate(synthetic):
            a['xyz'] = np.array(h.GetConformer().GetAtomPosition(i))
        self.assertTrue(all(r['status']=='MATCH' for r in stereo_details(mol,synthetic)))

    def test_declared_LLP_single_bond_is_general_failure(self):
        doc = gemmi.cif.read_file(FIXTURE)
        loop = doc.sole_block().init_loop('_chem_comp_bond.',['comp_id','atom_id_1','atom_id_2','value_order'])
        for edge in self.case['polymer_modifications'][0]['bonds']:
            a,b = edge['atoms']
            order = 'SING' if {a,b}=={"C4'",'NZ'} else edge['order']
            loop.add_row(['LLP',gemmi.cif.quote(a),gemmi.cif.quote(b),order])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'bad.cif'; doc.write_file(str(p))
            r = evaluate(p,'1ET0')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertIn('DECLARED_LLP_CHEMISTRY_MISMATCH',r['general']['failure_tags'])

    def test_short_reactive_contact_not_exempt_from_general(self):
        c = self.geometry(1.5)
        def change(s):
            for role in ['ADC_1','ADC_2']:
                n = c['ligands'][role][12]
                s[0][n['chain']][0].find_atom(n['atom_name'],'\x00').pos = gemmi.Position(*n['xyz'])
        r = self.variant(change)
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertIn('F05_NONCOVALENT_SHORT_CONTACT',r['general']['failure_tags'])
        self.assertTrue(any(d['tag']=='COVALENT_LIKE_REACTIVE_CONTACT' for d in r['general']['case_specific_checks']['diagnostics']))

    def test_declared_new_substrate_bond_is_general_only(self):
        doc = gemmi.cif.read_file(FIXTURE)
        loop = doc.sole_block().init_loop('_struct_conn.',[
            'id','conn_type_id','ptnr1_label_asym_id','ptnr1_label_comp_id','ptnr1_label_seq_id','ptnr1_label_atom_id',
            'ptnr2_label_asym_id','ptnr2_label_comp_id','ptnr2_label_seq_id','ptnr2_label_atom_id'])
        loop.add_row(['new','covale','C','LIG_C','.','N1','B','LLP','140',gemmi.cif.quote("C4'")])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'new_bond.cif'; doc.write_file(str(p))
            r = evaluate(p,'1ET0')
        self.assertEqual(r['mechanism_status'],'PASS')
        self.assertIn('UNDECLARED_SUBSTRATE_COVALENT_LINK',r['general']['failure_tags'])

    def test_chain_and_author_numbering_independence(self):
        def change(s):
            for chain in s[0]:
                chain.name = 'NEW_'+chain.name
                for i,res in enumerate(chain):
                    res.seqid = gemmi.SeqId(600+i,' ')
                    atoms = list(res.clone())
                    for j in range(len(res)-1,-1,-1):del res[j]
                    for a in reversed(atoms):res.add_atom(a)
        r = self.variant(change)
        self.assertEqual(r['mechanism_status'],'PASS')
        np.testing.assert_allclose(sorted(s['N_C4prime_A'] for s in r['mechanism']['sites']),
                                   sorted(s['N_C4prime_A'] for s in evaluate_mechanism(self.context,self.case)['sites']),atol=1e-10)

    def test_rigid_transform(self):
        def change(s):
            for chain in s[0]:
                for res in chain:
                    for a in res:
                        a.pos = gemmi.Position(-a.pos.y+10,a.pos.x-20,a.pos.z+30)
        r = self.variant(change,'mechanism')
        self.assertEqual(r['mechanism_status'],'PASS')
        np.testing.assert_allclose(sorted(s['N_C4prime_A'] for s in r['mechanism']['sites']),
                                   sorted(s['N_C4prime_A'] for s in evaluate_mechanism(self.context,self.case)['sites']),atol=1e-10)

    def test_unknown_ADC_names_require_map(self):
        mapping = {role:{k:v for k,v in row.items() if k!='mapping_method'} for role,row in self.context['atom_mapping'].items()}
        for spec in mapping.values():spec['atom_names'] = ['X'+n for n in spec['atom_names']]
        def change(s):
            for chain in ['C','D']:
                for a in s[0][chain][0]:a.name = 'X'+a.name
        r = self.variant(change,'mechanism')
        self.assertIn('ATOM_MAPPING_REQUIRED',r['input_problem'])
        self.assertEqual(self.variant(change,'mechanism',mapping)['mechanism_status'],'PASS')

    def test_stage_isolation(self):
        with patch.object(API,'run_general',side_effect=AssertionError('not requested')):
            r = evaluate(FIXTURE,'1ET0','mechanism')
            self.assertEqual(r['general_status'],'NOT_RUN')
        with patch.dict(API.MECHANISMS,{'plp_transaldimination':lambda *_:(_ for _ in ()).throw(AssertionError('not requested'))}):
            r = evaluate(FIXTURE,'1ET0','general')
            self.assertEqual(r['mechanism_status'],'NOT_RUN')

    def test_stage_error_does_not_erase_other_stage(self):
        with patch.object(API,'run_general',side_effect=ValueError('synthetic general error')):
            r = evaluate(FIXTURE,'1ET0')
            self.assertEqual((r['mechanism_status'],r['general_status']),('PASS','ERROR'))

    def test_missing_ADC_copy_is_explicit(self):
        r = self.variant(lambda s:s[0].remove_chain('C'))
        self.assertEqual(r['input_status'],'FAIL')
        self.assertIn('LIGAND_COMPOSITION_OR_COPY_MISMATCH',r['input_problem'])


if __name__ == '__main__':
    unittest.main()
