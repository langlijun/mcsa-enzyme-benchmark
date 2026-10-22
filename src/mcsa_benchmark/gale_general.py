"""1XEL stereo audit with explicit, chemistry-checked ordinary-phosphate exclusions."""
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from .general import atom_id
from .structure import InputProblem


def labels(mol):
    rdCIPLabeler.AssignCIPLabels(mol)
    return {a.GetIdx(): a.GetProp('_CIPCode') for a in mol.GetAtoms() if a.HasProp('_CIPCode')}


def review_stereochemistry(role, atoms, molecule, metrics, case):
    rules = case['case_general_requirements']
    expected = Chem.Mol(molecule)
    excluded = set(rules['phosphate_formal_stereo_exclusions'][role])
    for index in excluded:
        if not 0 <= index < expected.GetNumAtoms():
            raise InputProblem('INVALID_STEREOCHEMISTRY_EXCEPTION', 'Exclusion index is outside the input graph.')
        atom = expected.GetAtomWithIdx(index)
        terminal = [o for o in atom.GetNeighbors() if o.GetSymbol() == 'O' and o.GetDegree() == 1]
        if atom.GetSymbol() != 'P' or atom.GetDegree() != 4 or len(terminal) != 2 or any(o.GetIsotope() for o in terminal):
            raise InputProblem('INVALID_STEREOCHEMISTRY_EXCEPTION', 'Only the registered ordinary phosphate representation can be excluded.')
    observed = Chem.Mol(expected)
    observed.RemoveAllConformers()
    conformer = Chem.Conformer(len(atoms))
    conformer.Set3D(True)
    for i, atom in enumerate(atoms):
        conformer.SetAtomPosition(i, tuple(float(v) for v in atom['xyz']))
    observed.AddConformer(conformer)
    Chem.RemoveStereochemistry(observed)
    Chem.AssignAtomChiralTagsFromStructure(observed, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistryFrom3D(observed, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistry(observed, force=True, cleanIt=True)
    ex, ob = labels(expected), labels(observed)
    centers = [{'index': i, 'element': expected.GetAtomWithIdx(i).GetSymbol(), 'atom': atom_id(atoms[i]),
                'expected_CIP': v, 'observed_CIP': ob.get(i), 'matches': ob.get(i) == v,
                'excluded_formal_nonbridging_P_label': i in excluded} for i, v in ex.items()]
    failures = [c for c in centers if not c['matches'] and not c['excluded_formal_nonbridging_P_label']]
    if role == case['reaction_atoms']['donor_role']:
        index = rules['sugar_C4_index']
        if ex.get(index) != rules['sugar_C4_expected_CIP']:
            raise InputProblem('INPUT_STEREOCHEMISTRY_CONTRACT_MISMATCH', 'Expected UDP-galactose C4 identity changed.')
    reviewed = dict(metrics)
    reviewed['raw_tetrahedral_stereochemistry_valid'] = metrics['tetrahedral_stereochemistry_valid']
    reviewed['tetrahedral_stereochemistry_valid'] = not failures
    reviewed['gale_stereochemistry'] = {
        'method': rules['stereo_reporter'], 'role': role, 'centers': centers,
        'meaningful_mismatches': failures,
        'excluded_indices': sorted(excluded), 'basis': rules['phosphate_exclusion_basis'],
        'scope': 'Only tetrahedral labels are reviewed; all bond, angle, E/Z, planarity and contact checks are unchanged.',
    }
    return reviewed
