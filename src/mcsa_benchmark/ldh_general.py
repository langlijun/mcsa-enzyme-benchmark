"""True input stereochemistry for every NAD and lactate copy, independent of sites."""
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler, rdDistGeom

from .general import atom_id, distance
from .structure import InputProblem


def labels(mol):
    rdCIPLabeler.AssignCIPLabels(mol)
    return {a.GetIdx(): a.GetProp('_CIPCode') for a in mol.GetAtoms() if a.HasProp('_CIPCode')}


def review_stereochemistry(role, atoms, molecule, metrics, case):
    rules = case['case_general_requirements']
    expected = Chem.Mol(molecule)
    excluded = set(rules['phosphate_formal_stereo_exclusions'][role])
    for i in excluded:
        if not 0 <= i < expected.GetNumAtoms():
            raise InputProblem('INVALID_STEREOCHEMISTRY_EXCEPTION', 'Exclusion index is outside input graph.')
        a = expected.GetAtomWithIdx(i)
        terminal = [o for o in a.GetNeighbors() if o.GetSymbol() == 'O' and o.GetDegree() == 1]
        if a.GetSymbol() != 'P' or a.GetDegree() != 4 or len(terminal) != 2 or any(o.GetIsotope() for o in terminal):
            raise InputProblem('INVALID_STEREOCHEMISTRY_EXCEPTION', 'Only registered ordinary phosphate labels may be excluded.')
    observed = Chem.Mol(expected)
    observed.RemoveAllConformers()
    conformer = Chem.Conformer(len(atoms))
    conformer.Set3D(True)
    for i, a in enumerate(atoms):
        conformer.SetAtomPosition(i, tuple(float(v) for v in a['xyz']))
    observed.AddConformer(conformer)
    Chem.RemoveStereochemistry(observed)
    Chem.AssignAtomChiralTagsFromStructure(observed, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistryFrom3D(observed, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistry(observed, force=True, cleanIt=True)
    ex, ob = labels(expected), labels(observed)
    if case['ligands'][role]['copy_group'] == 'lactate':
        if ex.get(rules['lactate_C2_input_index']) != rules['lactate_C2_required_CIP']:
            raise InputProblem('INPUT_STEREOCHEMISTRY_CONTRACT_MISMATCH', 'Expected L-lactate S configuration changed.')
    centers = [{'index': i, 'atom': atom_id(atoms[i]), 'element': expected.GetAtomWithIdx(i).GetSymbol(),
                'expected_CIP': v, 'observed_CIP': ob.get(i), 'matches': ob.get(i) == v,
                'excluded_formal_nonbridging_P_label': i in excluded} for i, v in ex.items()]
    mismatches = [r for r in centers if not r['matches'] and not r['excluded_formal_nonbridging_P_label']]
    reviewed = dict(metrics)
    reviewed['raw_tetrahedral_stereochemistry_valid'] = metrics['tetrahedral_stereochemistry_valid']
    reviewed['tetrahedral_stereochemistry_valid'] = not mismatches
    reviewed['ldh_stereochemistry'] = {'role': role, 'centers': centers, 'meaningful_mismatches': mismatches,
                                     'excluded_indices': sorted(excluded), 'basis': rules['basis'],
                                     'counted_centers': sum(not c['excluded_formal_nonbridging_P_label'] for c in centers),
                                     'matched_counted_centers': sum(c['matches'] and not c['excluded_formal_nonbridging_P_label'] for c in centers),
                                     'method': 'rdCIPLabeler.AssignCIPLabels for input and 3D-derived structure'}
    outliers = []
    if metrics['bond_length_outlier_count']:
        bounds = rdDistGeom.GetMoleculeBoundsMatrix(molecule)
        for bond in molecule.GetBonds():
            i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            lo, hi = .75*float(bounds[j,i]), 1.25*float(bounds[i,j])
            observed_distance = distance(atoms[i],atoms[j])
            if not lo <= observed_distance <= hi:
                outliers.append({'input_indices': [i,j], 'atoms': [atom_id(atoms[i]),atom_id(atoms[j])],
                                 'distance_A': observed_distance, 'lower_bound_A': lo, 'upper_bound_A': hi,
                                 'basis': 'Unchanged shared RDKit distance-geometry bounds scaled by 0.75 and 1.25'})
    if len(outliers) != metrics['bond_length_outlier_count']:
        raise InputProblem('BOND_DETAIL_COUNT_MISMATCH', 'Detailed bond evidence disagrees with the shared kernel.')
    reviewed['ldh_bond_length_outliers'] = outliers
    return reviewed
