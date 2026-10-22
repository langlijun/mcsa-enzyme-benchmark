"""Case-specific intact SAM bond and consistently labeled input stereochemistry."""

from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler, rdDistGeom

from .general import atom_id, distance
from .mechanisms.adenylate import inside
from .structure import InputProblem


def cip_labels(molecule):
    return {a.GetIdx(): a.GetProp('_CIPCode') for a in molecule.GetAtoms() if a.HasProp('_CIPCode')}


@lru_cache(maxsize=8)
def sulfur_bounds(smiles, sulfur_index):
    molecule = Chem.MolFromSmiles(smiles)
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(molecule)
    rows = []
    for a in molecule.GetAtomWithIdx(sulfur_index).GetNeighbors():
        i, j = sorted([sulfur_index, a.GetIdx()])
        lo, hi = float(bounds[j, i]), float(bounds[i, j])
        rows.append({'indices': [sulfur_index, a.GetIdx()], 'DG_lower_A': lo, 'DG_upper_A': hi,
                     'shared_scaled_lower_A': lo*0.75, 'shared_scaled_upper_A': hi*1.25})
    return rows


def evaluate_sam_checks(context, case):
    roles, rules = case['reaction_atoms'], case['case_general_requirements']
    role = roles['donor_role']
    atoms = context['ligands'][role]
    methyl, sulfur = atoms[roles['methyl_C_index']], atoms[roles['sulfur_index']]
    d = distance(methyl, sulfur)
    bond_ok = inside(d, rules['SAM_methyl_C_S_A'])
    checks = {'SAM.intact_methyl_C_S': 'PASS' if bond_ok else 'FAIL'}
    failures = [] if bond_ok else [{'tag': 'SAM_METHYL_C_S_BOND_OUT_OF_RANGE', 'component': role,
                                    'atoms': [atom_id(methyl), atom_id(sulfur)], 'distance_A': d,
                                    'rule': rules['SAM_methyl_C_S_A']}]
    smiles = case['ligands'][role]['smiles']
    expected = Chem.MolFromSmiles(smiles)
    observed = Chem.Mol(expected)
    observed.RemoveAllConformers()
    conformer = Chem.Conformer(len(atoms))
    for i, a in enumerate(atoms):
        conformer.SetAtomPosition(i, tuple(float(x) for x in a['xyz']))
    observed.AddConformer(conformer)
    Chem.RemoveStereochemistry(observed)
    Chem.AssignAtomChiralTagsFromStructure(observed, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistryFrom3D(observed, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistry(observed, force=True, cleanIt=True)
    legacy_expected, legacy_observed = cip_labels(expected), cip_labels(observed)
    rdCIPLabeler.AssignCIPLabels(expected)
    rdCIPLabeler.AssignCIPLabels(observed)
    modern_expected, modern_observed = cip_labels(expected), cip_labels(observed)
    legacy_bad = sorted(i for i, value in legacy_expected.items() if legacy_observed.get(i) != value)
    modern_bad = sorted(i for i, value in modern_expected.items() if modern_observed.get(i) != value)
    if legacy_bad != modern_bad:
        raise InputProblem('STEREO_LABELING_DISAGREEMENT', 'SAM mismatch atom indices differ between legacy and accurate CIP labelers; do not compare labels across methods.')
    centers = [{'input_atom_index': i, 'atom': atom_id(atoms[i]), 'input_CIP': value,
                'observed_CIP': modern_observed.get(i), 'matches': modern_observed.get(i) == value,
                'legacy_input_label': legacy_expected.get(i), 'legacy_observed_label': legacy_observed.get(i)}
               for i, value in sorted(modern_expected.items())]
    checks['SAM.input_stereochemistry_CIPLabeler'] = 'FAIL' if modern_bad else 'PASS'
    for i in modern_bad:
        failures.append({'tag': 'INPUT_STEREOCHEMISTRY_MISMATCH', 'component': role,
                         'atom': atom_id(atoms[i]), 'input_CIP': modern_expected[i],
                         'observed_CIP': modern_observed.get(i), 'labeler': 'rdCIPLabeler.AssignCIPLabels'})
    return {
        'checks': checks, 'failures': failures,
        'intact_C_S': {'distance_A': d, 'status': checks['SAM.intact_methyl_C_S'],
                      'bounds_A': rules['SAM_methyl_C_S_A'], 'basis': rules['C_S_basis']},
        'stereochemistry': {'method': 'rdCIPLabeler.AssignCIPLabels', 'centers': centers,
                            'legacy_and_modern_mismatch_indices_agree': True, 'mismatch_indices': modern_bad,
                            'note': 'The same valid SAM input has legacy sulfur label R and accurate CIP label S. Compare input and prediction under the same algorithm.'},
        'diagnostics': [{'tag': 'SAM_SULFONIUM_DG_BOUNDS', 'affects_pass': False,
                         'bonds': sulfur_bounds(smiles, roles['sulfur_index']),
                         'note': rules['RDKit_bounds_note'], 'energy_computed': False}],
    }
