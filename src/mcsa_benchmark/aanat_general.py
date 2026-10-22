"""Named reactant-bond evidence using the unchanged general chemical tolerances."""

import gemmi
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from . import _kernels as k
from .general import atom_id, distance
from .plp_general import substrate_crosslinks


def evaluate_aanat_checks(context, case, path):
    role = case['reaction_atoms']['acetyl_CoA_role']
    atoms = context['ligands'][role]
    molecule = Chem.MolFromSmiles(case['ligands'][role]['smiles'])
    matrix = rdDistGeom.GetMoleculeBoundsMatrix(molecule)
    lower_factor,upper_factor = case['general_policy']['bond_and_angle_bounds_factors']
    checks,failures,diagnostics,bonds = {},[],[],[]
    for i,j in case['case_general_requirements']['existing_bonds']:
        name = 'C_S' if atoms[j]['element']=='S' else 'C_O'
        low = float(matrix[max(i,j),min(i,j)])*lower_factor
        high = float(matrix[min(i,j),max(i,j)])*upper_factor
        d = distance(atoms[i],atoms[j])
        passed = low <= d <= high
        row = {'id':name,'atoms':[atom_id(atoms[i]),atom_id(atoms[j])],
               'distance_A':d,'lower_A':low,'upper_A':high,'status':'PASS' if passed else 'FAIL',
               'basis':'Same input-graph RDKit bounds and tolerance factors as shared general bond checks.'}
        bonds.append(row)
        checks[f'acetyl_CoA.existing_{name}'] = row['status']
        if not passed:failures.append({'tag':f'ACYL_EXISTING_{name}_BOND_OUT_OF_RANGE',**row})
    indices = case['case_general_requirements']['reactive_carbonyl_plane_indices']
    plane = k.plane_max_deviation([atoms[i]['xyz'] for i in indices])
    limit = case['general_policy']['flatness_max_A']
    checks['acetyl_CoA.thioester_carbonyl_planarity'] = 'PASS' if plane <= limit else 'FAIL'
    if plane > limit:
        failures.append({'tag':'THIOESTER_CARBONYL_NONPLANAR','max_deviation_A':plane,'upper_A':limit,
                         'atoms':[atom_id(atoms[i]) for i in indices]})
    metadata = []
    if str(path).lower().endswith(('.cif','.mmcif','.cif.gz','.mmcif.gz')):
        block = gemmi.cif.read_file(str(path)).sole_block()
        supplied = {}
        for r in block.find('_chem_comp_bond.',['comp_id','atom_id_1','atom_id_2','value_order']):
            if r[0].upper()==atoms[0]['residue_name'].upper():
                pair = tuple(sorted([gemmi.cif.as_string(r[1]),gemmi.cif.as_string(r[2])]))
                supplied.setdefault(pair,set()).add(r[3].upper())
        for i,j in case['case_general_requirements']['existing_bonds']:
            pair = tuple(sorted([atoms[i]['atom_name'],atoms[j]['atom_name']]))
            orders = supplied.get(pair)
            if orders:
                wanted = {'DOUB','DOUBLE','2'} if atoms[j]['element']=='O' else {'SING','SINGLE','1'}
                good = orders <= wanted
                metadata.append({'atoms':list(pair),'orders':sorted(orders),'status':'PASS' if good else 'FAIL'})
                if not good:failures.append({'tag':'DECLARED_ACYL_BOND_ORDER_MISMATCH',**metadata[-1]})
    checks['acetyl_CoA.explicit_reactive_bond_orders'] = ('NOT_APPLICABLE' if not metadata else 'PASS' if all(r['status']=='PASS' for r in metadata) else 'FAIL')
    links = substrate_crosslinks(path,context)
    checks['reactants.no_declared_crosscomponent_covalent_bond'] = 'FAIL' if links else 'PASS'
    if links:failures.append({'tag':'UNDECLARED_REACTANT_COVALENT_LINK','connections':links})
    n = context['ligands'][case['reaction_atoms']['serotonin_role']][case['reaction_atoms']['serotonin_N_index']]
    c = atoms[case['reaction_atoms']['acetyl_C_index']]
    if distance(n,c)<1.8:
        diagnostics.append({'tag':'COVALENT_LIKE_REACTIVE_CONTACT','distance_A':distance(n,c),
                            'atoms':[atom_id(n),atom_id(c)],'affects_pass':False,
                            'note':'Not evidence of a new bond. Shared noncovalent clash checks remain active.'})
    return {'checks':checks,'failures':failures,'diagnostics':diagnostics,'existing_bonds':bonds,
            'thioester_carbonyl_plane':{'atoms':[atom_id(atoms[i]) for i in indices],
                                        'max_deviation_A':plane,'upper_A':limit},
            'explicit_bond_metadata':metadata,'declared_crosslinks':links,
            'scope':'Intact input thioester and ordinary chemical geometry; no His distance gate or water-wire reconstruction.'}
