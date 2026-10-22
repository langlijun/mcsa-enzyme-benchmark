"""ADC-to-PLP approach, independent of the existing LLP covalent bond checks."""

import numpy as np

from ..general import atom_id, distance
from .adenylate import angle, EPS
from .ugm_n5 import assign_ligands


def evaluate_mechanism(context, case):
    roles, criteria = case['reaction_atoms'], case['mechanism_criteria']
    centers, site_atoms, missing = {}, {}, []
    for chain, residues in sorted(context['protein_chains'].items()):
        residue = residues[roles['LLP_position']-1]
        atoms = {a['atom_name']: a for a in residue['atoms']}
        required = roles['PLP_ring_names'] + [roles['PLP_C4prime_name']]
        absent = [n for n in required if n not in atoms]
        if residue['name'] != 'LLP' or absent:
            missing.append({'protein_chain':chain, 'sequence_position':roles['LLP_position'],
                            'residue_name':residue['name'], 'missing_atoms':absent})
            continue
        centers[chain] = np.mean([atoms[n]['xyz'] for n in roles['PLP_ring_names']], axis=0)
        site_atoms[chain] = atoms
    base = {'criteria_id':criteria['criteria_id'], 'distance_upper_A':criteria['distance_upper_A'],
            'distance_lower_A':None, 'declared_sites':2, 'general_checks_included':False,
            'water_required':False, 'scope':criteria['scope'], 'passing_sites':0, 'sites':[],
            'input_state_contract':case['input_state_contract']}
    if missing:
        return {**base, 'status':'FAIL', 'failure_tags':['REQUIRED_PLP_REACTION_SITE_MISSING'],
                'missing_site_evidence':missing, 'all_candidate_axes':[]}
    assignment = assign_ligands(context, roles['ADC_roles'], centers, [roles['ADC_N_index']], criteria['site_assignment'])
    candidates = []
    for role in roles['ADC_roles']:
        n = context['ligands'][role][roles['ADC_N_index']]
        for chain, atoms in sorted(site_atoms.items()):
            c, nz = atoms[roles['PLP_C4prime_name']], atoms.get(roles['Lys_NZ_name'])
            d = distance(n,c)
            theta = angle(n,c,nz) if nz else None
            candidates.append({'ADC_role':role, 'substrate_chain':n['chain'], 'protein_chain':chain,
                               'atoms':[atom_id(n),atom_id(c)], 'N_C4prime_A':d,
                               'N_C4prime_NZ_deg':theta, 'N_to_NZ_A':distance(n,nz) if nz else None,
                               'auxiliary_NZ_atom':atom_id(nz) if nz else None,
                               'angle_status':'DIAGNOSTIC_ONLY' if theta is not None else 'UNAVAILABLE_DIAGNOSTIC_ONLY',
                               'status':'PASS' if d <= criteria['distance_upper_A']+EPS else 'FAIL',
                               'failure_tags':[] if d <= criteria['distance_upper_A']+EPS else ['REACTION_DISTANCE_TOO_LONG'],
                               'distance_to_upper_boundary_A':criteria['distance_upper_A']-d,
                               'sub_1p8_A_contact_diagnostic':d < 1.8})
    if assignment['status'] != 'PASS':
        return {**base, 'status':assignment['status'], 'failure_tags':assignment['failure_tags'],
                'site_assignment':assignment, 'all_candidate_axes':candidates}
    sites = [row for row in candidates if assignment['selected_assignment'][row['ADC_role']] == row['protein_chain']]
    passed = sum(row['status']=='PASS' for row in sites)
    tags = sorted({tag for row in sites for tag in row['failure_tags']})
    if passed == 1:
        tags.append('PARTIAL_SITE_RECOVERY')
    return {**base, 'status':'PASS' if passed == 2 else 'FAIL', 'failure_tags':tags,
            'passing_sites':passed, 'sites':sites, 'all_candidate_axes':candidates,
            'site_assignment':assignment,
            'distance_upper_sensitivity':{str(t):{'passing_sites':sum(row['N_C4prime_A'] <= t+EPS for row in sites),
                'status':'PASS' if all(row['N_C4prime_A'] <= t+EPS for row in sites) else 'FAIL'}
                for t in criteria['sensitivity_upper_A']}}
