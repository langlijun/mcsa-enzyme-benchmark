"""Three same-site ornithine N -> carbamoyl C=O tuples; no proton-transfer gate."""

from collections import defaultdict
from itertools import product

import numpy as np
from scipy.spatial import cKDTree

from ..general import atom_id, distance
from .adenylate import angle, inside, EPS


def evaluate_mechanism(context, case):
    roles, criteria = case['reaction_atoms'], case['mechanism_criteria']
    limits = criteria['main_axis']
    protein_atoms = {chain: [a for r in residues for a in r['atoms']]
                     for chain, residues in sorted(context['protein_chains'].items())}
    trees = {chain: cKDTree(np.array([a['xyz'] for a in atoms])) for chain, atoms in protein_atoms.items()}
    ligands = context['ligands']
    by_role = {role: [name for name in ligands if case['ligands'][name]['copy_group'] == role]
               for role in ('ORN', 'CP')}
    assignments = []
    occupancy = {chain: defaultdict(list) for chain in protein_atoms}
    ambiguous = False
    for instance, atoms in ligands.items():
        costs, witnesses = {}, {}
        for chain, tree in trees.items():
            distances, indices = tree.query(np.array([a['xyz'] for a in atoms]))
            closest = int(np.argmin(distances))
            costs[chain] = float(distances[closest])
            witnesses[chain] = {'ligand_atom': atom_id(atoms[closest]),
                                'protein_atom': atom_id(protein_atoms[chain][int(indices[closest])])}
        ordered = sorted(costs, key=costs.get)
        margin = costs[ordered[1]] - costs[ordered[0]]
        tied = margin <= criteria['site_assignment']['tie_tolerance_A']
        chain = None if tied else ordered[0]
        group = case['ligands'][instance]['copy_group']
        if chain is not None:
            occupancy[chain][group].append(instance)
        ambiguous |= tied
        assignments.append({'ligand_instance': instance, 'role': group, 'ligand_chain': atoms[0]['chain'],
                            'protein_chain': chain, 'status': 'AMBIGUOUS' if tied else 'ASSIGNED',
                            'costs_A': costs, 'closest_contact_witnesses': witnesses, 'margin_A': margin,
                            'reaction_distance_or_angle_used': False})
    candidates, lookup = [], {}
    for orn_name, cp_name in product(by_role['ORN'], by_role['CP']):
        orn, cp = ligands[orn_name], ligands[cp_name]
        nu = orn[roles['ORN_terminal_N_index']]
        carbon, oxygen = cp[roles['CP_carbonyl_C_index']], cp[roles['CP_carbonyl_O_index']]
        d, theta = distance(nu, carbon), angle(nu, carbon, oxygen)
        tags = []
        if d < limits['distance_A']['min'] - EPS:
            tags.append('REACTION_DISTANCE_TOO_SHORT_FOR_REACTANT_TASK')
        if d > limits['distance_A']['max'] + EPS:
            tags.append('REACTION_DISTANCE_TOO_LONG')
        if not inside(theta, limits['angle_deg']):
            tags.append('DEGENERATE_REACTION_AXIS' if theta is None else 'ATTACK_DIRECTION_MISMATCH')
        row = {'ORN_instance': orn_name, 'CP_instance': cp_name, 'ORN_chain': nu['chain'], 'CP_chain': carbon['chain'],
               'atoms': [atom_id(a) for a in (nu, carbon, oxygen)],
               'distance_A': d, 'angle_deg': theta,
               'distance_status': 'PASS' if inside(d, limits['distance_A']) else 'FAIL',
               'angle_status': 'PASS' if inside(theta, limits['angle_deg']) else 'FAIL',
               'status': 'FAIL' if tags else 'PASS', 'failure_tags': tags,
               'CP_carbonyl_bond_A_diagnostic': distance(carbon, oxygen),
               'CP_bridge_bond_A_diagnostic': distance(carbon, cp[roles['CP_bridge_O_index']])}
        candidates.append(row)
        lookup[orn_name, cp_name] = row
    sites, selected = [], []
    for chain, occupied in occupancy.items():
        tags = []
        for role in ('ORN', 'CP'):
            n = len(occupied[role])
            if n == 0:
                tags.append(f'{role}_SITE_UNOCCUPIED')
            elif n > 1:
                tags.append(f'MULTIPLE_{role}_AT_SITE')
        measured = len(occupied['ORN']) == len(occupied['CP']) == 1
        if measured:
            axis = {**lookup[occupied['ORN'][0], occupied['CP'][0]], 'protein_chain': chain}
            tags.extend(axis['failure_tags'])
            selected.append(axis)
        sites.append({'protein_chain': chain, 'ORN_instances': occupied['ORN'], 'CP_instances': occupied['CP'],
                      'measured': measured, 'status': 'FAIL' if tags else 'PASS', 'failure_tags': sorted(set(tags))})
    tags = sorted({t for site in sites for t in site['failure_tags']})
    if ambiguous:
        tags.append('OCCUPIED_SITE_ASSIGNMENT_AMBIGUOUS')
    return {
        'criteria_id': criteria['criteria_id'], 'status': 'ERROR' if ambiguous else 'FAIL' if tags else 'PASS',
        'failure_tags': tags, 'declared_site_count': case['protein_copies'],
        'passing_site_count': sum(site['status'] == 'PASS' for site in sites),
        'site_results': sites, 'site_assignments': assignments,
        'selected_axes': selected, 'all_candidate_axes': candidates, 'cutoffs': limits,
        'scope': criteria['scope'], 'input_state_note': criteria['input_state_note'],
        'proton_transfer_assessed': False, 'general_checks_included': False,
    }
