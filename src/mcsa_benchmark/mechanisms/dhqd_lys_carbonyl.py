"""1QFE two-site carbonyl approach, independent of input chemical validity."""

from collections import Counter
from itertools import permutations

import numpy as np

from ..general import atom_id, distance
from ..structure import InputProblem
from .adenylate import angle, inside, EPS


def relay_diagnostics(context, case):
    """Keep both named His nitrogen contacts; neither determines the verdict."""
    output = []
    for chain, residues in sorted(context['protein_chains'].items()):
        glu = {a['atom_name']: a for a in residues[case['diagnostic_residues']['Glu'] - 1]['atoms']}
        his = {a['atom_name']: a for a in residues[case['diagnostic_residues']['His'] - 1]['atoms']}
        lys = {a['atom_name']: a for a in residues[case['reaction_atoms']['protein_sequence_position_1based'] - 1]['atoms']}
        for nitrogen in ['ND1', 'NE2']:
            for oxygen in ['OE1', 'OE2']:
                a, b = glu.get(oxygen), his.get(nitrogen)
                output.append({'protein_chain': chain, 'contact': f'Glu86.{oxygen}-His143.{nitrogen}',
                               'atoms': [atom_id(x) for x in (a, b) if x is not None],
                               'distance_A': distance(a, b) if a is not None and b is not None else None,
                               'status': 'MEASURED' if a is not None and b is not None else 'ATOM_MISSING',
                               'affects_pass': False})
            a, b = his.get(nitrogen), lys.get('NZ')
            output.append({'protein_chain': chain, 'contact': f'His143.{nitrogen}-Lys170.NZ',
                           'atoms': [atom_id(x) for x in (a, b) if x is not None],
                           'distance_A': distance(a, b) if a is not None and b is not None else None,
                           'status': 'MEASURED' if a is not None and b is not None else 'ATOM_MISSING',
                           'affects_pass': False})
    return output


def evaluate_mechanism(context, case):
    roles, criteria = case['reaction_atoms'], case['mechanism_criteria']
    bounds = criteria['main_axis']
    sites = {}
    for chain, residues in sorted(context['protein_chains'].items()):
        residue = residues[roles['protein_sequence_position_1based'] - 1]
        if residue['name'] != roles['protein_residue_name']:
            raise InputProblem('MECHANISM_RESIDUE_MISMATCH', f'{chain}: expected Lys170')
        atoms = {a['atom_name']: a for a in residue['atoms']}
        for name in roles['anchor_names'] + [roles['nucleophile_name']]:
            if name not in atoms or atoms[name]['element'] != name[0]:
                raise InputProblem('MECHANISM_REQUIRED_ATOM_MISSING', f'{chain}: Lys170 {name}', True)
        sites[chain] = {'nu': atoms[roles['nucleophile_name']],
                        'anchor': np.mean([atoms[n]['xyz'] for n in roles['anchor_names']], axis=0)}
    ligands = context['ligands']
    if len(sites) != 2 or len(ligands) != 2:
        raise InputProblem('MECHANISM_SITE_COUNT_MISMATCH', 'Require two protein copies and two free DHQ molecules.', True)
    chains, instances = sorted(sites), sorted(ligands)
    costs, candidates, lookup = {}, [], {}
    for instance in instances:
        ligand = ligands[instance]
        center, oxygen = ligand[roles['carbonyl_C_index']], ligand[roles['carbonyl_O_index']]
        anchor = np.mean([ligand[i]['xyz'] for i in roles['nonreactive_indices']], axis=0)
        costs[instance] = {}
        for chain in chains:
            site = sites[chain]
            costs[instance][chain] = float(np.linalg.norm(anchor - site['anchor']))
            d, theta = distance(site['nu'], center), angle(site['nu'], center, oxygen)
            distance_pass = d <= bounds['distance_A']['max'] + EPS
            angle_pass = inside(theta, bounds['angle_deg'])
            tags = []
            if not distance_pass:
                tags.append('REACTION_DISTANCE_TOO_LONG')
            if not angle_pass:
                tags.append('DEGENERATE_REACTION_AXIS' if theta is None else 'ATTACK_DIRECTION_MISMATCH')
            row = {'protein_chain': chain, 'ligand_instance': instance, 'ligand_chain': center['chain'],
                   'atoms': [atom_id(a) for a in (site['nu'], center, oxygen)],
                   'distance_A': d, 'angle_deg': theta,
                   'carbonyl_C_O_A_diagnostic': distance(center, oxygen),
                   'distance_status': 'PASS' if distance_pass else 'FAIL',
                   'angle_status': 'PASS' if angle_pass else 'FAIL',
                   'status': 'FAIL' if tags else 'PASS', 'failure_tags': tags}
            candidates.append(row)
            lookup[instance, chain] = row
    options = []
    for perm in permutations(chains):
        assignment = dict(zip(instances, perm))
        passing = sum(lookup[i, c]['status'] == 'PASS' for i, c in assignment.items())
        options.append({'cost_A': sum(costs[i][c] for i, c in assignment.items()),
                        'assignment': assignment, 'passing_site_count': passing,
                        'status': 'PASS' if passing == 2 else 'FAIL'})
    options.sort(key=lambda o: (o['cost_A'], tuple(o['assignment'].values())))
    primary = options[0]
    selected = [dict(lookup[i, c]) for i, c in primary['assignment'].items()]
    margin = options[1]['cost_A'] - primary['cost_A']
    tied = margin <= criteria['site_assignment']['tie_tolerance_A'] + EPS
    tags = sorted({t for row in selected for t in row['failure_tags']})
    if 0 < primary['passing_site_count'] < 2:
        tags.append('PARTIAL_SITE_RECOVERY')
    if tied:
        tags.append('SITE_ASSIGNMENT_NUMERIC_TIE')
    nearest = {i: min(costs[i], key=costs[i].get) for i in instances}
    return {'criteria_id': criteria['criteria_id'], 'status': 'ERROR' if tied else primary['status'],
            'failure_tags': sorted(tags), 'declared_site_count': 2,
            'passing_site_count': primary['passing_site_count'], 'selected_axes': selected,
            'all_candidate_axes': candidates, 'cutoffs': bounds,
            'site_assignment': {'basis': criteria['site_assignment'], 'cost_matrix_A': costs,
                                'selected_assignment': primary['assignment'], 'options': options,
                                'next_best_margin_A': margin, 'numeric_tie': tied,
                                'independent_nearest_site_diagnostic': nearest,
                                'independent_nearest_occupancy_diagnostic': dict(Counter(nearest.values())),
                                'forced_away_from_nearest_diagnostic': [i for i in instances if primary['assignment'][i] != nearest[i]],
                                'reaction_atoms_used_for_pairing': False},
            'relay_diagnostics': relay_diagnostics(context, case),
            'scope': criteria['scope'], 'water_required': False, 'general_checks_included': False,
            'distance_lower_bound_applied': False, 'relay_gates_applied': False}
