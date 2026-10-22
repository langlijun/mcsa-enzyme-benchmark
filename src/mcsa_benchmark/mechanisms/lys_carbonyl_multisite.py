"""Fixed four-site pre-addition task; assignment never optimizes reaction scores."""

from collections import Counter
from itertools import permutations

import numpy as np

from ..general import atom_id, distance
from ..structure import InputProblem
from .adenylate import angle, inside, EPS


def evaluate_mechanism(context, case):
    roles, criteria = case['reaction_atoms'], case['mechanism_criteria']
    bounds = criteria['main_axis']
    sites = {}
    for chain, residues in sorted(context['protein_chains'].items()):
        residue = residues[roles['protein_sequence_position_1based'] - 1]
        if residue['name'] != roles['protein_residue_name']:
            raise InputProblem('MECHANISM_RESIDUE_MISMATCH', f'{chain}: expected Lys161')
        atoms = {a['atom_name']: a for a in residue['atoms']}
        for name in roles['anchor_names'] + [roles['nucleophile_name']]:
            if name not in atoms or atoms[name]['element'] != name[0]:
                raise InputProblem('MECHANISM_REQUIRED_ATOM_MISSING', f'{chain}: Lys161 {name}')
        sites[chain] = {'nu': atoms[roles['nucleophile_name']],
                        'anchor': np.mean([atoms[n]['xyz'] for n in roles['anchor_names']], axis=0)}
    ligands = context['ligands']
    if len(sites) != 4 or len(ligands) != 4:
        raise InputProblem('MECHANISM_SITE_COUNT_MISMATCH', 'Require four protein copies and four pyruvates.')
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
            tags = []
            if d < bounds['distance_A']['min'] - EPS:
                tags.append('REACTION_DISTANCE_TOO_SHORT_FOR_REACTANT_TASK')
            if d > bounds['distance_A']['max'] + EPS:
                tags.append('REACTION_DISTANCE_TOO_LONG')
            if not inside(theta, bounds['angle_deg']):
                tags.append('DEGENERATE_REACTION_AXIS' if theta is None else 'ATTACK_DIRECTION_MISMATCH')
            row = {'protein_chain': chain, 'ligand_instance': instance, 'ligand_chain': center['chain'],
                   'atoms': [atom_id(a) for a in (site['nu'], center, oxygen)],
                   'distance_A': d, 'angle_deg': theta, 'carbonyl_C_O_A_diagnostic': distance(center, oxygen),
                   'distance_status': 'PASS' if inside(d, bounds['distance_A']) else 'FAIL',
                   'angle_status': 'PASS' if inside(theta, bounds['angle_deg']) else 'FAIL',
                   'status': 'FAIL' if tags else 'PASS', 'failure_tags': tags}
            candidates.append(row)
            lookup[instance, chain] = row
    # Enumeration is small (4!); retain every alternative rather than hiding ties.
    options = []
    for perm in permutations(chains):
        assignment = dict(zip(instances, perm))
        rows = [lookup[i, c] for i, c in assignment.items()]
        passing = sum(r['status'] == 'PASS' for r in rows)
        options.append({'cost_A': sum(costs[i][c] for i, c in assignment.items()),
                        'assignment': assignment, 'passing_site_count': passing,
                        'status': 'PASS' if passing == 4 else 'FAIL'})
    options.sort(key=lambda o: (o['cost_A'], tuple(o['assignment'].values())))
    primary = options[0]
    selected = [dict(lookup[i, c]) for i, c in primary['assignment'].items()]
    margin = options[1]['cost_A'] - primary['cost_A']
    close = [o for o in options if o['cost_A'] - primary['cost_A'] <= criteria['site_assignment']['near_tie_margin_A'] + EPS]
    ambiguous_verdict = len({o['status'] for o in close}) > 1
    tags = sorted({tag for row in selected for tag in row['failure_tags']})
    if ambiguous_verdict:
        tags.append('SITE_ASSIGNMENT_CHANGES_VERDICT')
    nearest = {i: min(costs[i], key=costs[i].get) for i in instances}
    return {
        'criteria_id': criteria['criteria_id'],
        'status': 'ERROR' if ambiguous_verdict else primary['status'],
        'failure_tags': tags, 'declared_site_count': 4,
        'passing_site_count': primary['passing_site_count'], 'selected_axes': selected,
        'all_candidate_axes': candidates, 'cutoffs': bounds,
        'site_assignment': {
            'basis': criteria['site_assignment'], 'cost_matrix_A': costs,
            'selected_assignment': primary['assignment'], 'options': options,
            'next_best_margin_A': margin, 'near_tie': len(close) > 1,
            'near_optimal_assignments': len(close),
            'near_optimal_verdict_invariant': not ambiguous_verdict,
            'near_optimal_passing_site_count_range': [min(o['passing_site_count'] for o in close), max(o['passing_site_count'] for o in close)],
            'all_24_verdict_invariant': len({o['status'] for o in options}) == 1,
            'all_24_passing_site_count_range': [min(o['passing_site_count'] for o in options), max(o['passing_site_count'] for o in options)],
            'independent_nearest_site_diagnostic': nearest,
            'independent_nearest_occupancy_diagnostic': dict(Counter(nearest.values())),
            'forced_away_from_nearest_diagnostic': [i for i in instances if primary['assignment'][i] != nearest[i]],
            'reactive_atoms_used_for_pairing': False,
            'note': 'One-to-one comparison assignment is not proof of physical occupancy. A forced long-range pair can represent an unoccupied site.'},
        'scope': criteria['scope'], 'water_required': False, 'general_checks_included': False,
    }
