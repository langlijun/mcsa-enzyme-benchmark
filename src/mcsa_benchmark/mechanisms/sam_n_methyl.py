"""A single glycine N / SAM methyl-C / S tuple; no general checks are called."""

from ..general import atom_id, distance
from .adenylate import angle, inside, EPS


def evaluate_mechanism(context, case):
    roles, criteria = case['reaction_atoms'], case['mechanism_criteria']
    limits = criteria['main_axis']
    nu = context['ligands'][roles['acceptor_role']][roles['acceptor_N_index']]
    donor = context['ligands'][roles['donor_role']]
    methyl, sulfur = donor[roles['methyl_C_index']], donor[roles['sulfur_index']]
    d, theta, cs = distance(nu, methyl), angle(nu, methyl, sulfur), distance(methyl, sulfur)
    tags = []
    if d < limits['distance_A']['min'] - EPS:
        tags.append('REACTION_DISTANCE_TOO_SHORT_FOR_REACTANT_TASK')
    if d > limits['distance_A']['max'] + EPS:
        tags.append('REACTION_DISTANCE_TOO_LONG')
    if d > criteria['separation_diagnostic_A']:
        tags.append('REACTION_PARTNERS_SEPARATED')
    if not inside(theta, limits['angle_deg']):
        tags.append('DEGENERATE_REACTION_AXIS' if theta is None else 'ATTACK_DIRECTION_MISMATCH')
    axis_pass = inside(d, limits['distance_A']) and inside(theta, limits['angle_deg'])
    state_bounds = case['case_general_requirements']['SAM_methyl_C_S_A']
    return {
        'criteria_id': criteria['criteria_id'], 'status': 'PASS' if axis_pass else 'FAIL',
        'failure_tags': tags, 'cutoffs': limits,
        'measurements': {'N_C_distance_A': d, 'N_C_S_angle_deg': theta,
                         'atoms': [atom_id(a) for a in (nu, methyl, sulfur)],
                         'distance_status': 'PASS' if inside(d, limits['distance_A']) else 'FAIL',
                         'angle_status': 'PASS' if inside(theta, limits['angle_deg']) else 'FAIL'},
        'SAM_C_S_state_diagnostic': {'distance_A': cs, 'atoms': [atom_id(a) for a in (methyl, sulfur)],
                                    'bounds_A': state_bounds, 'status': 'PASS' if inside(cs, state_bounds) else 'FAIL',
                                    'affects_mechanism_status': False, 'scoring_stage': 'general'},
        'legacy_three_condition_status': 'PASS' if axis_pass and inside(cs, state_bounds) else 'FAIL',
        'declared_reaction_pairs': 1, 'protein_copies': case['protein_copies'],
        'pairing': 'Unique declared SAM and glycinate; no search for a better subunit or ligand pair.',
        'general_checks_included': False, 'water_required': False, 'scope': criteria['scope'],
    }
