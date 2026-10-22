"""One declared sugar C4 / nicotinamide C4 pair; general checks remain separate."""
from ..general import atom_id, distance
from .adenylate import EPS, inside
from ..structure import InputProblem


def evaluate_mechanism(context, case):
    roles, criteria = case['reaction_atoms'], case['mechanism_criteria']
    if case['protein_copies'] != 1 or len(context['protein_chains']) != 1:
        raise InputProblem('PROTEIN_COPY_COUNT_MISMATCH', 'The 1XEL task requires one full subunit.', True)
    donor = context['ligands'][roles['donor_role']][roles['donor_C_index']]
    acceptor = context['ligands'][roles['acceptor_role']][roles['acceptor_C_index']]
    if donor['element'] != 'C' or acceptor['element'] != 'C':
        raise InputProblem('REACTION_ATOM_ELEMENT_MISMATCH', 'Both hydride reporter atoms must be carbon.')
    d = distance(donor, acceptor)
    bounds = criteria['main_axis']['distance_A']
    passed = inside(d, bounds)
    tags = []
    if d < bounds['min'] - EPS:
        tags.append('REACTION_C4_C4_TOO_SHORT')
    if d > bounds['max'] + EPS:
        tags.append('REACTION_C4_C4_TOO_LONG')
    return {
        'criteria_id': criteria['criteria_id'], 'status': 'PASS' if passed else 'FAIL',
        'failure_tags': tags, 'cutoffs': criteria['main_axis'],
        'measurements': {'C4_C4_distance_A': d, 'atoms': [atom_id(donor), atom_id(acceptor)],
                         'distance_status': 'PASS' if passed else 'FAIL'},
        'declared_reaction_pairs': 1, 'protein_copies': 1,
        'pairing': 'Unique input UDP-galactose and oxidized NAD; no minimum-distance pairing search.',
        'general_checks_included': False, 'water_required': False, 'angles_scored': False,
        'residue_contacts_scored': False, 'scope': criteria['scope'],
    }
