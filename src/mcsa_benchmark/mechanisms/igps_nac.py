"""Published IGPS near-attack geometry; independent of general validity checks."""

import math

from ..general import atom_id, distance
from .adenylate import angle, inside


def evaluate_mechanism(context, case):
    roles = case['reaction_atoms']
    rules = case['mechanism_criteria']
    substrate = context['ligands']['substrate']
    c1, c2p, c6, o2p = [substrate[roles[key + '_index']]
                        for key in ('C1', 'C2prime', 'C6', 'O2prime')]
    d = distance(c1, c2p)
    if min(d, distance(c1, c6)) <= 1e-12:
        raise ValueError('Undefined NAC angle: two defining atoms coincide.')
    theta = angle(c2p, c1, c6)
    if not all(math.isfinite(v) for v in (d, theta)):
        raise ValueError('Nonfinite NAC geometry.')
    distance_ok = d <= rules['distance_A']['max'] + 1e-9
    angle_ok = inside(theta, rules['angle_deg'])
    status = 'PASS' if distance_ok and angle_ok else 'FAIL'
    tags = ([] if distance_ok else ['RING_CLOSURE_DISTANCE_TOO_LONG'])
    if not angle_ok:
        tags.append('RING_CLOSURE_DIRECTION_MISMATCH')
    measurements = {
        'C1_C2prime_distance_A': d,
        'C2prime_C1_C6_angle_deg': theta,
        'distance_atoms': [atom_id(a) for a in (c1, c2p)],
        'angle_atoms': [atom_id(a) for a in (c2p, c1, c6)],
        'distance_status': 'PASS' if distance_ok else 'FAIL',
        'angle_status': 'PASS' if angle_ok else 'FAIL',
    }
    # Missing support atoms must not erase the independently measurable NAC.
    chain, residues = next(iter(context['protein_chains'].items()))
    residue = residues[roles['protein_sequence_position_1based'] - 1]
    nz = [a for a in residue['atoms'] if a['atom_name'] == roles['protein_atom_name'] and a['element'] == 'N']
    lys = {'hard_gate': False, 'basis': rules['lys110_support'], 'protein_chain': chain}
    legacy = rules['legacy_operational_sensitivity']
    if residue['name'] == roles['protein_residue_name'] and len(nz) == 1:
        contact = distance(nz[0], o2p)
        contact_ok = inside(contact, legacy['lys110_distance_A'])
        lys.update(status='MEASURED', distance_A=contact, atoms=[atom_id(nz[0]), atom_id(o2p)],
                   operational_window_status='PASS' if contact_ok else 'FAIL')
    else:
        lys.update(status='UNAVAILABLE', reason='Required Lys110 NZ support atom absent or misidentified.')
        contact_ok = None
    lower_ok = inside(d, legacy['reaction_distance_A'])
    legacy_status = ('FAIL' if not (lower_ok and angle_ok) or contact_ok is False else
                     'UNAVAILABLE' if contact_ok is None else 'PASS')
    return {
        'criteria_id': rules['criteria_id'], 'status': status, 'failure_tags': tags,
        'measurements': measurements, 'lys110_support': lys,
        'legacy_three_gate_sensitivity': {'status': legacy_status, 'rules': legacy,
                                        'reaction_distance_status': 'PASS' if lower_ok else 'FAIL',
                                        'changes_primary_verdict': False},
        'cutoffs': {'distance_A': rules['distance_A'], 'angle_deg': rules['angle_deg']},
        'source': rules['source'], 'scope': rules['scope'],
        'general_checks_included': False,
        'water_required': False,
    }
