"""1IM5 step-1 geometry and named non-water Zn contacts, independent of general."""

from ..general import atom_id, distance
from ..structure import InputProblem
from .adenylate import angle, inside, EPS


def evaluate_mechanism(context, case):
    criteria, roles = case['mechanism_criteria'], case['reaction_atoms']
    cutoffs = criteria['cutoffs']
    if len(context['protein_chains']) != 1:
        raise InputProblem('MECHANISM_SITE_AMBIGUOUS', '1IM5 requires its declared single protein copy.')
    chain, residues = next(iter(context['protein_chains'].items()))
    substrate = context['ligands']['substrate']
    carbon = substrate[roles['substrate_carbonyl_C']]
    oxygen = substrate[roles['substrate_carbonyl_O']]
    ring_n = substrate[roles['substrate_ring_N']]
    checks, measured, tags, errors = {}, {}, [], []

    def protein_atom(position, residue_name, name):
        residue = residues[position - 1]
        if residue['name'] != residue_name:
            raise InputProblem('MECHANISM_RESIDUE_MISMATCH', f'{chain} sequence position {position}: expected {residue_name}.')
        matches = [a for a in residue['atoms'] if a['atom_name'] == name]
        if len(matches) != 1 or matches[0]['element'] != name[0]:
            raise InputProblem('MECHANISM_REQUIRED_ATOM_MISSING', f'{chain}/{residue_name}{position}/{name}')
        return matches[0]

    def sulfur():
        return protein_atom(roles['cysteine_position'], 'CYS', 'SG')

    def zinc():
        if 'Zn' not in context['metals']:
            raise InputProblem('MECHANISM_ZINC_MISSING_OR_AMBIGUOUS', 'Exactly one Zn is required; another element cannot substitute.')
        return context['metals']['Zn']

    def pair(a, b):
        return {'atoms': [atom_id(a), atom_id(b)], 'value': distance(a, b), 'unit': 'angstrom'}

    def carboxylate(position, partner):
        values = [pair(protein_atom(position, 'ASP', name), partner) for name in ('OD1', 'OD2')]
        witness = min(values, key=lambda row: row['value'])
        return {**witness, 'alternatives': values, 'selection': 'Nearest oxygen of this specified carboxylate only.'}

    def attack_angle():
        s = sulfur()
        value = angle(s, carbon, oxygen)
        if value is None:
            raise InputProblem('DEGENERATE_REACTION_ANGLE', 'Coincident atoms make SG-C-O undefined.')
        return {'atoms': [atom_id(a) for a in (s, carbon, oxygen)], 'value': value, 'unit': 'degree'}

    def check(name, measurement, limits, failure_prefix):
        try:
            record = measurement()
        except InputProblem as exc:
            checks[name] = 'ERROR'
            errors.append({'check': name, 'tag': exc.code, 'detail': str(exc)})
            return
        passed = inside(record['value'], limits)
        checks[name] = 'PASS' if passed else 'FAIL'
        measured[name] = {**record, 'limits': limits, 'status': checks[name],
                          'margin_to_lower': record['value'] - limits['min'],
                          'margin_to_upper': limits['max'] - record['value']}
        if not passed:
            if name == 'attack_angle':
                tags.append('ATTACK_ANGLE_OUTSIDE_OPERATIONAL_WINDOW')
            else:
                direction = 'TOO_SHORT' if record['value'] < limits['min'] - EPS else 'TOO_LONG'
                tags.append(f'{failure_prefix}_{direction}')

    check('reaction_distance', lambda: pair(sulfur(), carbon), cutoffs['reaction_distance_A'], 'REACTION_DISTANCE')
    check('attack_angle', attack_angle, cutoffs['attack_angle_deg'], 'ATTACK_ANGLE')
    check('asp10_activation_contact', lambda: carboxylate(roles['base_position'], sulfur()), cutoffs['base_contact_A'], 'CATALYTIC_BASE_CONTACT')
    check('Zn_substrate_ring_N', lambda: pair(zinc(), ring_n), cutoffs['Zn_contact_A'], 'ZN_SUBSTRATE_CONTACT')
    check('Zn_Asp52_O', lambda: carboxylate(roles['zinc_asp_position'], zinc()), cutoffs['Zn_contact_A'], 'ZN_ASP52_CONTACT')
    for position in roles['zinc_histidine_positions']:
        check(f'Zn_His{position}_NE2', lambda: pair(zinc(), protein_atom(position, 'HIS', 'NE2')),
              cutoffs['Zn_contact_A'], f'ZN_HIS{position}_CONTACT')

    diagnostics = {'oxyanion_hole': []}
    for position, resname in zip(roles['oxyanion_backbone_positions'], ('ALA', 'CYS')):
        try:
            record = pair(protein_atom(position, resname, 'N'), oxygen)
            diagnostics['oxyanion_hole'].append({**record, 'sequence_position': position, 'affects_pass': False})
        except InputProblem as exc:
            diagnostics['oxyanion_hole'].append({'sequence_position': position, 'affects_pass': False,
                                                'status': 'NOT_MEASURED', 'reason': str(exc)})
    diagnostics['original_carbonyl_C_O_A'] = distance(carbon, oxygen)
    diagnostics['original_amide_C_N_A'] = distance(carbon, substrate[roles['substrate_amide_N']])
    axis_names = ('reaction_distance', 'attack_angle')
    legacy_axis = ('ERROR' if any(checks[name] == 'ERROR' for name in axis_names) else
                   'PASS' if all(checks[name] == 'PASS' for name in axis_names) else 'FAIL')
    status = 'ERROR' if errors else 'PASS' if all(v == 'PASS' for v in checks.values()) else 'FAIL'
    return {
        'criteria_id': criteria['criteria_id'], 'status': status, 'checks': checks,
        'measurements': measured, 'failure_tags': sorted(set(tags + [e['tag'] for e in errors])),
        'observed_geometry_failure_tags': tags, 'errors': errors,
        'legacy_axis_status': legacy_axis, 'diagnostics': diagnostics,
        'site_assignment': {'protein_chain': chain, 'policy': 'Exactly one declared protein copy; no best-site search.'},
        'metal_atom': atom_id(context['metals']['Zn']) if 'Zn' in context['metals'] else None,
        'scope': 'Selected reactant S-C-O geometry, Asp10 activation proximity and four named non-water Zn contacts. No complete water shell, proton transfer, catalytic activity or full hydrolysis claimed.',
        'angle_interpretation': 'Outside an author operational window is not proof of a reversed trajectory; retain the unrounded angle and margins.',
        'cutoffs': cutoffs,
    }
