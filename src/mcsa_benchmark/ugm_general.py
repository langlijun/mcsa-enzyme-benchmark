"""Intact UDP-Galp C1-O bond checks, independent of mechanism/site assignment."""

from .general import atom_id, distance
from .mechanisms.adenylate import inside


def evaluate_ugm_checks(context, case):
    roles, rules = case['reaction_atoms'], case['case_general_requirements']
    rows, checks, failures = [], {}, []
    for role in roles['UDP_roles']:
        ligand = context['ligands'][role]
        carbon, oxygen = ligand[roles['UDP_C1_index']], ligand[roles['UDP_leaving_O_index']]
        d = distance(carbon, oxygen)
        passed = inside(d, rules['C1_OUDP_A'])
        row = {'component':role,'atoms':[atom_id(carbon),atom_id(oxygen)],'distance_A':d,
               'bounds_A':rules['C1_OUDP_A'],'status':'PASS' if passed else 'FAIL'}
        rows.append(row)
        checks[role+'.intact_C1_OUDP'] = row['status']
        if not passed:
            failures.append({'tag':'UDP_GLYCOSIDIC_BOND_OUT_OF_RANGE',**row})
    return {'checks':checks,'failures':failures,'bonds':rows,
            'basis':rules['basis'],
            'diagnostics':[{'tag':'DECLARED_FAD_MICROSTATE_SCOPE','affects_pass':False,
                            'contract':case['input_state_contract']}],
            'site_assignment_used':False}
