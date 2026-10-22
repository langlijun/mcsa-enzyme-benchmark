"""One fixed serotonin/acetyl-CoA carbonyl axis; protein contacts are auxiliary."""

from ..general import atom_id, distance
from .adenylate import angle, inside, EPS


def evaluate_mechanism(context, case):
    roles, criteria = case['reaction_atoms'], case['mechanism_criteria']
    coa, serotonin = context['ligands'][roles['acetyl_CoA_role']], context['ligands'][roles['serotonin_role']]
    n,c,o,s = serotonin[roles['serotonin_N_index']],coa[roles['acetyl_C_index']],coa[roles['carbonyl_O_index']],coa[roles['CoA_S_index']]
    d,theta = distance(n,c),angle(n,c,o)
    distance_pass = d <= criteria['distance_upper_A']+EPS
    angle_pass = inside(theta,criteria['angle_deg'])
    tags = []
    if not distance_pass:tags.append('REACTION_DISTANCE_TOO_LONG')
    if not angle_pass:tags.append('DEGENERATE_REACTION_AXIS' if theta is None else 'ATTACK_DIRECTION_MISMATCH')
    auxiliary = []
    chain,residues = next(iter(context['protein_chains'].items()))
    for definition in criteria['auxiliary_contacts']:
        residue = residues[definition['position']-1]
        atom = next((a for a in residue['atoms'] if a['atom_name']==definition['atom']),None)
        target = context['ligands'][definition['target_role']][definition['target_index']]
        item = {'id':definition['id'],'protein_chain':chain,'sequence_position':definition['position'],
                'affects_mechanism_status':False,'hard_cutoff_A':None}
        if residue['name'] != definition['residue'] or atom is None:
            item.update(status='UNAVAILABLE_DIAGNOSTIC_ONLY',distance_A=None,
                        reason='Required auxiliary atom or residue identity is absent.',
                        expected_residue=definition['residue'],expected_atom=definition['atom'])
        else:
            item.update(status='MEASURED_DIAGNOSTIC_ONLY',distance_A=distance(atom,target),
                        atoms=[atom_id(atom),atom_id(target)])
        auxiliary.append(item)
    sensitivities = []
    for upper in criteria['sensitivity_distance_upper_A']:
        for lower_a,upper_a in criteria['sensitivity_angle_windows_deg']:
            good = d <= upper+EPS and inside(theta,{'min':lower_a,'max':upper_a})
            sensitivities.append({'distance_upper_A':upper,'angle_lower_deg':lower_a,'angle_upper_deg':upper_a,
                                  'status':'PASS' if good else 'FAIL'})
    return {'criteria_id':criteria['criteria_id'],'status':'PASS' if distance_pass and angle_pass else 'FAIL',
            'failure_tags':tags,'declared_sites':1,'passing_sites':int(distance_pass and angle_pass),
            'axis':{'atoms':[atom_id(a) for a in [n,c,o]],'N_C_A':d,'N_C_O_deg':theta,
                    'distance_status':'PASS' if distance_pass else 'FAIL','angle_status':'PASS' if angle_pass else 'FAIL',
                    'distance_to_upper_boundary_A':criteria['distance_upper_A']-d,
                    'angle_distance_to_nearest_boundary_deg':min(theta-criteria['angle_deg']['min'],criteria['angle_deg']['max']-theta) if theta is not None else None},
            'cutoffs':{'distance_upper_A':criteria['distance_upper_A'],'distance_lower_A':None,'angle_deg':criteria['angle_deg']},
            'site_assignment':{'basis':'One chemically distinct copy of each substrate in the declared monomer input.',
                               'acetyl_CoA_chain':c['chain'],'serotonin_chain':n['chain'],'reaction_score_used':False},
            'auxiliary_contacts':auxiliary,'auxiliary_affects_status':False,
            'diagnostics':{'N_C_S_deg':angle(n,c,s),'existing_C_S_A':distance(c,s),'existing_C_O_A':distance(c,o),
                           'sub_1p8_A_reactive_contact':d<1.8,'old_bond_checks_belong_to_general':True},
            'sensitivity':sensitivities,'general_checks_included':False,
            'water_required_for_scored_axis':False,'proton_wire_evaluated':False,
            'input_state_contract':case['input_state_contract'],'scope':criteria['scope']}
