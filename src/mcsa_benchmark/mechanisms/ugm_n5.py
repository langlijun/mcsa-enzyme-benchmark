"""Two independently assigned flavin N5 / sugar C1 / leaving O reaction axes."""

from collections import Counter
from itertools import permutations

import numpy as np

from ..general import atom_id, distance
from ..structure import InputProblem, SIDECHAINS
from .adenylate import angle, inside, EPS


def assign_ligands(context, ligand_roles, centers, excluded, policy):
    chains = sorted(centers)
    costs = {}
    for role in ligand_roles:
        atoms = context['ligands'][role]
        anchor = np.mean([a['xyz'] for i,a in enumerate(atoms) if i not in excluded], axis=0)
        costs[role] = {c:float(np.linalg.norm(anchor-centers[c])) for c in chains}
    options = sorted((sum(costs[r][c] for r,c in zip(ligand_roles,p)),p) for p in permutations(chains))
    nearest, ties = {}, []
    for role in ligand_roles:
        ordered = sorted(chains,key=costs[role].get)
        if costs[role][ordered[1]]-costs[role][ordered[0]] <= policy['nearest_tie_tolerance_A']:
            ties.append(role)
        nearest[role] = ordered[0]
    counts = Counter(nearest.values())
    margin = options[1][0]-options[0][0]
    selected = dict(zip(ligand_roles,options[0][1]))
    tags = []
    if ties:
        status = 'ERROR'
        tags = ['LIGAND_POCKET_ASSIGNMENT_TIED']
    elif any(counts[c] != 1 for c in chains):
        status = 'FAIL'
        tags = ['MULTIPLE_LIGANDS_IN_ONE_POCKET','REQUIRED_POCKET_UNOCCUPIED']
    elif margin < policy['minimum_total_margin_A']:
        status = 'ERROR'
        tags = ['LIGAND_POCKET_ASSIGNMENT_LOW_MARGIN']
    elif selected != nearest:
        status = 'ERROR'
        tags = ['GLOBAL_AND_LOCAL_POCKET_ASSIGNMENTS_DISAGREE']
    else:
        status = 'PASS'
    return {'status':status,'failure_tags':tags,'cost_matrix_A':costs,'nearest_assignment':nearest,
            'nearest_occupancy_counts':dict(counts),'selected_assignment':selected if status=='PASS' else None,
            'options':[{'cost_A':v,'assignment':dict(zip(ligand_roles,p))} for v,p in options],
            'total_assignment_margin_A':margin,'tied_ligands':ties,'reactive_atoms_used':False}


def contact(left, right, limits, missing=None):
    if missing or not left or not right:
        return {'status':'UNAVAILABLE','reason':'Required auxiliary atoms are missing.',
                'missing':missing or [],'affects_mechanism_status':False}
    d,a,b = min(((distance(a,b),a,b) for a in left for b in right),key=lambda r:r[0])
    return {'status':'PASS' if inside(d,limits) else 'FAIL','distance_A':d,
            'atoms':[atom_id(a),atom_id(b)],'bounds_A':limits,'affects_mechanism_status':False}


def evaluate_mechanism(context, case):
    roles, criteria = case['reaction_atoms'], case['mechanism_criteria']
    limits, aux_rules = criteria['main_axis'], criteria['auxiliary']
    proteins = context['protein_chains']
    if len(proteins) != 2:
        raise InputProblem('MECHANISM_SITE_COUNT_MISMATCH','Two protein pockets are required.')
    centers, pocket_atoms, pocket_missing = {}, {}, {}
    for chain, residues in sorted(proteins.items()):
        anchors, aa, missing = [], [], []
        for pos, name in zip(roles['pocket_sequence_positions'],roles['pocket_residue_names']):
            residue = residues[pos-1]
            if residue['name'] != name:
                raise InputProblem('MECHANISM_RESIDUE_MISMATCH',f'{chain}: expected {name} at sequence position {pos}.')
            atoms = {a['atom_name']:a for a in residue['atoms']}
            if 'CA' not in atoms:
                raise InputProblem('MECHANISM_SITE_ANCHOR_MISSING',f'{chain}: sequence position {pos} CA.')
            anchors.append(atoms['CA']['xyz'])
            aa.extend(residue['atoms'])
            required = set(('N CA C O '+SIDECHAINS[name]).split())
            missing.extend(f'{chain}:{pos}:{n}' for n in sorted(required-set(atoms)))
        centers[chain] = np.mean(anchors,axis=0)
        pocket_atoms[chain], pocket_missing[chain] = aa, missing
    fad_assignment = assign_ligands(context,roles['FAD_roles'],centers,[roles['FAD_N5_index']],criteria['site_assignment'])
    udp_assignment = assign_ligands(context,roles['UDP_roles'],centers,[roles['UDP_C1_index'],roles['UDP_leaving_O_index']],criteria['site_assignment'])
    candidates, by_pair = [], {}
    for fr in roles['FAD_roles']:
        for ur in roles['UDP_roles']:
            fad, udp = context['ligands'][fr],context['ligands'][ur]
            n,c,o = fad[roles['FAD_N5_index']],udp[roles['UDP_C1_index']],udp[roles['UDP_leaving_O_index']]
            d,theta = distance(n,c),angle(n,c,o)
            tags = []
            if d < limits['distance_A']['min']-EPS: tags.append('REACTION_DISTANCE_TOO_SHORT_FOR_REACTANT_TASK')
            if d > limits['distance_A']['max']+EPS: tags.append('REACTION_DISTANCE_TOO_LONG')
            if not inside(theta,limits['angle_deg']):
                tags.append('DEGENERATE_REACTION_AXIS' if theta is None else 'ATTACK_DIRECTION_MISMATCH')
            row = {'FAD_role':fr,'UDP_role':ur,'FAD_chain':fad[0]['chain'],'UDP_chain':udp[0]['chain'],
                   'atoms':[atom_id(a) for a in (n,c,o)],'N5_C1_A':d,'N5_C1_OUDP_deg':theta,
                   'distance_status':'PASS' if inside(d,limits['distance_A']) else 'FAIL',
                   'angle_status':'PASS' if inside(theta,limits['angle_deg']) else 'FAIL',
                   'status':'FAIL' if tags else 'PASS','failure_tags':tags,
                   'C1_OUDP_A_diagnostic':distance(c,o),
                   'distance_to_upper_boundary_A':limits['distance_A']['max']-d}
            candidates.append(row)
            by_pair[fr,ur] = row
    assignment_statuses = {fad_assignment['status'],udp_assignment['status']}
    assignment_tags = sorted(set(fad_assignment['failure_tags']+udp_assignment['failure_tags']))
    sites, auxiliary = [], []
    if assignment_statuses == {'PASS'}:
        fs = {c:r for r,c in fad_assignment['selected_assignment'].items()}
        us = {c:r for r,c in udp_assignment['selected_assignment'].items()}
        for chain in sorted(proteins):
            axis = dict(by_pair[fs[chain],us[chain]])
            axis['protein_chain'] = chain
            sites.append(axis)
            fad, udp = context['ligands'][fs[chain]],context['ligands'][us[chain]]
            arg, missing = [], []
            for pos in roles['Arg_sequence_positions']:
                atoms = {a['atom_name']:a for a in proteins[chain][pos-1]['atoms']}
                for name in roles['Arg_donor_names']:
                    if name not in atoms: missing.append(f'{chain}:{pos}:{name}')
                    else: arg.append(atoms[name])
            support = contact(arg,[udp[i] for i in roles['UDP_phosphate_oxygen_indices']],aux_rules['Arg_contact_A'],missing)
            fc = contact([a for i,a in enumerate(fad) if i!=roles['FAD_N5_index']],pocket_atoms[chain],aux_rules['FAD_pocket_contact_A'],pocket_missing[chain])
            uc = contact([a for i,a in enumerate(udp) if i not in (roles['UDP_C1_index'],roles['UDP_leaving_O_index'])],pocket_atoms[chain],aux_rules['UDP_pocket_contact_A'],pocket_missing[chain])
            bond_pass = inside(axis['C1_OUDP_A_diagnostic'],case['case_general_requirements']['C1_OUDP_A'])
            aux_states = [s['status'] for s in (support,fc,uc)]
            legacy = ('UNAVAILABLE' if 'UNAVAILABLE' in aux_states else
                      'PASS' if axis['status']=='PASS' and bond_pass and all(s=='PASS' for s in aux_states) else 'FAIL')
            auxiliary.append({'protein_chain':chain,'Arg_support':support,'FAD_pocket_contact':fc,'UDP_pocket_contact':uc,
                              'intact_bond_diagnostic_pass':bond_pass,'legacy_six_check_status':legacy,
                              'affects_mechanism_status':False})
    status = ('FAIL' if 'FAIL' in assignment_statuses else 'ERROR' if 'ERROR' in assignment_statuses
              else 'PASS' if all(s['status']=='PASS' for s in sites) else 'FAIL')
    main_tags = sorted(set(assignment_tags+[tag for s in sites for tag in s['failure_tags']]))
    legacy_states = [a['legacy_six_check_status'] for a in auxiliary]
    legacy_status = ('UNAVAILABLE' if len(legacy_states)!=2 or 'UNAVAILABLE' in legacy_states else
                     'PASS' if all(s=='PASS' for s in legacy_states) else 'FAIL')
    return {'criteria_id':criteria['criteria_id'],'status':status,'failure_tags':main_tags,
            'cutoffs':limits,'declared_sites':2,'passing_sites':sum(s['status']=='PASS' for s in sites),
            'sites':sites,'all_candidate_axes':candidates,'auxiliary_contacts':auxiliary,
            'site_assignment':{'FAD':fad_assignment,'UDP_Galp':udp_assignment},
            'legacy_six_check_status':legacy_status,'legacy_is_primary':False,
            'general_checks_included':False,'water_required':False,
            'input_state_contract':case['input_state_contract'],'scope':criteria['scope']}
