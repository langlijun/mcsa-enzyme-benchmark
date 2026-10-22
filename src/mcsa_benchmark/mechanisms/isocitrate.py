"""5ICD initial oxidation with score-independent copy and interface assignment."""

from itertools import permutations

import numpy as np
from scipy.spatial.distance import cdist

from ..general import atom_id, distance
from ..structure import InputProblem

EPS = 1e-9


def assignment(costs, tolerance=1e-6):
    choices = sorted((sum(costs[i][j] for i, j in enumerate(p)), p)
                     for p in permutations(range(len(costs[0])), len(costs)))
    margin = choices[1][0] - choices[0][0] if len(choices) > 1 else None
    evidence = {'costs': costs, 'all_assignments': choices, 'margin': margin,
                'scored_geometry_or_outcomes_used': False}
    if margin is not None and margin <= tolerance:
        raise InputProblem('SITE_ASSIGNMENT_AMBIGUOUS', f'Independent assignment tied: {evidence}')
    return choices[0][1], evidence


def protein_atom(context, chain, position, residue_name, name):
    res = context['protein_chains'][chain][position - 1]
    if res['name'] != residue_name:
        raise InputProblem('MECHANISM_RESIDUE_MISMATCH', f'{chain} sequence position {position}: expected {residue_name}')
    hits = [a for a in res['atoms'] if a['atom_name'] == name and a['element'] == name[0]]
    if len(hits) != 1:
        raise InputProblem('MECHANISM_REQUIRED_ATOM_MISSING', f'{chain}/{residue_name}@sequence{position}/{name}')
    return hits[0]


def paired_ligands(context, case):
    def collect(kind):
        return [(name, atoms) for name, atoms in context['ligands'].items()
                if case['ligands'][name]['chemical_role'] == kind]
    donors, acceptors = collect('isocitrate'), collect('nadp')
    roles = case['reaction_atoms']
    def center(atoms, excluded):
        return np.mean([a['xyz'] for i, a in enumerate(atoms) if i != excluded], axis=0)
    costs = [[float(np.linalg.norm(center(d, roles['donor_C']) - center(a, roles['acceptor_C'])))
              for _, a in acceptors] for _, d in donors]
    order, evidence = assignment(costs)
    evidence['method'] = 'NONREACTIVE_HEAVY_CENTROIDS_ONE_TO_ONE'
    evidence['donor_roles'] = [name for name, _ in donors]
    evidence['acceptor_roles'] = [name for name, _ in acceptors]
    pairs = [{'donor_role': name, 'donor': d, 'acceptor_role': acceptors[j][0], 'acceptor': acceptors[j][1]}
             for (name, d), j in zip(donors, order)]
    return pairs, evidence


def assign_protein(context, case, pairs):
    roles = case['reaction_atoms']
    chains = list(context['protein_chains'])
    costs, details = [], []
    for pair in pairs:
        anchors = [pair['acceptor'][i]['xyz'] for i in roles['nadp_protein_anchor_indices']]
        row, table = [], {}
        for chain in chains:
            protein = [(i, a) for i, res in enumerate(context['protein_chains'][chain], 1)
                       if i not in roles['protein_assignment_excluded_positions'] for a in res['atoms']]
            nearest = cdist(np.array([a['xyz'] for _, a in protein]), np.array(anchors)).min(axis=1)
            count = len({i for (i, _), d in zip(protein, nearest) if d <= 4.5})
            minimum = float(nearest.min())
            row.append(-count + minimum/(1+minimum)*0.001)
            table[chain] = {'contact_residue_count_4p5': count, 'minimum_A': minimum}
        costs.append(row)
        details.append(table)
    order, evidence = assignment(costs)
    evidence.update(method='NADP_SCAFFOLD_NONREACTIVE_PROTEIN_CONTACTS_ONE_TO_ONE',
                    chains=chains, contact_tables=details,
                    excluded_sequence_positions=roles['protein_assignment_excluded_positions'])
    for pair, j in zip(pairs, order):
        pair['local_chain'] = chains[j]
        pair['opposite_chain'] = next(c for c in chains if c != chains[j])
    return evidence


def assign_metals(context, case, pairs):
    metals = context['metal_copies'].get('Mg', [])
    if len(metals) != len(pairs):
        raise InputProblem('MECHANISM_MG_MISSING_OR_AMBIGUOUS', f'Require {len(pairs)} Mg, found {len(metals)}')
    if len(metals) == 1:
        pairs[0]['metal'] = metals[0]
        return {'method': 'SINGLE_DECLARED_MG', 'scored_geometry_or_outcomes_used': False}
    roles = case['reaction_atoms']
    centers = []
    for pair in pairs:
        local = protein_atom(context, pair['local_chain'], roles['asp_local_position'], 'ASP', 'CA')
        other = protein_atom(context, pair['opposite_chain'], roles['asp_opposite_position'], 'ASP', 'CA')
        centers.append((local['xyz'] + other['xyz'])/2)
    costs = [[float(np.linalg.norm(center-m['xyz'])) for m in metals] for center in centers]
    order, evidence = assignment(costs)
    evidence.update(method='MG_TO_INTERFACE_BACKBONE_CA_ANCHORS_ONE_TO_ONE',
                    metal_atoms=[atom_id(m) for m in metals])
    for pair, j in zip(pairs, order):
        pair['metal'] = metals[j]
    return evidence


def face_measurement(donor, c4, c3, c5):
    normal = -np.cross(c3['xyz']-c4['xyz'], c5['xyz']-c4['xyz'])
    size = float(np.linalg.norm(normal))
    if size < 1e-10:
        raise InputProblem('DEGENERATE_NICOTINAMIDE_FACE', 'C3N/C4N/C5N do not define a plane')
    normal /= size
    approach = donor['xyz']-c4['xyz']
    length = float(np.linalg.norm(approach))
    if length < 1e-10:
        raise InputProblem('DEGENERATE_HYDRIDE_AXIS', 'Coincident donor and acceptor')
    height = float(normal @ approach)
    theta = float(np.degrees(np.arccos(np.clip(height/length, -1, 1))))
    return {'atoms': [atom_id(a) for a in (donor, c4, c3, c5)], 'value': height,
            'unit': 'angstrom', 'approach_to_re_normal_deg_diagnostic_only': theta}


def evaluate_mechanism(context, case):
    roles, criteria = case['reaction_atoms'], case['mechanism_criteria']
    cuts = criteria['cutoffs']
    pairs, ligand_evidence = paired_ligands(context, case)
    pairing = {'ligands': ligand_evidence}
    protein_error = metal_error = None
    try:
        pairing['protein'] = assign_protein(context, case, pairs)
    except InputProblem as exc:
        protein_error = exc
        pairing['protein'] = {'status': 'ERROR', 'detail': str(exc)}
    try:
        if protein_error and len(pairs) > 1:
            raise protein_error
        pairing['metal'] = assign_metals(context, case, pairs)
    except InputProblem as exc:
        metal_error = exc
        pairing['metal'] = {'status': 'ERROR', 'detail': str(exc)}
    sites = []
    for index, site in enumerate(pairs, 1):
        checks, measured, errors, tags = {}, {}, [], []
        donor, nadp = site['donor'], site['acceptor']
        carbon, acceptor = donor[roles['donor_C']], nadp[roles['acceptor_C']]

        def aa(role, residue_name, name, opposite=False):
            if protein_error:
                raise protein_error
            chain = site['opposite_chain' if opposite else 'local_chain']
            return protein_atom(context, chain, roles[role], residue_name, name)

        def metal():
            if metal_error:
                raise metal_error
            return site['metal']

        def pair(a, b):
            return {'atoms': [atom_id(a), atom_id(b)], 'value': distance(a, b), 'unit': 'angstrom'}

        def minimum_oxygen(partner, oxygens):
            alternatives = [pair(partner, atom) for atom in oxygens]
            chosen = min(alternatives, key=lambda r: r['value'])
            return {**chosen, 'alternatives': alternatives, 'selection': 'Minimum within this specified carboxylate; both bounds applied to minimum'}

        def check(name, measure, limits, prefix):
            try:
                value = measure()
            except InputProblem as exc:
                checks[name] = 'ERROR'
                errors.append({'check': name, 'tag': exc.code, 'detail': str(exc)})
                return
            v = value['value']
            passed = (('min' not in limits or v >= limits['min']-EPS)
                      and ('max' not in limits or v <= limits['max']+EPS)
                      and ('strict_min' not in limits or v > limits['strict_min']))
            checks[name] = 'PASS' if passed else 'FAIL'
            measured[name] = {**value, 'limits': limits, 'status': checks[name]}
            if 'max' in limits:
                measured[name]['margin_to_upper'] = limits['max']-v
            if 'min' in limits:
                measured[name]['margin_to_lower'] = v-limits['min']
            if not passed:
                if 'strict_min' in limits:
                    tags.append('HYDRIDE_ACCEPTOR_FACE_WRONG_OR_COPLANAR')
                else:
                    direction = 'TOO_SHORT' if 'min' in limits and v < limits['min']-EPS else 'TOO_LONG'
                    tags.append(prefix+'_'+direction)

        check('hydride_distance', lambda: pair(carbon, acceptor), cuts['hydride_distance_A'], 'HYDRIDE_DISTANCE')
        check('acceptor_re_face', lambda: face_measurement(carbon, acceptor, nadp[roles['nadp_C3N']], nadp[roles['nadp_C5N']]),
              cuts['re_face_height_A'], 'HYDRIDE_FACE')
        check('lys230_prime_OH', lambda: pair(aa('lys_position','LYS','NZ',True), donor[roles['substrate_OH']]),
              cuts['lys_OH_distance_A'], 'LYS230_PRIME_OH_CONTACT')
        check('Mg_substrate_OH', lambda: pair(metal(),donor[roles['substrate_OH']]), cuts['Mg_O_distance_A'], 'MG_SUBSTRATE_OH')
        check('Mg_substrate_carboxylate', lambda: minimum_oxygen(metal(),[donor[i] for i in roles['substrate_carboxylate_O']]),
              cuts['Mg_O_distance_A'], 'MG_SUBSTRATE_CARBOXYLATE')
        for name, role, opposite in [('Mg_Asp307_local','asp_local_position',False),('Mg_Asp283_prime','asp_opposite_position',True)]:
            check(name, lambda: minimum_oxygen(metal(),[aa(role,'ASP',a,opposite) for a in ('OD1','OD2')]),
                  cuts['Mg_O_distance_A'], name.upper())
        status = 'ERROR' if errors else 'PASS' if all(v=='PASS' for v in checks.values()) else 'FAIL'
        sites.append({'site':index,'donor_role':site['donor_role'],'acceptor_role':site['acceptor_role'],
                      'local_chain':site.get('local_chain'),'opposite_chain':site.get('opposite_chain'),
                      'metal_atom':atom_id(site['metal']) if 'metal' in site else None,
                      'status':status,'checks':checks,'measurements':measured,'errors':errors,
                      'failure_tags':sorted(set(tags+[e['tag'] for e in errors])),
                      'legacy_distance_3_4_status':'PASS' if 3.0<=measured['hydride_distance']['value']<=4.0 else 'FAIL'})
    status = 'ERROR' if any(s['status']=='ERROR' for s in sites) else 'PASS' if all(s['status']=='PASS' for s in sites) else 'FAIL'
    return {'criteria_id':criteria['criteria_id'],'status':status,'sites':sites,'site_assignment':pairing,
            'failure_tags':sorted({tag for s in sites for tag in s['failure_tags']}),
            'legacy_distance_status':'PASS' if all(s['legacy_distance_3_4_status']=='PASS' for s in sites) else 'FAIL',
            'cutoffs':cuts, 'scope':criteria['scope'],
            'interpretation':'All declared sites required. No full water shell, activity claim, lower organic distance bound or C-H-C angle. General independent.'}
