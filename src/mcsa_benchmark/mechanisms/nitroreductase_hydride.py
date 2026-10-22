"""Two-site NADH-to-FMN proximity, with score-independent pocket registration."""
from itertools import permutations

import numpy as np

from ..general import atom_id, distance
from ..structure import InputProblem

EPS = 1e-9


def kabsch(mobile, target):
    x, y = np.asarray(mobile, dtype=float), np.asarray(target, dtype=float)
    if len(x) < 3 or np.linalg.matrix_rank(x-x.mean(axis=0)) < 2:
        raise InputProblem('PROTEIN_ALIGNMENT_DEGENERATE', 'Insufficient independent CA anchors.')
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    u, _, vt = np.linalg.svd((x-xm).T @ (y-ym))
    rotation = u @ np.diag([1., 1., 1. if np.linalg.det(u @ vt) > 0 else -1.]) @ vt
    translation = ym - xm @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum((x @ rotation+translation-y)**2, axis=1))))
    return rotation, translation, rmsd


def registration(context, case):
    chains = sorted(context['protein_chains'])
    refs = case['reference_sites']
    options = []
    for order in permutations(sorted(refs)):
        assignment = dict(zip(chains, order))
        x, y = [], []
        for chain, ref in assignment.items():
            for pos, xyz in sorted(refs[ref]['CA_by_sequence_position'].items(), key=lambda p: int(p[0])):
                atoms = context['protein_chains'][chain][int(pos)-1]['atoms']
                ca = [a for a in atoms if a['atom_name'] == 'CA']
                if len(ca) != 1:
                    raise InputProblem('PROTEIN_CA_ANCHOR_MISSING', f'{chain} sequence position {pos}', True)
                x.append(ca[0]['xyz']); y.append(xyz)
        rotation, translation, rmsd = kabsch(x, y)
        options.append((rmsd, tuple(order), assignment, rotation, translation, len(x)))
    options.sort(key=lambda o: (o[0], o[1]))
    best = options[0]
    info = {'protein_to_reference_chain': best[2], 'CA_RMSD_A_diagnostic': best[0],
            'CA_count': best[5], 'options': [{'mapping':o[2], 'CA_RMSD_A':o[0]} for o in options],
            'note':'Homodimer chain swaps are label symmetry, not different ligand pairing hypotheses.'}
    return best[3], best[4], info


def assign_pockets(centers, pockets, tolerance):
    roles = sorted(centers)
    costs = {r: {p: float(np.linalg.norm(centers[r]-xyz)) for p,xyz in pockets.items()} for r in roles}
    choices = []
    for order in permutations(sorted(pockets)):
        mapping = dict(zip(roles, order))
        choices.append({'assignment':mapping, 'total_cost_A':sum(costs[r][p] for r,p in mapping.items())})
    choices.sort(key=lambda c: (c['total_cost_A'], tuple(c['assignment'].values())))
    margin = choices[1]['total_cost_A'] - choices[0]['total_cost_A']
    return {'assignment':choices[0]['assignment'], 'cost_matrix_A':costs, 'options':choices,
            'margin_A':margin, 'numeric_tie':margin <= tolerance+EPS,
            'nearest_pocket_diagnostic':{r:min(costs[r], key=costs[r].get) for r in roles}}


def plane(points):
    points = np.asarray(points)
    _, singular, vt = np.linalg.svd(points-points.mean(axis=0))
    if singular[1] < 1e-8:
        return None, None
    n = vt[-1]
    return n, float(np.sqrt(np.mean(((points-points.mean(axis=0)) @ n)**2)))


def orientation_diagnostics(donor, flavin, atoms):
    dn, dr = plane([donor[i]['xyz'] for i in atoms['NADH_ring_indices']])
    fn, fr = plane([flavin[i]['xyz'] for i in atoms['FMN_ring_indices']])
    vector = donor[atoms['donor_C4_index']]['xyz']-flavin[atoms['acceptor_N5_index']]['xyz']
    length = np.linalg.norm(vector)
    approach = None if fn is None or length < 1e-8 else float(np.degrees(np.arccos(np.clip(abs(vector @ fn)/length, 0., 1.))))
    interplane = None if dn is None or fn is None else float(np.degrees(np.arccos(np.clip(abs(dn @ fn), 0., 1.))))
    return {'donor_vector_to_FMN_plane_normal_deg':approach, 'unsigned_ring_plane_angle_deg':interplane,
            'NADH_ring_plane_rms_A':dr, 'FMN_ring_plane_rms_A':fr, 'affects_pass':False,
            'face_status':'NOT_CERTIFIED', 'reason':'Unsigned plane metrics do not identify re/si or the stereospecific transferable hydrogen. No numeric angle gate is calibrated.'}


def evaluate_mechanism(context, case):
    criteria, atoms = case['mechanism_criteria'], case['reaction_atoms']
    ligands = context['ligands']
    donors = sorted(r for r in ligands if case['ligands'][r]['copy_group'] == 'NADH')
    flavins = sorted(r for r in ligands if case['ligands'][r]['copy_group'] == 'FMN')
    if len(donors) != 2 or len(flavins) != 2:
        raise InputProblem('HYDRIDE_COMPONENT_COUNT_MISMATCH', 'Two NADH and two FMN molecules are required.', True)
    rotation, translation, align = registration(context, case)
    pockets = {k:np.asarray(v['pocket_centroid_excluding_N5']) for k,v in case['reference_sites'].items()}
    centers_d = {r:np.mean([a['xyz'] for i,a in enumerate(ligands[r]) if i != atoms['donor_C4_index']],axis=0) @ rotation + translation for r in donors}
    centers_f = {r:np.mean([ligands[r][i]['xyz'] for i in atoms['FMN_ring_indices'] if i != atoms['acceptor_N5_index']],axis=0) @ rotation + translation for r in flavins}
    da = assign_pockets(centers_d, pockets, criteria['tie_tolerance_A'])
    fa = assign_pockets(centers_f, pockets, criteria['tie_tolerance_A'])
    all_pairs, selected = [], []
    for donor in donors:
        for flavin in flavins:
            d, f = ligands[donor], ligands[flavin]
            c4, n5 = d[atoms['donor_C4_index']], f[atoms['acceptor_N5_index']]
            value = distance(c4, n5)
            passed = value <= criteria['distance_A']['max']+EPS
            record = {'donor_instance':donor, 'flavin_instance':flavin, 'donor_chain':c4['chain'],
                      'flavin_chain':n5['chain'], 'atoms':[atom_id(c4), atom_id(n5)], 'distance_A':value,
                      'distance_upper_A':criteria['distance_A']['max'], 'margin_to_upper_A':criteria['distance_A']['max']-value,
                      'status':'PASS' if passed else 'FAIL', 'failure_tags':[] if passed else ['HYDRIDE_DONOR_ACCEPTOR_TOO_FAR'],
                      'orientation_diagnostics':orientation_diagnostics(d, f, atoms)}
            all_pairs.append(record)
            if da['assignment'][donor] == fa['assignment'][flavin]:
                pocket = da['assignment'][donor]
                occupancy_ok = (da['nearest_pocket_diagnostic'][donor] == pocket and
                                fa['nearest_pocket_diagnostic'][flavin] == pocket)
                site_tags = list(record['failure_tags'])
                if not occupancy_ok:
                    site_tags.append('SITE_COPY_OCCUPANCY_MISMATCH')
                selected.append(dict(record, reference_pocket=pocket, proximity_status=record['status'],
                                     distinct_site_assignment_compatible=occupancy_ok,
                                     status='PASS' if passed and occupancy_ok else 'FAIL', failure_tags=site_tags))
    assert len(selected) == 2
    passing = sum(s['status'] == 'PASS' for s in selected)
    tags = sorted({t for s in selected for t in s['failure_tags']})
    if passing == 1:
        tags.append('PARTIAL_SITE_RECOVERY')
    distinct = (set(da['nearest_pocket_diagnostic'].values()) == set(pockets) and
                set(fa['nearest_pocket_diagnostic'].values()) == set(pockets))
    if not distinct:
        tags.append('DISTINCT_POCKET_OCCUPANCY_NOT_RECOVERED')
    if da['numeric_tie'] or fa['numeric_tie']:
        tags.append('SITE_ASSIGNMENT_NUMERIC_TIE')
    return {'criteria_id':criteria['criteria_id'], 'status':'ERROR' if 'SITE_ASSIGNMENT_NUMERIC_TIE' in tags else 'PASS' if passing == 2 else 'FAIL',
            'failure_tags':tags, 'selected_axes':selected, 'all_candidate_pairs':all_pairs,
            'passing_site_count':passing, 'declared_site_count':2,
            'proximity_passing_pair_count':sum(a['proximity_status'] == 'PASS' for a in selected),
            'site_assignment':{'protein_alignment':align, 'NADH':da, 'FMN':fa, 'reaction_atoms_used':False,
                               'distinct_nearest_pocket_occupancy_recovered':distinct},
            'distance_lower_bound_applied':False, 'general_checks_included':False,
            'water_required':False, 'orientation_is_hard_gate':False,
            'scope':criteria['scope'], 'success_label':'PASSES_DECLARED_HYDRIDE_PROXIMITY_SCREEN'}
