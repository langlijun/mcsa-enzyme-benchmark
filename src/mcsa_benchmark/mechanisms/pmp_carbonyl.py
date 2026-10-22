"""PMP amine addition to GSA aldehyde: declared two-site geometric task."""
import math

import numpy as np

from ..structure import InputProblem


def transform(reference, target):
    x, y = np.asarray(reference, dtype=float), np.asarray(target, dtype=float)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid CA coordinates for site assignment')
    xc, yc = x.mean(axis=0), y.mean(axis=0)
    u, _, vt = np.linalg.svd((x-xc).T @ (y-yc))
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    rotation = u @ vt
    translation = yc-xc @ rotation
    rmsd = np.sqrt(np.mean(np.sum((x @ rotation + translation-y)**2, axis=1)))
    return rotation, translation, float(rmsd)


def geometry(n, c, o):
    u, v = n-c, o-c
    if not np.isfinite(np.concatenate([u, v])).all():
        raise ValueError('Nonfinite reaction coordinates')
    d, co = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if min(d, co) <= 1e-12:
        raise ValueError('Reaction atoms coincide; angle is undefined')
    angle = math.degrees(math.atan2(float(np.linalg.norm(np.cross(u, v))), float(np.dot(u, v))))
    return d, angle, co


def judge_axis(distance, angle, criteria):
    if not math.isfinite(distance) or not math.isfinite(angle) or distance <= 0 or not 0 <= angle <= 180:
        raise ValueError('Invalid distance or angle')
    rules = criteria['cutoffs']
    tags = []
    if distance > rules['distance_A']['max'] + 1e-10:
        tags.append('REACTION_DISTANCE_TOO_LONG')
    if angle < rules['angle_deg']['min'] - 1e-10:
        tags.append('ATTACK_ANGLE_BELOW_WINDOW')
    if angle > rules['angle_deg']['max'] + 1e-10:
        tags.append('ATTACK_ANGLE_ABOVE_WINDOW')
    return tags


def atom_id(atom):
    return {key: value for key, value in atom.items() if key != 'xyz'}


def evaluate_mechanism(context, case):
    criteria, indices = case['mechanism_criteria'], case['reaction_atoms']
    reference = case['site_reference']
    anchors, fits = {}, {}
    for chain, residues in sorted(context['protein_chains'].items()):
        ca = []
        for residue in residues:
            atoms = [a for a in residue['atoms'] if a['atom_name'] == 'CA']
            if len(atoms) != 1:
                raise InputProblem('SITE_ASSIGNMENT_CA_MISSING', f'Chain {chain} lacks a unique CA')
            ca.append(atoms[0]['xyz'])
        rotation, translation, rmsd = transform(reference['positive_occupancy_CA_coordinates'], ca)
        anchors[chain] = {kind: np.asarray(reference[key]) @ rotation + translation
                          for kind, key in [('PMP', 'PMP_ring_centroid'), ('GSA', 'PMP_N4A')]}
        fits[chain] = rmsd
    assignments = {}
    for role, atoms in context['ligands'].items():
        kind = case['ligands'][role]['copy_group']
        ii = indices['PMP_ring_indices'] if kind == 'PMP' else indices['GSA_assignment_indices']
        point = np.mean([atoms[i]['xyz'] for i in ii], axis=0)
        costs = sorted((float(np.linalg.norm(point-v[kind])), chain) for chain, v in anchors.items())
        margin = costs[1][0]-costs[0][0]
        assignments[role] = {'site': costs[0][1], 'kind': kind, 'costs_A': {ch:d for d,ch in costs},
                             'margin_A': margin, 'ambiguous': margin < case['site_assignment_margin_A']}
    axes = []
    for pmp in (r for r in assignments if assignments[r]['kind'] == 'PMP'):
        for gsa in (r for r in assignments if assignments[r]['kind'] == 'GSA'):
            n = context['ligands'][pmp][indices['PMP_N_index']]
            c = context['ligands'][gsa][indices['GSA_C_index']]
            o = context['ligands'][gsa][indices['GSA_O_index']]
            distance, angle, co = geometry(n['xyz'], c['xyz'], o['xyz'])
            tags = judge_axis(distance, angle, criteria)
            axes.append({'PMP_copy':pmp, 'GSA_copy':gsa, 'N_atom':atom_id(n), 'C_atom':atom_id(c),
                         'O_atom':atom_id(o), 'distance_A':distance, 'angle_deg':angle, 'aldehyde_CO_A':co,
                         'status':'FAIL' if tags else 'PASS', 'failure_tags':tags,
                         'tight_center_diagnostic':distance < 2.5, 'tight_center_is_hard_failure':False})
    sites, all_tags = [], set()
    for chain in sorted(anchors):
        pmp = [r for r,a in assignments.items() if a['site'] == chain and a['kind'] == 'PMP']
        gsa = [r for r,a in assignments.items() if a['site'] == chain and a['kind'] == 'GSA']
        if len(pmp) != 1 or len(gsa) != 1:
            site = {'site':chain, 'status':'FAIL', 'PMP_copies':pmp, 'GSA_copies':gsa,
                    'failure_tags':['DECLARED_SITE_COMPONENT_COUNT_MISMATCH']}
        else:
            site = {'site':chain, **next(a for a in axes if a['PMP_copy'] == pmp[0] and a['GSA_copy'] == gsa[0])}
        sites.append(site)
        all_tags.update(site['failure_tags'])
    ambiguous = [r for r,a in assignments.items() if a['ambiguous']]
    if ambiguous:
        result_status = 'ERROR'
        all_tags.add('SITE_ASSIGNMENT_AMBIGUOUS')
    else:
        result_status = 'FAIL' if all_tags else 'PASS'
    return {'criteria_id':criteria['id'], 'status':result_status, 'failure_tags':sorted(all_tags),
            'sites':sites, 'candidate_axes':axes, 'site_assignments':assignments,
            'CA_fit_RMSD_to_reference_A':fits, 'ambiguous_components':ambiguous,
            'numeric_reference_calibrated':False,
            'assignment_basis':'Reference protein CA fit, PMP ring centroid, GSA non-aldehyde centroid. No reaction-distance optimization or forced bijection.',
            'scope':'Both declared sites satisfy an author operational Step 1 distance/angle screen. Not activity, electronic state, complete mechanism, or native ligand pose proof.'}
