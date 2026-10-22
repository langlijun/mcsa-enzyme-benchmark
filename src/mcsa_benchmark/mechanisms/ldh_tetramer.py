"""Four protein-frame sites with separately reported hydride and His contacts."""
from itertools import permutations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..general import atom_id, distance
from ..structure import InputProblem
from .adenylate import EPS


def fit_frame(reference, target):
    x, y = np.asarray(reference, dtype=float), np.asarray(target, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise InputProblem('INVALID_PROTEIN_FRAME', 'Reference and predicted CA arrays must be complete and finite.')
    xc, yc = x.mean(0), y.mean(0)
    u, singular, vt = np.linalg.svd((x-xc).T @ (y-yc))
    if singular[1] <= 1e-9:
        raise InputProblem('DEGENERATE_PROTEIN_FRAME', 'CA positions do not define a noncollinear protein frame.')
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    shift = yc-xc @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum((x @ rotation+shift-y)**2, axis=1))))
    return rotation, shift, rmsd


def assign_group(costs, chains, roles, tolerance):
    costs = np.asarray(costs, dtype=float)
    if costs.shape != (4, 4) or not np.isfinite(costs).all():
        raise InputProblem('INVALID_SITE_ASSIGNMENT_COSTS', 'Require four finite pocket anchors and four components.')
    ii, jj = linear_sum_assignment(costs)
    best = tuple(int(j) for j in jj)
    alternatives = sorted((float(sum(costs[i, p[i]] for i in range(4))), p) for p in permutations(range(4)))
    total = float(costs[ii, jj].sum())
    if abs(total-alternatives[0][0]) > 1e-8:
        raise InputProblem('SITE_ASSIGNMENT_SOLVER_DISAGREEMENT', 'Assignment solvers disagree.')
    margin = alternatives[1][0] - alternatives[0][0]
    if margin <= tolerance:
        raise InputProblem('SITE_MAPPING_AMBIGUOUS', 'Two protein-frame assignments have equivalent cost; provide --site-map from independent evidence.')
    mapping = {chain: roles[best[i]] for i, chain in enumerate(chains)}
    return mapping, {'protein_chains': chains, 'component_roles': roles, 'costs_A': costs.tolist(),
                     'best_total_cost_A': total, 'second_best_total_cost_A': alternatives[1][0], 'margin_A': margin}


def assign_sites(context, case):
    chains = sorted(context['protein_chains'])
    groups = {g: sorted(r for r, d in case['ligands'].items() if d['copy_group'] == g) for g in ('NAD', 'lactate')}
    if len(chains) != 4 or any(len(r) != 4 for r in groups.values()):
        raise InputProblem('SITE_COUNT_MISMATCH', 'Require four full protein chains, four NADs and four lactates.', True)
    explicit = context.get('explicit_site_map')
    if explicit is not None:
        if not isinstance(explicit, dict) or set(explicit) != set(chains):
            raise InputProblem('INVALID_SITE_MAP', 'Provide exactly one mapping for every predicted protein chain.')
        if any(not isinstance(v, dict) or set(v) != {'NAD', 'lactate'} for v in explicit.values()):
            raise InputProblem('INVALID_SITE_MAP', 'Each site must contain NAD and lactate role names.')
        if any(not isinstance(v[g], str) for v in explicit.values() for g in ('NAD', 'lactate')):
            raise InputProblem('INVALID_SITE_MAP', 'Component role names must be strings.')
        for group, roles in groups.items():
            if sorted(v[group] for v in explicit.values()) != roles:
                raise InputProblem('INVALID_SITE_MAP', 'Every component role must be used exactly once.')
        return {ch: dict(explicit[ch]) for ch in chains}, {'method': 'USER_VERIFIED_EXPLICIT_SITE_MAP', 'mapping': explicit,
                                                         'reaction_distances_used_to_choose_assignment': False}
    cfg = case['binding_assignment']
    anchors, fits = {}, {}
    for chain in chains:
        target = []
        for residue in context['protein_chains'][chain]:
            atoms = [a for a in residue['atoms'] if a['atom_name'] == 'CA']
            if len(atoms) != 1:
                raise InputProblem('SITE_MAPPING_CA_MISSING', f'Chain {chain}: a unique CA is missing; provide a verified site map.')
            target.append(atoms[0]['xyz'])
        rotation, shift, rmsd = fit_frame(cfg['reference_CA'], target)
        anchors[chain] = {group: np.asarray(cfg['reference_anchors'][key]) @ rotation+shift
                          for group, key in [('NAD', 'NAD_nonreactive_centroid'), ('lactate', 'substrate_carboxyl_centroid')]}
        fits[chain] = {'CA_RMSD_A': rmsd, 'anchors_A': {g: v.tolist() for g, v in anchors[chain].items()}}
    mapping = {chain: {} for chain in chains}
    evidence = {}
    for group, roles in groups.items():
        centers = []
        for role in roles:
            atoms = context['ligands'][role]
            indices = ([i for i in range(len(atoms)) if i not in cfg['NAD_anchor_excluded_indices']]
                       if group == 'NAD' else cfg['lactate_anchor_indices'])
            centers.append(np.mean([atoms[i]['xyz'] for i in indices], axis=0))
        costs = [[float(np.linalg.norm(anchors[ch][group]-c)) for c in centers] for ch in chains]
        assigned, evidence[group] = assign_group(costs, chains, roles, cfg['ambiguity_tolerance_A'])
        for chain, role in assigned.items():
            mapping[chain][group] = role
    return mapping, {'method': 'PROTEIN_FRAME_NONREACTIVE_ANCHORS_ONE_TO_ONE', 'mapping': mapping,
                     'protein_fits': fits, 'groups': evidence, 'reaction_distances_used_to_choose_assignment': False}


def evaluate_mechanism(context, case):
    mapping, evidence = assign_sites(context, case)
    rules, roles = case['mechanism_criteria'], case['reaction_atoms']
    bounds = rules['hydride_distance']
    sites = []
    for chain, slot in mapping.items():
        nad, lactate = context['ligands'][slot['NAD']], context['ligands'][slot['lactate']]
        donor, acceptor = lactate[roles['lactate_C2_index']], nad[roles['NAD_C4_index']]
        oxygen = lactate[roles['lactate_OH_index']]
        if donor['element'] != 'C' or acceptor['element'] != 'C' or oxygen['element'] != 'O':
            raise InputProblem('REACTION_ATOM_ELEMENT_MISMATCH', 'Declared LDH reporter atoms must be C, C, O.')
        cc = distance(donor, acceptor)
        hydride_pass = bounds['min_A']-EPS <= cc <= bounds['max_A']+EPS
        tags = []
        if cc < bounds['min_A']-EPS:
            tags.append('HYDRIDE_DISTANCE_TOO_SHORT')
        if cc > bounds['max_A']+EPS:
            tags.append('HYDRIDE_DISTANCE_TOO_LONG')
        residue = context['protein_chains'][chain][roles['His_sequence_position_1based']-1]
        his_atoms = [a for a in residue['atoms'] if a['atom_name'] == roles['His_atom_name']]
        if residue['name'] != roles['His_residue_name'] or len(his_atoms) != 1:
            no, his, his_pass = None, None, False
            tags.append('CATALYTIC_HIS193_NE2_MISSING')
        elif his_atoms[0]['element'] != 'N':
            no, his, his_pass = None, his_atoms[0], False
            tags.append('CATALYTIC_HIS193_NE2_ELEMENT_MISMATCH')
        else:
            his = his_atoms[0]
            no = distance(oxygen, his)
            his_pass = no <= rules['catalytic_base_contact']['max_A']+EPS
            if not his_pass:
                tags.append('HIS193_CONTACT_TOO_LONG')
        sites.append({'protein_chain': chain, 'NAD_role': slot['NAD'], 'lactate_role': slot['lactate'],
                      'hydride_distance_A': cc, 'His_contact_distance_A': no,
                      'hydride_atoms': [atom_id(donor), atom_id(acceptor)],
                      'His_contact_atoms': [atom_id(oxygen), atom_id(his) if his is not None else None],
                      'hydride_status': 'PASS' if hydride_pass else 'FAIL',
                      'His_contact_status': 'PASS' if his_pass else 'FAIL',
                      'status': 'PASS' if hydride_pass and his_pass else 'FAIL', 'failure_tags': tags})
    counts = {'hydride_sites_passed': sum(s['hydride_status'] == 'PASS' for s in sites),
              'His_contact_sites_passed': sum(s['His_contact_status'] == 'PASS' for s in sites),
              'both_sites_passed': sum(s['status'] == 'PASS' for s in sites)}
    all_pass = counts['both_sites_passed'] == 4
    tags = sorted({t for s in sites for t in s['failure_tags']})
    if not all_pass:
        tags.append('MULTISITE_RECOVERY_INCOMPLETE')
    return {'criteria_id': rules['criteria_id'], 'status': 'PASS' if all_pass else 'FAIL', 'failure_tags': tags,
            'sites': sites, 'site_counts': counts, 'sites_required': 4, 'assignment': evidence,
            'hydride_status': 'PASS' if counts['hydride_sites_passed'] == 4 else 'FAIL',
            'His_contact_status': 'PASS' if counts['His_contact_sites_passed'] == 4 else 'FAIL',
            'cutoffs': {'hydride_distance_A': [bounds['min_A'], bounds['max_A']],
                        'His_contact_max_A': rules['catalytic_base_contact']['max_A']},
            'general_checks_included': False, 'angles_scored': False, 'water_required': False,
            'lower_bound_interpretation': rules['lower_bound_interpretation'], 'scope': rules['scope']}
