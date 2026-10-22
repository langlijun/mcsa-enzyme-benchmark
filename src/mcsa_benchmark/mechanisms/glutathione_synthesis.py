"""1GSA reactant organization: one axis and one two-Mg system per declared site."""
import itertools

import numpy as np

from ..general import atom_id, distance
from ..structure import InputProblem
from .adenylate import angle, EPS
from .pmp_carbonyl import transform


def within(d, window):
    return window[0]-EPS <= d <= window[1]+EPS


def donor_atoms(residues, position, residue_name, names):
    residue=residues[position-1]
    atoms={a['atom_name']:a for a in residue['atoms']}
    if residue['name']!=residue_name or any(n not in atoms or atoms[n]['element']!='O' for n in names):
        raise InputProblem('REQUIRED_METAL_DONOR_MISSING',f'{residue_name}{position} {names}')
    return [atoms[n] for n in names]


def protein_core(m1,m2,residues,reference,criteria):
    asp=donor_atoms(residues,273,'ASP',['OD1','OD2'])
    glu=donor_atoms(residues,281,'GLU',['OE1','OE2'])
    asn=donor_atoms(residues,283,'ASN',['OD1'])[0]
    expected=reference['donor_distances_A']
    candidates=[]
    # Equivalent carboxylate oxygens may swap labels, but the bridging Glu
    # oxygen must be the same physical atom for both metal contacts.
    for e,d in itertools.product(glu,asp):
        edges=[('Mg1_Glu',m1,e),('Mg1_Asp',m1,d),('Mg2_Glu',m2,e),('Mg2_Asn',m2,asn)]
        rows=[{'edge':name,'metal':atom_id(m),'donor':atom_id(a),'distance_A':distance(m,a),
               'reference_A':expected[name],'delta_A':distance(m,a)-expected[name]} for name,m,a in edges]
        candidates.append({'cost_A2':sum(r['delta_A']**2 for r in rows),'edges':rows,
                           'shared_Glu_oxygen':atom_id(e)})
    chosen=min(candidates,key=lambda c:c['cost_A2'])
    failure=[]
    for row in chosen['edges']:
        row['status']='PASS' if row['delta_A']<=criteria['maximum_positive_deviation_A']+EPS and row['distance_A']<=criteria['maximum_Mg_O_A']+EPS else 'FAIL'
        if row['status']=='FAIL':
            failure.append('MG_PROTEIN_DONOR_CONTACT_TOO_LONG')
    all_distances=[{'metal':atom_id(m),'donor':atom_id(a),'distance_A':distance(m,a)} for m in [m1,m2] for a in asp+glu+[asn]]
    other=next(e for e in glu if atom_id(e)!=chosen['shared_Glu_oxygen'])
    return {'status':'FAIL' if failure else 'PASS','failure_tags':sorted(set(failure)),
            **chosen,'equivalent_oxygen_candidates':candidates,'all_named_donor_distances':all_distances,
            'two_sided_reference_recovery_diagnostic':{'status':'PASS' if all(abs(r['delta_A'])<=criteria['maximum_positive_deviation_A']+EPS for r in chosen['edges']) else 'FAIL','affects_mechanism_status':False},
            'Mg2_other_Glu_O_A_diagnostic':distance(m2,other),
            'Mg_Mg_A_diagnostic':distance(m1,m2),'Mg_Mg_is_hard_gate':False}


def phosphate_coupling(m1,m2,atp,indices,criteria):
    groups={}
    for label,m in [('Mg1',m1),('Mg2',m2)]:
        groups[label]={}
        for phosphate,ii in indices.items():
            rows=[{'atom':atom_id(atp[i]),'input_index':i,'distance_A':distance(m,atp[i])} for i in ii]
            for r in rows:
                r['within_proximity_upper_limit']=r['distance_A']<=criteria['maximum_Mg_O_A']+EPS
            groups[label][phosphate]={'contacts':rows,'has_proximal_contact':any(r['within_proximity_upper_limit'] for r in rows)}
    checks={label+'_ATP_phosphate':any(groups[label][p]['has_proximal_contact'] for p in criteria['each_Mg_any_of_groups']) for label in ['Mg1','Mg2']}
    diagnostic=criteria['analog_topology_diagnostic']
    diagnostic_checks={f'Mg1_{p}':groups['Mg1'][p]['has_proximal_contact'] for p in diagnostic['Mg1_required_groups']}
    diagnostic_checks['Mg2_beta_or_gamma']=any(groups['Mg2'][p]['has_proximal_contact'] for p in diagnostic['Mg2_any_of_groups'])
    tags=[]
    if not checks['Mg1_ATP_phosphate']:
        tags.append('MG1_ATP_PHOSPHATE_CONTACT_MISSING')
    if not checks['Mg2_ATP_phosphate']:
        tags.append('MG2_ATP_PHOSPHATE_CONTACT_MISSING')
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'failure_tags':tags,
            'phosphate_groups':groups,
            'ADP_topology_diagnostic':{'checks':diagnostic_checks,'status':'PASS' if all(diagnostic_checks.values()) else 'FAIL','affects_primary_pass':False},
            'scope':'Declared partial phosphate-coupling topology, not a full experimental six-coordinate reactant shell.'}


def evaluate_mechanism(context,case):
    reference,rules,roles=case['site_reference'],case['mechanism_criteria'],case['reaction_atoms']
    anchors,fits={},{}
    for chain,residues in sorted(context['protein_chains'].items()):
        moving=[]
        for item in reference['CA']:
            ca=[a for a in residues[item['sequence_position']-1]['atoms'] if a['atom_name']=='CA']
            if len(ca)!=1:
                raise InputProblem('SITE_ASSIGNMENT_CA_MISSING',chain)
            moving.append(ca[0]['xyz'])
        rotation,translation,rmsd=transform([r['xyz'] for r in reference['CA']],moving)
        anchors[chain]={k:np.asarray(reference[k])@rotation+translation for k in ['ATP_anchor','GGC_anchor','Mg1','Mg2']}
        fits[chain]=rmsd
    assignments={}
    for role,atoms in context['ligands'].items():
        kind=case['ligands'][role]['copy_group']
        if kind not in ('ATP','GGC'):
            continue
        point=np.mean([atoms[i]['xyz'] for i in roles[kind+'_assignment_indices']],axis=0)
        costs=sorted((float(np.linalg.norm(point-a[kind+'_anchor'])),chain) for chain,a in anchors.items())
        margin=costs[1][0]-costs[0][0]
        assignments[role]={'kind':kind,'site':costs[0][1],'costs_A':{ch:d for d,ch in costs},'margin_A':margin,
                           'ambiguous':margin<rules['site_assignment']['ligand_margin_A']}
    mg=context['metal_copies'].get('MG',[])
    metal_assignments=[]
    for index,m in enumerate(mg):
        costs=sorted((float(np.linalg.norm(m['xyz']-(a['Mg1']+a['Mg2'])/2)),chain) for chain,a in anchors.items())
        margin=costs[1][0]-costs[0][0]
        metal_assignments.append({'index':index,'metal':atom_id(m),'site':costs[0][1],
                                  'costs_A':{ch:d for d,ch in costs},'margin_A':margin,
                                  'ambiguous':margin<rules['site_assignment']['ligand_margin_A']})
    sites=[]
    for chain in anchors:
        aa=[r for r,a in assignments.items() if a['site']==chain and a['kind']=='ATP']
        gg=[r for r,a in assignments.items() if a['site']==chain and a['kind']=='GGC']
        mm=[r for r in metal_assignments if r['site']==chain]
        site={'protein_chain':chain,'ATP_copies':aa,'GGC_copies':gg,'Mg_count':len(mm)}
        if len(aa)==len(gg)==1:
            atp,ggc=context['ligands'][aa[0]],context['ligands'][gg[0]]
            p,lg=atp[roles['Pgamma_index']],atp[roles['leaving_oxygen_index']]
            axes=[]
            for i in roles['GGC_nucleophile_indices']:
                nu=ggc[i];d,theta=distance(nu,p),angle(nu,p,lg)
                if theta is None or d<=EPS:
                    raise InputProblem('DEGENERATE_REACTION_AXIS','Cannot measure O-P-O')
                tags=[]
                if d>rules['main_axis']['distance_max_A']+EPS:tags.append('REACTION_DISTANCE_TOO_LONG')
                if theta<rules['main_axis']['angle_min_deg']-EPS:tags.append('ATTACK_DIRECTION_MISMATCH')
                axes.append({'atoms':[atom_id(a) for a in (nu,p,lg)],'distance_A':d,'angle_deg':theta,
                             'status':'FAIL' if tags else 'PASS','failure_tags':tags})
            passing=[a for a in axes if a['status']=='PASS']
            site['reaction_axis']={'status':'PASS' if passing else 'FAIL','candidates':axes,
                'witness':min(passing or axes,key=lambda a:a['distance_A']),
                'failure_tags':[] if passing else sorted({t for a in axes for t in a['failure_tags']}),
                'old_Pgamma_O_bond_A_diagnostic':distance(p,lg)}
        else:
            site['reaction_axis']={'status':'FAIL','failure_tags':['DECLARED_REACTANT_SITE_OCCUPANCY_MISMATCH']}
        if len(mm)==2 and len(mg)==4:
            choices=[]
            for ordering in itertools.permutations(mm):
                cost=sum(float(np.sum((mg[item['index']]['xyz']-anchors[chain][label])**2))
                         for item,label in zip(ordering,['Mg1','Mg2']))
                choices.append({'cost_A2':cost,'indices':[i['index'] for i in ordering]})
            choices.sort(key=lambda x:x['cost_A2'])
            delta=choices[1]['cost_A2']-choices[0]['cost_A2']
            m1,m2=[mg[i] for i in choices[0]['indices']]
            site['metal_role_assignment']={'method':'protein-fitted reference positions; no donor-pass optimization',
                'candidates':choices,'cost_margin_A2':delta,'Mg1':atom_id(m1),'Mg2':atom_id(m2)}
            if delta<rules['site_assignment']['Mg_role_cost_tie_A2']:
                site['two_Mg_site']={'status':'ERROR','failure_tags':['MG_ROLE_ASSIGNMENT_AMBIGUOUS']}
                site['Mg_ATP_coupling']={'status':'ERROR','failure_tags':['MG_ROLE_ASSIGNMENT_AMBIGUOUS']}
            else:
                try:
                    site['two_Mg_site']=protein_core(m1,m2,context['protein_chains'][chain],reference,rules['metal_core'])
                except InputProblem as exc:
                    site['two_Mg_site']={'status':'ERROR','failure_tags':[exc.code],'detail':exc.detail}
                if len(aa)==1:
                    site['Mg_ATP_coupling']=phosphate_coupling(m1,m2,context['ligands'][aa[0]],roles['phosphate_oxygen_indices'],rules['metal_atp'])
                else:
                    site['Mg_ATP_coupling']={'status':'FAIL','failure_tags':['DECLARED_ATP_SITE_OCCUPANCY_MISMATCH']}
        else:
            for key in ['two_Mg_site','Mg_ATP_coupling']:
                site[key]={'status':'FAIL','failure_tags':['DECLARED_TWO_MG_SITE_OCCUPANCY_MISMATCH']}
        site['failure_tags']=sorted({t for k in ['reaction_axis','two_Mg_site','Mg_ATP_coupling'] for t in site[k]['failure_tags']})
        site['status']=aggregate([site[k]['status'] for k in ['reaction_axis','two_Mg_site','Mg_ATP_coupling']])
        sites.append(site)
    subchecks={k:aggregate([s[k]['status'] for s in sites]) for k in ['reaction_axis','two_Mg_site','Mg_ATP_coupling']}
    ambiguous=any(a['ambiguous'] for a in assignments.values()) or any(a['ambiguous'] for a in metal_assignments)
    failure_tags=sorted({t for s in sites for t in s['failure_tags']})
    if ambiguous:failure_tags.append('CATALYTIC_SITE_ASSIGNMENT_AMBIGUOUS')
    return {'criteria_id':rules['criteria_id'],'status':'ERROR' if ambiguous else aggregate(list(subchecks.values())),
        'subchecks':subchecks,'sites':sites,'failure_tags':failure_tags,
        'ligand_site_assignments':assignments,'metal_site_assignments':metal_assignments,
        'protein_CA_fit_RMSD_A':fits,'native_reactant_reference_calibrated':False,
        'scope':'Two declared dimer sites: carboxylate activation axis, shared-Glu two-Mg protein core and partial same-ATP coupling. No glycine-attack or full water-shell criterion.'}


def aggregate(statuses):
    return 'ERROR' if 'ERROR' in statuses else 'FAIL' if 'FAIL' in statuses else 'PASS'
