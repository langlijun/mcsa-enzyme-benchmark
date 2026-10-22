"""PSP step1 organization with coherent attacking/metal-binding Asp11 oxygens."""
import numpy as np

from ..general import atom_id, distance
from ..structure import InputProblem

EPS=1e-9


def bound(value,limit,mode='max'):
    if value is None:
        return False
    return value<=limit+EPS if mode=='max' else value>=limit-EPS


def reaction_angle(first,center,last):
    x,y=first['xyz']-center['xyz'],last['xyz']-center['xyz']
    if np.linalg.norm(x)*np.linalg.norm(y)<1e-12:
        return None
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(x,y)),np.dot(x,y))))


def metric(name,first,second,limit):
    value=distance(first,second) if first is not None and second is not None else None
    return {'name':name,'atoms':[atom_id(a) for a in (first,second) if a is not None],
            'distance_A':value,'upper_A':limit,'margin_A':None if value is None else limit-value,
            'status':'NOT_MEASURABLE' if value is None else 'PASS' if bound(value,limit) else 'FAIL'}


def closest_contact(name,first,donors,limit):
    contacts=[metric(name,first,a,limit) for a in donors]
    selected=min(range(len(contacts)),key=lambda i:float('inf') if contacts[i]['distance_A'] is None else contacts[i]['distance_A'])
    return {'name':name,'candidates':contacts,'selected_index':selected,'selected':contacts[selected],
            'status':contacts[selected]['status'],'policy':'Equivalent oxygens within this declared chemical group only; all candidate distances retained.'}


def evaluate_mechanism(context,case):
    criteria=case['mechanism_criteria'];roles=case['reaction_atoms']
    chains=context['protein_chains']
    if len(chains)!=1 or set(context['ligands'])!={'PLS'}:
        raise InputProblem('PSP_COMPONENT_COUNT_MISMATCH','Require one protein and one free phosphoserine.',True)
    chain,residues=next(iter(chains.items()))
    def asp(position,names):
        r=residues[position-1]
        if r['name']!='ASP':
            raise InputProblem('MECHANISM_RESIDUE_MISMATCH',f'{chain} sequence position {position} must be ASP.',True)
        atoms={a['atom_name']:a for a in r['atoms']}
        for name in names:
            if name not in atoms or atoms[name]['element']!='O':
                raise InputProblem('MECHANISM_REQUIRED_ATOM_MISSING',f'{chain}:ASP:{position}:{name}',True)
        return atoms
    nu=asp(roles['nucleophile_residue_position'],['OD1','OD2'])
    acid=asp(roles['acid_residue_position'],['OD1','OD2','O'])
    metal_asp=asp(roles['Mg_ligand_residue_position'],['OD1','OD2'])
    ligand=context['ligands']['PLS'];p=ligand[roles['P_index']];lg=ligand[roles['leaving_O_index']]
    mg=context['metal_copies'].get('MG',[]);metal=mg[0] if len(mg)==1 else None
    upper=criteria['Mg_contact_max_A']
    acid_contact=closest_contact('ASP13_TO_LEAVING_O',lg,[acid['OD1'],acid['OD2']],criteria['acid_contact_max_A'])
    backbone=metric('MG_ASP13_BACKBONE_O',metal,acid['O'],upper)
    asp167=closest_contact('MG_ASP167_SIDECHAIN_O',metal,[metal_asp['OD1'],metal_asp['OD2']],upper)
    phosphate=closest_contact('MG_SUBSTRATE_NONBRIDGING_O',metal,[ligand[i] for i in roles['phosphate_nonbridging_O_indices']],upper)
    variants=[]
    for attack,coord in [('OD1','OD2'),('OD2','OD1')]:
        approach=metric('ASP11_O_TO_P',nu[attack],p,criteria['attack_distance_max_A'])
        theta=reaction_angle(nu[attack],p,lg)
        angle_ok=bound(theta,criteria['attack_angle_min_deg'],'min')
        mg_other=metric('MG_ASP11_NONATTACKING_O',metal,nu[coord],upper)
        axis_ok=approach['status']=='PASS' and angle_ok
        metal_checks=[mg_other,backbone,asp167['selected'],phosphate['selected']]
        metal_ok=len(mg)==1 and all(c['status']=='PASS' for c in metal_checks)
        checks={'attack_distance':approach['status']=='PASS','attack_direction':angle_ok,
                'acid_contact':acid_contact['status']=='PASS','required_Mg_count':len(mg)==1,
                **{c['name']:c['status']=='PASS' for c in metal_checks}}
        tags=[]
        if not checks['attack_distance']:tags.append('REACTION_DISTANCE_TOO_LONG')
        if not angle_ok:tags.append('DEGENERATE_REACTION_AXIS' if theta is None else 'ATTACK_DIRECTION_MISMATCH')
        if not checks['acid_contact']:tags.append('ASP13_LEAVING_GROUP_CONTACT_TOO_LONG')
        if len(mg)!=1:tags.append('REQUIRED_MG_COUNT_MISMATCH')
        else:
            tags.extend(c['name']+'_TOO_LONG' for c in metal_checks if c['status']=='FAIL')
        variants.append({'attack_oxygen':atom_id(nu[attack]),'coupled_Asp11_Mg_oxygen':atom_id(nu[coord]),
                         'axis_atoms':[atom_id(a) for a in [nu[attack],p,lg]],'attack_contact':approach,
                         'angle_deg':theta,'angle_min_deg':criteria['attack_angle_min_deg'],
                         'angle_margin_deg':None if theta is None else theta-criteria['attack_angle_min_deg'],
                         'axis_status':'PASS' if axis_ok else 'FAIL','acid_status':acid_contact['status'],
                         'Mg_organization_status':'PASS' if metal_ok else 'FAIL',
                         'Mg_contacts':metal_checks,'checks':checks,'failure_tags':sorted(tags),
                         'status':'PASS' if all(checks.values()) else 'FAIL'})
    passing=[i for i,v in enumerate(variants) if v['status']=='PASS']
    # The witness explains a complete allowed variant, never a mix of checks.
    witness=passing[0] if passing else min(range(2),key=lambda i:(sum(not ok for ok in variants[i]['checks'].values()),
                                                                -(variants[i]['angle_deg'] or 0),i))
    selected=variants[witness]
    tags=[] if passing else sorted(set(selected['failure_tags']+['NO_COHERENT_FIRST_STEP_VARIANT']))
    diagnostics=[]
    diagnostic_position=roles['diagnostic_residue_position']
    for a in residues[diagnostic_position-1]['atoms']:
        if a['atom_name'] in ['OD1','OD2']:
            diagnostics.append({'name':'ASP171_MG_DISTANCE_ONLY','atom':atom_id(a),
                                'distance_A':None if metal is None else distance(metal,a),'affects_pass':False})
    all_oxygen=[a for a in context['protein_atoms']+ligand if a['element']=='O']
    minimum_contacts=[]
    for m in mg:
        if all_oxygen:
            donor=min(all_oxygen,key=lambda a:distance(m,a))
            minimum_contacts.append({'metal':atom_id(m),'donor':atom_id(donor),
                                     'distance_A':distance(m,donor),'affects_pass':False,
                                     'policy':'Continuous minimum only; no hard Mg lower bound in either stage.'})
    return {'criteria_id':criteria['criteria_id'],'status':'PASS' if passing else 'FAIL','failure_tags':tags,
            'protein_chain':chain,'declared_sites':1,'passing_site_count':int(bool(passing)),
            'variants':variants,'selected_variant_index':witness,'selected_variant':selected,'passing_variant_indices':passing,
            'axis_any_variant_status':'PASS' if any(v['axis_status']=='PASS' for v in variants) else 'FAIL',
            'acid_contact':acid_contact,'Mg_Asp167_candidates':asp167,'Mg_substrate_candidates':phosphate,
            'Mg_atoms':[atom_id(a) for a in mg],
            'preexisting_P_O_diagnostic':{'atoms':[atom_id(p),atom_id(lg)],'distance_A':distance(p,lg),'affects_mechanism_pass':False},
            'Asp171_diagnostics':diagnostics,'distance_lower_bound_applied':False,
            'Mg_minimum_nonwater_O_contacts_diagnostic':minimum_contacts,
            'general_checks_included':False,'water_required':False,'scope':criteria['success_scope']}
