"""1KQP adenylation and two fixed Mg roles; general chemistry remains independent."""

from itertools import permutations

import numpy as np
from scipy.spatial.distance import cdist

from ..general import atom_id, distance
from ..structure import InputProblem
from .adenylate import angle
from .isocitrate import assignment, protein_atom

EPS = 1e-9


def pair_components(context, case):
    def components(role):
        return [(key, atoms) for key, atoms in context['ligands'].items()
                if case['ligands'][key]['chemical_role'] == role]
    atps, naads = components('atp'), components('naad')
    roles = case['reaction_atoms']
    def center(atoms, excluded):
        return np.mean([a['xyz'] for i, a in enumerate(atoms) if i not in excluded], axis=0)
    costs = [[float(np.linalg.norm(center(n,roles['NaAD_oxygen_indices'])-
                                   center(a,[roles['ATP_alpha_P'],roles['ATP_alpha_beta_O']])))
              for _, a in atps] for _, n in naads]
    order, evidence = assignment(costs)
    evidence.update(method='NONREACTIVE_CENTROIDS_BOTH_EQUIVALENT_O_EXCLUDED',
                    atp_roles=[k for k,_ in atps], naad_roles=[k for k,_ in naads])
    sites = [{'naad_role':nk,'naad':n,'atp_role':atps[j][0],'atp':atps[j][1]}
             for (nk,n),j in zip(naads,order)]
    return sites,evidence


def assign_proteins(context,case,sites):
    chains=list(context['protein_chains']);roles=case['reaction_atoms']
    costs,tables=[],[]
    for site in sites:
        anchor=np.array([site['atp'][i]['xyz'] for i in roles['ATP_scaffold_indices']])
        row,table=[],{}
        for ch in chains:
            protein=[(i,a) for i,r in enumerate(context['protein_chains'][ch],1)
                     if i not in (roles['asp_position'],roles['glu_position']) for a in r['atoms']]
            d=cdist(np.array([a['xyz'] for _,a in protein]),anchor).min(axis=1)
            count=len({i for (i,_),v in zip(protein,d) if v<=4.5});minimum=float(d.min())
            row.append(-count+0.001*minimum/(1+minimum))
            table[ch]={'contact_residues_4p5':count,'minimum_A':minimum}
        costs.append(row);tables.append(table)
    order,evidence=assignment(costs)
    evidence.update(method='ATP_ADENOSINE_SCAFFOLD_CONTACTS_ONE_TO_ONE',chains=chains,contact_tables=tables)
    for site,j in zip(sites,order):site['protein_chain']=chains[j]
    return evidence


def align_reference_metals(context,case,chain):
    reference=case['metal_assignment_reference']
    original=np.array([a['xyz'] for a in reference['anchors']])
    observed=np.array([protein_atom(context,chain,a['sequence_position'],a['residue'],'CA')['xyz']
                       for a in reference['anchors']])
    xmean,ymean=original.mean(axis=0),observed.mean(axis=0)
    x,y=original-xmean,observed-ymean
    u,s,vt=np.linalg.svd(x.T@y)
    if np.linalg.det(u@vt)<0:u[:,-1]*=-1
    rotation=u@vt
    positions=[(np.array(reference[name+'_xyz'])-xmean)@rotation+ymean for name in ('Mg1','Mg2')]
    return positions,{'protein_chain':chain,'anchor_count':len(original),
                       'rmsd_A':float(np.sqrt(np.mean(np.sum((x@rotation-y)**2,axis=1)))),
                       'predicted_reference_positions':[p.tolist() for p in positions]}


def assign_metals(context,case,sites):
    metals=context['metal_copies'].get('Mg',[])
    if len(metals)!=2*len(sites):
        raise InputProblem('MECHANISM_MG_COUNT_MISMATCH',f'Require {2*len(sites)} Mg, found {len(metals)}')
    targets,alignments=[],[]
    for site in sites:
        positions,report=align_reference_metals(context,case,site['protein_chain'])
        targets.extend(positions);alignments.append(report)
    costs=[[float(np.linalg.norm(p-m['xyz'])) for m in metals] for p in targets]
    order,evidence=assignment(costs)
    evidence.update(method='BACKBONE_ALIGNED_REFERENCE_METAL_ROLES_ONE_TO_ONE',alignments=alignments,
                    metal_atoms=[atom_id(m) for m in metals],
                    reference_ligand_coordinates_used=False,scored_O_distances_used=False)
    for i,site in enumerate(sites):
        site['Mg1'],site['Mg2']=metals[order[2*i]],metals[order[2*i+1]]
    return evidence


def pair_measurement(a,b):
    return {'atoms':[atom_id(a),atom_id(b)],'value':distance(a,b),'unit':'angstrom'}


def oxygen_group(metal,atoms):
    alternatives=[pair_measurement(metal,a) for a in atoms]
    witness=min(alternatives,key=lambda x:x['value'])
    return {**witness,'alternatives':alternatives,'selection':'Minimum in specified equivalent O group; lower bound applies to minimum'}


def inside(v,limits):
    return ('min' not in limits or v>=limits['min']-EPS) and ('max' not in limits or v<=limits['max']+EPS)


def gamma_pair(mg1,mg2,oxygens,limits):
    first=[pair_measurement(mg1,o) for o in oxygens]
    second=[pair_measurement(mg2,o) for o in oxygens]
    minima=[min(first,key=lambda r:r['value']),min(second,key=lambda r:r['value'])]
    candidates=[]
    for i,j in permutations(range(len(oxygens)),2):
        records=[first[i],second[j]]
        passes=[inside(r['value'],limits) and minima[k]['value']>=limits['min']-EPS for k,r in enumerate(records)]
        candidates.append({'indices':[i,j],'records':records,'passes':passes,'joint_pass':all(passes)})
    # Search only equivalent O label permutations. Metal roles have already
    # been fixed independently and are never swapped to improve this score.
    candidates.sort(key=lambda c:(-sum(c['passes']),sum(r['value'] for r in c['records']),c['indices']))
    chosen=candidates[0]
    records=[]
    for i,r in enumerate(chosen['records']):
        records.append({**r,'minimum_in_group':minima[i],'all_group_distances':first if i==0 else second,
                        'short_group_guard_pass':minima[i]['value']>=limits['min']-EPS,
                        'selection':'Coherent distinct gamma O witnesses; equivalent atoms only'})
    return records,{'distinct_gamma_oxygen_required':True,'candidates':candidates,
                    'joint_pass':chosen['joint_pass'],'scored_Mg_roles_changed':False}


def evaluate_site(context,case,site,protein_error=None,metal_error=None):
    roles=case['reaction_atoms'];cuts=case['mechanism_criteria']['cutoffs']
    atp,naad=site['atp'],site['naad'];p=atp[roles['ATP_alpha_P']];lg=atp[roles['ATP_alpha_beta_O']]
    checks,measurements,tags,errors={},{},[],[]
    axes=[]
    for index in roles['NaAD_oxygen_indices']:
        o=naad[index];d=distance(o,p);theta=angle(o,p,lg)
        dpass=inside(d,cuts['reaction_distance_A']);apass=theta is not None and inside(theta,cuts['attack_angle_deg'])
        axes.append({'NaAD_input_index':index,'atoms':[atom_id(a) for a in (o,p,lg)],
                     'distance_A':d,'angle_deg':theta,'distance_pass':dpass,'angle_pass':apass,
                     'joint_pass':dpass and apass})
    checks['reaction_distance']='PASS' if any(a['distance_pass'] for a in axes) else 'FAIL'
    checks['attack_angle']='PASS' if any(a['angle_pass'] for a in axes) else 'FAIL'
    checks['reaction_axis_same_oxygen']='PASS' if any(a['joint_pass'] for a in axes) else 'FAIL'
    if checks['reaction_distance']=='FAIL':tags.append('REACTION_DISTANCE_TOO_LONG')
    if checks['attack_angle']=='FAIL':tags.append('ATTACK_ANGLE_OUTSIDE_OPERATIONAL_WINDOW')
    if checks['reaction_axis_same_oxygen']=='FAIL':tags.append('REACTION_AXIS_NO_PASSING_O_TUPLE')
    if all(a['angle_deg'] is None for a in axes):
        checks['attack_angle']=checks['reaction_axis_same_oxygen']='ERROR'
        errors.append({'check':'attack_angle','tag':'DEGENERATE_REACTION_AXIS','detail':'Both O-P-O candidates have undefined angles'})

    def mg(name):
        if metal_error:raise metal_error
        return site[name]

    def protein_group(position,residue_name,names):
        if protein_error:raise protein_error
        return [protein_atom(context,site['protein_chain'],position,residue_name,n) for n in names]

    def checked(name,fn):
        try:r=fn()
        except InputProblem as exc:
            checks[name]='ERROR';errors.append({'check':name,'tag':exc.code,'detail':str(exc)});return
        limit=cuts['Mg_O_A'];passed=inside(r['value'],limit) and r.get('short_group_guard_pass',True)
        checks[name]='PASS' if passed else 'FAIL'
        measurements[name]={**r,'limits':limit,'status':checks[name],
                            'margin_to_lower':r['value']-limit['min'],'margin_to_upper':limit['max']-r['value']}
        if not passed:
            short=r.get('minimum_in_group',r)['value']<limit['min']-EPS
            tags.append(name.upper()+('_TOO_SHORT' if short else '_TOO_LONG'))
    checked('Mg1_Asp50_O',lambda:oxygen_group(mg('Mg1'),protein_group(roles['asp_position'],'ASP',['OD1','OD2'])))
    checked('Mg1_Glu162_O',lambda:oxygen_group(mg('Mg1'),protein_group(roles['glu_position'],'GLU',['OE1','OE2'])))
    checked('Mg1_ATP_beta_O',lambda:oxygen_group(mg('Mg1'),[atp[i] for i in roles['ATP_nonbridging_O']['beta']]))
    checked('Mg2_ATP_alpha_O',lambda:oxygen_group(mg('Mg2'),[atp[i] for i in roles['ATP_nonbridging_O']['alpha']]))
    gamma_evidence={}
    try:
        records,gamma_evidence=gamma_pair(mg('Mg1'),mg('Mg2'),[atp[i] for i in roles['ATP_nonbridging_O']['gamma']],cuts['Mg_O_A'])
        checked('Mg1_ATP_gamma_O',lambda:records[0]);checked('Mg2_ATP_gamma_O',lambda:records[1])
    except InputProblem as exc:
        for name in ['Mg1_ATP_gamma_O','Mg2_ATP_gamma_O']:
            checks[name]='ERROR';errors.append({'check':name,'tag':exc.code,'detail':str(exc)})
    status='ERROR' if errors else 'PASS' if all(v=='PASS' for v in checks.values()) else 'FAIL'
    return {'protein_chain':site.get('protein_chain'),'atp_role':site['atp_role'],'naad_role':site['naad_role'],
            'Mg1':atom_id(site['Mg1']) if 'Mg1' in site else None,'Mg2':atom_id(site['Mg2']) if 'Mg2' in site else None,
            'status':status,'axis_status':checks['reaction_axis_same_oxygen'],'checks':checks,'axes':axes,
            'measurements':measurements,'gamma_oxygen_mapping':gamma_evidence,'errors':errors,
            'failure_tags':sorted(set(tags+[e['tag'] for e in errors])),
            'Mg_Mg_distance_A_diagnostic':distance(site['Mg1'],site['Mg2']) if 'Mg1' in site and 'Mg2' in site else None}


def evaluate_mechanism(context,case):
    sites,ligand_evidence=pair_components(context,case);evidence={'ligands':ligand_evidence}
    protein_error=metal_error=None
    try:evidence['protein']=assign_proteins(context,case,sites)
    except InputProblem as exc:
        protein_error=exc;evidence['protein']={'status':'ERROR','detail':str(exc)}
    try:
        if protein_error:raise protein_error
        evidence['metals']=assign_metals(context,case,sites)
    except InputProblem as exc:
        metal_error=exc;evidence['metals']={'status':'ERROR','detail':str(exc)}
    records=[{'site':i,**evaluate_site(context,case,s,protein_error,metal_error)} for i,s in enumerate(sites,1)]
    status='ERROR' if any(r['status']=='ERROR' for r in records) else 'PASS' if all(r['status']=='PASS' for r in records) else 'FAIL'
    axis='ERROR' if any(r['axis_status']=='ERROR' for r in records) else 'PASS' if all(r['axis_status']=='PASS' for r in records) else 'FAIL'
    return {'criteria_id':case['mechanism_criteria']['criteria_id'],'status':status,'axis_status':axis,'sites':records,
            'site_assignment':evidence,'failure_tags':sorted({t for r in records for t in r['failure_tags']}),
            'scope':case['mechanism_criteria']['scope'],'cutoffs':case['mechanism_criteria']['cutoffs'],
            'ammonium_scope':'Presence and general geometry only; no subsequent attack trajectory in this step-1 task'}
