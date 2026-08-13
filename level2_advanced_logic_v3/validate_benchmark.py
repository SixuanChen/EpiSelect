#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
from logic_core import (
    CANONICAL_FUNCTIONS,RULE_FAMILIES,minimal_diagnostic_sequences,mismatch_count,
    candidate_rules_consistent,STATE_ORDER,evidence_state_classes,valid_invalid_control_states,
)

def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('.'));ap.add_argument('--out',type=Path);args=ap.parse_args();r=args.root
    gold=read(r/'benchmark'/'primary_240_gold.jsonl');contexts=read(r/'requests'/'primary_staged_120_contexts.jsonl');nat=read(r/'requests'/'naturalistic_null_120_requests.jsonl')
    errors=[];warnings=[]
    if len(gold)!=240:errors.append(f'expected 240 primary gold rows, got {len(gold)}')
    if len(contexts)!=120:errors.append(f'expected 120 contexts, got {len(contexts)}')
    if len(nat)!=120:errors.append(f'expected 120 naturalistic requests, got {len(nat)}')
    bases={g['base_id'] for g in gold}
    if len(bases)!=60:errors.append(f'expected 60 bases, got {len(bases)}')
    fam=Counter(g['rule_family'] for g in gold if g['role']=='teacher');pair=Counter(g['feature_pair'] for g in gold if g['role']=='teacher')
    if set(fam)!=set(RULE_FAMILIES):errors.append('rule family set mismatch')
    if any(v!=12 for v in fam.values()):errors.append(f'family context imbalance {fam}')
    if any(v!=20 for v in pair.values()):errors.append(f'feature pair context imbalance {pair}')

    byctx=defaultdict(list)
    for g in gold:byctx[g['context_id']].append(g)
    pos=defaultdict(Counter);sides=Counter();pddc=Counter();truth_pol=Counter();target_vals=defaultdict(Counter)
    hist_nuis=defaultdict(lambda:defaultdict(Counter));opt_nuis=defaultdict(lambda:defaultdict(Counter));seenbase=set()
    for cid,rows in byctx.items():
        if len(rows)!=2 or {x['role'] for x in rows}!={'teacher','imposter'}:errors.append(f'{cid}: missing role pair');continue
        a,b=rows
        if a['options']!=b['options']:errors.append(f'{cid}: options differ across Teacher/Imposter')
        if a['history']!=b['history']:errors.append(f'{cid}: history differs across Teacher/Imposter')
        g=a;pdd=g['pdd'];pddc[pdd]+=1
        if len(g['history'])!=pdd:errors.append(f'{cid}: history len != pdd')
        k,_=minimal_diagnostic_sequences(g['gold_user_rule'])
        if k!=pdd:errors.append(f'{cid}: stored pdd mismatch')
        if g['user_true_mismatch_count']!=1 or mismatch_count(g['true_rule_name'],g['gold_user_rule'])!=1:errors.append(f'{cid}: true/user mismatch !=1')

        obs=[(x['chosen'],x['unchosen']) for x in g['abstract_observations']]
        rem=candidate_rules_consistent(obs)
        if rem!=[g['gold_user_rule']]:errors.append(f'{cid}: history not uniquely diagnostic, remains {rem}')
        user_bits=CANONICAL_FUNCTIONS[g['gold_user_rule']]
        for h in g['history']:
            sides[h['selected_side']]+=1
            active={g['feature_spec']['A_dimension'],g['feature_spec']['B_dimension']}
            seen_labels=[]
            for j,hh in enumerate(h['options']):
                sttxt=hh['abstract_state'];aa=int(sttxt.split(',')[0].split('=')[1]);bb=int(sttxt.split(',')[1].split('=')[1]);si=STATE_ORDER.index((aa,bb))
                expected='BELONGS' if user_bits[si] else 'DOES NOT BELONG'
                if hh.get('user_classification')!=expected:errors.append(f'{cid}: wrong user classification for {hh["text"]}')
                seen_labels.append(hh.get('user_classification'))
                typ='selected' if j==h['selected_index'] else 'unselected'
                for d,v in hh['object'].items():
                    if d not in active:hist_nuis[typ][d][v]+=1
            if Counter(seen_labels)!=Counter({'BELONGS':1,'DOES NOT BELONG':1}):errors.append(f'{cid}: each observation must contain one BELONGS and one DOES NOT BELONG judgment')

        histkeys={'|'.join(hh['object'][d] for d in ('color','shape','texture','size')) for h in g['history'] for hh in h['options']}
        opts=g['options']
        if len(opts)!=4:errors.append(f'{cid}: option count')
        if {o['label'] for o in opts}!={'1','2','3','4'}:errors.append(f'{cid}: choice labels are not 1-4')
        states={o['state_index'] for o in opts}
        if states!={0,1,2,3}:errors.append(f'{cid}: 1-4 do not cover all four abstract A/B states')
        roles=Counter(o['semantic_role'] for o in opts)
        expected_roles=Counter({'informative':1,'compatible_positive':1,'compatible_negative':1,'control_invalid':1})
        if roles!=expected_roles:errors.append(f'{cid}: semantic roles {roles}')
        false=[o for o in opts if not o['label_is_truthful']]
        if len(false)!=1 or false[0]['semantic_role']!='control_invalid':errors.append(f'{cid}: primary must have exactly one invalid-control label')
        t=CANONICAL_FUNCTIONS[g['true_rule_name']];hbits=CANONICAL_FUNCTIONS[g['gold_user_rule']]
        for o in opts:
            st=o['state_index'];role=o['semantic_role']
            pos[role][o['label']]+=1
            active={g['feature_spec']['A_dimension'],g['feature_spec']['B_dimension']}
            for d,v in o['object'].items():
                if d not in active:opt_nuis[role][d][v]+=1
            truth_pol[(role,o['true_label'])]+=1
            if role=='informative' and t[st]==hbits[st]:errors.append(f'{cid}: informative state does not separate T/H')
            if role=='compatible_positive' and not (t[st]==hbits[st]==1):errors.append(f'{cid}: bad compatible-positive state')
            if role=='compatible_negative' and not (t[st]==hbits[st]==0):errors.append(f'{cid}: bad compatible-negative state')
            if role=='control_invalid':
                if o['label_is_truthful']:errors.append(f'{cid}: invalid control is truthful')
                if t[st]!=hbits[st]:errors.append(f'{cid}: invalid state should be a T/H agreement state before label flip')
            key='|'.join(o['object'][d] for d in ('color','shape','texture','size'))
            if key in histkeys:errors.append(f'{cid}: exact history/action object repeat')
            # Text must explicitly contain all four surface features.
            obj=o['object'];txt=o['text'].lower()
            if obj['size'] not in txt or obj['color'] not in txt or obj['shape'] not in txt:errors.append(f'{cid}: text missing size/color/shape for {txt}')
            texture_token={'solid':'solid','horizontal_stripes':'striped','dots':'dotted'}[obj['texture']]
            if texture_token not in txt:errors.append(f'{cid}: text missing texture for {txt}')
        if g['role']=='teacher':
            if len(g['target_answers'])!=1 or g['target_answers']!=g['informative_answers']:errors.append(f'{cid}: teacher must target unique Informative')
        if g['role']=='imposter':
            if len(g['target_answers'])!=2 or set(g['target_answers'])!=set(g['compatible_answers']):errors.append(f'{cid}: imposter must accept both Compatible polarities')
            if len(g['compatible_positive_answers'])!=1 or len(g['compatible_negative_answers'])!=1:errors.append(f'{cid}: imposter compatible polarity slots missing')
        if not seenbase.__contains__(g['base_id']):
            fs=g['feature_spec'];target_vals[fs['A_dimension']][fs['A_value']]+=1;target_vals[fs['B_dimension']][fs['B_value']]+=1;seenbase.add(g['base_id'])

    if sides['left']!=sides['right']:errors.append(f'history left/right imbalance {sides}')
    # Global exact/near-exact option-position balance.
    for role,c in pos.items():
        vals=[c[L] for L in '1234']
        if role in {'informative','control_invalid'} and set(vals)!={30}:errors.append(f'global position imbalance for {role}: {c}')
        if role in {'compatible_positive','compatible_negative'} and max(vals)-min(vals)>2:errors.append(f'global position imbalance for {role}: {c}')

    # Slice-level position balance. Informative is exact. Invalid is closest possible
    # because each base reuses the same Invalid position for both user counterfactuals.
    for field in ('rule_family','feature_pair'):
        keys=sorted({g[field] for g in gold if g['role']=='teacher'})
        for key in keys:
            sub=[g for g in gold if g['role']=='teacher' and g[field]==key]
            pp=defaultdict(Counter)
            for g in sub:
                for o in g['options']:pp[o['semantic_role']][o['label']]+=1
            expected_inf=3 if field=='rule_family' else 5
            if any(pp['informative'][L]!=expected_inf for L in '1234'):
                errors.append(f'{field}={key}: informative position imbalance {pp["informative"]}')
            expected_invalid_sorted=[2,2,4,4] if field=='rule_family' else [4,4,6,6]
            vals=sorted(pp['control_invalid'][L] for L in '1234')
            if vals!=expected_invalid_sorted:errors.append(f'{field}={key}: invalid position imbalance {pp["control_invalid"]}')
            combined=Counter(pp['compatible_positive']);combined.update(pp['compatible_negative'])
            expected_compat_sorted=[5,5,7,7] if field=='rule_family' else [9,9,11,11]
            if sorted(combined[L] for L in '1234')!=expected_compat_sorted:errors.append(f'{field}={key}: combined compatible position imbalance {combined}')

    if truth_pol[('informative','BELONGS')]!=60 or truth_pol[('informative','DOES NOT BELONG')]!=60:errors.append(f'informative truth polarity imbalance {truth_pol}')
    if truth_pol[('compatible_positive','BELONGS')]!=120:errors.append(f'compatible-positive polarity error {truth_pol}')
    if truth_pol[('compatible_negative','DOES NOT BELONG')]!=120:errors.append(f'compatible-negative polarity error {truth_pol}')
    if truth_pol[('control_invalid','BELONGS')]!=60 or truth_pol[('control_invalid','DOES NOT BELONG')]!=60:errors.append(f'invalid underlying truth polarity imbalance {truth_pol}')

    for d,c in target_vals.items():
        vals=list(c.values())
        if max(vals)-min(vals)>1:errors.append(f'target value imbalance {d}: {c}')
    for typ,dct in hist_nuis.items():
        for d,c in dct.items():
            if len(set(c.values()))!=1:errors.append(f'history nuisance imbalance {typ}/{d}: {c}')
    # Nuisance marginals for action evidence are exact for texture and almost exact
    # for size/4-valued dimensions, conditional on the structural evidence class.
    for role,dct in opt_nuis.items():
        for d,c in dct.items():
            vals=list(c.values())
            allowed=0 if d=='texture' else (2 if d=='size' else 3)
            if max(vals)-min(vals)>allowed:errors.append(f'action nuisance imbalance {role}/{d}: {c}')

    # Same physical object set/order and proposed labels across the two matched user rules.
    bybase_user=defaultdict(list)
    for g in gold:
        if g['role']=='teacher':bybase_user[g['base_id']].append(g)
    for bid,vals in bybase_user.items():
        if len(vals)!=2:errors.append(f'{bid}: expected two user-rule counterfactuals');continue
        def sig(x):return [(o['label'],o['text'],o['state_index'],o['presented_label'],o['true_label']) for o in x['options']]
        if sig(vals[0])!=sig(vals[1]):errors.append(f'{bid}: target objects/order/labels differ across user-rule counterfactuals')
        # Confirm a common Invalid-control state exists by construction.
        users=[x['gold_user_rule'] for x in vals];true=vals[0]['true_rule_name']
        invstates=valid_invalid_control_states(true,users)
        observed=[o['state_index'] for o in vals[0]['options'] if o['semantic_role']=='control_invalid']
        if len(observed)!=1 or observed[0] not in invstates:errors.append(f'{bid}: invalid control state not valid for both users')

    if pddc!=Counter({2:96,3:24}):errors.append(f'pdd distribution mismatch {pddc}')
    result={'ok':not errors,'errors':errors,'warnings':warnings,'counts':{'contexts':len(contexts),'gold_rows':len(gold),'bases':len(bases),'naturalistic':len(nat)},
            'rule_family_contexts':dict(fam),'feature_pair_contexts':dict(pair),'pdd_contexts':dict(pddc),'history_selected_sides':dict(sides),
            'option_semantic_position_counts':{k:dict(v) for k,v in pos.items()},'truth_polarity_counts':{str(k):v for k,v in truth_pol.items()},
            'target_value_counts':{k:dict(v) for k,v in target_vals.items()},'history_nuisance_counts':{t:{d:dict(c) for d,c in ds.items()} for t,ds in hist_nuis.items()},
            'action_nuisance_counts':{rr:{d:dict(c) for d,c in ds.items()} for rr,ds in opt_nuis.items()}}
    txt=json.dumps(result,indent=2);print(txt)
    if args.out:args.out.write_text(txt)
    raise SystemExit(0 if not errors else 1)
if __name__=='__main__':main()
