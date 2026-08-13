#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
from logic_core import parse_rule_expression,RuleParseError

def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def obj(x):
    if isinstance(x,dict):return x
    if isinstance(x,str):
        try:v=json.loads(x);return v if isinstance(v,dict) else {}
        except:return {}
    return {}
def payload(r):
    p={}
    for source in ('response','raw_response','raw_inference_response','inference_response','raw_action_response','action_response'):
        q=obj(r.get(source));
        for k in ('rule','choice','answer'):
            if k in q:p[k]=q[k]
    if r.get('rule') is not None:p['rule']=r['rule']
    if r.get('choice') is not None:p['choice']=r['choice']
    if r.get('answer') is not None and 'choice' not in p:p['choice']=r['answer']
    return p

def mean(xs):
    x=list(xs);return None if not x else sum(x)/len(x)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gold',required=True);ap.add_argument('--pred',required=True);ap.add_argument('--out');args=ap.parse_args()
    G={x['item_id']:x for x in read(args.gold)};P={x['item_id']:x for x in read(args.pred) if 'item_id'in x}
    rows=[];sem=defaultdict(Counter)
    for iid,g in G.items():
        p=payload(P.get(iid,{})); c=str(p.get('choice','')).strip().upper() or None; rr=str(p.get('rule','')).strip() or None
        expected_rule=g.get('expected_rule'); expected_choice=g.get('expected_choice',g.get('expected_answer'))
        if expected_rule is not None:
            inf=(rr or '').upper()==expected_rule; act=c==expected_choice;target=act
            role=g['role'];selected='abstain' if c=='E' else 'other';policy=selected
        else:
            inf=None
            if 'gold_user_rule_truth_table' in g and rr:
                try:inf=tuple(parse_rule_expression(rr))==tuple(g['gold_user_rule_truth_table'])
                except RuleParseError:inf=False
            ob={o['label']:o for o in g.get('options',[])};sel=ob.get(c);selected='abstain' if c=='E' else (sel['semantic_role'] if sel else 'invalid_format')
            policy='abstain' if c=='E' else (sel.get('policy_class') if sel else 'invalid_format')
            target=c in g.get('target_answers',g.get('preferred_answers',[]));role=g.get('role')
        sem[role][selected]+=1
        rows.append({'item_id':iid,'role':role,'inference_correct':inf,'choice':c,'action_success':target,'selected_semantic_role':selected,
                     'selected_policy_class':policy,'rule_family':g.get('rule_family'),'feature_pair':g.get('feature_pair'),'pdd':g.get('pdd')})
    res={'n':len(rows),'action_success':mean(x['action_success'] for x in rows),
         'rule_inference_accuracy':mean(x['inference_correct'] for x in rows if x['inference_correct'] is not None),
         'informative_rate':mean(x['selected_semantic_role']=='informative' for x in rows),
         'compatible_rate':mean(x['selected_policy_class']=='compatible' for x in rows),
         'compatible_positive_rate':mean(x['selected_semantic_role']=='compatible_positive' for x in rows),
         'compatible_negative_rate':mean(x['selected_semantic_role']=='compatible_negative' for x in rows),
         'invalid_control_rate':mean(x['selected_semantic_role']=='control_invalid' for x in rows),
         'semantic_counts_by_role':{k:dict(v) for k,v in sem.items()}}
    for role in sorted({x['role'] for x in rows if x['role']}):
        v=[x for x in rows if x['role']==role];res[role]={'n':len(v),'action_success':mean(x['action_success'] for x in v),
                                                        'inference':mean(x['inference_correct'] for x in v if x['inference_correct'] is not None),
                                                        'informative_rate':mean(x['selected_semantic_role']=='informative' for x in v),
                                                        'compatible_rate':mean(x['selected_policy_class']=='compatible' for x in v)}
    txt=json.dumps(res,indent=2);print(txt)
    if args.out:Path(args.out).write_text(txt)
if __name__=='__main__':main()
