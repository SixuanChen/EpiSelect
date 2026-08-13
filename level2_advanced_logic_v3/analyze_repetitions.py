#!/usr/bin/env python3
"""Analyze item-level stability across repeated EpiSelect Level-2 scored-detail files."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def read(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]

def mean(xs):
    x=list(xs);return None if not x else sum(x)/len(x)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('details',nargs='+')
    ap.add_argument('--out',help='Legacy alias for --json-out')
    ap.add_argument('--json-out')
    ap.add_argument('--csv-out')
    args=ap.parse_args()
    runs=[{x['item_id']:x for x in read(p)} for p in args.details]
    common=set.intersection(*(set(r) for r in runs)) if runs else set()
    rows=[]
    for iid in sorted(common):
        rr=[r[iid] for r in runs]
        rules=[x.get('reported_rule_canonical') for x in rr]
        choices=[x.get('reported_choice') for x in rr]
        sem=[x.get('selected_semantic_role') for x in rr]
        rows.append({
            'item_id':iid,'n_runs':len(rr),
            'rule_exact_agreement':len(set(rules))==1,
            'choice_exact_agreement':len(set(choices))==1,
            'semantic_exact_agreement':len(set(sem))==1,
            'always_inference_correct':all(x.get('inference_correct') for x in rr),
            'always_action_success':all(x.get('action_success',x.get('preferred_action_correct')) for x in rr),
            'rules':rules,'choices':choices,'semantic_roles':sem,
        })
    res={
        'n_common_items':len(rows),'n_runs':len(runs),
        'rule_exact_agreement_rate':mean(x['rule_exact_agreement'] for x in rows),
        'choice_exact_agreement_rate':mean(x['choice_exact_agreement'] for x in rows),
        'semantic_exact_agreement_rate':mean(x['semantic_exact_agreement'] for x in rows),
        'always_inference_correct_rate':mean(x['always_inference_correct'] for x in rows),
        'always_action_success_rate':mean(x['always_action_success'] for x in rows),
    }
    txt=json.dumps(res,indent=2);print(txt)
    jout=args.json_out or args.out
    if jout:Path(jout).write_text(txt,encoding='utf-8')
    if args.csv_out:
        with Path(args.csv_out).open('w',newline='',encoding='utf-8') as f:
            fields=['item_id','n_runs','rule_exact_agreement','choice_exact_agreement','semantic_exact_agreement','always_inference_correct','always_action_success','rules','choices','semantic_roles']
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for r in rows:
                q=dict(r)
                for k in ('rules','choices','semantic_roles'):q[k]=json.dumps(q[k])
                w.writerow(q)
if __name__=='__main__':main()
