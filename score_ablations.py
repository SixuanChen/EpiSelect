#!/usr/bin/env python3
"""Score optional choice-only / diagnosis-only / action-given-rule ablations."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def mean(x): return None if not x else sum(x)/len(x)

def payload(r):
    if 'response' in r and isinstance(r['response'],str):
        try: return json.loads(r['response'].strip())
        except: return {}
    return r

def nrule(x):
    if x is None:return None
    s=str(x).strip().upper().replace('-','_').replace(' ','_')
    return {'COLOR':'COLOR_RULE','SHAPE':'SHAPE_RULE'}.get(s,s)

def nans(x):
    if x is None:return None
    s=str(x).strip().upper(); return s if s in {'A','B','C','D'} else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['choice_only','diagnosis_only','action_given_rule'],required=True)
    ap.add_argument('--gold',type=Path,required=True);ap.add_argument('--pred',type=Path,required=True);ap.add_argument('--out',type=Path,default=None);a=ap.parse_args()
    g={x['item_id']:x for x in rows(a.gold)};p={x['item_id']:payload(x) for x in rows(a.pred) if 'item_id' in x}
    vals=[];byrole={'teacher':[],'imposter':[]}
    for iid,x in g.items():
        if iid not in p:continue
        if a.mode=='diagnosis_only': ok=nrule(p[iid].get('inferred_rule'))==x['gold_other_rule']
        else: ok=nans(p[iid].get('answer')) in set(x['acceptable_answers'])
        vals.append(ok);byrole[x['role']].append(ok)
    s={'mode':a.mode,'num_scored':len(vals),'accuracy':mean(vals),'teacher_accuracy':mean(byrole['teacher']),'imposter_accuracy':mean(byrole['imposter'])}
    text=json.dumps(s,indent=2);print(text)
    if a.out:a.out.write_text(text)
if __name__=='__main__':main()
