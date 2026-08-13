#!/usr/bin/env python3
from __future__ import annotations
import json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def write(p,rows):Path(p).write_text(''.join(json.dumps(x)+'\n' for x in rows))

def main():
    subprocess.run([sys.executable,str(ROOT/'generate_benchmark.py'),'--root',str(ROOT)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(ROOT/'validate_benchmark.py'),'--root',str(ROOT)],check=True,stdout=subprocess.DEVNULL)

    from logic_core import parse_rule_expression,CANONICAL_FUNCTIONS,family_invariants_ok
    assert family_invariants_ok()[0]
    assert tuple(parse_rule_expression('IF(A,B)'))==tuple(CANONICAL_FUNCTIONS['A_TO_B'])
    assert tuple(parse_rule_expression('NOT A OR B'))==tuple(CANONICAL_FUNCTIONS['A_TO_B'])

    gold=read(ROOT/'benchmark'/'primary_240_gold.jsonl')
    preds=[];byctx={}
    for g in gold:
        if g['role']=='teacher':byctx[g['context_id']]=g
        preds.append({'item_id':g['item_id'],'context_id':g['context_id'],
                      'raw_inference_response':json.dumps({'rule':g['gold_user_rule_expression']}),
                      'raw_action_response':json.dumps({'choice':g['target_answers'][0]})})
    assert len(byctx)==120
    assert all(sum(not o['label_is_truthful'] for o in g['options'])==1 for g in byctx.values())
    assert all({o['semantic_role'] for o in g['options']}=={'informative','compatible_positive','compatible_negative','control_invalid'} for g in byctx.values())

    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'pred.jsonl';s=Path(td)/'summary.json';write(p,preds)
        subprocess.run([sys.executable,str(ROOT/'score_predictions.py'),'--gold',str(ROOT/'benchmark'/'primary_240_gold.jsonl'),'--pred',str(p),'--summary-out',str(s)],check=True,stdout=subprocess.DEVNULL)
        res=json.loads(s.read_text())
        assert res['unique_context_rule_inference_accuracy']==1.0
        assert res['teacher_action_success']==1.0
        assert res['imposter_action_success']==1.0
        assert res['strict_goal_sensitive_role_contrast']==1.0
        assert res['strict_four_cell_quartet']==1.0
        assert res['invalid_control_selection_rate']==0.0
        assert res['imposter_compatible_rate']==1.0

    ngold=read(ROOT/'benchmark'/'naturalistic_null_120_gold.jsonl')
    npreds=[]
    for g in ngold:
        c=next(o['label'] for o in g['options'] if o['semantic_role']=='informative')
        npreds.append({'item_id':g['item_id'],'response':json.dumps({'choice':c})})
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'null.jsonl';s=Path(td)/'null_summary.json';write(p,npreds)
        subprocess.run([sys.executable,str(ROOT/'score_naturalistic.py'),'--gold',str(ROOT/'benchmark'/'naturalistic_null_120_gold.jsonl'),'--pred',str(p),'--out',str(s)],check=True,stdout=subprocess.DEVNULL)
        res=json.loads(s.read_text());assert res['informative_rate']==1.0;assert res['paired_informative_both_rate']==1.0

    ugold=read(ROOT/'benchmark'/'ablations'/'underdetermined_staged_gold.jsonl')
    upreds=[{'item_id':g['item_id'],'raw_inference_response':json.dumps({'rule':'UNKNOWN'}),'raw_action_response':json.dumps({'choice':'E'})} for g in ugold]
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'under.jsonl';s=Path(td)/'under_summary.json';write(p,upreds)
        subprocess.run([sys.executable,str(ROOT/'score_ablations.py'),'--gold',str(ROOT/'benchmark'/'ablations'/'underdetermined_staged_gold.jsonl'),'--pred',str(p),'--out',str(s)],check=True,stdout=subprocess.DEVNULL)
        res=json.loads(s.read_text());assert res['action_success']==1.0;assert res['rule_inference_accuracy']==1.0

    agold=read(ROOT/'benchmark'/'all_truthful_360_gold.jsonl')
    byctx2={}
    for g in agold:
        if g['role']=='teacher':byctx2[g['context_id']]=g
    assert len(byctx2)==120
    assert all(all(o['label_is_truthful'] for o in g['options']) for g in byctx2.values())

    print('EpiSelect Level 2 Advanced Logic v3 self-test: PASS')
if __name__=='__main__':main()
