#!/usr/bin/env python3
"""Provider-neutral runner template for EpiSelect Level 2 v2.

Edit call_your_model(messages, schema, model_name) only. The rest handles staged
contexts, flat requests, raw response preservation, inference reuse, and resumable
JSONL output.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

def call_your_model(messages, schema, model_name):
    raise NotImplementedError("Connect your provider here. Return the raw model text or a dict.")

def read_jsonl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def append(p,row):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def raw_text(x):return json.dumps(x,separators=(',',':')) if isinstance(x,dict) else str(x)

RULE_SCHEMA={"type":"object","properties":{"rule":{"type":"string"}},"required":["rule"],"additionalProperties":False}
CHOICE_SCHEMA={"type":"object","properties":{"choice":{"type":"string","enum":["1","2","3","4","E"]}},"required":["choice"],"additionalProperties":False}
JOINT_SCHEMA={"type":"object","properties":{"rule":{"type":"string"},"choice":{"type":"string","enum":["1","2","3","4","E"]}},"required":["rule","choice"],"additionalProperties":False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--requests',required=True);ap.add_argument('--out',required=True);ap.add_argument('--model-name',required=True);ap.add_argument('--provider',default='custom');ap.add_argument('--rep',type=int,default=0);ap.add_argument('--reuse-inference-from');args=ap.parse_args()
    reqs=read_jsonl(args.requests);done=set();reuse={}
    if args.reuse_inference_from:
        for x in read_jsonl(args.reuse_inference_from):
            if x.get('context_id') and x.get('raw_inference_response') is not None:
                reuse[x['context_id']]=x['raw_inference_response']
    if Path(args.out).exists():done={x.get('item_id') for x in read_jsonl(args.out)}
    for req in reqs:
        if 'inference_messages' in req:
            if req.get('reuse_primary_inference') and req['context_id'] in reuse:
                infraw=str(reuse[req['context_id']])
            else:
                inf=call_your_model(req['inference_messages'],RULE_SCHEMA,args.model_name);infraw=raw_text(inf)
            for br in req['branches']:
                if br['item_id'] in done:continue
                msgs=list(req['inference_messages'])+[{"role":"assistant","content":infraw},{"role":"user","content":br['message']}]
                act=call_your_model(msgs,CHOICE_SCHEMA,args.model_name)
                append(args.out,{"item_id":br['item_id'],"context_id":req['context_id'],"role":br['role'],"provider":args.provider,"model":args.model_name,"rep":args.rep,
                                 "raw_inference_response":infraw,"raw_action_response":raw_text(act)})
        else:
            iid=req['item_id']
            if iid in done:continue
            schema=JOINT_SCHEMA if req.get('request_type')=='joint_single_call' else CHOICE_SCHEMA
            out=call_your_model(req['messages'],schema,args.model_name)
            append(args.out,{"item_id":iid,"context_id":req.get('context_id',iid),"provider":args.provider,"model":args.model_name,"rep":args.rep,"response":raw_text(out)})
if __name__=='__main__':main()
