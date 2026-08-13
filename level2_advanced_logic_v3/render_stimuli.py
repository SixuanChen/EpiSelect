#!/usr/bin/env python3
"""Optional deterministic renderer for EpiSelect Level 2 symbolic objects."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
try:
    from PIL import Image, ImageDraw
except ImportError as e:
    raise SystemExit('Install Pillow: python -m pip install pillow') from e

COLORS={
    'red':'#D62728','blue':'#1F77B4','green':'#2CA02C','purple':'#9467BD'
}
SIZES={'small':72,'large':128}
CANVAS=256;CENTER=128;OUTLINE='#333333';OUTLINE_W=3

def star_points(cx,cy,r_outer,r_inner,n=5):
    pts=[]
    for i in range(n*2):
        ang=-math.pi/2+i*math.pi/n
        r=r_outer if i%2==0 else r_inner
        pts.append((cx+r*math.cos(ang),cy+r*math.sin(ang)))
    return pts

def shape_mask(shape,size):
    im=Image.new('L',(CANVAS,CANVAS),0);d=ImageDraw.Draw(im)
    half=size/2;box=(CENTER-half,CENTER-half,CENTER+half,CENTER+half)
    if shape=='circle':d.ellipse(box,fill=255)
    elif shape=='square':d.rectangle(box,fill=255)
    elif shape=='triangle':
        h=size;pts=[(CENTER,CENTER-h/2),(CENTER-size/2,CENTER+h/2),(CENTER+size/2,CENTER+h/2)]
        d.polygon(pts,fill=255)
    elif shape=='star':d.polygon(star_points(CENTER,CENTER,size/2,size*0.22),fill=255)
    else:raise ValueError(shape)
    return im

def texture_image(texture,color):
    im=Image.new('RGB',(CANVAS,CANVAS),'white');d=ImageDraw.Draw(im)
    if texture=='solid':
        d.rectangle((0,0,CANVAS,CANVAS),fill=color)
    elif texture=='horizontal_stripes':
        d.rectangle((0,0,CANVAS,CANVAS),fill='white')
        stripe=6;period=12
        for y in range(0,CANVAS,period):d.rectangle((0,y,CANVAS,y+stripe-1),fill=color)
    elif texture=='dots':
        d.rectangle((0,0,CANVAS,CANVAS),fill='white')
        spacing=12;r=3
        for y in range(6,CANVAS,spacing):
            for x in range(6,CANVAS,spacing):d.ellipse((x-r,y-r,x+r,y+r),fill=color)
    else:raise ValueError(texture)
    return im

def render(obj,out):
    size=SIZES[obj['size']];mask=shape_mask(obj['shape'],size)
    bg=Image.new('RGB',(CANVAS,CANVAS),'white')
    tex=texture_image(obj['texture'],COLORS[obj['color']])
    bg.paste(tex,(0,0),mask)
    d=ImageDraw.Draw(bg)
    half=size/2;box=(CENTER-half,CENTER-half,CENTER+half,CENTER+half)
    if obj['shape']=='circle':d.ellipse(box,outline=OUTLINE,width=OUTLINE_W)
    elif obj['shape']=='square':d.rectangle(box,outline=OUTLINE,width=OUTLINE_W)
    elif obj['shape']=='triangle':
        h=size;pts=[(CENTER,CENTER-h/2),(CENTER-size/2,CENTER+h/2),(CENTER+size/2,CENTER+h/2)]
        d.line(pts+[pts[0]],fill=OUTLINE,width=OUTLINE_W,joint='curve')
    elif obj['shape']=='star':
        pts=star_points(CENTER,CENTER,size/2,size*0.22);d.line(pts+[pts[0]],fill=OUTLINE,width=OUTLINE_W,joint='curve')
    Path(out).parent.mkdir(parents=True,exist_ok=True);bg.save(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--object-json');ap.add_argument('--out');ap.add_argument('--all',action='store_true');ap.add_argument('--outdir',default='rendered_universe');args=ap.parse_args()
    if args.all:
        from generate_benchmark import OBJECT_UNIVERSE
        for o in OBJECT_UNIVERSE:
            obj={'color':o.color,'shape':o.shape,'texture':o.texture,'size':o.size}
            name=f"{o.size}_{o.color}_{o.texture}_{o.shape}.png"
            render(obj,Path(args.outdir)/name)
        print(f'Rendered {len(OBJECT_UNIVERSE)} objects to {args.outdir}')
    else:
        if not args.object_json or not args.out:raise SystemExit('Use --object-json JSON --out FILE or --all')
        render(json.loads(args.object_json),args.out)
if __name__=='__main__':main()
