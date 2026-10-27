"""Compact paper figure; frozen paired scores, no new evaluation or selection."""
from pathlib import Path
import json, hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

OUT=Path(__file__).resolve().parent
ROOT=OUT
source=OUT/'data.json'
digest=hashlib.sha256(source.read_bytes()).hexdigest()
data=json.loads(source.read_text())
for p in (OUT/'assets').glob('EBGaramond-*.ttf'):
    font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'EB Garamond','font.size':8.5,
    'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False})
W,H=7.16,2.45
SX=3.5/W
def fy(y): return (.25+1.9*y)/H
INK='#293344'; SFT='#BCCADA'; OURS='#9B84BC'; RED='#AE4049'
fig=plt.figure(figsize=(W,H))
ax=fig.add_axes([.13*SX,.61/H,.855*SX,1.22/H])
ax.set(xlim=(-.48,2.48),ylim=(0,10),xticks=[],yticks=[0,2,4,6,8,10])
ax.set_ylabel('Mean score',color=INK,fontsize=9,labelpad=5)
ax.tick_params(axis='y',length=0,pad=3,labelsize=8,colors='#697586')
ax.set_axisbelow(True)
ax.grid(axis='y',color='#E8ECF1',linewidth=.45)
for name in ['left','right','top']:ax.spines[name].set_visible(False)
ax.spines['bottom'].set(color='#B6C0CD',linewidth=.65)
labels=['Qwen3-VL\n8B-Thinking','Doubao\nSeed 2.1 Pro','GPT-6 Astra\nSession review']
audit=[]
def rounded_top_bar(x, height, color, width=.27, radius_pt=2.2):
    # Round only the upper corners in physical units. Exact mean and zero
    # baseline remain the top and bottom bounds of the data-space path.
    rx=min(width/2, radius_pt/72 * (ax.get_xlim()[1]-ax.get_xlim()[0]) / (W*ax.get_position().width))
    ry=radius_pt/72 * (ax.get_ylim()[1]-ax.get_ylim()[0]) / (H*ax.get_position().height)
    left,right=x-width/2,x+width/2
    vertices=[(left,0),(right,0),(right,height-ry),
              (right,height),(right-rx,height),(left+rx,height),
              (left,height),(left,height-ry),(left,0),(left,0)]
    codes=[MplPath.MOVETO,MplPath.LINETO,MplPath.LINETO,
           MplPath.CURVE3,MplPath.CURVE3,MplPath.LINETO,
           MplPath.CURVE3,MplPath.CURVE3,MplPath.LINETO,MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices,codes),facecolor=color,edgecolor='none',zorder=3))
for i,g in enumerate(data['groups']):
    for arm,offset,color in [('sft',-.16,SFT),('r2',.16,OURS)]:
        stat=g[arm];m=stat['mean'];lo,hi=stat['ci95'];x=i+offset
        rounded_top_bar(x,m,color)
        ax.errorbar(x,m,yerr=[[m-lo],[hi-m]],fmt='none',color='#4A5363',
                    elinewidth=.65,capsize=1.8,capthick=.65,zorder=4)
        ax.text(x,hi+.13,f'{m:.2f}',ha='center',va='bottom',fontsize=8.5,
                color=INK,fontweight='bold')
        audit.append({'judge':g['id'],'arm':arm,'mean':m,'ci95':[lo,hi]})
    gain=100*(g['r2']['mean']/g['sft']['mean']-1)
    assert abs(gain-g['relative_gain_percent'])<1e-9
    # Comparison bracket denotes relative mean gain, not a significance test.
    ax.plot([i-.30,i-.30,i+.30,i+.30],[9.8,10.2,10.2,9.8],color=RED,
            linewidth=.65,clip_on=False)
    ax.text(i,10.35,f'+{gain:.1f}%',ha='center',va='bottom',color=RED,
            fontsize=9,fontweight='bold',clip_on=False)
    xf=(.13+.855*(i+.48)/2.96)*SX
    im=Image.open(OUT/'assets'/f"{g['id']}.png").convert('RGBA')
    im=im.crop(im.getchannel('A').getbbox());im.thumbnail((100,100))
    canvas=Image.new('RGBA',(110,110),(255,255,255,0))
    canvas.alpha_composite(im,((110-im.width)//2,(110-im.height)//2))
    badge=AnnotationBbox(OffsetImage(np.asarray(canvas),zoom=.105),
        (xf,.47/H),xycoords=fig.transFigure,frameon=False,pad=0)
    fig.add_artist(badge)
    fig.text(xf,.16/H,labels[i],ha='center',va='bottom',fontsize=8.3,
             color=INK,linespacing=.96)
fig.legend(handles=[Patch(facecolor=SFT,label='SFT'),Patch(facecolor=OURS,label='RoboImagine')],
    loc='upper center',bbox_to_anchor=(.56*SX,fy(1.015)),ncol=2,frameon=False,
    fontsize=9,handlelength=1.15,handletextpad=.45,columnspacing=1.35,borderaxespad=0)
# Right panel: exact substrings of archived final-answer evaluation rationales.
from matplotlib.patches import FancyBboxPatch
import textwrap
excerpts=json.loads((OUT/'excerpts.json').read_text())
fig.text(.10/W,2.40/H,'(a) Mean evaluation scores',fontsize=10,fontweight='bold',color=INK,va='top')
fig.text(3.67/W,2.40/H,'(b) Actual evaluator excerpts',fontsize=10,fontweight='bold',color=INK,va='top')
fig.text(3.67/W,2.17/H,'Illustrative case · “Open the bottom drawer.”',fontsize=8.7,color=INK,va='top')
fig.text(3.67/W,1.96/H,'Evaluator',fontsize=8.5,fontweight='bold',color=INK,va='top')
fig.text(4.49/W,1.96/H,'SFT',fontsize=8.8,fontweight='bold',color='#607895',va='top')
fig.text(5.82/W,1.96/H,'RoboImagine',fontsize=8.8,fontweight='bold',color='#80639E',va='top')
fig.canvas.draw()
quote_renderer=fig.canvas.get_renderer()
quote_prop=font_manager.FontProperties(family='EB Garamond',size=8.2)
def wrap_quote(text):
    lines=[];line=''
    for word in text.split():
        candidate=(line+' '+word).strip()
        if line and quote_renderer.get_text_width_height_descent(candidate,quote_prop,False)[0]>1.15*fig.dpi:
            lines.append(line);line=word
        else:line=candidate
    if line:lines.append(line)
    assert len(lines)<=3,(text,lines)
    return '\n'.join(lines)
names=['Qwen3-VL\n8B-Thinking','Doubao\nSeed 2.1 Pro','GPT-6 Astra\nSession review']
for row,name in enumerate(['qwen','doubao','session']):
    top=1.79-row*.57
    for arm,x,col,bg in [('sft',4.45,'#607895','#F2F5F9'),('hybrid',5.78,'#80639E','#F7F4FB')]:
        record=next(r for r in excerpts['records'] if r['evaluator']==name and r['arm']==arm)
        patch=FancyBboxPatch((x/W,(top-.49)/H),1.27/W,.53/H,boxstyle='round,pad=0,rounding_size=0.006',transform=fig.transFigure,facecolor=bg,edgecolor='none',zorder=0)
        fig.add_artist(patch)
        fig.text((x+.055)/W,top/H,f"{record['total']}/10",fontsize=8.6,fontweight='bold',color=col,va='top')
        quote='“'+record['excerpt']+'”'
        wrapped=wrap_quote(quote)
        t=fig.text((x+.055)/W,(top-.145)/H,wrapped,fontsize=8.2,color=INK,va='top',linespacing=1.03)
    im=Image.open(OUT/'assets'/f'{name}.png').convert('RGBA');im=im.crop(im.getchannel('A').getbbox());im.thumbnail((100,100))
    canvas=Image.new('RGBA',(110,110),(255,255,255,0));canvas.alpha_composite(im,((110-im.width)//2,(110-im.height)//2))
    fig.add_artist(AnnotationBbox(OffsetImage(np.asarray(canvas),zoom=.085),(4.025/W,(top-.045)/H),xycoords=fig.transFigure,frameon=False,pad=0))
    fig.text(4.025/W,(top-.17)/H,names[row],fontsize=8.0,color=INK,ha='center',va='top',linespacing=1.0)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for t in list(fig.texts)+list(ax.texts):
    box=t.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    assert box.x0>=0 and box.x1<=1 and box.y0>=0 and box.y1<=1,(t.get_text(),box)
from matplotlib.transforms import Bbox
export_box=Bbox.from_bounds(0,.10,W,H-.10)
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'figure.{ext}',dpi=600,facecolor='white',bbox_inches=export_box)
(OUT/'figure.editable.svg').write_bytes((OUT/'figure.svg').read_bytes())
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
(OUT/'validation.json').write_text(json.dumps({'source_sha256':digest,'size_inches':[W,H-.10],
 'layout':'Double-column mean scores plus paired actual excerpts','displayed_statistics':audit,
 'unchanged_cohorts':[g['n'] for g in data['groups']],
 'error_bars':'95% trajectory-bootstrap confidence intervals',
 'macro_gain_displayed':False,'new_scoring_calls':0},indent=2))
print('Saved compact figure:',W,H)
