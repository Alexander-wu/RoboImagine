#!/usr/bin/env python3
"""Semicircular native-unit scales, not part-to-whole pie charts."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge

BLUE, PURPLE, RED = '#225D91', '#9B84BC', '#BF3F45'
INK, GRAY, TRACK = '#253141', '#7A8898', '#EDF0F4'
FONT_DIR = HERE.parents[1]/'common/fonts'

def main():
    raw=(HERE/'data.json').read_bytes()
    data=json.loads(raw)
    c=data['cohorts']['heldout_test128']
    assert c['n']==11 and len(set(c['ids']))==11
    horizons=[64,128]
    fonts={}
    for face in ['Regular','Bold','Italic','BoldItalic']:
        p=FONT_DIR/f'EBGaramond-{face}.otf'
        assert p.is_file()
        font_manager.fontManager.addfont(str(p))
        fonts[face]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    font_manager.findfont('EB Garamond',fallback_to_default=False)
    plt.rcParams.update({'font.family':'EB Garamond','font.size':7.5,'text.color':INK,
                         'savefig.facecolor':'white','svg.hashsalt':'r2-native-unit-semicircle-v3-singlecolumn-h64-h128'})
    fig,axes=plt.subplots(2,3,figsize=(3.5,2.35))
    fig.subplots_adjust(left=.005,right=.995,bottom=.012,top=.88,wspace=.06,hspace=.025)
    specs=[
        ('psnr','PSNR ↑',30.,[0,10,20,30],2),
        ('ssim','SSIM ↑',1.,[0,.25,.5,.75,1],4),
        ('lpips','LPIPS ↓',.4,[0,.1,.2,.3,.4],4),
    ]
    items=[]
    for index,ax in enumerate(axes.flat):
        horizon=horizons[index//3]
        stats=c['prefixes'][str(horizon)]['means']
        key,title,upper,ticks,decimals=specs[index%3]
        title=f'({chr(97+index)}) {title}'
        ax.set_aspect('equal')
        ax.set_xlim(-1.36,1.36)
        ax.set_ylim(-.48,1.62)
        ax.axis('off')
        ax.text(0,1.50,title,ha='center',va='center',fontsize=8.1,weight='bold')
        records={}
        for arm,color,radius in [('current_only',BLUE,1.),('full_w6',PURPLE,.82)]:
            value=stats[arm][key]
            assert 0<=value<=upper
            angle=180*value/upper
            ax.add_patch(Wedge((0,0),radius,0,180,width=.14,facecolor=TRACK,edgecolor='none'))
            ax.add_patch(Wedge((0,0),radius,180-angle,180,width=.14,facecolor=color,edgecolor='none'))
            records[arm]=dict(value=value,angle_degrees=angle,radius=radius)
        for tick in ticks:
            angle=math.radians(180*(1-tick/upper))
            cos,sin=math.cos(angle),math.sin(angle)
            ax.plot([1.04*cos,1.09*cos],[1.04*sin,1.09*sin],color='#A6B0BD',lw=.65)
            label=f'{tick:g}'
            ax.text(1.22*cos,1.22*sin,label,ha='center',va='center',fontsize=6.2,color=GRAY)
        b,f=stats['current_only'][key],stats['full_w6'][key]
        delta=(f/b-1)*100 if key=='lpips' else f-b
        gain=f'−{abs(delta):.2f}%' if key=='lpips' else f'+{delta:.2f} dB' if key=='psnr' else f'+{delta:.4f}'
        ax.text(0,.29,gain,ha='center',va='center',fontsize=8.2,weight='bold',color=RED)
        ax.text(0,.055,f'{horizon} frames',ha='center',va='center',fontsize=6.8,color=INK,style='italic')
        ax.text(-.16,-.30,f'{b:.{decimals}f}',ha='right',va='center',fontsize=7.2,weight='bold',color=BLUE)
        ax.text(0,-.30,'→',ha='center',va='center',fontsize=7.2,color=GRAY)
        ax.text(.16,-.30,f'{f:.{decimals}f}',ha='left',va='center',fontsize=7.2,weight='bold',color='#80639E')
        items.append(dict(metric=key,horizon=horizon,axis_min=0,axis_max=upper,ticks=ticks,
                          encoding='Arc angle = 180 * metric value / displayed axis maximum; concentric radii distinguish methods only.',
                          values=records,gain=dict(label=gain,value=delta,unit='relative percent' if key=='lpips' else 'dB' if key=='psnr' else 'absolute')))
    legend=[Line2D([],[],lw=3.2,color=color,label=label,solid_capstyle='butt')
            for color,label in [(BLUE,'Current-frame only'),(PURPLE,'Full memory')]]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.99),ncol=2,
               frameon=False,fontsize=7.5,handlelength=1.4,columnspacing=1.4,handletextpad=.5)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bb=fig.get_window_extent(renderer)
    for ax in axes.flat:
        for t in ax.texts:
            tb=t.get_window_extent(renderer)
            assert tb.x0>=bb.x0-1 and tb.y0>=bb.y0-1 and tb.x1<=bb.x1+1 and tb.y1<=bb.y1+1
    for name,mode in [('figure.editable.svg','none'),('figure.svg','path')]:
        with plt.rc_context({'svg.fonttype':mode}):
            fig.savefig(HERE/name,format='svg',metadata={'Creator':'RoboAlign full-cohort semicircular metric scales','Date':None})
    plt.close(fig)
    sys.path.insert(0,str(HERE.parents[1]/"common"))
    from cairo_support import configure_cairo
    configure_cairo()
    import cairosvg
    cairosvg.svg2pdf(url=str(HERE/'figure.svg'),write_to=str(HERE/'figure.pdf'))
    cairosvg.svg2png(url=str(HERE/'figure.svg'),write_to=str(HERE/'figure.png'),output_width=2800)
    caption=('Full-memory comparison at 64- and 128-frame prediction horizons. The top and bottom rows report cumulative means over the first 64 and all 128 predicted frames, respectively, of the same 11 test trajectories. '
             'The outer blue and inner purple arcs show current-frame-only and full-memory results on the displayed metric-specific angular scales; arc areas and ring radii do not encode performance. '
             'Red labels denote absolute PSNR/SSIM gains and relative LPIPS reduction at the corresponding horizon. Each metric uses the same angular scale in both rows; scales differ across metrics and are plotting ranges, not universal metric bounds. '
             'Full memory retains the fixed dynamic anchor and latest six dynamic states. Both configurations share the initial scene, actions, instruction, model checkpoint, and decoding settings.\n')
    (HERE/'caption.txt').write_text(caption)
    metadata=dict(data_sha256=hashlib.sha256(raw).hexdigest(),source_summary=data['source_summary'],
                  source_summary_sha256=data['source_summary_sha256'],n=11,cohort='heldout_test128',horizons=horizons,
                  statistic='Cumulative means over the first64/all128 predicted frames of the same11 trajectories; not22 independent samples',
                  trajectory_ids=c['ids'],checkpoint_digest=data['checkpoint_digest'],protocol_sha256=data['protocol_sha256'],
                  no_outcome_selection=True,not_part_to_whole=True,panels=items,fonts=fonts,canvas_inches=[3.5,2.35],
                  scope='Single-column typography adaptation, same data and layout as approved 64/128 figure',
                  uncertainty='Means only; official paired bootstrap intervals retained in data.json')
    (HERE/'render_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'state':'RENDERED','output':str(HERE),'panels':items},indent=2))

if __name__=='__main__':main()
