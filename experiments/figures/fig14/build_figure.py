"""Preview only: verified means -> heatmaps + zero-baseline grouped bars."""
from pathlib import Path
import os,json,hashlib
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm,colors
from matplotlib.patches import FancyBboxPatch,Patch
from matplotlib.ticker import FormatStrFormatter
import numpy as np
H=[16,32,48,64,96];W=[1,3,6,12]
s=json.loads((ROOT/'data.json').read_text());assert s['unique_trajectories']==30 and s['quality_records']==750
D={(r['horizon'],r['arm']):r['statistics'] for r in s['table']}
for p in (ROOT/'fonts').glob('*.otf'):fm.fontManager.addfont(str(p))
reg=fm.FontProperties(fname=str(ROOT/'fonts/EBGaramond-Regular.otf'))
bold=fm.FontProperties(fname=str(ROOT/'fonts/EBGaramond-Bold.otf'))
plt.rcParams.update({'font.family':reg.get_name(),'font.size':8.4,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'rq4-heatbars-hatched','hatch.linewidth':.40,'text.color':'#202936','axes.labelcolor':'#202936','xtick.color':'#394454','ytick.color':'#394454'})
INK='#202936';C=['#3784E5','#52BCCF','#8A72D9','#F18D83']
width,height=7.1,2.75
fig=plt.figure(figsize=(width,height))
# Shared vertical geometry for each horizon across all panels.
lefts=[.35,2.02,3.75,5.48];widths=[1.43,1.43,1.35,1.35];bottom=.53;top=2.18
axs=[fig.add_axes([l/width,bottom/height,w/widths[0]*widths[0]/width,(top-bottom)/height]) for l,w in zip(lefts,widths)]
plotted={}
for j,(ax,key,title,palette,lo,hi,fmt) in enumerate(zip(axs[:2],['psnr','lpips'],['(a) PSNR ↑','(b) LPIPS ↓'],[['#EEF4FD','#B3D2F6','#548FDD','#255CAF'],['#F2EFFA','#CEBFEF','#9D84D4','#6E4CB1']],[18,.14],[24,.23],['.2f','.4f'])):
 vals=np.array([[D[h,f'w{w}'][key]['mean'] for w in W] for h in H]);plotted[key]=vals.tolist()
 cmap=colors.LinearSegmentedColormap.from_list(key,palette if key=='psnr' else palette[::-1]);norm=colors.Normalize(lo,hi)
 for row in range(5):
  win=int(np.argmax(vals[row]) if key=='psnr' else np.argmin(vals[row]))
  for col in range(4):
   value=vals[row,col];rgba=cmap(norm(value));lum=sum(a*b for a,b in zip(rgba[:3],[.2126,.7152,.0722]))
   rect=FancyBboxPatch((col+.035,row+.035),.93,.93,boxstyle='round,pad=0,rounding_size=.08',fc=rgba,ec='none',zorder=2);ax.add_patch(rect)
   ax.text(col+.5,row+.50,format(value,fmt),ha='center',va='center',fontsize=8.0 if key=='psnr' else 7.2,color='white' if lum<.55 else '#27334A',fontproperties=bold if col==win else reg,zorder=4)
   if col==win:
    ax.add_patch(FancyBboxPatch((col+.04,row+.04),.92,.92,boxstyle='round,pad=0,rounding_size=.08',fc='none',ec='#26354B',lw=.85,zorder=3))
 ax.set_xlim(0,4);ax.set_ylim(5,0);ax.set_xticks(np.arange(4)+.5,[f'W{w}' for w in W]);ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,pad=3)
 ax.set_yticks(np.arange(5)+.5,[str(h) for h in H] if j==0 else [])
 if j==0:ax.set_ylabel('Frames',fontsize=8.5,labelpad=5)
 for sp in ax.spines.values():sp.set_visible(False)
 ax.set_title(title,loc='left',fontproperties=bold,fontsize=10,pad=24)
 cbax=fig.add_axes([lefts[j]/width,.28/height,widths[j]/width,.055/height]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cbax,orientation='horizontal')
 cb.set_ticks([18,21,24] if key=='psnr' else [.14,.18,.23]);cb.ax.tick_params(labelsize=7.3,pad=2,length=2,width=.5);cb.outline.set_visible(False);cb.solids.set_rasterized(False)
# Bars preserve a zero origin, with one shared scale per metric over all horizons.
for j,(ax,key,title,xlim,ticks,unit) in enumerate(zip(axs[2:],['latency_s','peak_GiB'],['(c) Latency ↓','(d) GPU memory ↓'],[108,2.85],[[0,50,100],[0,1,2]],['Seconds / rollout','Peak allocated GiB'])):
 for row,h in enumerate(H):
  if row%2==0:ax.axhspan(row,row+1,color='#F5F7FB',zorder=0)
  for wi,w in enumerate(W):
   stat=D[h,f'w{w}'][key];v=stat['mean'];y=row+.17+wi*.22;barh=.155
   # Rounded corners in display space; data width transformed to a fraction of the axis.
   xx=v/xlim
   patch=FancyBboxPatch((0,1-(y+barh/2)/5),xx,barh/5,boxstyle='round,pad=0,rounding_size=.008',transform=ax.transAxes,fc=C[wi],ec=(1,1,1,.65),linewidth=0,hatch='////',zorder=3)
   ax.add_patch(patch)
   low,high=stat['ci95'];ax.errorbar([v],[y],xerr=[[v-low],[high-v]],fmt='none',ecolor='#3E495B',elinewidth=.55,capsize=.8,capthick=.5,zorder=4)
 ax.set_xlim(0,xlim);ax.set_ylim(5,0);ax.set_xticks(ticks);ax.set_yticks(np.arange(5)+.5,[]);ax.grid(axis='x',color='#E3E8F0',lw=.6);ax.set_axisbelow(True)
 ax.spines[['left','right','top']].set_visible(False);ax.spines['bottom'].set_color('#A7B0C0');ax.tick_params(axis='x',length=2,pad=3,labelsize=7.8);ax.tick_params(axis='y',length=0)
 ax.set_title(title,loc='left',fontproperties=bold,fontsize=10,pad=24);ax.set_xlabel(unit,fontsize=8.1,labelpad=5)
handles=[Patch(facecolor=c,edgecolor=(1,1,1,.65),linewidth=0,hatch='////',label=f'W={w}') for w,c in zip(W,C)]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.745,.935),frameon=False,ncol=4,handlelength=.85,handleheight=.65,handletextpad=.3,columnspacing=.7,fontsize=7.6)
for ext in ['pdf','png','svg']:fig.savefig(ROOT/f'figure14_heatmap_bars.{ext}',dpi=400,facecolor='white',bbox_inches='tight',pad_inches=.015)
with plt.rc_context({'svg.fonttype':'path'}):fig.savefig(ROOT/'figure14_heatmap_bars.outlined.svg',bbox_inches='tight',pad_inches=.015)
plt.close(fig)
fcap=r'\textbf{History-window sensitivity and inference cost.} Cells show mean quality; outlines mark the best mean at each length. Bars show whole-rollout latency and peak allocated GPU memory, with 95\% trajectory-bootstrap intervals. Three timing repeats are averaged within each trajectory. \emph{All five lengths use the same 30 long-sequence-eligible RT-1 test trajectories, rather than the 128 trajectories in Table 2. Their 32-frame means are therefore not directly comparable (e.g., $W=6$: 22.20 versus 22.57 dB).}'
(ROOT/'figure14_caption.tex').write_text('\\caption{'+fcap+'}\n')
(ROOT/'caption.txt').write_text(fcap+'\n')
table=(ROOT/'table5_preview.tex').read_text().splitlines();body='\n'.join(table[1:-1]).replace('\\caption{','\\captionof{table}{')
tex=r'''\documentclass[10pt]{article}
\usepackage[paperwidth=8.3in,paperheight=10.4in,left=.6in,right=.6in,top=.26in,bottom=.2in]{geometry}
\usepackage{booktabs,graphicx,amsmath,xparse,colortbl,xcolor,caption}
\usepackage[T1]{fontenc}
\definecolor{codegray}{HTML}{757575}
\input{table_style}
\captionsetup{font=small,skip=7pt}
\captionsetup[table]{labelsep=newline,labelfont=sc}
\renewcommand{\tablename}{TABLE}\renewcommand{\figurename}{Fig.}
\pagestyle{empty}
\begin{document}
\setcounter{table}{4}\setcounter{figure}{13}
'''+body+'\n\\vspace{8pt}\n\\begin{center}\n\\includegraphics[width=\\textwidth]{figure14_heatmap_bars.pdf}\n\\captionof{figure}{'+fcap+'}\n\\end{center}\n\\end{document}\n'
(ROOT/'preview.tex').write_text(tex)
(ROOT/'plotted_data.json').write_text(json.dumps({'heatmaps':plotted,'bars':{k:[[D[h,f'w{w}'][k] for w in W] for h in H] for k in ['latency_s','peak_GiB']},'horizons':H,'windows':W},indent=2))
(ROOT/'validation.json').write_text(json.dumps({'source_sha256':hashlib.sha256((ROOT/'data.json').read_bytes()).hexdigest(),'raw_means_no_normalization':True,'colormap_ranges':{'psnr':[18,24],'lpips':[.14,.23]},'bar_axes_start_zero':True,'n_trajectories':30,'all_horizons_and_windows':True,'paper_modified':False,'visual_review_pending':True},indent=2))
print('Heatmap + bar preview generated.')
