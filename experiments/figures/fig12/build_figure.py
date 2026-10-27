"""Integrated recorded search storyboard; original pixels and ancestry unchanged."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch,Circle
R=Path(__file__).resolve().parent
for f in (R/'assets').glob('*.ttf'):fm.fontManager.addfont(str(f))
plt.rcParams.update({'font.family':'EB Garamond','pdf.fonttype':42,'svg.fonttype':'none','font.size':15})
raw=json.loads((R/'raw_traces.json').read_text());m=json.loads((R/'asset_manifest.json').read_text())
r=next(v['record'] for v in raw['arms']['diverse']['first_batch'] if v['record']['rank']==6)
for n,v in m['assets'].items():
 if n.startswith('diverse_'):
  f=R/'assets'/n;assert hashlib.sha256(f.read_bytes()).hexdigest()==v['sha256']
  assert hashlib.sha256(np.asarray(Image.open(f)).tobytes()).hexdigest()==v['pixel_sha256']
parents=[f['frame_parent_indices'] for f in r['trace']['frames']];paths=[]
for g in range(4):
 slot=g*2;p=[slot]
 for pp in parents[::-1]:slot=pp[slot];p.append(slot)
 paths.append(p[::-1]);assert all(k//2==g for k in paths[-1])
colors=['#1668B7','#008577','#CF6819','#7543AE'];ink='#151515';gray='#151515'
X=np.array([112,235,300,365,430,490,541,592,643,694,745])
W,H=1200,1050;fig=plt.figure(figsize=(9.6,7.2),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,150));ax.axis('off')
def text(x,y,s,size=15,color=ink,weight='normal',ha='left',style='normal'):
 return ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va='center',style=style,zorder=12)
def image(name,x,y,w,h,color='#253141',radius=7):
 ax.add_patch(FancyBboxPatch((x+3,y+4),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',fc='#26394D',ec='none',alpha=.12,zorder=6))
 clip=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',fc='white',ec=color,lw=1.5,zorder=9)
 ax.add_patch(clip);im=ax.imshow(np.asarray(Image.open(R/'assets'/name)),extent=(x,x+w,y+h,y),zorder=10,interpolation='none');im.set_clip_path(clip)
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',fc='none',ec=color,lw=1.9,zorder=11))
text(35,169,'Group',16,weight='bold')
for x,s in [(235,'Frame +1'),(490,'Frame +5'),(745,'Frame +10')]:text(x,169,s,18,weight='bold',ha='center')
text(1030,169,'Feedback',20,weight='bold',ha='center')
ax.plot([875,875],[187,994],color='#CDD3D9',lw=.9)
sem=np.asarray(r['semantic']);vis=np.asarray([v['reward'] for v in r['metrics']]);fused=np.asarray(r['rewards'])
for g,y in enumerate([215,415,615,815]):
 c=colors[g]
 # A compact dark group badge; no background panels.
 ax.add_patch(FancyBboxPatch((25,y-19),70,38,boxstyle='round,pad=0,rounding_size=9',fc=c,ec='none',zorder=8));text(60,y,f'G{g+1}',20,color='white',weight='bold',ha='center')
 def yy(slot):return y+27*(slot%2)
 for f,pp in enumerate(parents,1):
  for slot in [g*2,g*2+1]:
   parent=pp[slot];assert parent//2==g
   ax.plot([X[f-1],X[f]],[yy(parent),yy(slot)],color='#A5ADB7',lw=1.15,zorder=2)
   ax.add_patch(Circle((X[f],yy(slot)),3.6,fc='white',ec='#8D99A7',lw=.8,zorder=3))
 path=paths[g];ys=[yy(k) for k in path]
 ax.plot(X,ys,color=c,lw=3.2,solid_capstyle='round',solid_joinstyle='round',zorder=4)
 for f in range(11):
  radius=11.5 if f in [1,5,10] else 4.1
  ax.add_patch(Circle((X[f]+.9,ys[f]+1.6),radius,fc='#1E3144',ec='none',alpha=.14,zorder=4))
  ax.add_patch(Circle((X[f],ys[f]),radius,fc=c if f in [1,5,10] else 'white',ec=c,lw=1.6,zorder=5))
  if f in [1,5,10]:text(X[f],ys[f],str(f),10,color='white',weight='bold',ha='center')
  elif g==0:text(X[f],ys[f]-15,str(f),10.5,color=ink,ha='center')
 # Genuine retained-slot changes, without inventing extra token-tree branches.
 changes=[f for f in range(1,11) if path[f]%2!=path[f-1]%2]
 for f in changes:
  if path[f]%2==1:
   ax.add_patch(Circle((X[f],ys[f]),9.5,fc='none',ec=c,lw=1,zorder=5))
   # Double ring marks the logged beam-slot change.
 for frame in [1,5,10]:
  x=X[frame];ax.plot([x,x],[ys[frame]+11,y+35],color=c,lw=1.1,zorder=3)
  image(f'diverse_c{g+1}_t{frame:02d}.png',x-87,y+35,174,139.2,c)
 text(810,y,f'C{g+1}',19,color=c,weight='bold',ha='center')
 ax.annotate('',xy=(859,y),xytext=(839,y),arrowprops=dict(arrowstyle='-|>',color=c,lw=1.3),zorder=6)
 # A compact three-row score table aligned with the candidate thumbnails.
 for dy,vals,label,co in [(55,sem,'Semantic','#6A3598'),(99,vis,'Visual','#007A68'),(143,fused,'Fused',ink)]:
  cy=y+dy;best=g==int(np.argmax(vals))
  text(898,cy,label,16.5,color=co)
  text(1158,cy,f'{vals[g]:+.3f}',23,color=co,weight='bold' if best else 'normal',ha='right')
  if best:ax.plot([1085,1158],[cy+17,cy+17],color=co,lw=1.7)
 if g<3:ax.plot([895,1178],[y+196,y+196],color='#DBE0E5',lw=.7)
ax.plot([20,53],[1020,1020],color=colors[0],lw=2.5,solid_capstyle='round');text(66,1020,'Selected path',20,color=ink,weight='bold')
ax.plot([280,313],[1020,1020],color='#64717E',lw=1.2);text(326,1020,'Retained beam',20,color=ink,weight='bold')
ax.add_patch(Circle((647,1020),6,fc='white',ec=colors[3],lw=1.2));ax.add_patch(Circle((647,1020),9,fc='none',ec=colors[3],lw=.8));text(668,1020,'Slot switch',20,color=ink,weight='bold')
text(1175,1020,'Higher is better',19,color=ink,weight='bold',ha='right')
for ext in ['png','pdf','editable.svg']:fig.savefig(R/f'gdbs_case_study.{ext}',dpi=200,facecolor='white',pad_inches=0)
with plt.rc_context({'svg.fonttype':'path'}):fig.savefig(R/'gdbs_case_study.svg',pad_inches=0)
plt.close(fig)
old=json.loads((R/'expected_ancestry.json').read_text());assert old['returned_ancestry']==paths
assert old['semantic']==sem.tolist() and old['visual']==vis.tolist() and old['fused_recorded_reward']==fused.tolist()
old.update(version=7,layout='Integrated frame-boundary routes with attached unchanged keyframes; dark group colors and compact per-candidate score tables',visual_review='PENDING',outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('gdbs_case_study.*')})
(R/'validation.json').write_text(json.dumps(old,indent=2)+'\n')
print('Source/data/ancestry unchanged; v7 exported')
