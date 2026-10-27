"""Full-cohort Figure 11 preview. Fails closed without complete verified scores."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import Bbox
R=Path(__file__).resolve().parent
METHODS=['sft','mcts500','ordinary500','diverse500']
N=128
BASELINE=R/'quality_reference.json'
NAMES={'sft':'SFT','mcts500':'Likelihood-guided MCTS (16)','ordinary500':'Ordinary beam','diverse500':'Grouped diverse beam'}

def main():
    d=json.loads((R/'results.json').read_text())
    assert d['state']=='SUCCEEDED' and d['methods']==METHODS
    assert d['generation_updates']==dict(zip(METHODS,[0,500,500,500]))
    assert len(d['per_case'])==N and d['indices']==list(range(N))
    assert len({c['id'] for c in d['per_case']})==N
    assert [c['index'] for c in d['per_case']]==list(range(N))
    assert d['judge_completion']['frozen_witness_unchanged']
    assert d['judge_completion']['records']==1536 and len(d['raw_judgments'])==1536
    dist=np.asarray([c['token_distance_pct'] for c in d['per_case']],dtype=float)
    sem=np.asarray([c['semantic_feedback']['scores'] for c in d['per_case']],dtype=float)
    metrics={a:{k:np.asarray([c['metrics'][a][k] for c in d['per_case']],dtype=float) for k in ['psnr','ssim','lpips','mae','mse']} for a in METHODS}
    vis=np.stack([-(metrics[a]['mae']+metrics[a]['lpips']) for a in METHODS],axis=1)
    assert dist.shape==(N,4,4) and sem.shape==vis.shape==(N,4)
    assert np.isfinite(dist).all() and np.isfinite(sem).all() and np.isfinite(vis).all()
    assert np.allclose(dist,dist.transpose(0,2,1),rtol=0,atol=1e-12)
    assert np.allclose(dist[:,np.arange(4),np.arange(4)],0,rtol=0,atol=1e-12)
    assert (dist>=0).all() and (dist<=100).all()
    assert np.allclose(sem.sum(axis=1),0,rtol=0,atol=1e-8)
    for c in d['per_case']:
        ans=c['semantic_feedback'];assert len(ans['pairs'])==6
        assert {(p['i'],p['j']) for p in ans['pairs']}=={(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)}
        reconstructed=np.zeros(4)
        for p in ans['pairs']:
            a,b=p['forward'],p['reverse']
            assert abs(.5*((a['A']-a['B'])-(b['A']-b['B']))-p['margin'])<1e-9
            reconstructed[p['i']]+=p['margin']/3
            reconstructed[p['j']]-=p['margin']/3
        assert np.allclose(reconstructed,ans['scores'],rtol=0,atol=1e-9)
    D,S,V=dist.mean(0),sem.mean(0),vis.mean(0)
    assert np.allclose(D,d['mean_token_distance_pct'],rtol=0,atol=1e-10)
    assert np.allclose(S,d['mean_semantic'],rtol=0,atol=1e-10)
    baseline=json.loads(BASELINE.read_text());published={r['method']:r for r in baseline['table']}
    for a in METHODS:
        for k,arr in metrics[a].items():
            assert np.isfinite(arr).all()
            assert abs(arr.mean()-d['mean_metrics'][a][k])<1e-10
            assert abs(arr.mean()-published[NAMES[a]][k])<1e-8,(a,k,arr.mean(),published[NAMES[a]][k])
    rng=np.random.default_rng(20261026);samples=rng.integers(0,N,(20000,N))
    sci=np.quantile(sem[samples].mean(1),[.025,.975],axis=0)
    vci=np.quantile(vis[samples].mean(1),[.025,.975],axis=0)
    sd=sem-sem[:,0:1];vd=vis-vis[:,0:1]
    sdci=np.quantile(sd[samples].mean(1),[.025,.975],axis=0)
    vdci=np.quantile(vd[samples].mean(1),[.025,.975],axis=0)
    rows=[]
    for j,a in enumerate(METHODS):
        rows.append(dict(method=a,semantic=float(S[j]),semantic_ci_low=float(sci[0,j]),semantic_ci_high=float(sci[1,j]),semantic_delta_vs_sft=float(sd[:,j].mean()),semantic_delta_ci_low=float(sdci[0,j]),semantic_delta_ci_high=float(sdci[1,j]),visual=float(V[j]),visual_ci_low=float(vci[0,j]),visual_ci_high=float(vci[1,j]),visual_delta_vs_sft=float(vd[:,j].mean()),visual_delta_ci_low=float(vdci[0,j]),visual_delta_ci_high=float(vdci[1,j]),**{k:float(v.mean()) for k,v in metrics[a].items()}))
    pair_rows=[]
    for i in range(4):
        for j in range(i+1,4):
            arr=dist[:,i,j];ci=np.quantile(arr[samples].mean(1),[.025,.975])
            pair_rows.append(dict(method_i=METHODS[i],method_j=METHODS[j],mean_token_distance_percent=float(arr.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
    summary=dict(trajectories=N,generation_updates=dict(zip(METHODS,[0,500,500,500])),frames=32,rows=rows,mean_token_distance_pct=D.tolist(),distance_pairs=pair_rows,bootstrap='20000 paired trajectory resamples; seed20261026. Single-training-seed uncertainty.',semantic_definition='Frozen training Jev, both orders of six method pairs; mean antisymmetric margin against other three methods, then mean across128 trajectories. Signed relative preferences, not absolute task quality.',visual_definition='Negative of (MAE + VGG-LPIPS), calculated on unchanged float predictions; higher is better.',distance_definition='Between-model dynamics-token normalized Hamming distance; not within-search diversity or a quality score.')
    (R/'summary.json').write_text(json.dumps(summary,indent=2))
    for name,data in [('summary.csv',rows),('distance_pairs.csv',pair_rows)]:
        with (R/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    per_case=[dict(index=c['index'],id=c['id'],method=a,semantic=float(sem[i,j]),visual=float(vis[i,j]),**c['metrics'][a]) for i,c in enumerate(d['per_case']) for j,a in enumerate(METHODS)]
    with (R/'per_case.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(per_case[0]));w.writeheader();w.writerows(per_case)
    for f in (R/'assets').glob('*.ttf'):fm.fontManager.addfont(str(f))
    plt.rcParams.update({'font.family':'EB Garamond','font.size':8,'text.color':'#263343','pdf.fonttype':42,'svg.fonttype':'none'})
    W,H=3.85,2.20
    fig=plt.figure(figsize=(W,H));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off')
    cm=LinearSegmentedColormap.from_list('shared',['#F0F3F9','#C5D4EE','#9CACE0','#8272BC','#604296']);norm=Normalize(0,100)
    labels=['SFT','MCTS','Ordinary','GDBS'];short=['SFT','MCTS','Ord.','GDBS'];colors=['#34485E','#A85B2D','#205F99','#653C96']
    x0,y0,cell=.63,.37,.34
    for i in range(4):
        y=y0+(3-i)*cell
        ax.text(x0-.29,y+cell/2,labels[i],ha='center',va='center',rotation=45,fontsize=8,color=colors[i],weight='bold')
        for j in range(4):
            x=x0+j*cell
            ax.add_patch(FancyBboxPatch((x+.008,y+.008),cell-.016,cell-.016,boxstyle='round,pad=0,rounding_size=.025',facecolor=cm(norm(D[i,j])),edgecolor='none'))
            ax.text(x+cell/2,y+cell/2,'—' if i==j else f'{D[i,j]:.1f}',ha='center',va='center',fontsize=8.5,color='#FFFFFF' if i!=j and D[i,j]>=45 else '#263343',weight='bold' if i!=j else 'normal')
    for j in range(4):ax.text(x0+(j+.5)*cell,y0+4*cell+.035,short[j],ha='center',va='bottom',fontsize=8,color=colors[j])
    ax.text(1.31,2.08,'Token distance (%)',ha='center',va='center',fontsize=9,weight='bold')
    ax.text(2.94,2.08,'Mean feedback ↑',ha='center',va='center',fontsize=9,weight='bold')
    for j,(values,title,fg,bg) in enumerate([(S,'Semantic','#593385','#E7DCF3'),(V,'Visual','#12695B','#D8EEE6')]):
        cx=2.52+j*.72;ax.text(cx,y0+4*cell+.035,title,ha='center',va='bottom',fontsize=8.5,weight='bold',color=fg)
        for i,value in enumerate(values):
            y=y0+(3-i)*cell
            ax.add_patch(FancyBboxPatch((cx-.335,y+.008),.67,cell-.016,boxstyle='round,pad=0,rounding_size=.025',facecolor=bg,edgecolor='none'))
            ax.text(cx,y+cell/2,f'{value:+.4f}' if j==0 else f'{value:.4f}',ha='center',va='center',fontsize=8.5,color=fg,weight='bold' if i==int(np.argmax(values)) else 'normal')
    ax.plot([2.10,2.10],[.38,1.92],color='#BCC5D3',lw=.7)
    cax=fig.add_axes([((3.755-1.36)/2)/W,.16/H,1.36/W,.065/H]);cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cm),cax=cax,orientation='horizontal');cb.set_ticks([0,50,100]);cb.ax.tick_params(labelsize=7,length=1.5,pad=1);cb.outline.set_visible(False);cb.solids.set_rasterized(False)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    assert all(0<=t.get_window_extent(renderer).x0 and t.get_window_extent(renderer).x1<=fig.bbox.width and 0<=t.get_window_extent(renderer).y0 and t.get_window_extent(renderer).y1<=fig.bbox.height for t in ax.texts)
    crop=Bbox.from_extents(.145,0,3.61,2.16)
    for ext in ['png','pdf']:fig.savefig(R/f'figure.{ext}',dpi=420,facecolor='white',bbox_inches=crop,pad_inches=0)
    fig.savefig(R/'figure.editable.svg',facecolor='white',bbox_inches=crop,pad_inches=0)
    with plt.rc_context({'svg.fonttype':'path'}):fig.savefig(R/'figure.svg',facecolor='white',bbox_inches=crop,pad_inches=0)
    plt.close(fig)
    validation=dict(all128included=True,final500_checkpoints=True,source_visual_metrics_unchanged=True,matrix_symmetric_zero_diagonal=True,semantic_pair_margins_reconstructed=True,raw_ordered_judgments=1536,frozen_judge_verified=True,source_results_sha256=hashlib.sha256((R/'results.json').read_bytes()).hexdigest(),final_quality_summary_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),bounds_check='PASS',preview_only=True,figure_inches=[W,H])
    (R/'validation.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
