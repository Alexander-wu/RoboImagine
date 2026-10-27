"""Download a released checkpoint and run causal RGB/action-conditioned prediction."""
import argparse,json,random,time
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['rt1','bridge','robocasa'],default='rt1')
    p.add_argument('--variant',choices=['main','sft'],default='main')
    p.add_argument('--bundle',type=Path,help='Existing assembled bundle; otherwise download from Hugging Face')
    p.add_argument('--weights',default='weights');p.add_argument('--example-index',type=int,default=0)
    p.add_argument('--input',type=Path,help='NPZ with observations uint8[2,256,320,3], actions float32[H+1,A], instruction scalar string')
    p.add_argument('--reference',type=Path,help='Optional saved main-model prediction for parity checking')
    p.add_argument('--text-model',help='Optional locally downloaded pinned T5-base directory')
    p.add_argument('--output',type=Path,default=Path('predictions'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--horizon',type=int,default=32)
    p.add_argument('--beam-width',type=int,default=4);p.add_argument('--window',type=int,default=6)
    p.add_argument('--no-anchor',action='store_true');p.add_argument('--seed',type=int,default=20261011)
    p.add_argument('--download-only',action='store_true');p.add_argument('--metrics',action='store_true')
    a=p.parse_args();reference=a.reference
    if a.bundle:
        bundle=a.bundle
        if not a.input:p.error('--bundle requires --input')
        case=a.input
    else:
        from .hub import download_bundle
        bundle,case,ref=download_bundle(a.dataset,a.variant,a.weights,example_index=a.example_index)
        case=a.input or case
        if reference is None and a.variant=='main' and a.input is None:reference=ref
    if a.download_only:print(json.dumps({'bundle':str(bundle),'example':str(case)}));return
    import torch
    from .pipeline import WorldPipeline
    from .window import predict_with_window
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    with np.load(case,allow_pickle=False) as z:
        observations=z['observations'].copy();actions=z['actions'].copy();instruction=str(z['instruction'])
        truth=z['ground_truth'].copy() if 'ground_truth' in z else None
    pipe=WorldPipeline(bundle,a.dataset,a.variant,a.device,a.text_model)
    pipe.condition(instruction)
    if a.device.startswith('cuda'):torch.cuda.synchronize()
    started=time.perf_counter()
    result=predict_with_window(pipe,observations,actions,instruction,horizon=a.horizon,beam_width=a.beam_width,recent_states=a.window,retain_anchor=not a.no_anchor)
    if a.device.startswith('cuda'):torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output/'prediction.npz',prediction=result['prediction'],tokens=result['tokens'])
    import imageio.v2 as iio
    rgb=np.round(np.clip(result['prediction'],0,1)*255).astype('uint8')
    iio.mimwrite(a.output/'prediction.mp4',rgb,fps=4,format='FFMPEG',codec='libx264')
    report={'dataset':a.dataset,'variant':a.variant,'horizon':a.horizon,'window':a.window,'retain_anchor':not a.no_anchor,'beam_width':a.beam_width,'seconds_single_unwarmed_run':elapsed,'precision':'float32','seed':a.seed,'torch':torch.__version__}
    if truth is not None and len(truth)>=a.horizon:
        iio.mimwrite(a.output/'ground_truth.mp4',truth[:a.horizon],fps=4,format='FFMPEG',codec='libx264')
        mse=np.mean((result['prediction'].astype('float64')-truth[:a.horizon]/255.)**2,axis=(1,2,3))
        report['psnr_db']=float(np.mean(-10*np.log10(np.maximum(mse,1e-12))))
        report['mae']=float(np.mean(np.abs(result['prediction']-truth[:a.horizon]/255.)))
        if a.metrics:
            from .metrics import Metrics
            report['visual_metrics']=Metrics(a.device)(result['prediction'],truth[:a.horizon])
    if reference is not None:
        with np.load(reference,allow_pickle=False) as z:
            if z['prediction'].shape==result['prediction'].shape:
                report['reference_check']={'exact_tokens':bool(np.array_equal(z['tokens'],result['tokens'])),'maximum_pixel_error':float(np.max(np.abs(z['prediction']-result['prediction'])))}
    (a.output/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'memory_trace.json').write_text(json.dumps(result['trace'],indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
