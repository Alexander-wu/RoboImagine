"""128 paired H32 rollouts, identical to R2; resumable per-case artifacts."""
from sft_common import *
import time,traceback,gc
from safetensors.torch import load_file
from futureworlds.evaluation import Metrics

rank,world=init_ddp(); setup()
selection,rows,witnesses=verified_inputs()
training=json.loads((E/'run/training_complete.json').read_text())
assert training['updates']==500 and training['source']==own_seal()
assert training['protocol']==protocol() and training['initial_digest']==EXPECTED_INITIAL
O=E/'evaluation'; R=O/f'rank{rank}'; R.mkdir(parents=True,exist_ok=True)
pipe=load_initial(); cp=Path(training['checkpoint'])
receipt=json.loads((cp/'receipt.json').read_text())
assert all(sha(cp/name)==expected for name,expected in receipt['files'].items())
pipe.model.core.load_state_dict(load_file(str(cp/'model.safetensors'),device='cuda:0'),strict=True)
pipe.model.adapters.load_state_dict(load_file(str(cp/'text_adapters.safetensors'),device='cuda:0'),strict=True)
assert agree(pipe.model)==training['final_digest']
pipe.model.eval().requires_grad_(False); metric=Metrics('cuda:0')
evaluations=[]; started=time.monotonic(); computed=0
try:
    for i,row in enumerate(selection['dev']):
        if i%world!=rank: continue
        rp=R/f'dev_{i:03d}.json'; npz=R/f'dev_{i:03d}.npz'
        if rp.exists() and npz.exists():
            rec=json.loads(rp.read_text())
            assert rec['id']==row['id'] and rec['prepared_input_sha256']==row['sha256']
            assert rec['model_digest']==training['final_digest'] and rec['artifact_sha256']==sha(npz)
            assert rec['source']==own_seal()
        else:
            setup(SEED+10000+i); images,actions,text=pixels(row); tick=time.monotonic()
            result=pipe.predict(images[:2].copy(),actions[:33].copy(),text,horizon=32,beam_width=4,memory='full')
            prediction=np.asarray(result['prediction'],dtype='float32'); gt=images[2:34]
            assert prediction.shape==(32,256,320,3) and np.isfinite(prediction).all()
            torch.cuda.synchronize(); infer=time.monotonic()-tick
            scored=metric(prediction,gt,[10,20,32])
            if computed==0:
                ident=metric(gt.astype('float32')/255,gt,[32])['metrics']['32']
                assert abs(ident['ssim']-1)<1e-5 and abs(ident['lpips'])<1e-5
            tmp=npz.with_name(npz.stem+'.partial.npz')
            np.savez_compressed(tmp,prediction=prediction,tokens=result['tokens'],ground_truth=gt,
                                observations=images[:2],instruction=np.array(text)); os.replace(tmp,npz)
            rec=dict(id=row['id'],index=i,metrics=scored['metrics'],per_frame=scored['per_frame'],
                     frame32={k:v[-1] for k,v in scored['per_frame'].items()},inference_seconds=infer,
                     seconds=time.monotonic()-tick,artifact_sha256=sha(npz),prepared_input_sha256=row['sha256'],
                     model_digest=training['final_digest'],source=own_seal())
            write(rp,rec); computed+=1
        evaluations.append(rec)
        write(R/'status.json',dict(state='EVALUATING',cases=len(evaluations),total=128//world,new_cases=computed))
        print('SFT_EVAL',rank,len(evaluations),128//world,flush=True)
        pipe.text_cache.clear()
    assert len(evaluations)==128//world and agree(pipe.model)==training['final_digest']
    own=dict(rank=rank,evaluation=evaluations,eval_seconds_this_attempt=time.monotonic()-started,new_cases=computed)
    write(R/'complete.json',own); parts=gather(own)
    if rank==0:
        rs=sorted([r for part in parts for r in part['evaluation']],key=lambda r:r['index'])
        baseline=json.loads(BASELINE.read_text()); assert [r['index'] for r in rs]==list(range(128))
        assert [r['id'] for r in rs]==[r['id'] for r in baseline['evaluation']]
        assert all(r['prepared_input_sha256']==b['prepared_input_sha256'] for r,b in zip(rs,baseline['evaluation']))
        mean32={k:float(np.mean([r['metrics']['32'][k] for r in rs])) for k in rs[0]['metrics']['32']}
        contrasts={}
        for key in ['psnr','ssim','lpips']:
            delta=np.array([r['metrics']['32'][key]-b['metrics']['32'][key] for r,b in zip(rs,baseline['evaluation'])])
            rng=np.random.default_rng(20261023); boot=np.mean(delta[rng.integers(0,128,(5000,128))],axis=1)
            contrasts[key]=dict(delta=float(delta.mean()),ci95=[float(x) for x in np.quantile(boot,[.025,.975])])
        prepared=json.loads((E/'prepared.json').read_text())
        write(E/'complete.json',dict(state='SUCCEEDED',arm='sft_continuation500',cases=128,updates=500,
                                     initial_digest=EXPECTED_INITIAL,final_digest=training['final_digest'],
                                     mean_frames1_32=mean32,evaluation=rs,paired_delta_vs_sft=contrasts,
                                     train_seconds=training['train_seconds'],train_gpu_hours=training['train_gpu_hours'],
                                     preparation_seconds=prepared['preparation_seconds'],
                                     eval_seconds_this_attempt=max(p['eval_seconds_this_attempt'] for p in parts),
                                     protocol=protocol(),source=own_seal(),baseline_sha256=sha(BASELINE),
                                     statistics='paired trajectory bootstrap5000; not training-seed uncertainty'))
        print('SFT_COMPLETE',json.dumps(mean32),flush=True)
    write(R/'status.json',dict(state='SUCCEEDED',cases=128//world,total=128//world)); dist.barrier()
except BaseException:
    write(R/'ERROR.json',dict(error=traceback.format_exc())); traceback.print_exc(); os._exit(1)
finally:
    dist.destroy_process_group()
