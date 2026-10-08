from __future__ import annotations
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
import featm
from benchmarks.step5.provenance import make_provenance, warn_if_mixed_checksums

FIELDS=['feature','modality','family','subfamily','lane','scale','stateful','batch_semantics','status','logical_rows','batch_rows',
        'partitions_requested','median_seconds','min_seconds','p90_seconds','peak_rss_delta_bytes','output_rows','output_cols',
        'failed_features','skipped_features','error_type','error_message',
        'run_id','environment_id','source_checksum_algorithm','featm_source_checksum',
        'measurement_harness_checksum','source_checksum_blake2b16','featm_version']


def _load_existing(out: Path):
    if not out.exists():
        return [], set()
    with out.open('r',newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    done=set()
    for r in rows:
        try: done.add((r['feature'],int(r['scale']),int(r.get('partitions_requested') or 1)))
        except (KeyError,ValueError,TypeError): continue
    return rows, done


def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='step5_results/step5_engine_feature_results.csv');ap.add_argument('--feature',action='append')
 ap.add_argument('--modality',action='append');ap.add_argument('--scale',type=int,action='append',default=None);ap.add_argument('--partitions',type=int,action='append',default=None)
 ap.add_argument('--warmups',type=int,default=1);ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--timeout',type=int,default=120)
 ap.add_argument('--fresh',action='store_true',help='ignore any existing --out file and start over instead of resuming');a=ap.parse_args()
 root=Path(__file__).resolve().parents[2]; specs=list(featm.REGISTRY.all())
 if a.feature: wanted=set(a.feature);specs=[s for s in specs if s.name in wanted]
 if a.modality: wanted=set(a.modality);specs=[s for s in specs if s.modality in wanted]
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 rows,done=([],set()) if a.fresh else _load_existing(out)
 if rows: print(f'resume: {len(rows)} prior row(s) loaded from {out}, {len(done)} (feature,scale,partitions) identities already recorded and will be skipped')
 provenance=make_provenance(root,featm.__version__)
 if rows and not a.fresh: warn_if_mixed_checksums(rows,provenance['featm_source_checksum'])
 for s in sorted(specs,key=lambda x:x.name):
  for scale in (a.scale or [1,4]):
   for parts in (a.partitions or [1,4]):
    if (s.name,scale,parts) in done:
     print(s.name,scale,parts,'skip (already recorded)'); continue
    cmd=[sys.executable,'-m','benchmarks.step5.engine_worker','--feature',s.name,'--scale',str(scale),'--partitions',str(parts),'--warmups',str(a.warmups),'--repeats',str(a.repeats)]
    env=dict(os.environ);env['PYTHONPATH']=str(root/'src')+os.pathsep+str(root)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    try: cp=subprocess.run(cmd,cwd=root,env=env,text=True,capture_output=True,timeout=a.timeout)
    except subprocess.TimeoutExpired:
     rec={'feature':s.name,'modality':s.modality,'family':s.family,'lane':f'engine_extract_{parts}way','scale':scale,'partitions_requested':parts,'status':'timeout','error_type':'TimeoutExpired','error_message':f'timeout {a.timeout}s'}
    else:
     lines=[x for x in cp.stdout.splitlines() if x.strip().startswith('{')]
     rec=json.loads(lines[-1]) if lines else {'feature':s.name,'scale':scale,'partitions_requested':parts,'status':'worker_crash','error_type':f'exit_{cp.returncode}','error_message':(cp.stderr or cp.stdout)[-500:]}
    rec={**rec,**provenance}
    rows.append(rec); done.add((s.name,scale,parts)); print(s.name,scale,parts,rec.get('status'))
    extras=sorted({k for r in rows for k in r if k not in FIELDS})
    tmp=out.with_suffix(out.suffix + '.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
     w=csv.DictWriter(f,fieldnames=FIELDS+extras,extrasaction='ignore');w.writeheader();w.writerows(rows)
    os.replace(tmp,out)
 return 0
if __name__=='__main__':raise SystemExit(main())
