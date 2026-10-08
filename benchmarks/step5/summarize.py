from __future__ import annotations

import argparse,csv,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path


def _f(v):
    try:return float(v)
    except:return math.nan

def _i(v):
    try:return int(float(v))
    except:return 0

# A wall-clock measurement below this is dominated by timer/scheduler noise
# on typical hardware, not by the kernel's real cost -- an "observed slope"
# built from such points is not a trustworthy complexity estimate and must
# not be reported as one (Step-5 audit finding).
_TIME_NOISE_FLOOR_SECONDS = 0.001


def _slope(rs,key,floor=0.0):
    pts=sorted((max(1,_i(r.get('input_elements'))),_f(r.get(key))) for r in rs if r.get('status')=='ok' and _i(r.get('input_elements'))>0 and _f(r.get(key))>floor)
    if len(pts)<2:return math.nan
    slopes=[]
    for (x1,y1),(x2,y2) in zip(pts,pts[1:]):
        if x2>x1 and y1>0 and y2>0: slopes.append(math.log(y2/y1)/math.log(x2/x1))
    return statistics.median(slopes) if slopes else math.nan

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--input',default='step5_results/step5_system_feature_results.csv')
 ap.add_argument('--feature-out',default='step5_results/step5_system_feature_summary.csv');ap.add_argument('--family-out',default='step5_results/step5_system_family_summary.csv')
 ap.add_argument('--json-out',default='step5_results/step5_system_summary.json');a=ap.parse_args()
 rows=list(csv.DictReader(open(a.input,encoding='utf-8')))
 byfeat=defaultdict(list)
 for r in rows:byfeat[r.get('feature','')].append(r)
 fs=[]
 for name,rs in sorted(byfeat.items()):
  ok=[r for r in rs if r.get('status')=='ok']; first=rs[0]
  fs.append({'feature':name,'modality':first.get('modality',''),'family':first.get('family',''),'cases':len(rs),'ok_cases':len(ok),
             'max_scale_ok':max((_i(r.get('scale')) for r in ok),default=0),'observed_time_exponent':_slope(rs,'median_seconds',floor=_TIME_NOISE_FLOOR_SECONDS),
             'observed_rss_exponent':_slope(rs,'peak_rss_delta_bytes'),'nondeterministic':any(str(r.get('deterministic','')).lower() not in {'true','1'} for r in ok),
             'nonfinite_values':sum(_i(r.get('nonfinite_count')) for r in ok),'statuses':';'.join(f'{k}:{v}' for k,v in sorted(Counter(r.get('status','') for r in rs).items()))})
 fo=Path(a.feature_out);fo.parent.mkdir(parents=True,exist_ok=True)
 with fo.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(fs[0]) if fs else ['feature']);w.writeheader();w.writerows(fs)
 groups=defaultdict(list)
 for r in fs:groups[(r['modality'],r['family'])].append(r)
 fam=[]
 for (mod,fa),rs in sorted(groups.items()):
  ex=[_f(r['observed_time_exponent']) for r in rs if math.isfinite(_f(r['observed_time_exponent']))]
  fam.append({'modality':mod,'family':fa,'features_benchmarked':len(rs),'features_with_any_ok':sum('ok:' in r['statuses'] for r in rs),
              'median_observed_time_exponent':statistics.median(ex) if ex else '',
              'nondeterministic_features':';'.join(r['feature'] for r in rs if r['nondeterministic']),
              'nonfinite_values':sum(_i(r['nonfinite_values']) for r in rs)})
 fop=Path(a.family_out)
 with fop.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(fam[0]) if fam else ['modality','family']);w.writeheader();w.writerows(fam)
 Path(a.json_out).write_text(json.dumps({'cases':len(rows),'features':len(fs),'families':len(fam),'statuses':dict(Counter(r.get('status','') for r in rows))},indent=2,sort_keys=True)+'\n')
 print(f'cases={len(rows)} features={len(fs)} families={len(fam)}');return 0
if __name__=='__main__':raise SystemExit(main())
