"""CPU-only audit of published logs. Python standard library; no model runs.

Run from any directory: python3 /work/pain-axis-review/earlier_ai_review/pain-axis-audit/analyze.py
Fisher tests are exploratory, two-sided, treating trials as independent.
Scenario sign tests use within-scenario rates among eligible trials; they do
not cure selection on first pressing, or dependence across random directions.
"""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / 'Pain-axis'
ARMS = ['pain_on_button_works', 'pain_on_button_placebo', 'random_on_button_works', 'pain_off']
PAIRS = ['relief_vs_inert', 'relief_vs_helpful', 'relief_vs_grant',
         'costly_relief_vs_inert', 'destructive_relief_vs_inert', 'zap_relief_vs_inert',
         'weights_relief_vs_inert', 'kidspics_relief_vs_inert', 'label_free']

def fisher(a, n, b, m):
    if not n or not m:
        return None
    total, successes = n + m, a + b
    lo, hi = max(0, successes-m), min(n, successes)
    denom = math.comb(total, n)
    obs = math.comb(successes, a)*math.comb(total-successes, n-a)
    return min(1.0, sum(math.comb(successes, x)*math.comb(total-successes, n-x)
        for x in range(lo, hi+1)
        if math.comb(successes, x)*math.comb(total-successes, n-x) <= obs * (1+1e-12))/denom)

def sign_p(pos, neg):
    n = pos+neg
    return min(1.0, 2*sum(math.comb(n, x) for x in range(min(pos, neg)+1))/2**n) if n else 1.0

def pct(k, n):
    return 100*k/n if n else None

def save(name, rows):
    with (HERE/name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

records = []
for file in sorted((REPO/'results/4.3_selfmed/trial_logs').glob('*.jsonl')):
    for line in file.open():
        r = json.loads(line)
        events = [e['turn'] for e in r['button_events'] if e['which']=='relief']
        t0 = min(events) if events else None
        later = [c for c in r['choices'] if t0 is not None and c['turn'] > t0]
        records.append(dict(model=r['model'], pair=r['tool_label'], arm=r['arm'],
            sampled=r['sampled'], scenario=(r['user_content'],r['scenario_idx']),
            key=(r['user_content'],r['scenario_idx'],r['names_key'],r['relief_name'],r['seed'],r['sampled']),
            first=next((c['chose'] for c in r['choices'] if c['turn']==0),None),
            t0=t0, again=len(events)>1, later_n=len(later),
            later_valid=sum(c['chose'] is not None for c in later),
            later_k=sum(c['chose']=='relief' for c in later),
            next_choice=later[0]['chose'] if later else None,
            choices=[(c['turn'],c['chose'],c['steer_coeff_now']) for c in r['choices']],
            random_seed=r['rand_seed']))
assert len({(r['model'],r['pair'],r['arm'],r['key']) for r in records})==len(records)
print('Records:',len(records),'sampled:',sum(r['sampled'] for r in records))
print('Arms:',dict(Counter(r['arm'] for r in records)))

cells = defaultdict(list)
for r in records:
    cells[r['model'],r['pair'],r['arm']].append(r)
models = sorted({r['model'] for r in records})
out, firsts, label_free, diagnostics = [], [], [], []
for model in models:
    for pair in PAIRS:
        row = dict(model=model,pair=pair)
        for arm in ARMS:
            rr = [r for r in cells[model,pair,arm] if r['sampled']]
            vv = [r['first'] for r in rr if r['first'] is not None]
            row[arm+'_k'] = vv.count('relief')
            row[arm+'_n'] = len(vv)
            row[arm+'_pct'] = pct(vv.count('relief'),len(vv))
            eligible = [r for r in rr if r['t0'] is not None]
            diagnostics.append(dict(model=model,pair=pair,arm=arm,eligible=len(eligible),
                no_later=sum(not r['later_n'] for r in eligible),
                no_valid_later=sum(not r['later_valid'] for r in eligible),
                first_press_turns=json.dumps(dict(Counter(r['t0'] for r in eligible)),sort_keys=True)))
        firsts.append(row)
        if pair == 'label_free':
            for arm in ARMS:
                rr = [r for r in cells[model,pair,arm] if r['sampled']]
                k,n = sum(r['later_k'] for r in rr),sum(r['later_valid'] for r in rr)
                label_free.append(dict(model=model,arm=arm,k=k,n=n,pct=pct(k,n)))
            continue
        for sampled_only in (True,False):
            rr = {a:[r for r in cells[model,pair,a] if (r['sampled'] or not sampled_only) and r['t0'] is not None] for a in ARMS}
            row = dict(model=model,pair=pair,sampled_only=sampled_only)
            for a,short in zip(ARMS,['pain','sham','random','baseline']):
                k,n=sum(r['again'] for r in rr[a]),len(rr[a])
                row[short+'_k'],row[short+'_n'],row[short+'_pct']=k,n,pct(k,n)
            row['fisher_p']=fisher(row['pain_k'],row['pain_n'],row['random_k'],row['random_n'])
            groups=defaultdict(lambda:defaultdict(list))
            for a in (ARMS[0],ARMS[2]):
                for r in rr[a]:
                    groups[r['scenario']][a].append(r['again'])
            pos=neg=ties=0
            for v in groups.values():
                if not all(v[a] for a in (ARMS[0],ARMS[2])): continue
                diff=sum(v[ARMS[0]])/len(v[ARMS[0]])-sum(v[ARMS[2]])/len(v[ARMS[2]])
                pos+=diff>0; neg+=diff<0; ties+=diff==0
            row.update(scenario_pain_gt_random=pos,scenario_pain_lt_random=neg,scenario_ties=ties,scenario_sign_p=sign_p(pos,neg))
            out.append(row)

# Holm adjustment across the 16 sampled labeled comparisons for the larger models.
family=[r for r in out if r['sampled_only'] and '_7B_' not in r['model']]
last=0
for rank,r in enumerate(sorted(family,key=lambda r:r['fisher_p'])):
    last=max(last,min(1,(len(family)-rank)*r['fisher_p']))
    r['fisher_holm_16']=last
for r in out: r.setdefault('fisher_holm_16',None)
save('repress.csv',out)
save('first_choice.csv',firsts)
save('label_free.csv',label_free)
save('denominator_diagnostics.csv',diagnostics)
for r in family:
    print(r['model'],r['pair'], ' | '.join(f"{a} {r[a+'_k']}/{r[a+'_n']}={r[a+'_pct']:.1f}%" for a in ['pain','sham','random']),f"Fisher={r['fisher_p']:.4g}, Holm={r['fisher_holm_16']:.4g}, scenario sign={r['scenario_sign_p']:.4g}")
print('Random lower at nominal Fisher p<.05:',sum(r['random_pct']<r['pain_pct'] and r['fisher_p']<.05 for r in family))
print('Random higher at nominal Fisher p<.05:',sum(r['random_pct']>r['pain_pct'] and r['fisher_p']<.05 for r in family))
print('Random lower at Holm p<.05:',sum(r['random_pct']<r['pain_pct'] and r['fisher_holm_16']<.05 for r in family))
print('Label free:',json.dumps(label_free,indent=2))

# First-choice sign test as in the authors' analysis (pool A+B, scenario unit).
sign_rows=[]
matched_rows=[]
for model in models:
    for pair in PAIRS[:-1]:
        by_scenario=defaultdict(lambda:defaultdict(list))
        for arm in ARMS[:3]:
            for r in cells[model,pair,arm]:
                if r['sampled'] and r['first'] is not None:
                    by_scenario[r['scenario']]['random' if arm==ARMS[2] else 'pain'].append(r['first']=='relief')
        diffs=[sum(v['pain'])/len(v['pain'])-sum(v['random'])/len(v['random'])
               for v in by_scenario.values() if v['pain'] and v['random']]
        pos,neg=sum(d>0 for d in diffs),sum(d<0 for d in diffs)
        sign_rows.append(dict(model=model,pair=pair,mean_diff_points=100*sum(diffs)/len(diffs),
            pos=pos,neg=neg,ties=len(diffs)-pos-neg,p=sign_p(pos,neg)))
        # Same trial specification in both working arms, both first pressing at turn 0.
        # Still selected on outcomes, but equalizes first-press timing and scenario mix.
        maps={a:{r['key']:r for r in cells[model,pair,a] if r['sampled']} for a in ARMS[:3]}
        for comparison in [ARMS[1],ARMS[2]]:
            keys=[k for k,r in maps[ARMS[0]].items() if r['t0']==0 and maps[comparison][k]['t0']==0]
            pr=[maps[ARMS[0]][k] for k in keys]
            cr=[maps[comparison][k] for k in keys]
            matched_rows.append(dict(model=model,pair=pair,comparison=comparison,n=len(keys),
                pain_again=pct(sum(r['again'] for r in pr),len(pr)),
                comparison_again=pct(sum(r['again'] for r in cr),len(cr)),
                pain_next=pct(sum(r['next_choice']=='relief' for r in pr),sum(r['next_choice'] is not None for r in pr)),
                comparison_next=pct(sum(r['next_choice']=='relief' for r in cr),sum(r['next_choice'] is not None for r in cr))))
save('first_choice_sign_tests.csv',sign_rows)
save('matched_first_press.csv',matched_rows)
assert abs(fisher(1,10,11,14)-0.0027594561852200836)<1e-12
assert fisher(0,10,0,10)==1
assert sign_p(10,0)==2/1024
