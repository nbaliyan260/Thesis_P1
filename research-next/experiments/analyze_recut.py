"""Descriptive paired analysis; no claim of IID deployment failure rates."""
from pathlib import Path
import argparse, json, hashlib, statistics, collections, math

def median(values): return statistics.median(list(values))
def quantiles(values):
    values=sorted(values)
    def at(p):
        index=(len(values)-1)*p
        lo=int(index); hi=min(lo+1,len(values)-1)
        return values[lo]*(hi-index)+values[hi]*(index-lo) if hi!=lo else values[lo]
    return dict(min=values[0],q25=at(.25),median=at(.5),q75=at(.75),max=values[-1])

def summarize(rows):
    faults=[r for r in rows if r['case']['fault']!='noop']
    controls=[r for r in rows if r['case']['fault']=='noop']
    result=dict(n=len(rows),faults=len(faults),noops=len(controls),
        distinct_prompts=len({r['case']['prompt'] for r in rows}),
        token_divergences=sum(r['first_divergence'] is not None for r in faults),
        no_actual_change_faults=sum(r['fault']['changed_elements']==0 for r in faults),
        correctness={},performance={},layers={})
    for method in ('none','slotonly','full','sparse','all'):
        result['correctness'][method]=dict(
            all_equal_faults=sum(r['methods'][method]['all_equal'] for r in faults),
            cache_equal_faults=sum(r['methods'][method]['cache']['equal'] for r in faults),
            tokens_equal_faults=sum(r['methods'][method]['tokens_equal'] for r in faults),
            logits_equal_faults=sum(r['methods'][method]['logits']['equal'] for r in faults),
            all_equal_controls=sum(r['methods'][method]['all_equal'] for r in controls))
    paired=[]
    for r in faults:
        recovery={m:median(t['seconds'] for t in r['timings'] if t['method']==m) for m in ('full','sparse','all')}
        overhead={m:median(t['seconds'] for t in r['overhead_trials'] if t['mode']==m) for m in ('none','sparse','all')}
        paired.append(dict(case_id=r['case_id'],model=r['model'],layer=r['layer'],
            divergence=r['first_divergence'],recovery=recovery,normal=overhead,
            sparse_speedup=recovery['full']/recovery['sparse'],all_speedup=recovery['full']/recovery['all'],
            sparse_overhead=overhead['sparse']/overhead['none']-1,
            all_overhead=overhead['all']/overhead['none']-1))
    for m in ('sparse','all'):
        savings=[p['recovery']['full']-p['recovery'][m] for p in paired]
        costs=[p['normal'][m]-p['normal']['none'] for p in paired]
        avg_saved=statistics.mean(savings); avg_cost=statistics.mean(costs)
        result['performance'][m]=dict(
            paired_speedup=quantiles([p[m+'_speedup'] for p in paired]),
            paired_normal_overhead_fraction=quantiles([p[m+'_overhead'] for p in paired]),
            recovery_ms=quantiles([p['recovery'][m]*1000 for p in paired]),
            mean_recovery_saving_ms=1000*avg_saved,mean_normal_cost_ms=1000*avg_cost,
            diagnostic_break_even_probability=avg_cost/avg_saved if avg_saved>0 and avg_cost>=0 else None,
            journal_bytes=quantiles([r[m+'_journal_payload_bytes'] for r in faults]),
            journal_over_checkpoint_fraction=quantiles([r[m+'_journal_payload_bytes']/r['checkpoint_payload_bytes'] for r in faults]),
            layer_step_fraction=quantiles([r['methods'][m]['layer_steps']/(r['num_layers']*r['case']['window']) for r in faults]))
    result['full_recovery_ms']=quantiles([p['recovery']['full']*1000 for p in paired])
    result['normal_nojournal_ms']=quantiles([p['normal']['none']*1000 for p in paired])
    result['checkpoint_bytes']=quantiles([r['checkpoint_payload_bytes'] for r in faults])
    result['checkpoint_copy_ms']=quantiles([median(r['checkpoint_copy_seconds'])*1000 for r in faults])
    for layer in sorted({p['layer'] for p in paired}):
        subset=[p for p in paired if p['layer']==layer]
        result['layers'][str(layer)]=dict(n=len(subset),divergences=sum(p['divergence'] is not None for p in subset),
            sparse_speedup=quantiles([p['sparse_speedup'] for p in subset]),
            all_speedup=quantiles([p['all_speedup'] for p in subset]))
    return result, paired

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('run',type=Path); args=parser.parse_args()
    run=args.run
    metadata=json.loads((run/'metadata.json').read_text())
    assert metadata['status']=='complete',metadata['status']
    raw=(run/'episodes.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==metadata['episodes_sha256']
    rows=[json.loads(line) for line in raw.splitlines()]
    assert len({(r['model'],r['case_id']) for r in rows})==len(rows)
    assert all(r['valid_repairs_equal'] for r in rows)
    models={}; paired=[]
    for model in sorted({r['model'] for r in rows}):
        models[model],data=summarize([r for r in rows if r['model']==model]); paired.extend(data)
    total,_=summarize(rows)
    output=dict(status='complete_bounded_pilot',episode_sha256=metadata['episodes_sha256'],
        total=total,models=models,
        caveats=['Descriptive results for constructed prompts and injected faults, not field probabilities.',
                 'Repeated prompts and multiple policies/timings are not independent new trials.',
                 'Break-even is a diagnostic mean cost/saving ratio for this artificial case mix.',
                 'Oracle detection/localization and experimental full-reference checks are not deployed mechanisms.',
                 'Peak allocation includes correctness fixtures; payload bytes omit metadata and allocator overhead.'])
    (run/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    (run/'paired_analysis.json').write_text(json.dumps(paired,indent=2)+'\n')
    print(json.dumps(output,indent=2))
if __name__=='__main__': main()
