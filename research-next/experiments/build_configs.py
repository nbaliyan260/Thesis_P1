"""Prospective deterministic mechanism-test matrix, no outcome-based selection."""
from pathlib import Path
import itertools, json, hashlib
from datetime import datetime, timezone
root = Path(__file__).resolve().parent
prompts = [
    'A science museum keeps a record of the weather. On Monday the wind came from the north and the temperature was mild. The guide explained that careful measurements help people compare changes over time. In summary,',
    'The store has three boxes of pencils and four notebooks. Each box contains twelve pencils. A student buys one box and two notebooks. The inventory after the purchase is',
    'Record A: city=Abu Dhabi; category=library; opening=09:00. Record B: city=Al Ain; category=museum; opening=10:00. The records should be described in clear English. Record A describes',
    'def total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result\n\nThis function adds a sequence of numbers. For the input [2, 3, 4], the result is',
]
cases=[]
for index,(p,l,w,k,f) in enumerate(itertools.product(range(4),(0.0,0.5,1.0),(4,12),('K','V'),('scalar','vector'))):
    cases.append(dict(id=f'primary-{index:03d}',prompt=prompts[p],prefix_tokens=64 if p%2==0 else 128,
        layer_fraction=l,window=w,kind=k,fault=f,seed=22000+index))
for p,l in itertools.product(range(4),(0.0,0.5,1.0)):
    index=len(cases)
    cases.append(dict(id=f'primary-{index:03d}',prompt=prompts[p],prefix_tokens=64 if p%2==0 else 128,
        layer_fraction=l,window=4 if p%2==0 else 12,kind='K',fault='noop',seed=22000+index))
primary=dict(protocol='RECUT-v0.1-frozen-primary',cases=cases,repetitions=3,
    overhead_repetitions=5,audit_max_bytes=52428800,description='Constructed mechanism workloads; not a task-quality or field-error benchmark.')
debug=dict(protocol='RECUT-v0.1-debug',repetitions=1,overhead_repetitions=2,audit_max_bytes=10485760,cases=[])
for i,(l,k,f,w) in enumerate([(0.,'V','scalar',1),(.5,'K','vector',4),(1.,'V','vector',4),(.5,'K','noop',4)]):
    debug['cases'].append(dict(id=f'debug-{i}',prompt='A blue bird sat beside a quiet river. The bird looked at the water and',prefix_tokens=32,layer_fraction=l,kind=k,fault=f,window=w,seed=11000+i))
records={}
for name,data in [('config_primary.json',primary),('config_debug.json',debug)]:
    path=root/name
    if path.exists(): raise RuntimeError('Do not overwrite an existing protocol')
    path.write_text(json.dumps(data,indent=2)+'\n')
    records[name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cases_per_model=len(data['cases']))
(root/'protocol_freeze.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
    files=records,notes='Frozen before pretrained-model experiments. Any revisions require a new filename and deviation note.'),indent=2)+'\n')
print(json.dumps(records,indent=2))
