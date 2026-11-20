#!/usr/bin/env python3
"""Turn the curated video catalogue into auditable research tasks."""
import hashlib, json
from pathlib import Path
p=Path('research_catalog/video_catalog.json'); data=json.loads(p.read_text(encoding='utf-8'))
out=[]
for collection in data['collections']:
  for topic in collection['topics']:
    task_id=hashlib.sha256((collection['url']+'|'+topic).encode()).hexdigest()[:12]
    out.append({'task_id':task_id,'topic':topic,'source_title':collection['title'],'source_url':collection['url'],
      'status':'HYPOTHESIS_PENDING','required_evidence':['causal_formula','purged_walk_forward','base_cost','stress_cost','ablation','shadow']})
Path('research_catalog/research_backlog.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'tasks':len(out),'output':'research_catalog/research_backlog.json'},ensure_ascii=False))
