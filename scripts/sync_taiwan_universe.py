#!/usr/bin/env python3
import argparse, json
from taiwan_stock.universe_registry import TaiwanUniverseRegistry
p=argparse.ArgumentParser(); p.add_argument('--include-emerging',action='store_true'); p.add_argument('--out',default='taiwan_stock/universe_snapshots')
a=p.parse_args(); reg=TaiwanUniverseRegistry(a.out); records=reg.fetch_current(a.include_emerging); print(json.dumps(reg.write_snapshot(records),ensure_ascii=False,indent=2))
