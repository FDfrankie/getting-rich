import json, sqlite3
import numpy as np, pandas as pd
from taiwan_stock.universe_registry import parse_tpex_json, TaiwanUniverseRegistry, Security
from crypto_trading.indicator_library import add_indicators
from crypto_trading.brain_feature_bridge import BrainFeatureBridge

def test_tpex_parser_and_snapshot(tmp_path):
    rows=parse_tpex_json(json.dumps([{"SecuritiesCompanyCode":"6488","CompanyName":"環球晶"}]).encode())
    assert rows[0].symbol=='6488'
    m=TaiwanUniverseRegistry(tmp_path).write_snapshot(rows)
    assert m['rows']==1 and len(m['sha256'])==64

def test_indicators_are_backward_looking():
    n=100; df=pd.DataFrame({'Close':np.arange(n)+100.,'High':np.arange(n)+101.,'Low':np.arange(n)+99.,'Volume':1000})
    full=add_indicators(df); prefix=add_indicators(df.iloc[:80])
    pd.testing.assert_series_equal(full.loc[:79,'rsi_14'],prefix['rsi_14'])

def test_brain_bridge_applies_availability_lag(tmp_path):
    db=tmp_path/'a.db'
    with sqlite3.connect(db) as c:
      c.execute('CREATE TABLE alpha_experiments(completed_at TEXT,sharpe REAL,fitness REAL,turnover REAL,margin REAL,self_correlation REAL,simulation_status TEXT)')
      c.execute("INSERT INTO alpha_experiments VALUES('2026-01-01T12:00:00+00:00',1.5,1.2,.2,.01,.3,'BRAIN_COMPLETE')")
    market=pd.DataFrame({'timestamp':pd.to_datetime(['2026-01-01T18:00:00Z','2026-01-03T00:00:00Z'])})
    out=BrainFeatureBridge(db,lag='1D').merge_asof(market)
    assert pd.isna(out.loc[0,'brain_completed']) and out.loc[1,'brain_completed']==1
