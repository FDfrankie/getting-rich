from worldquant_brain.optimization_policy import *
from worldquant_brain.objective_aware_scheduler import *


def perf(**kwargs):
    data=dict(returns=.10,net_returns=.09,sharpe=1.3,fitness=1.1,turnover=.3,margin=.001,drawdown=.08,volatility=.12,previous_net_returns=.08,previous_turnover=.2)
    data.update(kwargs)
    return AlphaPerformance(**data)


def test_productive_turnover_is_preserved():
    m=perf(net_returns=.18,turnover=.60,previous_net_returns=.08,previous_turnover=.20)
    actions={a.action for a in recommend_actions(m,ObjectivePolicy(OptimizationObjective.RETURN))}
    assert 'PRESERVE_TURNOVER_AND_STRESS_COST' in actions
    assert 'APPLY_HUMP' not in actions


def test_wasteful_turnover_recommends_hump():
    m=perf(net_returns=.085,turnover=.60,previous_net_returns=.08,previous_turnover=.20)
    actions={a.action for a in recommend_actions(m,ObjectivePolicy(OptimizationObjective.BALANCED))}
    assert 'APPLY_HUMP' in actions and 'ADD_TRADE_WHEN' in actions


def test_low_turnover_stale_signal_recommends_lower_decay():
    m=perf(net_returns=.03,returns=.04,sharpe=.7,turnover=.04,previous_net_returns=.03,previous_turnover=.04)
    actions={a.action for a in recommend_actions(m,ObjectivePolicy(OptimizationObjective.RETURN))}
    assert 'LOWER_DECAY' in actions


def test_lane_budget_and_selection():
    candidates=[]
    for i,obj in enumerate((OptimizationObjective.RETURN,OptimizationObjective.SHARPE,OptimizationObjective.CAPACITY,OptimizationObjective.BALANCED)*3):
        candidates.append(LaneCandidate(str(i),obj,.7,.7,.7,.7,.7,.7,.7,.7,.1,.1,3))
    selected=select_candidates(candidates,total_slots=8)
    assert len(selected)==8
    assert sum(1 for x in selected if x.objective==OptimizationObjective.RETURN)>=2
