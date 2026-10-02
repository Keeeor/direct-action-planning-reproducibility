import numpy as np

from dap.agents.rules import AggressiveRule, ConservativeRule, NoIntervention


def observation(load=6.0, queue=0.0, growth=0.0, risk=0.0):
    obs = np.zeros(14, dtype=np.float32)
    obs[0] = load
    obs[2] = queue
    obs[3] = growth
    obs[8] = 1.0 + risk * 4.0
    obs[9] = float(risk > 0.5)
    return obs


def test_no_intervention_always_uses_minimum_action() -> None:
    policy = NoIntervention()
    assert {policy.act(observation(queue=q)) for q in (0, 10, 100)} == {0}


def test_aggressive_rule_acts_no_later_than_conservative_rule() -> None:
    conservative, aggressive = ConservativeRule(), AggressiveRule()
    states = [
        observation(load=4, queue=0),
        observation(load=8, queue=5, growth=3, risk=0.4),
        observation(load=12, queue=20, growth=8, risk=0.9),
    ]
    for state in states:
        assert aggressive.act(state) >= conservative.act(state)


def test_rules_escalate_with_risk() -> None:
    for policy in (ConservativeRule(), AggressiveRule()):
        actions = [policy.act(observation(load=6 + i * 3, queue=i * 8, growth=i * 2, risk=i / 3)) for i in range(4)]
        assert actions == sorted(actions)
