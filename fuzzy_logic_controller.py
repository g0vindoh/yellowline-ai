"""
YellowLine AI — Fuzzy Logic Controller v8.0
Expanded rule set: edge count, dwell time, buffer count, fall bonus, surge bonus.
Graceful fallback to linear scoring if skfuzzy is unavailable.
"""

import numpy as np

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    _SKFUZZY_OK = True
except ImportError:
    _SKFUZZY_OK = False
    print("[FLC] skfuzzy not available — using linear fallback scorer.")

# ─── Fallback linear scorer ───────────────────────────────────────────────────
def _linear_risk(in_edge, in_buffer, max_dwell, fall=False, surge=False):
    score = 0
    score += min(in_edge * 18, 45)
    score += min(in_buffer * 5, 15)
    score += min(max_dwell * 3, 20)
    if fall:   score += 30
    if surge:  score += 20
    return min(int(score), 100)


# ─── Build FIS ────────────────────────────────────────────────────────────────
def _build_fis():
    in_zone   = ctrl.Antecedent(np.arange(0, 11, 1),    'in_zone')
    dwell     = ctrl.Antecedent(np.arange(0, 61, 0.5),  'dwell')
    in_buf    = ctrl.Antecedent(np.arange(0, 16, 1),    'in_buf')
    risk      = ctrl.Consequent(np.arange(0, 101, 1),   'risk')

    # in_zone membership
    in_zone['none']     = fuzz.trimf(in_zone.universe, [0, 0, 1])
    in_zone['low']      = fuzz.trimf(in_zone.universe, [1, 2, 4])
    in_zone['moderate'] = fuzz.trimf(in_zone.universe, [3, 5, 7])
    in_zone['high']     = fuzz.trimf(in_zone.universe, [5, 8, 10])

    # dwell membership
    dwell['short']    = fuzz.trimf(dwell.universe, [0, 0, 5])
    dwell['medium']   = fuzz.trimf(dwell.universe, [3, 8, 15])
    dwell['long']     = fuzz.trimf(dwell.universe, [10, 20, 35])
    dwell['critical'] = fuzz.trimf(dwell.universe, [25, 45, 60])

    # buffer membership
    in_buf['low']  = fuzz.trimf(in_buf.universe, [0, 0, 4])
    in_buf['high'] = fuzz.trimf(in_buf.universe, [3, 8, 15])

    # risk output membership
    risk['safe']     = fuzz.trimf(risk.universe, [0, 0, 30])
    risk['caution']  = fuzz.trimf(risk.universe, [20, 40, 60])
    risk['danger']   = fuzz.trimf(risk.universe, [50, 70, 85])
    risk['critical'] = fuzz.trimf(risk.universe, [75, 90, 100])

    rules = [
        # safe scenarios
        ctrl.Rule(in_zone['none'] & dwell['short'],                 risk['safe']),
        ctrl.Rule(in_zone['none'] & in_buf['low'],                  risk['safe']),
        # caution
        ctrl.Rule(in_zone['low'] & dwell['short'],                  risk['caution']),
        ctrl.Rule(in_zone['none'] & in_buf['high'],                 risk['caution']),
        ctrl.Rule(in_zone['low'] & in_buf['low'],                   risk['caution']),
        # danger
        ctrl.Rule(in_zone['low'] & dwell['medium'],                 risk['danger']),
        ctrl.Rule(in_zone['moderate'] & dwell['short'],             risk['danger']),
        ctrl.Rule(in_zone['low'] & dwell['long'],                   risk['danger']),
        ctrl.Rule(in_zone['moderate'] & in_buf['high'],             risk['danger']),
        # critical
        ctrl.Rule(in_zone['moderate'] & dwell['medium'],            risk['critical']),
        ctrl.Rule(in_zone['high'],                                   risk['critical']),
        ctrl.Rule(in_zone['moderate'] & dwell['long'],              risk['critical']),
        ctrl.Rule(dwell['critical'],                                 risk['critical']),
        ctrl.Rule(in_zone['low'] & dwell['critical'],               risk['critical']),
        ctrl.Rule(in_zone['moderate'] & dwell['critical'],          risk['critical']),
    ]

    cs = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(cs)


_sim = None

def _get_sim():
    global _sim
    if _sim is None and _SKFUZZY_OK:
        try:
            _sim = _build_fis()
        except Exception as e:
            print(f"[FLC] FIS build failed: {e} — using linear fallback.")
    return _sim


# ─── Public API ───────────────────────────────────────────────────────────────
def get_risk(in_edge_cnt: int, in_buffer_cnt: int, max_dwell: float,
             fall: bool = False, surge: bool = False, edge_loss: bool = False) -> int:
    """Compute platform risk score 0–100. Returns int for backwards compat."""
    return get_risk_detail(in_edge_cnt, in_buffer_cnt, max_dwell, fall, surge, edge_loss)["score"]


def get_risk_detail(in_edge_cnt: int, in_buffer_cnt: int, max_dwell: float,
                    fall: bool = False, surge: bool = False, edge_loss: bool = False) -> dict:
    """
    Full breakdown of risk computation — used by dashboard AI panel.
    Returns:
      score       — final 0-100 risk score
      base        — FIS/linear score before bonuses
      bonus       — sum of event bonuses
      engine      — "fuzzy_fis" | "linear_fallback"
      inputs      — {in_zone, dwell, in_buf} clipped values fed to FIS
      bonuses     — {fall, surge, edge_loss} bonus amounts
      active_rule — human-readable description of dominant rule
    """
    sim = _get_sim()
    inputs = {
        "in_zone": float(np.clip(in_edge_cnt,   0, 10)),
        "dwell":   float(np.clip(max_dwell,     0, 60)),
        "in_buf":  float(np.clip(in_buffer_cnt, 0, 15)),
    }

    engine = "fuzzy_fis"
    if sim is not None:
        try:
            sim.input['in_zone'] = inputs["in_zone"]
            sim.input['dwell']   = inputs["dwell"]
            sim.input['in_buf']  = inputs["in_buf"]
            sim.compute()
            base = int(sim.output['risk'])
        except Exception:
            base   = _linear_risk(in_edge_cnt, in_buffer_cnt, max_dwell)
            engine = "linear_fallback"
    else:
        base   = _linear_risk(in_edge_cnt, in_buffer_cnt, max_dwell)
        engine = "linear_fallback"

    bonuses = {
        "fall":      30 if fall      else 0,
        "surge":     20 if surge     else 0,
        "edge_loss": 35 if edge_loss else 0,
    }
    bonus = sum(bonuses.values())
    score = min(base + bonus, 100)

    # Human-readable dominant rule
    if edge_loss:
        rule = "Edge loss detected — person disappeared at danger zone"
    elif fall:
        rule = "Fall confirmed — emergency scoring active"
    elif surge:
        rule = "Crowd surge — aligned velocity vectors detected"
    elif inputs["dwell"] >= 25:
        rule = "Critical dwell — person stationary at edge too long"
    elif inputs["in_zone"] >= 5:
        rule = "High occupancy — multiple persons in edge zone"
    elif inputs["in_zone"] >= 3 and inputs["dwell"] >= 8:
        rule = "Moderate occupancy + dwell — escalating risk"
    elif inputs["in_zone"] >= 1:
        rule = "Low edge occupancy — caution threshold"
    else:
        rule = "No edge occupancy — platform clear"

    return {
        "score":       score,
        "base":        base,
        "bonus":       bonus,
        "engine":      engine,
        "inputs":      inputs,
        "bonuses":     bonuses,
        "active_rule": rule,
    }
