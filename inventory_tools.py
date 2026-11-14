"""
inventory_tools.py

Deterministic Python calculation and verification tools for the
Supply Chain Analytics Inventory Decision Agent.

Implemented Tools:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

Every calculation tool returns a comprehensive structured dictionary containing:
1. inputs_extracted
2. variable_definitions
3. formula
4. formula_with_substituted_values
5. intermediate_calculations
6. final_answer
7. verification_checks
8. assumptions

Design Principles:
- Validate all inputs strictly
- Never invent missing numerical inputs
- All numerical steps come strictly from deterministic Python code
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional

try:
    from scipy.stats import norm as _scipy_norm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_STD_NORM = NormalDist(mu=0.0, sigma=1.0)


def _require_non_none(params: Dict[str, Any]) -> None:
    """Ensure required parameters are explicitly provided (never invent missing numerical inputs)."""
    missing = [name for name, val in params.items() if val is None]
    if missing:
        raise ValueError(
            f"Missing required numerical input(s): {', '.join(missing)}. "
            "Tools never invent missing numerical inputs."
        )


def _validate_numeric(name: str, value: Any) -> float:
    """Validate that a value is a real number (not bool, None, or NaN/Inf)."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Parameter '{name}' must be a valid number, got {value!r}.")
    fval = float(value)
    if math.isnan(fval) or math.isinf(fval):
        raise ValueError(f"Parameter '{name}' must be a finite number, got {value!r}.")
    return fval


def norm_ppf(p: float) -> float:
    """Standard normal inverse CDF (z-score from probability)."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"Probability/service level must be strictly between 0 and 1, got {p}.")
    if _HAS_SCIPY:
        return float(_scipy_norm.ppf(p))
    return float(_STD_NORM.inv_cdf(p))


def norm_pdf(z: float) -> float:
    """Standard normal probability density function."""
    if _HAS_SCIPY:
        return float(_scipy_norm.pdf(z))
    return float(_STD_NORM.pdf(z))


def norm_cdf(z: float) -> float:
    """Standard normal cumulative distribution function."""
    if _HAS_SCIPY:
        return float(_scipy_norm.cdf(z))
    return float(_STD_NORM.cdf(z))


def _normalize_service_level(service_level: float) -> float:
    """Normalize service level provided either as a percentage (e.g., 95) or decimal (0.95)."""
    sl = _validate_numeric("service_level", service_level)
    if 1.0 < sl < 100.0:
        return sl / 100.0
    if not (0.0 < sl < 1.0):
        raise ValueError(f"service_level must be between 0 and 1 (or 0 and 100%), got {service_level}.")
    return sl


# =====================================================================
# 1. Economic Order Quantity (EOQ)
# =====================================================================
def calculate_eoq(
    annual_demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    unit_cost: Optional[float] = None,
    holding_rate: Optional[float] = None,
    working_days: Optional[float] = None,
    stated_demand: Optional[float] = None,
    demand_period: Optional[str] = None,
    demand_conversion_factor: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Calculate Economic Order Quantity: Q* = sqrt(2 * D * S / H).

    Supports demand time unit normalization:
    - If demand was stated non-annually (e.g. monthly, weekly, daily), it is normalized to an annual basis.

    Returns 8 verification elements:
    1. inputs_extracted
    2. variable_definitions
    3. formula
    4. formula_with_substituted_values
    5. intermediate_calculations
    6. final_answer
    7. verification_checks
    8. assumptions
    """
    _require_non_none({"annual_demand": annual_demand, "ordering_cost": ordering_cost})
    d_val = _validate_numeric("annual_demand", annual_demand)
    s_val = _validate_numeric("ordering_cost", ordering_cost)

    if d_val <= 0:
        raise ValueError("annual_demand (D) must be strictly greater than 0.")
    if s_val <= 0:
        raise ValueError("ordering_cost (S) must be strictly greater than 0.")

    c_val: Optional[float] = None
    if unit_cost is not None:
        c_val = _validate_numeric("unit_cost", unit_cost)
        if c_val <= 0:
            raise ValueError("unit_cost (C) must be strictly greater than 0.")

    rate_val: Optional[float] = None
    if holding_rate is not None:
        raw_rate = _validate_numeric("holding_rate", holding_rate)
        rate_val = raw_rate / 100.0 if raw_rate > 1.0 else raw_rate
        if rate_val <= 0:
            raise ValueError("holding_rate (I) must be strictly greater than 0.")

    if holding_cost is not None:
        h_val = _validate_numeric("holding_cost", holding_cost)
        holding_cost_source = "Fixed holding cost per unit per year provided directly."
    elif c_val is not None and rate_val is not None:
        h_val = c_val * rate_val
        holding_cost_source = f"Calculated as holding_rate ({rate_val:.4f}) * unit_cost (${c_val:.2f}) = ${h_val:.4f}/unit/year."
    else:
        raise ValueError(
            "Missing required holding cost input: provide either holding_cost (H) "
            "or both unit_cost (C) and holding_rate (I)."
        )

    if h_val <= 0:
        raise ValueError("holding_cost (H) must be strictly greater than 0.")

    w_days: Optional[float] = None
    if working_days is not None:
        w_days = _validate_numeric("working_days", working_days)
        if w_days <= 0:
            raise ValueError("working_days must be strictly greater than 0.")

    # Core EOQ numerical calculations (Python deterministic)
    numerator = 2.0 * d_val * s_val
    eoq = math.sqrt(numerator / h_val)
    average_inventory = eoq / 2.0
    orders_per_year = d_val / eoq
    annual_ordering_cost = orders_per_year * s_val
    annual_holding_cost = average_inventory * h_val
    total_relevant_cost = annual_ordering_cost + annual_holding_cost

    annual_purchase_cost = (d_val * c_val) if c_val is not None else None
    total_annual_cost = (
        total_relevant_cost + annual_purchase_cost
        if annual_purchase_cost is not None
        else total_relevant_cost
    )
    cycle_time_days = (w_days / orders_per_year) if w_days is not None else None

    # Determine if demand was normalized from a non-annual period
    is_demand_normalized = (
        stated_demand is not None
        and demand_period is not None
        and demand_period.lower().strip() not in ("year", "annual", "annually", "annum")
    )
    mult = demand_conversion_factor if demand_conversion_factor else (d_val / stated_demand if stated_demand and stated_demand > 0 else 1.0)

    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any] = {
        "annual_demand": d_val,
        "ordering_cost": s_val,
        "holding_cost": round(h_val, 4),
        "unit_cost": c_val,
        "holding_rate": rate_val,
        "working_days": w_days,
    }
    if is_demand_normalized:
        inputs_extracted["stated_demand"] = stated_demand
        inputs_extracted["demand_period"] = demand_period
        inputs_extracted["demand_conversion_factor"] = mult

    # 2. Variable Definitions
    d_def = (
        f"Annual demand (units/year) [Normalized from {stated_demand:,.2f} units/{demand_period} to {d_val:,.2f} units/year]"
        if is_demand_normalized
        else "Annual demand (units/year)"
    )
    variable_definitions = {
        "D": d_def,
        "S": "Ordering / setup cost per order ($/order)",
        "H": "Holding (carrying) cost per unit per year ($/unit/year)",
        "C": "Unit purchase price / cost ($/unit)",
        "I": "Annual holding cost rate (% of unit purchase cost)",
        "Q*": "Economic Order Quantity (optimal order quantity in units)",
        "N": "Order frequency / orders per year: D / Q* (orders/year)",
        "T_cycle": "Cycle time between orders: working_days / N (days)",
        "TOC": "Annual ordering cost: (D / Q*) * S ($/year)",
        "THC": "Annual holding cost: (Q* / 2) * H ($/year)",
        "TRC": "Total relevant inventory cost: TOC + THC ($/year)",
        "TC": "Total annual cost: TRC + (D * C) ($/year)",
    }

    # 3. Formula
    formula: Dict[str, str] = {
        "eoq": "Q* = sqrt((2 * D * S) / H)",
        "orders_per_year": "N = D / Q*",
        "cycle_time_days": "T_cycle = working_days / (D / Q*)",
        "annual_ordering_cost": "TOC = (D / Q*) * S",
        "annual_holding_cost": "THC = (Q* / 2) * H",
        "total_relevant_cost": "TRC = (D / Q*) * S + (Q* / 2) * H",
        "total_annual_cost": "TC = TRC + (D * C)",
    }
    if is_demand_normalized:
        formula["demand_normalization"] = f"D = stated_demand * conversion_factor ({demand_period}s/year)"

    # 4. Formula with Substituted Values
    formula_with_substituted_values: Dict[str, str] = {
        "eoq": f"Q* = sqrt((2 * {d_val:,.2f} * {s_val:,.2f}) / {h_val:,.4f}) = sqrt({numerator:,.2f} / {h_val:,.4f}) = {eoq:,.4f} units",
        "orders_per_year": f"N = {d_val:,.2f} / {eoq:,.4f} = {orders_per_year:,.4f} orders/year",
        "cycle_time_days": f"T_cycle = {w_days:,.0f} / {orders_per_year:,.4f} = {cycle_time_days:,.2f} days" if w_days else "T_cycle = working_days / N (working days not specified)",
        "annual_ordering_cost": f"TOC = ({d_val:,.2f} / {eoq:,.4f}) * ${s_val:,.2f} = {orders_per_year:,.4f} * ${s_val:,.2f} = ${annual_ordering_cost:,.2f}",
        "annual_holding_cost": f"THC = ({eoq:,.4f} / 2) * ${h_val:,.4f} = {average_inventory:,.4f} * ${h_val:,.4f} = ${annual_holding_cost:,.2f}",
        "total_relevant_cost": f"TRC = ${annual_ordering_cost:,.2f} + ${annual_holding_cost:,.2f} = ${total_relevant_cost:,.2f}",
        "total_annual_cost": f"TC = ${total_relevant_cost:,.2f} + ({d_val:,.2f} * ${c_val:,.2f}) = ${total_relevant_cost:,.2f} + ${annual_purchase_cost:,.2f} = ${total_annual_cost:,.2f}" if c_val else f"TC = TRC = ${total_relevant_cost:,.2f} (unit purchase cost not specified)",
    }
    if is_demand_normalized:
        formula_with_substituted_values["demand_normalization"] = (
            f"D = {stated_demand:,.2f} units/{demand_period} * {mult:g} {demand_period}s/year = {d_val:,.2f} units/year"
        )

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any] = {
        "two_D_S": round(numerator, 4),
        "holding_cost_per_unit": round(h_val, 4),
        "holding_cost_derivation": holding_cost_source,
        "eoq_unrounded": round(eoq, 6),
        "average_inventory": round(average_inventory, 4),
        "orders_per_year": round(orders_per_year, 4),
        "cycle_time_days": round(cycle_time_days, 4) if cycle_time_days is not None else None,
        "annual_ordering_cost": round(annual_ordering_cost, 4),
        "annual_holding_cost": round(annual_holding_cost, 4),
        "total_relevant_cost": round(total_relevant_cost, 4),
        "annual_purchase_cost": round(annual_purchase_cost, 4) if annual_purchase_cost is not None else None,
        "total_annual_cost": round(total_annual_cost, 4),
    }
    if is_demand_normalized:
        intermediate_calculations["stated_demand"] = stated_demand
        intermediate_calculations["demand_period"] = demand_period
        intermediate_calculations["demand_conversion_factor"] = mult
        intermediate_calculations["demand_normalization_note"] = (
            f"Demand was stated as {stated_demand:,.2f} units per {demand_period} and normalized to "
            f"{d_val:,.2f} units per year ({stated_demand:,.2f} * {mult:g} {demand_period}s/year) "
            "to ensure dimensional consistency with annual holding cost (H)."
        )

    # 6. Final Answer
    final_answer = {
        "eoq": round(eoq, 4),
        "eoq_rounded": round(eoq),
        "orders_per_year": round(orders_per_year, 4),
        "cycle_time_days": round(cycle_time_days, 4) if cycle_time_days is not None else None,
        "total_relevant_cost": round(total_relevant_cost, 4),
        "total_annual_cost": round(total_annual_cost, 4),
        "recommendation": f"Order {eoq:,.2f} units (approx. {round(eoq):,} whole units) per order. This results in {orders_per_year:,.2f} orders/year and minimizes total relevant inventory cost at ${total_relevant_cost:,.2f}/year.",
    }

    # 8. Assumptions
    assumptions = [
        "Demand rate (D) is constant, deterministic, and known over the entire planning horizon.",
        "Ordering / setup cost (S) is fixed per order regardless of the quantity ordered.",
        "Holding cost (H) is linear, constant per unit per year, and applies to average cycle inventory (Q/2).",
        "Replenishment lead time is zero (or constant), and the entire order arrives simultaneously in a single batch.",
        "Stockouts, backorders, and shortages are not permitted.",
        "No quantity discounts apply (unit purchase cost C is constant).",
    ]
    if is_demand_normalized:
        assumptions.insert(
            1,
            f"Demand was stated on a {demand_period}ly basis ({stated_demand:,.2f} units/{demand_period}) "
            f"and converted to {d_val:,.2f} units/year ({stated_demand:,.2f} × {mult:g}) to match the annual holding cost period.",
        )

    results: Dict[str, Any] = {
        "model": "eoq",
        "inputs_extracted": inputs_extracted,
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "assumptions": assumptions,
        # Top-level convenience fields for backwards compatibility
        "eoq": round(eoq, 4),
        "eoq_rounded": round(eoq),
        "average_inventory": round(average_inventory, 4),
        "orders_per_year": round(orders_per_year, 4),
        "cycle_time_days": round(cycle_time_days, 4) if cycle_time_days is not None else None,
        "annual_ordering_cost": round(annual_ordering_cost, 4),
        "annual_holding_cost": round(annual_holding_cost, 4),
        "total_relevant_cost": round(total_relevant_cost, 4),
        "annual_purchase_cost": round(annual_purchase_cost, 4) if annual_purchase_cost is not None else None,
        "total_annual_cost": round(total_annual_cost, 4),
    }

    # 7. Verification Checks
    verification = verify_calculation("eoq", results["inputs_extracted"], results)
    results["verification_checks"] = verification
    results["verification"] = verification
    return results


# =====================================================================
# 2. Quantity Discount Model
# =====================================================================
def calculate_quantity_discount(
    annual_demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    tiers: Optional[List[Dict[str, Any]]] = None,
    holding_rate: Optional[float] = None,
    holding_cost: Optional[float] = None,
    stated_demand: Optional[float] = None,
    demand_period: Optional[str] = None,
    demand_conversion_factor: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Evaluate feasible order quantities across price-break tiers and compare total annual cost.
    Supports both fixed holding cost (holding_cost) and percentage holding cost (holding_rate).

    Returns 8 verification elements:
    1. inputs_extracted
    2. variable_definitions
    3. formula
    4. formula_with_substituted_values
    5. intermediate_calculations
    6. final_answer
    7. verification_checks
    8. assumptions
    """
    _require_non_none({
        "annual_demand": annual_demand,
        "ordering_cost": ordering_cost,
        "tiers": tiers,
    })
    d_val = _validate_numeric("annual_demand", annual_demand)
    s_val = _validate_numeric("ordering_cost", ordering_cost)

    if d_val <= 0:
        raise ValueError("annual_demand (D) must be strictly greater than 0.")
    if s_val <= 0:
        raise ValueError("ordering_cost (S) must be strictly greater than 0.")
    if not isinstance(tiers, list) or len(tiers) == 0:
        raise ValueError("tiers must be a non-empty list of price break dictionaries.")

    norm_holding_rate: Optional[float] = None
    if holding_rate is not None:
        raw_rate = _validate_numeric("holding_rate", holding_rate)
        norm_holding_rate = raw_rate / 100.0 if raw_rate > 1.0 else raw_rate
        if norm_holding_rate <= 0:
            raise ValueError("holding_rate must be strictly greater than 0.")

    fixed_h: Optional[float] = None
    if holding_cost is not None:
        fixed_h = _validate_numeric("holding_cost", holding_cost)
        if fixed_h <= 0:
            raise ValueError("holding_cost must be strictly greater than 0.")

    validated_tiers: List[Dict[str, Any]] = []
    for idx, t in enumerate(tiers, start=1):
        if not isinstance(t, dict):
            raise ValueError(f"Tier {idx} must be a dictionary.")
        if "min_qty" not in t or t["min_qty"] is None:
            raise ValueError(f"Tier {idx} is missing required 'min_qty'.")
        if "unit_cost" not in t or t["unit_cost"] is None:
            raise ValueError(f"Tier {idx} is missing required 'unit_cost'.")

        min_q = _validate_numeric(f"tier[{idx}].min_qty", t["min_qty"])
        if min_q <= 0:
            raise ValueError(f"Tier {idx} min_qty must be > 0, got {min_q}.")
        u_cost = _validate_numeric(f"tier[{idx}].unit_cost", t["unit_cost"])
        if u_cost <= 0:
            raise ValueError(f"Tier {idx} unit_cost must be > 0, got {u_cost}.")

        raw_max = t.get("max_qty", None)
        max_q = _validate_numeric(f"tier[{idx}].max_qty", raw_max) if raw_max is not None else float("inf")
        if max_q < min_q:
            raise ValueError(f"Tier {idx} max_qty ({max_q}) cannot be less than min_qty ({min_q}).")

        tier_h = _validate_numeric(f"tier[{idx}].holding_cost", t["holding_cost"]) if t.get("holding_cost") is not None else None
        validated_tiers.append({
            "min_qty": min_q,
            "max_qty": max_q,
            "unit_cost": u_cost,
            "holding_cost": tier_h,
        })

    sorted_tiers = sorted(validated_tiers, key=lambda x: x["min_qty"])
    tier_evaluations: List[Dict[str, Any]] = []

    for idx, tier in enumerate(sorted_tiers, start=1):
        min_qty = tier["min_qty"]
        max_qty = tier["max_qty"]
        unit_cost = tier["unit_cost"]

        if tier["holding_cost"] is not None:
            h_i = tier["holding_cost"]
            holding_mode = "tier_specific_fixed"
        elif norm_holding_rate is not None:
            h_i = norm_holding_rate * unit_cost
            holding_mode = "percentage_of_unit_cost"
        elif fixed_h is not None:
            h_i = fixed_h
            holding_mode = "fixed_constant"
        else:
            raise ValueError(
                "Missing holding cost input: provide holding_rate, holding_cost, or tier-specific holding_cost."
            )

        if h_i <= 0:
            raise ValueError(f"Tier {idx} holding cost must be > 0.")

        unconstrained_eoq = math.sqrt((2.0 * d_val * s_val) / h_i)

        if unconstrained_eoq < min_qty:
            evaluated_q = min_qty
            feasible = True
            adjustment_reason = "Unconstrained EOQ < min_qty; adjusted up to tier minimum quantity."
        elif unconstrained_eoq > max_qty:
            evaluated_q = max_qty
            feasible = False
            adjustment_reason = "Unconstrained EOQ > max_qty; infeasible in this tier (dominated by a higher-quantity tier)."
        else:
            evaluated_q = unconstrained_eoq
            feasible = True
            adjustment_reason = "Unconstrained EOQ lies within the tier range [min_qty, max_qty]."

        ann_ordering = (d_val / evaluated_q) * s_val
        ann_holding = (evaluated_q / 2.0) * h_i
        ann_purchase = d_val * unit_cost
        total_cost = ann_ordering + ann_holding + ann_purchase

        tier_evaluations.append({
            "tier_index": idx,
            "min_qty": min_qty,
            "max_qty": None if math.isinf(max_qty) else max_qty,
            "unit_cost": round(unit_cost, 4),
            "holding_cost": round(h_i, 4),
            "holding_cost_mode": holding_mode,
            "unconstrained_eoq": round(unconstrained_eoq, 4),
            "evaluated_order_qty": round(evaluated_q, 4),
            "feasible": feasible,
            "adjustment_reason": adjustment_reason,
            "annual_ordering_cost": round(ann_ordering, 4),
            "annual_holding_cost": round(ann_holding, 4),
            "annual_purchase_cost": round(ann_purchase, 4),
            "total_annual_cost": round(total_cost, 4),
        })

    feasible_tiers = [t for t in tier_evaluations if t["feasible"]]
    candidates = feasible_tiers if feasible_tiers else tier_evaluations
    best_tier = min(candidates, key=lambda t: t["total_annual_cost"])

    # 1. Inputs Extracted
    inputs_extracted = {
        "annual_demand": d_val,
        "ordering_cost": s_val,
        "holding_rate": norm_holding_rate,
        "holding_cost": fixed_h,
        "tiers": [
            {
                "min_qty": t["min_qty"],
                "max_qty": None if math.isinf(t["max_qty"]) else t["max_qty"],
                "unit_cost": t["unit_cost"],
                "holding_cost": t["holding_cost"],
            }
            for t in sorted_tiers
        ],
    }

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual demand (units/year)",
        "S": "Ordering cost per order ($/order)",
        "i": "Price break tier index (1, 2, ...)",
        "C_i": "Unit purchase price in tier i ($/unit)",
        "H_i": "Unit holding cost per year in tier i ($/unit/year)",
        "I": "Annual holding cost rate (% of unit purchase cost)",
        "min_qty_i": "Minimum order quantity required to qualify for tier i price (units)",
        "max_qty_i": "Maximum order quantity for tier i price (units)",
        "Q*_i": "Unconstrained EOQ in tier i: sqrt((2 * D * S) / H_i)",
        "Q_i": "Feasible order quantity evaluated for tier i (units)",
        "TOC_i": "Annual ordering cost in tier i: (D / Q_i) * S ($/year)",
        "THC_i": "Annual holding cost in tier i: (Q_i / 2) * H_i ($/year)",
        "TPC_i": "Annual purchase cost in tier i: D * C_i ($/year)",
        "TC_i": "Total annual cost in tier i: TOC_i + THC_i + TPC_i ($/year)",
    }

    # 3. Formula
    formula = {
        "unconstrained_eoq": "Q*_i = sqrt((2 * D * S) / H_i)",
        "feasibility_rule": "If Q*_i < min_qty_i -> Q_i = min_qty_i; if min_qty_i <= Q*_i <= max_qty_i -> Q_i = Q*_i; if Q*_i > max_qty_i -> infeasible in tier i",
        "annual_ordering_cost": "TOC_i = (D / Q_i) * S",
        "annual_holding_cost": "THC_i = (Q_i / 2) * H_i",
        "annual_purchase_cost": "TPC_i = D * C_i",
        "total_annual_cost": "TC_i = TOC_i + THC_i + TPC_i",
        "optimal_decision": "Optimal Tier = argmin_{feasible i} (TC_i)",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "optimal_tier": f"Tier {best_tier['tier_index']}",
        "optimal_eoq_substitution": f"Q*_{best_tier['tier_index']} = sqrt((2 * {d_val:,.2f} * {s_val:,.2f}) / {best_tier['holding_cost']:,.4f}) = {best_tier['unconstrained_eoq']:,.2f} units",
        "optimal_quantity_adjusted": f"Q_{best_tier['tier_index']} = {best_tier['evaluated_order_qty']:,.2f} units ({best_tier['adjustment_reason']})",
        "ordering_cost_substitution": f"TOC_{best_tier['tier_index']} = ({d_val:,.2f} / {best_tier['evaluated_order_qty']:,.2f}) * ${s_val:,.2f} = ${best_tier['annual_ordering_cost']:,.2f}",
        "holding_cost_substitution": f"THC_{best_tier['tier_index']} = ({best_tier['evaluated_order_qty']:,.2f} / 2) * ${best_tier['holding_cost']:,.4f} = ${best_tier['annual_holding_cost']:,.2f}",
        "purchase_cost_substitution": f"TPC_{best_tier['tier_index']} = {d_val:,.2f} * ${best_tier['unit_cost']:,.2f} = ${best_tier['annual_purchase_cost']:,.2f}",
        "total_cost_substitution": f"TC_{best_tier['tier_index']} = ${best_tier['annual_ordering_cost']:,.2f} + ${best_tier['annual_holding_cost']:,.2f} + ${best_tier['annual_purchase_cost']:,.2f} = ${best_tier['total_annual_cost']:,.2f}",
    }

    # 5. Intermediate Calculations
    intermediate_calculations = {
        "holding_cost_type": "percentage" if norm_holding_rate is not None else "fixed",
        "tier_evaluations": tier_evaluations,
        "feasible_tier_indices": [t["tier_index"] for t in feasible_tiers],
    }

    # 6. Final Answer
    final_answer = {
        "optimal_tier": best_tier["tier_index"],
        "optimal_order_quantity": best_tier["evaluated_order_qty"],
        "optimal_order_quantity_rounded": round(best_tier["evaluated_order_qty"]),
        "optimal_unit_cost": best_tier["unit_cost"],
        "min_total_annual_cost": best_tier["total_annual_cost"],
        "recommendation": f"Order {best_tier['evaluated_order_qty']:,.2f} units to qualify for Tier {best_tier['tier_index']} pricing at ${best_tier['unit_cost']:,.2f}/unit. This minimizes total annual cost (ordering + holding + purchase) at ${best_tier['total_annual_cost']:,.2f}/year.",
    }

    # 8. Assumptions
    assumptions = [
        "All-units quantity discount schedule: the discounted unit purchase price applies to all units in the order.",
        "Annual demand (D) is deterministic and constant throughout the year.",
        "Ordering cost (S) is fixed per order regardless of the order quantity.",
        "Holding cost is linear, applied to average inventory (Q/2), either fixed per unit or proportional to tier purchase price (H_i = I * C_i).",
        "Replenishment is instantaneous in a single batch, and shortages/backorders are not permitted.",
    ]

    results = {
        "model": "quantity_discount",
        "inputs_extracted": inputs_extracted,
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "assumptions": assumptions,
        # Top-level convenience fields
        "optimal_tier": best_tier["tier_index"],
        "optimal_order_quantity": best_tier["evaluated_order_qty"],
        "optimal_unit_cost": best_tier["unit_cost"],
        "optimal_holding_cost": best_tier["holding_cost"],
        "annual_ordering_cost": best_tier["annual_ordering_cost"],
        "annual_holding_cost": best_tier["annual_holding_cost"],
        "annual_purchase_cost": best_tier["annual_purchase_cost"],
        "min_total_annual_cost": best_tier["total_annual_cost"],
        "tier_evaluations": tier_evaluations,
    }

    # 7. Verification Checks
    verification = verify_calculation("quantity_discount", results["inputs_extracted"], results)
    results["verification_checks"] = verification
    results["verification"] = verification
    return results


# =====================================================================
# 3. Safety Stock and Reorder Point (ROP)
# =====================================================================
def calculate_rop(
    avg_demand: Optional[float] = None,
    lead_time: Optional[float] = None,
    std_demand: Optional[float] = None,
    std_lead_time: Optional[float] = None,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    safety_stock: Optional[float] = None,
    stated_lead_time: Optional[float] = None,
    lead_time_unit: Optional[str] = None,
    stated_std_lead_time: Optional[float] = None,
    demand_unit: Optional[str] = None,
    days_per_week: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Calculate Safety Stock (SS) and Reorder Point (ROP).

    Supports all 4 demand/lead-time variability cases:
    1. Constant demand (std_demand == 0) and constant lead time (std_lead_time == 0)
    2. Variable demand (std_demand > 0) and constant lead time (std_lead_time == 0)
    3. Constant demand (std_demand == 0) and variable lead time (std_lead_time > 0)
    4. Variable demand (std_demand > 0) and variable lead time (std_lead_time > 0)

    General formulas:
        sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
        SS = z * sigma_LT  (or directly specified)
        ROP = d * L + SS

    Returns 8 verification elements:
    1. inputs_extracted
    2. variable_definitions
    3. formula
    4. formula_with_substituted_values
    5. intermediate_calculations
    6. final_answer
    7. verification_checks
    8. assumptions
    """
    _require_non_none({"avg_demand": avg_demand, "lead_time": lead_time})
    if std_demand is None and std_lead_time is None and safety_stock is None:
        raise ValueError(
            "Missing variability inputs: provide std_demand (sigma_d) and/or std_lead_time (sigma_L), "
            "or provide safety_stock directly. Pass 0 for constant demand and/or constant lead time."
        )

    # Unit conversion / normalization metadata handling
    lead_time_converted = False
    op_days = days_per_week if days_per_week is not None else 7.0
    raw_lead_time = stated_lead_time if stated_lead_time is not None else lead_time
    raw_std_lead_time = stated_std_lead_time if stated_std_lead_time is not None else std_lead_time

    lt_unit_norm = (lead_time_unit or "").strip().lower()
    d_unit_norm = (demand_unit or "").strip().lower()
    if lt_unit_norm in ("week", "weeks") and (d_unit_norm in ("day", "days") or not d_unit_norm):
        lead_time_converted = True
        if lead_time is not None and (stated_lead_time is None or lead_time == stated_lead_time):
            lead_time = lead_time * op_days
        if std_lead_time is not None and (stated_std_lead_time is None or std_lead_time == stated_std_lead_time):
            std_lead_time = std_lead_time * op_days

    d_val = _validate_numeric("avg_demand", avg_demand)
    l_val = _validate_numeric("lead_time", lead_time)
    sigma_d = _validate_numeric("std_demand", std_demand) if std_demand is not None else 0.0
    sigma_l = _validate_numeric("std_lead_time", std_lead_time) if std_lead_time is not None else 0.0

    if d_val < 0:
        raise ValueError("avg_demand (d) must be non-negative.")
    if l_val <= 0:
        raise ValueError("lead_time (L) must be strictly greater than 0.")
    if sigma_d < 0:
        raise ValueError("std_demand (sigma_d) must be non-negative.")
    if sigma_l < 0:
        raise ValueError("std_lead_time (sigma_L) must be non-negative.")

    if sigma_d == 0.0 and sigma_l == 0.0:
        variability_case = "constant_demand_constant_lead_time"
    elif sigma_d > 0.0 and sigma_l == 0.0:
        variability_case = "variable_demand_constant_lead_time"
    elif sigma_d == 0.0 and sigma_l > 0.0:
        variability_case = "constant_demand_variable_lead_time"
    else:
        variability_case = "variable_demand_variable_lead_time"

    # Lead-time demand variance: sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
    variance_demand_component = l_val * (sigma_d ** 2)
    variance_lt_component = (d_val ** 2) * (sigma_l ** 2)
    sigma_lt = math.sqrt(variance_demand_component + variance_lt_component)

    if safety_stock is not None:
        ss_val = _validate_numeric("safety_stock", safety_stock)
        if ss_val < 0:
            raise ValueError("safety_stock must be non-negative.")
        calc_ss = ss_val
        if sigma_lt > 0:
            z = calc_ss / sigma_lt
            csl = norm_cdf(z)
        else:
            z = 0.0
            csl = 1.0
    elif z_score is not None:
        z = _validate_numeric("z_score", z_score)
        if z < 0:
            raise ValueError("z_score must be non-negative.")
        csl = norm_cdf(z)
        calc_ss = z * sigma_lt
    elif service_level is not None:
        csl = _normalize_service_level(service_level)
        z = norm_ppf(csl)
        calc_ss = z * sigma_lt
    elif sigma_lt == 0.0:
        z = 0.0
        csl = 1.0
        calc_ss = 0.0
    else:
        raise ValueError(
            "Missing required service level input: provide either service_level, z_score, "
            "or direct safety_stock when demand or lead time is variable."
        )

    expected_lt_demand = d_val * l_val
    reorder_point = expected_lt_demand + calc_ss

    # Alternative 5-day work week calculation when converted from weeks
    alt_5day = None
    if lead_time_converted and raw_lead_time is not None:
        raw_lt_val = float(raw_lead_time)
        raw_std_lt_val = float(raw_std_lead_time) if raw_std_lead_time is not None else 0.0
        alt_l = raw_lt_val * 5.0
        alt_std_l = raw_std_lt_val * 5.0
        alt_exp_lt = d_val * alt_l
        alt_var_d = alt_l * (sigma_d ** 2)
        alt_var_l = (d_val ** 2) * (alt_std_l ** 2)
        alt_sigma_lt = math.sqrt(alt_var_d + alt_var_l)
        alt_ss = z * alt_sigma_lt
        alt_rop = alt_exp_lt + alt_ss
        alt_5day = {
            "days_per_week": 5.0,
            "lead_time_days": round(alt_l, 4),
            "std_lead_time_days": round(alt_std_l, 4),
            "expected_demand_during_lead_time": round(alt_exp_lt, 4),
            "sigma_lt": round(alt_sigma_lt, 4),
            "safety_stock": round(alt_ss, 4),
            "safety_stock_rounded": round(alt_ss),
            "reorder_point": round(alt_rop, 4),
            "reorder_point_rounded": round(alt_rop),
        }

    # 1. Inputs Extracted
    inputs_extracted = {
        "avg_demand": d_val,
        "lead_time": l_val,
        "std_demand": sigma_d,
        "std_lead_time": sigma_l,
        "service_level": round(csl, 6) if csl is not None else None,
        "z_score": round(z, 6),
        "safety_stock": round(calc_ss, 4) if safety_stock is not None else None,
    }
    if lead_time_converted or raw_lead_time is not None:
        if raw_lead_time is not None:
            inputs_extracted["stated_lead_time"] = raw_lead_time
        if lead_time_unit:
            inputs_extracted["lead_time_unit"] = lead_time_unit
        if raw_std_lead_time is not None:
            inputs_extracted["stated_std_lead_time"] = raw_std_lead_time
        if demand_unit:
            inputs_extracted["demand_unit"] = demand_unit
        inputs_extracted["days_per_week_assumed"] = op_days

    # 2. Variable Definitions
    variable_definitions = {
        "d": "Average demand per period (units/period)",
        "L": "Average replenishment lead time (periods)",
        "sigma_d": "Standard deviation of demand per period (units/period)",
        "sigma_L": "Standard deviation of lead time (periods)",
        "CSL": "Cycle service level (target probability of no stockout during lead time)",
        "z": "Safety factor (standard normal z-score corresponding to CSL)",
        "sigma_LT": "Standard deviation of lead-time demand: sqrt(L * sigma_d^2 + d^2 * sigma_L^2) (units)",
        "d_LT": "Expected demand during lead time: d * L (units)",
        "SS": "Safety stock: z * sigma_LT (units)",
        "ROP": "Reorder point: d * L + SS (units)",
    }

    # 3. Formula
    formula = {
        "expected_lead_time_demand": "d_LT = d * L",
        "lead_time_demand_std": "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)",
        "safety_factor_z": "z = norm_ppf(CSL)",
        "safety_stock": "SS = z * sigma_LT",
        "reorder_point": "ROP = d_LT + SS = (d * L) + (z * sigma_LT)",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "expected_lead_time_demand": f"d_LT = {d_val:,.2f} * {l_val:,.2f} = {expected_lt_demand:,.2f} units",
        "lead_time_demand_std": f"sigma_LT = sqrt({l_val:,.2f} * ({sigma_d:,.2f})^2 + ({d_val:,.2f})^2 * ({sigma_l:,.2f})^2) = sqrt({variance_demand_component:,.4f} + {variance_lt_component:,.4f}) = {sigma_lt:,.4f} units",
        "safety_factor_z": f"z = norm_ppf({csl:.4f}) = {z:.4f}",
        "safety_stock": f"SS = {z:.4f} * {sigma_lt:,.4f} = {calc_ss:,.4f} units",
        "reorder_point": f"ROP = {expected_lt_demand:,.2f} + {calc_ss:,.4f} = {reorder_point:,.4f} units",
    }

    # 5. Intermediate Calculations
    intermediate_calculations = {
        "variability_case": variability_case,
        "expected_demand_during_lead_time": round(expected_lt_demand, 4),
        "demand_variance_component_L_sigma_d2": round(variance_demand_component, 4),
        "lead_time_variance_component_d2_sigma_L2": round(variance_lt_component, 4),
        "sigma_lt": round(sigma_lt, 4),
        "z_score": round(z, 4),
        "safety_stock": round(calc_ss, 4),
        "reorder_point": round(reorder_point, 4),
    }
    if alt_5day:
        intermediate_calculations["alternative_5day_work_week"] = alt_5day

    # 6. Final Answer
    rec_text = f"Maintain a safety stock of {calc_ss:,.2f} units (approx. {round(calc_ss):,} units). Trigger a replenishment order whenever inventory position drops to the Reorder Point of {reorder_point:,.2f} units (approx. {round(reorder_point):,} units) to satisfy a {csl * 100:.1f}% cycle service level."
    if alt_5day:
        rec_text += f" (Note: Based on standard 7 calendar days/week; if using a 5-day work week, Safety Stock is {alt_5day['safety_stock']:,.2f} units and ROP is {alt_5day['reorder_point']:,.2f} units)."

    final_answer = {
        "safety_stock": round(calc_ss, 4),
        "safety_stock_rounded": round(calc_ss),
        "reorder_point": round(reorder_point, 4),
        "reorder_point_rounded": round(reorder_point),
        "recommendation": rec_text,
    }

    # 8. Assumptions
    assumptions = [
        f"Variability regime: {variability_case.replace('_', ' ')}.",
        "Continuous review policy: inventory position is monitored continuously and an order is placed as soon as inventory position <= ROP.",
        "Demand per period and lead time are independent random variables.",
        "Lead-time demand follows a normal distribution (when variable).",
        "Time units for average demand, standard deviation of demand, and lead time are consistent and expressed on the same time basis.",
    ]
    if lead_time_converted and raw_lead_time is not None:
        assumptions.append(
            f"Lead-time normalization: Lead time was stated as {raw_lead_time:g} {lead_time_unit or 'weeks'} with standard deviation {raw_std_lead_time or 0.0:g} {lead_time_unit or 'weeks'}. Converted to daily basis using standard {op_days:g} calendar days/week (L = {l_val:g} days, sigma_L = {sigma_l:g} days)."
        )
        if alt_5day:
            assumptions.append(
                f"Alternative 5-day work week: If operations/replenishment follow a 5-day business week (L = {alt_5day['lead_time_days']:g} days, sigma_L = {alt_5day['std_lead_time_days']:g} days), Expected LTD = {alt_5day['expected_demand_during_lead_time']:,.2f} units, sigma_LT = {alt_5day['sigma_lt']:,.2f} units, Safety Stock = {alt_5day['safety_stock']:,.2f} units (approx. {alt_5day['safety_stock_rounded']:,} units), and Reorder Point = {alt_5day['reorder_point']:,.2f} units (approx. {alt_5day['reorder_point_rounded']:,} units)."
            )

    results = {
        "model": "safety_stock_rop",
        "inputs_extracted": inputs_extracted,
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "assumptions": assumptions,
        # Top-level convenience fields
        "variability_case": variability_case,
        "service_level": round(csl, 4) if csl is not None else None,
        "z_score": round(z, 4),
        "expected_demand_during_lead_time": round(expected_lt_demand, 4),
        "sigma_lt": round(sigma_lt, 4),
        "std_dev_demand_during_lead_time": round(sigma_lt, 4),
        "safety_stock": round(calc_ss, 4),
        "safety_stock_rounded": round(calc_ss),
        "reorder_point": round(reorder_point, 4),
        "reorder_point_rounded": round(reorder_point),
    }

    # 7. Verification Checks
    verification = verify_calculation("safety_stock_rop", results["inputs_extracted"], results)
    results["verification_checks"] = verification
    results["verification"] = verification
    return results


# Alias for backward compatibility
calculate_safety_stock_rop = calculate_rop


# =====================================================================
# 4. Fixed-Period Review Model (Periodic Review / P-System)
# =====================================================================
def calculate_periodic_review(
    avg_demand: Optional[float] = None,
    std_demand: Optional[float] = None,
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    inventory_position: Optional[float] = None,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    current_inventory: Optional[float] = None,
    safety_stock: Optional[float] = None,
    on_hand_inventory: Optional[float] = None,
    on_order_inventory: Optional[float] = None,
    backorders: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Calculate Target Inventory and Order Quantity for a Fixed-Period (Periodic) Review System.

    Formulas:
        Protection period = review_period + lead_time
        Demand during protection period = avg_demand * protection_period
        Std dev during protection period = std_demand * sqrt(protection_period)
        Safety stock = z * std_demand * sqrt(protection_period)  (or directly specified)
        Target inventory = demand during protection period + safety stock
        Inventory position = on_hand + on_order - backorders  (if components provided)
        Order quantity = max(0, target inventory - inventory position)

    Returns 8 verification elements:
    1. inputs_extracted
    2. variable_definitions
    3. formula
    4. formula_with_substituted_values
    5. intermediate_calculations
    6. final_answer
    7. verification_checks
    8. assumptions
    """
    inv_pos_val: Optional[float] = None
    if inventory_position is not None:
        inv_pos_val = _validate_numeric("inventory_position", inventory_position)
    elif current_inventory is not None:
        inv_pos_val = _validate_numeric("current_inventory", current_inventory)
    elif on_hand_inventory is not None:
        on_hand = _validate_numeric("on_hand_inventory", on_hand_inventory)
        on_order = _validate_numeric("on_order_inventory", on_order_inventory) if on_order_inventory is not None else 0.0
        bo = _validate_numeric("backorders", backorders) if backorders is not None else 0.0
        inv_pos_val = on_hand + on_order - bo

    _require_non_none({
        "avg_demand": avg_demand,
        "std_demand": std_demand,
        "review_period": review_period,
        "lead_time": lead_time,
        "inventory_position": inv_pos_val,
    })

    d_val = _validate_numeric("avg_demand", avg_demand)
    sigma_d = _validate_numeric("std_demand", std_demand)
    t_val = _validate_numeric("review_period", review_period)
    l_val = _validate_numeric("lead_time", lead_time)
    inv_pos = inv_pos_val

    if d_val < 0:
        raise ValueError("avg_demand must be non-negative.")
    if sigma_d < 0:
        raise ValueError("std_demand must be non-negative.")
    if t_val <= 0:
        raise ValueError("review_period must be strictly greater than 0.")
    if l_val < 0:
        raise ValueError("lead_time must be non-negative.")

    protection_period = t_val + l_val
    demand_protection_period = d_val * protection_period
    std_dev_protection_period = sigma_d * math.sqrt(protection_period)

    if safety_stock is not None:
        ss_val = _validate_numeric("safety_stock", safety_stock)
        if ss_val < 0:
            raise ValueError("safety_stock must be non-negative.")
        calc_ss = ss_val
        if std_dev_protection_period > 0:
            z = calc_ss / std_dev_protection_period
            csl = norm_cdf(z)
        else:
            z = 0.0
            csl = 1.0
    elif z_score is not None:
        z = _validate_numeric("z_score", z_score)
        if z < 0:
            raise ValueError("z_score must be non-negative.")
        csl = norm_cdf(z)
        calc_ss = z * std_dev_protection_period
    elif service_level is not None:
        csl = _normalize_service_level(service_level)
        z = norm_ppf(csl)
        calc_ss = z * std_dev_protection_period
    elif sigma_d == 0.0:
        z = 0.0
        csl = 1.0
        calc_ss = 0.0
    else:
        raise ValueError(
            "Missing required service level input: provide either service_level, z_score, "
            "or direct safety_stock."
        )

    target_inventory = demand_protection_period + calc_ss
    order_quantity = max(0.0, target_inventory - inv_pos)

    # 1. Inputs Extracted
    inputs_extracted = {
        "avg_demand": d_val,
        "std_demand": sigma_d,
        "review_period": t_val,
        "lead_time": l_val,
        "inventory_position": inv_pos,
        "current_inventory": inv_pos,
        "service_level": round(csl, 6) if csl is not None else None,
        "z_score": round(z, 6),
        "safety_stock": round(calc_ss, 4) if safety_stock is not None else None,
        "on_hand_inventory": float(on_hand_inventory) if on_hand_inventory is not None else None,
        "on_order_inventory": float(on_order_inventory) if on_order_inventory is not None else None,
        "backorders": float(backorders) if backorders is not None else None,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "d": "Average demand per period (units/period)",
        "sigma_d": "Standard deviation of demand per period (units/period)",
        "T": "Review period / review interval (periods)",
        "L": "Replenishment lead time (periods)",
        "T + L": "Protection period / exposure interval: T + L (periods)",
        "CSL": "Cycle service level (probability of no stockout during protection period)",
        "z": "Safety factor from standard normal inverse CDF: norm_ppf(CSL)",
        "d_PI": "Expected demand during protection period: d * (T + L) (units)",
        "sigma_PI": "Standard deviation of demand during protection period: sigma_d * sqrt(T + L) (units)",
        "SS": "Safety stock for protection period: z * sigma_PI (units)",
        "M": "Target inventory / order-up-to level: d_PI + SS (units)",
        "IP": "Inventory position: on_hand + on_order - backorders (units)",
        "Q": "Recommended order quantity: max(0, M - IP) (units)",
    }

    # 3. Formula
    formula = {
        "protection_period": "Protection Period = T + L",
        "demand_during_protection_period": "d_PI = d * (T + L)",
        "std_dev_protection_period": "sigma_PI = sigma_d * sqrt(T + L)",
        "safety_factor_z": "z = norm_ppf(CSL)",
        "safety_stock": "SS = z * sigma_PI = z * sigma_d * sqrt(T + L)",
        "target_inventory": "M = d_PI + SS = d * (T + L) + z * sigma_d * sqrt(T + L)",
        "inventory_position": "IP = on_hand + on_order - backorders",
        "order_quantity": "Q = max(0, M - IP)",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "protection_period": f"Protection Period = {t_val:,.2f} + {l_val:,.2f} = {protection_period:,.2f} periods",
        "demand_during_protection_period": f"d_PI = {d_val:,.2f} * ({t_val:,.2f} + {l_val:,.2f}) = {d_val:,.2f} * {protection_period:,.2f} = {demand_protection_period:,.2f} units",
        "std_dev_protection_period": f"sigma_PI = {sigma_d:,.2f} * sqrt({protection_period:,.2f}) = {std_dev_protection_period:,.4f} units",
        "safety_factor_z": f"z = norm_ppf({csl:.4f}) = {z:.4f}",
        "safety_stock": f"SS = {z:.4f} * {std_dev_protection_period:,.4f} = {calc_ss:,.4f} units",
        "target_inventory": f"M = {demand_protection_period:,.2f} + {calc_ss:,.4f} = {target_inventory:,.4f} units",
        "inventory_position": f"IP = {inv_pos:,.2f} units",
        "order_quantity": f"Q = max(0, {target_inventory:,.2f} - {inv_pos:,.2f}) = {order_quantity:,.4f} units",
    }

    # 5. Intermediate Calculations
    intermediate_calculations = {
        "protection_period": round(protection_period, 4),
        "demand_during_protection_period": round(demand_protection_period, 4),
        "std_dev_protection_period": round(std_dev_protection_period, 4),
        "z_score": round(z, 4),
        "safety_stock": round(calc_ss, 4),
        "target_inventory": round(target_inventory, 4),
        "inventory_position": round(inv_pos, 4),
        "raw_shortfall": round(target_inventory - inv_pos, 4),
    }

    # 6. Final Answer
    final_answer = {
        "protection_period": round(protection_period, 4),
        "safety_stock": round(calc_ss, 4),
        "target_inventory": round(target_inventory, 4),
        "target_inventory_rounded": round(target_inventory),
        "order_quantity": round(order_quantity, 4),
        "order_quantity_rounded": round(order_quantity),
        "recommendation": f"Set Order-Up-To Target Inventory to {target_inventory:,.2f} units (incorporating {calc_ss:,.2f} units of safety stock). With current inventory position of {inv_pos:,.2f} units, place an order for {order_quantity:,.2f} units (approx. {round(order_quantity):,} units) at this review.",
    }

    # 8. Assumptions
    assumptions = [
        "Inventory is reviewed periodically every T (review_period) time units.",
        "Protection period spans review_period + lead_time (T + L) until the order placed at the next review can arrive.",
        "Demand across periods is independent and normally distributed.",
        "Inventory position includes on-hand inventory plus scheduled receipts (on-order) minus backorders.",
        "Time units for average demand, standard deviation of demand, review period, and lead time are consistent.",
    ]

    results = {
        "model": "fixed_period_review",
        "inputs_extracted": inputs_extracted,
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "assumptions": assumptions,
        # Top-level convenience fields
        "service_level": round(csl, 4) if csl is not None else None,
        "z_score": round(z, 4),
        "protection_period": round(protection_period, 4),
        "protection_interval": round(protection_period, 4),
        "demand_during_protection_period": round(demand_protection_period, 4),
        "expected_demand_protection_interval": round(demand_protection_period, 4),
        "std_dev_protection_interval": round(std_dev_protection_period, 4),
        "safety_stock": round(calc_ss, 4),
        "target_inventory": round(target_inventory, 4),
        "target_inventory_level": round(target_inventory, 4),
        "target_inventory_level_rounded": round(target_inventory),
        "inventory_position": round(inv_pos, 4),
        "current_inventory": round(inv_pos, 4),
        "order_quantity": round(order_quantity, 4),
        "order_quantity_rounded": round(order_quantity),
    }

    # 7. Verification Checks
    verification = verify_calculation("fixed_period_review", results["inputs_extracted"], results)
    results["verification_checks"] = verification
    results["verification"] = verification
    return results


# Alias for backward compatibility
calculate_fixed_period_review = calculate_periodic_review


# =====================================================================
# 5. Newsvendor Model (Single-Period Inventory)
# =====================================================================
def calculate_newsvendor(
    mean_demand: Optional[float] = None,
    std_demand: Optional[float] = None,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
    selling_price: Optional[float] = None,
    unit_cost: Optional[float] = None,
    salvage_value: Optional[float] = None,
    goodwill_cost: Optional[float] = None,
    distribution: str = "normal",
    min_demand: Optional[float] = None,
    max_demand: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Calculate underage cost (Cu), overage cost (Co), critical ratio (CR),
    and recommended order quantity (Q*) for the Newsvendor model.

    Returns 8 verification elements:
    1. inputs_extracted
    2. variable_definitions
    3. formula
    4. formula_with_substituted_values
    5. intermediate_calculations
    6. final_answer
    7. verification_checks
    8. assumptions
    """
    assumptions: List[str] = [
        "Single-period selling season: inventory cannot be carried over to future periods.",
        "Order quantity Q must be placed before random demand is realized.",
    ]

    p_val: Optional[float] = None
    c_val: Optional[float] = None
    s_val: Optional[float] = None
    g_val: float = 0.0

    if underage_cost is not None and overage_cost is not None:
        cu = _validate_numeric("underage_cost", underage_cost)
        co = _validate_numeric("overage_cost", overage_cost)
        cost_derivation = "Direct underage_cost (Cu) and overage_cost (Co) provided."
        p_val = _validate_numeric("selling_price", selling_price) if selling_price is not None else None
        c_val = _validate_numeric("unit_cost", unit_cost) if unit_cost is not None else None
        s_val = _validate_numeric("salvage_value", salvage_value) if salvage_value is not None else None
        g_val = _validate_numeric("goodwill_cost", goodwill_cost) if goodwill_cost is not None else 0.0
    elif selling_price is not None and unit_cost is not None and salvage_value is not None:
        p_val = _validate_numeric("selling_price", selling_price)
        c_val = _validate_numeric("unit_cost", unit_cost)
        s_val = _validate_numeric("salvage_value", salvage_value)
        g_val = _validate_numeric("goodwill_cost", goodwill_cost) if goodwill_cost is not None else 0.0
        cu = p_val - c_val + g_val
        co = c_val - s_val
        cost_derivation = (
            f"Cu = selling_price (${p_val:.2f}) - unit_cost (${c_val:.2f}) + goodwill_cost (${g_val:.2f}) = ${cu:.2f}; "
            f"Co = unit_cost (${c_val:.2f}) - salvage_value (${s_val:.2f}) = ${co:.2f}."
        )
        if goodwill_cost is None:
            assumptions.append("No extra goodwill stockout penalty beyond lost contribution margin (goodwill_cost = 0).")
    else:
        raise ValueError(
            "Missing required cost inputs: provide either (underage_cost and overage_cost) "
            "or (selling_price, unit_cost, and salvage_value)."
        )

    if cu <= 0:
        raise ValueError(f"underage_cost (Cu) must be strictly greater than 0, got {cu}.")
    if co <= 0:
        raise ValueError(f"overage_cost (Co) must be strictly greater than 0, got {co}.")

    critical_ratio = cu / (cu + co)
    dist = distribution.strip().lower()

    if dist == "uniform":
        _require_non_none({"min_demand": min_demand, "max_demand": max_demand})
        a = _validate_numeric("min_demand", min_demand)
        b = _validate_numeric("max_demand", max_demand)
        if b <= a:
            raise ValueError("max_demand must be strictly greater than min_demand for uniform distribution.")
        mu = 0.5 * (a + b)
        sigma = (b - a) / math.sqrt(12.0)
        z = None
        optimal_q = a + critical_ratio * (b - a)
        expected_leftover = ((optimal_q - a) ** 2) / (2.0 * (b - a))
        expected_sales = optimal_q - expected_leftover
        expected_lost_sales = mu - expected_sales
        assumptions.append(f"Demand follows a continuous Uniform({a}, {b}) distribution.")
    elif dist == "normal":
        _require_non_none({"mean_demand": mean_demand, "std_demand": std_demand})
        mu = _validate_numeric("mean_demand", mean_demand)
        sigma = _validate_numeric("std_demand", std_demand)
        if mu < 0:
            raise ValueError("mean_demand must be non-negative.")
        if sigma < 0:
            raise ValueError("std_demand must be non-negative.")

        z = norm_ppf(critical_ratio)
        optimal_q = mu + z * sigma
        loss_z = norm_pdf(z) - z * (1.0 - critical_ratio)
        expected_lost_sales = sigma * loss_z
        expected_sales = mu - expected_lost_sales
        expected_leftover = optimal_q - expected_sales
        assumptions.append(f"Demand follows a Normal(mean={mu}, std={sigma}) distribution.")
    else:
        raise ValueError(f"Unsupported distribution '{distribution}'. Choose 'normal' or 'uniform'.")

    expected_mismatch_cost = co * expected_leftover + cu * expected_lost_sales
    if p_val is not None and c_val is not None:
        expected_profit = (p_val - c_val) * mu - expected_mismatch_cost
    else:
        expected_profit = cu * mu - expected_mismatch_cost

    # 1. Inputs Extracted
    inputs_extracted = {
        "distribution": dist,
        "mean_demand": mu,
        "std_demand": sigma,
        "min_demand": float(min_demand) if min_demand is not None else None,
        "max_demand": float(max_demand) if max_demand is not None else None,
        "underage_cost": round(cu, 4),
        "overage_cost": round(co, 4),
        "selling_price": p_val,
        "unit_cost": c_val,
        "salvage_value": s_val,
        "goodwill_cost": g_val,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "p": "Selling price per unit ($/unit)",
        "c": "Purchase or production cost per unit ($/unit)",
        "s": "Salvage value per unsold unit at end of season ($/unit)",
        "g": "Goodwill / stockout penalty cost per unit of unsatisfied demand ($/unit)",
        "Cu": "Cost of underage / understocking per unit: p - c + g ($/unit)",
        "Co": "Cost of overage / overstocking per unit: c - s ($/unit)",
        "CR": "Critical ratio / critical fractile: Cu / (Cu + Co)",
        "mu": "Mean demand for single-period season (units)",
        "sigma": "Standard deviation of demand (units)",
        "z": "Safety factor from standard normal inverse CDF: norm_ppf(CR)",
        "Q*": "Recommended optimal order quantity (units)",
        "E_Sales": "Expected sales during season (units)",
        "E_Leftover": "Expected leftover inventory at season end (units)",
        "E_LostSales": "Expected unsatisfied demand / lost sales (units)",
        "E_MismatchCost": "Expected inventory mismatch cost: Co * E_Leftover + Cu * E_LostSales ($)",
        "E_Profit": "Expected profit: (p - c) * mu - E_MismatchCost ($)",
    }

    # 3. Formula
    formula = {
        "underage_cost": "Cu = p - c + g",
        "overage_cost": "Co = c - s",
        "critical_ratio": "CR = Cu / (Cu + Co)",
        "safety_factor_z": "z = norm_ppf(CR)",
        "optimal_order_quantity": "Q* = mu + z * sigma",
        "expected_lost_sales": "E_LostSales = sigma * [norm_pdf(z) - z * (1 - CR)]",
        "expected_sales": "E_Sales = mu - E_LostSales",
        "expected_leftover": "E_Leftover = Q* - E_Sales",
        "expected_mismatch_cost": "E_MismatchCost = Co * E_Leftover + Cu * E_LostSales",
        "expected_profit": "E_Profit = (p - c) * mu - E_MismatchCost",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "underage_cost": f"Cu = ${p_val:,.2f} - ${c_val:,.2f} + ${g_val:,.2f} = ${cu:,.2f}" if p_val and c_val else f"Cu = ${cu:,.2f}",
        "overage_cost": f"Co = ${c_val:,.2f} - ${s_val:,.2f} = ${co:,.2f}" if c_val and s_val is not None else f"Co = ${co:,.2f}",
        "critical_ratio": f"CR = ${cu:,.2f} / (${cu:,.2f} + ${co:,.2f}) = ${cu:,.2f} / ${cu + co:,.2f} = {critical_ratio:,.4f}",
        "safety_factor_z": f"z = norm_ppf({critical_ratio:.4f}) = {z:.4f}" if z is not None else "N/A (uniform distribution)",
        "optimal_order_quantity": f"Q* = {mu:,.2f} + ({z:.4f} * {sigma:,.2f}) = {mu:,.2f} + {z * sigma:,.2f} = {optimal_q:,.4f} units" if z is not None else f"Q* = {min_demand:,.2f} + {critical_ratio:.4f} * ({max_demand:,.2f} - {min_demand:,.2f}) = {optimal_q:,.4f} units",
        "expected_sales": f"E_Sales = {mu:,.2f} - {expected_lost_sales:,.2f} = {expected_sales:,.2f} units",
        "expected_leftover": f"E_Leftover = {optimal_q:,.2f} - {expected_sales:,.2f} = {expected_leftover:,.2f} units",
        "expected_profit": f"E_Profit = ${expected_profit:,.2f}",
    }

    # 5. Intermediate Calculations
    intermediate_calculations = {
        "cost_derivation": cost_derivation,
        "underage_cost_cu": round(cu, 4),
        "overage_cost_co": round(co, 4),
        "critical_ratio": round(critical_ratio, 4),
        "z_score": round(z, 4) if z is not None else None,
        "expected_sales": round(expected_sales, 4),
        "expected_lost_sales": round(expected_lost_sales, 4),
        "expected_leftover_inventory": round(expected_leftover, 4),
        "expected_mismatch_cost": round(expected_mismatch_cost, 4),
        "expected_profit": round(expected_profit, 4),
    }

    # 6. Final Answer
    final_answer = {
        "underage_cost": round(cu, 4),
        "overage_cost": round(co, 4),
        "critical_ratio": round(critical_ratio, 4),
        "recommended_order_quantity": round(optimal_q, 4),
        "recommended_order_quantity_rounded": round(optimal_q),
        "expected_profit": round(expected_profit, 4),
        "recommendation": f"Order {optimal_q:,.2f} units (approx. {round(optimal_q):,} units) for the single-period season. This achieves the optimal service fractile of {critical_ratio * 100:.2f}% and maximizes expected profit at ${expected_profit:,.2f}.",
    }

    results = {
        "model": "newsvendor",
        "inputs_extracted": inputs_extracted,
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "assumptions": assumptions,
        # Top-level convenience fields
        "underage_cost": round(cu, 4),
        "overage_cost": round(co, 4),
        "critical_ratio": round(critical_ratio, 4),
        "in_stock_probability": round(critical_ratio, 4),
        "z_score": round(z, 4) if z is not None else None,
        "recommended_order_quantity": round(optimal_q, 4),
        "recommended_order_quantity_rounded": round(optimal_q),
        "optimal_order_quantity": round(optimal_q, 4),
        "optimal_order_quantity_rounded": round(optimal_q),
        "expected_sales": round(expected_sales, 4),
        "expected_lost_sales": round(expected_lost_sales, 4),
        "expected_leftover_inventory": round(expected_leftover, 4),
        "expected_mismatch_cost": round(expected_mismatch_cost, 4),
        "expected_profit": round(expected_profit, 4),
    }

    # 7. Verification Checks
    verification = verify_calculation("newsvendor", results["inputs_extracted"], results)
    results["verification_checks"] = verification
    results["verification"] = verification
    return results


# =====================================================================
# Verification Engine
# =====================================================================
def verify_calculation(
    model_name: str,
    inputs: Dict[str, Any],
    results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Perform mathematical and domain sanity checks on the calculated inventory outputs.
    """
    checks: List[Dict[str, Any]] = []

    if model_name == "eoq":
        eoq = results["eoq"]
        oc = results["annual_ordering_cost"]
        hc = results["annual_holding_cost"]
        trc = results["total_relevant_cost"]

        checks.append({
            "check": "Positive EOQ",
            "passed": eoq > 0,
            "detail": f"Q* = {eoq:.4f} > 0",
        })
        cost_balance = abs(oc - hc) < 1e-2
        checks.append({
            "check": "Ordering Cost equals Holding Cost at EOQ",
            "passed": cost_balance,
            "detail": f"Ordering Cost (${oc:,.2f}) == Holding Cost (${hc:,.2f})",
        })
        sum_check = abs((oc + hc) - trc) < 1e-2
        checks.append({
            "check": "Total Relevant Cost consistency",
            "passed": sum_check,
            "detail": f"TOC (${oc:,.2f}) + THC (${hc:,.2f}) = TRC (${trc:,.2f})",
        })

    elif model_name == "quantity_discount":
        opt_q = results["optimal_order_quantity"]
        min_tc = results["min_total_annual_cost"]
        evals = results["tier_evaluations"]
        feasible_evals = [t for t in evals if t["feasible"]] or evals
        lowest_cost = min(t["total_annual_cost"] for t in feasible_evals)
        opt_tier_obj = next(t for t in evals if t["tier_index"] == results["optimal_tier"])

        checks.append({
            "check": "Order quantity satisfies tier minimum",
            "passed": opt_q >= opt_tier_obj["min_qty"] - 1e-6,
            "detail": f"Q = {opt_q:.2f} >= min_qty ({opt_tier_obj['min_qty']:.2f})",
        })
        checks.append({
            "check": "Minimum Total Annual Cost selected across feasible tiers",
            "passed": abs(min_tc - lowest_cost) < 1e-2,
            "detail": f"Selected TC (${min_tc:,.2f}) is minimum across {len(feasible_evals)} feasible tier(s)",
        })

    elif model_name in ("safety_stock_rop", "rop"):
        csl = results["service_level"]
        ss = results["safety_stock"]
        rop = results["reorder_point"]
        exp_lt = results["expected_demand_during_lead_time"]

        checks.append({
            "check": "Valid Cycle Service Level (0 < CSL <= 1)",
            "passed": csl is not None and 0.0 < csl <= 1.0,
            "detail": f"CSL = {csl:.4f}" if csl is not None else "CSL missing",
        })
        checks.append({
            "check": "ROP equals Expected Lead-Time Demand + Safety Stock",
            "passed": abs((exp_lt + ss) - rop) < 1e-2,
            "detail": f"{exp_lt:.2f} + {ss:.2f} = {rop:.2f}",
        })
        if csl is not None and csl >= 0.5:
            checks.append({
                "check": "Non-negative Safety Stock for CSL >= 50%",
                "passed": ss >= -1e-6,
                "detail": f"SS = {ss:.2f} >= 0",
            })

    elif model_name in ("fixed_period_review", "periodic_review"):
        pi = results["protection_period"]
        t_val = inputs["review_period"]
        l_val = inputs["lead_time"]
        target = results["target_inventory"]
        exp_d = results["demand_during_protection_period"]
        ss = results["safety_stock"]
        q_val = results["order_quantity"]

        checks.append({
            "check": "Protection Period equals review_period + lead_time",
            "passed": abs(pi - (t_val + l_val)) < 1e-6,
            "detail": f"T + L = {t_val} + {l_val} = {pi}",
        })
        checks.append({
            "check": "Target Inventory equals Protection Demand + Safety Stock",
            "passed": abs((exp_d + ss) - target) < 1e-2,
            "detail": f"{exp_d:.2f} + {ss:.2f} = {target:.2f}",
        })
        checks.append({
            "check": "Order Quantity equals max(0, target_inventory - inventory_position)",
            "passed": q_val >= 0.0 and abs(q_val - max(0.0, target - inputs["inventory_position"])) < 1e-2,
            "detail": f"Q = {q_val:.2f} >= 0",
        })

    elif model_name == "newsvendor":
        cr = results["critical_ratio"]
        cu = results["underage_cost"]
        co = results["overage_cost"]
        q_opt = results["recommended_order_quantity"]
        exp_sales = results["expected_sales"]
        exp_leftover = results["expected_leftover_inventory"]

        checks.append({
            "check": "Valid Critical Ratio (0 < CR < 1)",
            "passed": 0.0 < cr < 1.0,
            "detail": f"CR = Cu / (Cu + Co) = {cu:.2f} / ({cu:.2f} + {co:.2f}) = {cr:.4f}",
        })
        checks.append({
            "check": "Inventory Balance (Q* = Expected Sales + Expected Leftover)",
            "passed": abs((exp_sales + exp_leftover) - q_opt) < 1e-2,
            "detail": f"{exp_sales:.2f} + {exp_leftover:.2f} = {q_opt:.2f}",
        })

    all_passed = all(c["passed"] for c in checks)
    return {
        "is_valid": all_passed,
        "checks": checks,
    }


TOOL_REGISTRY = {
    "eoq": calculate_eoq,
    "quantity_discount": calculate_quantity_discount,
    "safety_stock_rop": calculate_rop,
    "rop": calculate_rop,
    "fixed_period_review": calculate_periodic_review,
    "periodic_review": calculate_periodic_review,
    "newsvendor": calculate_newsvendor,
}
