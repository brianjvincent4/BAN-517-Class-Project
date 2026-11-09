"""
inventory_tools.py

Pure Python numerical calculation and verification tools for the
Supply Chain Analytics Inventory Decision Agent.

Supported Models:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model
3. Safety Stock and Reorder Point (Continuous Review)
4. Fixed-Period Review (Periodic Review / P-System)
5. Newsvendor Model (Single-Period Inventory)
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
    if service_level > 1.0 and service_level < 100.0:
        return service_level / 100.0
    return float(service_level)


# =====================================================================
# 1. Economic Order Quantity (EOQ)
# =====================================================================
def calculate_eoq(
    annual_demand: float,
    ordering_cost: float,
    holding_cost: Optional[float] = None,
    unit_cost: Optional[float] = None,
    holding_rate: Optional[float] = None,
    working_days: float = 365.0,
) -> Dict[str, Any]:
    """
    Calculate Economic Order Quantity (EOQ) and associated cost/cycle metrics.

    Args:
        annual_demand: Annual demand (D) in units (> 0).
        ordering_cost: Setup or ordering cost per order (S) (> 0).
        holding_cost: Annual holding cost per unit (H). Optional if unit_cost and holding_rate are given.
        unit_cost: Purchase cost per unit (C). Optional.
        holding_rate: Annual holding cost as a fraction of unit cost (I). Optional.
        working_days: Number of working days per year (default 365).

    Returns:
        Dictionary of EOQ results and verification checks.
    """
    if annual_demand <= 0:
        raise ValueError("annual_demand (D) must be greater than 0.")
    if ordering_cost <= 0:
        raise ValueError("ordering_cost (S) must be greater than 0.")

    if holding_cost is None:
        if unit_cost is not None and holding_rate is not None:
            rate = holding_rate / 100.0 if holding_rate > 1.0 else holding_rate
            holding_cost = unit_cost * rate
        else:
            raise ValueError(
                "Must provide either holding_cost (H) directly, or both unit_cost (C) and holding_rate (I)."
            )

    if holding_cost <= 0:
        raise ValueError("holding_cost (H) must be greater than 0.")
    if working_days <= 0:
        raise ValueError("working_days must be greater than 0.")

    eoq = math.sqrt((2.0 * annual_demand * ordering_cost) / holding_cost)
    orders_per_year = annual_demand / eoq
    cycle_time_days = working_days / orders_per_year
    average_inventory = eoq / 2.0

    annual_ordering_cost = orders_per_year * ordering_cost
    annual_holding_cost = average_inventory * holding_cost
    total_relevant_cost = annual_ordering_cost + annual_holding_cost

    c_val = float(unit_cost) if unit_cost is not None else 0.0
    annual_purchase_cost = annual_demand * c_val
    total_annual_cost = total_relevant_cost + annual_purchase_cost

    results = {
        "model": "eoq",
        "inputs": {
            "annual_demand": float(annual_demand),
            "ordering_cost": float(ordering_cost),
            "holding_cost": float(holding_cost),
            "unit_cost": c_val if unit_cost is not None else None,
            "holding_rate": float(holding_rate) if holding_rate is not None else None,
            "working_days": float(working_days),
        },
        "eoq": round(eoq, 4),
        "eoq_rounded": round(eoq),
        "average_inventory": round(average_inventory, 4),
        "orders_per_year": round(orders_per_year, 4),
        "cycle_time_days": round(cycle_time_days, 4),
        "annual_ordering_cost": round(annual_ordering_cost, 4),
        "annual_holding_cost": round(annual_holding_cost, 4),
        "total_relevant_cost": round(total_relevant_cost, 4),
        "annual_purchase_cost": round(annual_purchase_cost, 4),
        "total_annual_cost": round(total_annual_cost, 4),
    }
    results["verification"] = verify_calculation("eoq", results["inputs"], results)
    return results


# =====================================================================
# 2. Quantity Discount Model
# =====================================================================
def calculate_quantity_discount(
    annual_demand: float,
    ordering_cost: float,
    tiers: List[Dict[str, Any]],
    holding_rate: Optional[float] = None,
    holding_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate all price break tiers in an all-units quantity discount schedule
    and identify the order quantity that minimizes total annual cost.

    Args:
        annual_demand: Annual demand (D) in units (> 0).
        ordering_cost: Ordering cost per order (S) (> 0).
        tiers: List of dicts, each containing:
            - 'min_qty': Minimum order quantity for tier (>= 1)
            - 'max_qty': Optional maximum order quantity for tier (None or inf for top tier)
            - 'unit_cost': Unit purchase price (C_i) for this tier (> 0)
            - 'holding_cost': Optional explicit unit holding cost for this tier
        holding_rate: Holding cost rate (I) as decimal (e.g. 0.20) or percentage (e.g. 20).
        holding_cost: Fixed annual holding cost per unit (H) if constant across tiers.

    Returns:
        Dictionary with optimal order quantity, optimal tier, full tier breakdown, and verification.
    """
    if annual_demand <= 0:
        raise ValueError("annual_demand (D) must be greater than 0.")
    if ordering_cost <= 0:
        raise ValueError("ordering_cost (S) must be greater than 0.")
    if not tiers:
        raise ValueError("tiers list cannot be empty.")

    norm_holding_rate: Optional[float] = None
    if holding_rate is not None:
        norm_holding_rate = holding_rate / 100.0 if holding_rate > 1.0 else float(holding_rate)
        if norm_holding_rate <= 0:
            raise ValueError("holding_rate must be greater than 0.")

    sorted_tiers = sorted(tiers, key=lambda t: float(t.get("min_qty", 0)))
    tier_evaluations: List[Dict[str, Any]] = []

    for idx, tier in enumerate(sorted_tiers, start=1):
        min_qty = float(tier.get("min_qty", 1))
        if min_qty <= 0:
            min_qty = 1.0
        raw_max = tier.get("max_qty", None)
        max_qty = float(raw_max) if raw_max is not None else float("inf")
        unit_cost = float(tier["unit_cost"])
        if unit_cost <= 0:
            raise ValueError(f"Tier {idx} unit_cost must be greater than 0.")

        # Determine holding cost H_i for this tier
        if tier.get("holding_cost") is not None:
            h_i = float(tier["holding_cost"])
        elif norm_holding_rate is not None:
            h_i = norm_holding_rate * unit_cost
        elif holding_cost is not None:
            h_i = float(holding_cost)
        else:
            raise ValueError(
                "Must provide holding_rate, holding_cost, or tier-specific holding_cost."
            )

        if h_i <= 0:
            raise ValueError(f"Tier {idx} holding cost must be greater than 0.")

        unconstrained_eoq = math.sqrt((2.0 * annual_demand * ordering_cost) / h_i)

        # Check feasibility and adjust order quantity
        feasible = True
        if unconstrained_eoq < min_qty:
            adjusted_q = min_qty
            adjustment_reason = "Raised to tier minimum quantity (Q* < min_qty)"
        elif unconstrained_eoq > max_qty:
            adjusted_q = max_qty
            feasible = False
            adjustment_reason = "Unconstrained Q* exceeds tier maximum (dominated by higher tier)"
        else:
            adjusted_q = unconstrained_eoq
            adjustment_reason = "Unconstrained Q* is within tier range"

        ann_ordering = (annual_demand / adjusted_q) * ordering_cost
        ann_holding = (adjusted_q / 2.0) * h_i
        ann_purchase = annual_demand * unit_cost
        total_cost = ann_ordering + ann_holding + ann_purchase

        tier_evaluations.append({
            "tier_index": idx,
            "min_qty": min_qty,
            "max_qty": None if math.isinf(max_qty) else max_qty,
            "unit_cost": round(unit_cost, 4),
            "holding_cost": round(h_i, 4),
            "unconstrained_eoq": round(unconstrained_eoq, 4),
            "evaluated_order_qty": round(adjusted_q, 4),
            "feasible": feasible,
            "adjustment_reason": adjustment_reason,
            "annual_ordering_cost": round(ann_ordering, 4),
            "annual_holding_cost": round(ann_holding, 4),
            "annual_purchase_cost": round(ann_purchase, 4),
            "total_annual_cost": round(total_cost, 4),
        })

    # Filter feasible candidates (note: in all-units discount, if Q* > max_qty, a higher tier has lower C_i and >= Q)
    feasible_tiers = [t for t in tier_evaluations if t["feasible"]]
    candidates = feasible_tiers if feasible_tiers else tier_evaluations
    best_tier = min(candidates, key=lambda t: t["total_annual_cost"])

    results = {
        "model": "quantity_discount",
        "inputs": {
            "annual_demand": float(annual_demand),
            "ordering_cost": float(ordering_cost),
            "holding_rate": norm_holding_rate,
            "holding_cost": float(holding_cost) if holding_cost is not None else None,
            "tiers": sorted_tiers,
        },
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
    results["verification"] = verify_calculation("quantity_discount", results["inputs"], results)
    return results


# =====================================================================
# 3. Safety Stock and Reorder Point (Continuous Review)
# =====================================================================
def calculate_safety_stock_rop(
    avg_demand: Optional[float] = None,
    std_demand: Optional[float] = None,
    lead_time: Optional[float] = None,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    std_lead_time: float = 0.0,
    lead_time_demand_mean: Optional[float] = None,
    lead_time_demand_std: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate Safety Stock (SS) and Reorder Point (ROP) for a continuous review policy.

    Supports:
    - Constant lead time with variable demand: sigma_dLT = std_demand * sqrt(lead_time)
    - Variable demand and variable lead time: sigma_dLT = sqrt(L * std_demand^2 + avg_demand^2 * std_lead_time^2)
    - Directly specified lead-time demand mean and standard deviation.

    Args:
        avg_demand: Average demand per period (d_bar).
        std_demand: Standard deviation of demand per period (sigma_d).
        lead_time: Average replenishment lead time in periods (L).
        service_level: Cycle service level (e.g., 0.95 or 95).
        z_score: Explicit safety factor z (optional alternative to service_level).
        std_lead_time: Standard deviation of lead time (sigma_L, default 0.0).
        lead_time_demand_mean: Direct mean demand during lead time (optional).
        lead_time_demand_std: Direct std dev of demand during lead time (optional).

    Returns:
        Dictionary with safety stock, reorder point, lead-time demand stats, and verification.
    """
    if z_score is not None:
        z = float(z_score)
        csl = norm_cdf(z)
    elif service_level is not None:
        csl = _normalize_service_level(service_level)
        z = norm_ppf(csl)
    else:
        raise ValueError("Must provide either service_level or z_score.")

    if lead_time_demand_mean is not None and lead_time_demand_std is not None:
        if lead_time_demand_mean < 0 or lead_time_demand_std < 0:
            raise ValueError("Lead time demand mean and std must be non-negative.")
        expected_lt_demand = float(lead_time_demand_mean)
        sigma_dlt = float(lead_time_demand_std)
    else:
        if avg_demand is None or std_demand is None or lead_time is None:
            raise ValueError(
                "Must provide avg_demand, std_demand, and lead_time (or lead_time_demand_mean and lead_time_demand_std)."
            )
        if avg_demand < 0 or std_demand < 0 or lead_time <= 0 or std_lead_time < 0:
            raise ValueError(
                "avg_demand, std_demand, std_lead_time must be >= 0 and lead_time must be > 0."
            )
        expected_lt_demand = float(avg_demand) * float(lead_time)
        sigma_dlt = math.sqrt(
            float(lead_time) * (float(std_demand) ** 2)
            + (float(avg_demand) ** 2) * (float(std_lead_time) ** 2)
        )

    safety_stock = z * sigma_dlt
    reorder_point = expected_lt_demand + safety_stock

    results = {
        "model": "safety_stock_rop",
        "inputs": {
            "avg_demand": float(avg_demand) if avg_demand is not None else None,
            "std_demand": float(std_demand) if std_demand is not None else None,
            "lead_time": float(lead_time) if lead_time is not None else None,
            "std_lead_time": float(std_lead_time),
            "service_level": round(csl, 6),
            "z_score": round(z, 6),
            "lead_time_demand_mean": float(lead_time_demand_mean) if lead_time_demand_mean is not None else None,
            "lead_time_demand_std": float(lead_time_demand_std) if lead_time_demand_std is not None else None,
        },
        "service_level": round(csl, 4),
        "z_score": round(z, 4),
        "expected_demand_during_lead_time": round(expected_lt_demand, 4),
        "std_dev_demand_during_lead_time": round(sigma_dlt, 4),
        "safety_stock": round(safety_stock, 4),
        "safety_stock_rounded": round(safety_stock),
        "reorder_point": round(reorder_point, 4),
        "reorder_point_rounded": round(reorder_point),
    }
    results["verification"] = verify_calculation("safety_stock_rop", results["inputs"], results)
    return results


# =====================================================================
# 4. Fixed-Period Review Model (Periodic Review / P-System)
# =====================================================================
def calculate_fixed_period_review(
    avg_demand: float,
    std_demand: float,
    review_period: float,
    lead_time: float,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    current_inventory: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate Target Inventory Level (Order-Up-To Level), Safety Stock, and Order Quantity
    for a Fixed-Period (Periodic) Review inventory system.

    Args:
        avg_demand: Average demand per period (d_bar >= 0).
        std_demand: Standard deviation of demand per period (sigma_d >= 0).
        review_period: Time between inventory reviews in periods (T or P > 0).
        lead_time: Replenishment lead time in periods (L >= 0).
        service_level: Cycle service level (e.g., 0.95 or 95).
        z_score: Explicit z-score (optional alternative to service_level).
        current_inventory: Current inventory position (on-hand + on-order - backorders, default 0).

    Returns:
        Dictionary with protection interval stats, safety stock, target level, order quantity, and verification.
    """
    if avg_demand < 0:
        raise ValueError("avg_demand must be non-negative.")
    if std_demand < 0:
        raise ValueError("std_demand must be non-negative.")
    if review_period <= 0:
        raise ValueError("review_period (T) must be greater than 0.")
    if lead_time < 0:
        raise ValueError("lead_time (L) must be non-negative.")

    if z_score is not None:
        z = float(z_score)
        csl = norm_cdf(z)
    elif service_level is not None:
        csl = _normalize_service_level(service_level)
        z = norm_ppf(csl)
    else:
        raise ValueError("Must provide either service_level or z_score.")

    protection_interval = float(review_period) + float(lead_time)
    expected_demand_pi = float(avg_demand) * protection_interval
    std_dev_pi = float(std_demand) * math.sqrt(protection_interval)
    safety_stock = z * std_dev_pi
    target_inventory_level = expected_demand_pi + safety_stock
    order_quantity = max(0.0, target_inventory_level - float(current_inventory))

    results = {
        "model": "fixed_period_review",
        "inputs": {
            "avg_demand": float(avg_demand),
            "std_demand": float(std_demand),
            "review_period": float(review_period),
            "lead_time": float(lead_time),
            "service_level": round(csl, 6),
            "z_score": round(z, 6),
            "current_inventory": float(current_inventory),
        },
        "service_level": round(csl, 4),
        "z_score": round(z, 4),
        "protection_interval": round(protection_interval, 4),
        "expected_demand_protection_interval": round(expected_demand_pi, 4),
        "std_dev_protection_interval": round(std_dev_pi, 4),
        "safety_stock": round(safety_stock, 4),
        "target_inventory_level": round(target_inventory_level, 4),
        "target_inventory_level_rounded": round(target_inventory_level),
        "current_inventory": round(float(current_inventory), 4),
        "order_quantity": round(order_quantity, 4),
        "order_quantity_rounded": round(order_quantity),
    }
    results["verification"] = verify_calculation("fixed_period_review", results["inputs"], results)
    return results


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
    salvage_value: float = 0.0,
    goodwill_cost: float = 0.0,
    distribution: str = "normal",
    min_demand: Optional[float] = None,
    max_demand: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate the optimal order quantity and performance metrics for the single-period
    Newsvendor problem.

    Supports specifying underage_cost (Cu) and overage_cost (Co) directly, OR deriving them from:
        Cu = selling_price - unit_cost + goodwill_cost
        Co = unit_cost - salvage_value

    Args:
        mean_demand: Mean demand (mu) for normal distribution.
        std_demand: Standard deviation of demand (sigma) for normal distribution.
        underage_cost: Cost of understocking per unit (Cu > 0).
        overage_cost: Cost of overstocking per unit (Co > 0).
        selling_price: Retail price per unit (p).
        unit_cost: Purchase/production cost per unit (c).
        salvage_value: Salvage value per leftover unit (s, default 0.0).
        goodwill_cost: Penalty/loss of goodwill per stocked-out unit (g, default 0.0).
        distribution: 'normal' (default) or 'uniform'.
        min_demand: Minimum demand (a) if distribution == 'uniform'.
        max_demand: Maximum demand (b) if distribution == 'uniform'.

    Returns:
        Dictionary with critical ratio, z-score, optimal order quantity, expected sales/profit, and verification.
    """
    if underage_cost is None or overage_cost is None:
        if selling_price is not None and unit_cost is not None:
            cu = float(selling_price) - float(unit_cost) + float(goodwill_cost)
            co = float(unit_cost) - float(salvage_value)
        else:
            raise ValueError(
                "Must provide either (underage_cost and overage_cost) or (selling_price and unit_cost)."
            )
    else:
        cu = float(underage_cost)
        co = float(overage_cost)

    if cu <= 0:
        raise ValueError(f"Underage cost (Cu) must be > 0, got {cu}. Check that selling_price > unit_cost.")
    if co <= 0:
        raise ValueError(f"Overage cost (Co) must be > 0, got {co}. Check that unit_cost > salvage_value.")

    critical_ratio = cu / (cu + co)
    dist = distribution.strip().lower()

    if dist == "uniform":
        if min_demand is None or max_demand is None or max_demand <= min_demand:
            raise ValueError("Uniform distribution requires min_demand < max_demand.")
        a = float(min_demand)
        b = float(max_demand)
        mu = 0.5 * (a + b)
        sigma = (b - a) / math.sqrt(12.0)
        z = None
        optimal_q = a + critical_ratio * (b - a)
        expected_leftover = ((optimal_q - a) ** 2) / (2.0 * (b - a))
        expected_sales = optimal_q - expected_leftover
        expected_lost_sales = mu - expected_sales
    else:
        if mean_demand is None or std_demand is None:
            raise ValueError("Normal distribution requires mean_demand and std_demand.")
        if std_demand < 0:
            raise ValueError("std_demand must be non-negative.")
        mu = float(mean_demand)
        sigma = float(std_demand)
        z = norm_ppf(critical_ratio)
        optimal_q = mu + z * sigma

        # Standard normal loss function L(z) = phi(z) - z * (1 - Phi(z))
        loss_z = norm_pdf(z) - z * (1.0 - critical_ratio)
        expected_lost_sales = sigma * loss_z
        expected_sales = mu - expected_lost_sales
        expected_leftover = optimal_q - expected_sales

    expected_mismatch_cost = co * expected_leftover + cu * expected_lost_sales

    if selling_price is not None and unit_cost is not None:
        max_profit_under_certainty = (float(selling_price) - float(unit_cost)) * mu
        expected_profit = max_profit_under_certainty - expected_mismatch_cost
    else:
        expected_profit = cu * mu - expected_mismatch_cost

    results = {
        "model": "newsvendor",
        "inputs": {
            "distribution": dist,
            "mean_demand": mu,
            "std_demand": sigma,
            "min_demand": float(min_demand) if min_demand is not None else None,
            "max_demand": float(max_demand) if max_demand is not None else None,
            "underage_cost": round(cu, 4),
            "overage_cost": round(co, 4),
            "selling_price": float(selling_price) if selling_price is not None else None,
            "unit_cost": float(unit_cost) if unit_cost is not None else None,
            "salvage_value": float(salvage_value),
            "goodwill_cost": float(goodwill_cost),
        },
        "underage_cost": round(cu, 4),
        "overage_cost": round(co, 4),
        "critical_ratio": round(critical_ratio, 4),
        "in_stock_probability": round(critical_ratio, 4),
        "z_score": round(z, 4) if z is not None else None,
        "optimal_order_quantity": round(optimal_q, 4),
        "optimal_order_quantity_rounded": round(optimal_q),
        "expected_sales": round(expected_sales, 4),
        "expected_lost_sales": round(expected_lost_sales, 4),
        "expected_leftover_inventory": round(expected_leftover, 4),
        "expected_mismatch_cost": round(expected_mismatch_cost, 4),
        "expected_profit": round(expected_profit, 4),
    }
    results["verification"] = verify_calculation("newsvendor", results["inputs"], results)
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
            "detail": f"OC + HC = ${trc:,.2f}",
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

    elif model_name == "safety_stock_rop":
        csl = results["service_level"]
        ss = results["safety_stock"]
        rop = results["reorder_point"]
        exp_lt = results["expected_demand_during_lead_time"]

        checks.append({
            "check": "Valid Cycle Service Level (0 < CSL < 1)",
            "passed": 0.0 < csl < 1.0,
            "detail": f"CSL = {csl:.4f}",
        })
        checks.append({
            "check": "ROP equals Expected Lead-Time Demand + Safety Stock",
            "passed": abs((exp_lt + ss) - rop) < 1e-2,
            "detail": f"{exp_lt:.2f} + {ss:.2f} = {rop:.2f}",
        })
        if csl >= 0.5:
            checks.append({
                "check": "Non-negative Safety Stock for CSL >= 50%",
                "passed": ss >= -1e-6,
                "detail": f"SS = {ss:.2f} >= 0",
            })

    elif model_name == "fixed_period_review":
        pi = results["protection_interval"]
        t_val = inputs["review_period"]
        l_val = inputs["lead_time"]
        target = results["target_inventory_level"]
        exp_d = results["expected_demand_protection_interval"]
        ss = results["safety_stock"]
        q_val = results["order_quantity"]

        checks.append({
            "check": "Protection Interval equals T + L",
            "passed": abs(pi - (t_val + l_val)) < 1e-6,
            "detail": f"T + L = {t_val} + {l_val} = {pi}",
        })
        checks.append({
            "check": "Target Inventory equals Expected Protection Demand + Safety Stock",
            "passed": abs((exp_d + ss) - target) < 1e-2,
            "detail": f"{exp_d:.2f} + {ss:.2f} = {target:.2f}",
        })
        checks.append({
            "check": "Non-negative Order Quantity",
            "passed": q_val >= 0.0,
            "detail": f"Q = {q_val:.2f} >= 0",
        })

    elif model_name == "newsvendor":
        cr = results["critical_ratio"]
        cu = results["underage_cost"]
        co = results["overage_cost"]
        q_opt = results["optimal_order_quantity"]
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
    "safety_stock_rop": calculate_safety_stock_rop,
    "fixed_period_review": calculate_fixed_period_review,
    "newsvendor": calculate_newsvendor,
}
