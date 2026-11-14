"""
app.py

Supply Chain Analytics Inventory Decision Agent
Professional Business Analytics Dashboard & Decision Support System (DSS)

Architecture:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
"""

from __future__ import annotations

import io
import json
import math
import os
from typing import Any, Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import norm

from inventory_agent import MODEL_METADATA, InventoryDecisionAgent
from inventory_classification import (
    CLASSIFICATION_REQUIRED_COLUMNS,
    SCORE_MAP,
    WEIGHT_HITS,
    WEIGHT_SALES,
    assign_final_class_from_score,
    assign_pareto_class,
    classify_inventory_multicriteria,
    validate_classification_dataframe,
)
from inventory_tools import (
    calculate_eoq,
    calculate_newsvendor,
    calculate_periodic_review,
    calculate_quantity_discount,
    calculate_rop,
)

REQUIRED_COLUMNS: List[str] = ["Product", "Category", "Units", "Unit Cost"]

SAMPLE_QUESTIONS: Dict[str, str] = {
    "1. Economic Order Quantity (EOQ)": (
        "A distribution center has an annual demand of 10,000 units for a component. "
        "The ordering cost is $75 per order and the holding cost is $6 per unit per year "
        "(unit cost is $30). What is the Economic Order Quantity (EOQ) and total annual cost?"
    ),
    "2. Quantity Discount": (
        "A medical supply company faces an annual demand of 5,000 units. Ordering cost is $49 per order "
        "and the holding cost rate is 20% of unit cost. The supplier offers a quantity discount schedule: "
        "1 to 999 units at $5.00, 1,000 to 1,999 units at $4.80, and 2,000+ units at $4.75. "
        "What order quantity minimizes total annual cost?"
    ),
    "3. Safety Stock and Reorder Point": (
        "Daily demand for a warehouse item averages 150 units with a standard deviation of 25 units. "
        "Replenishment lead time is 4 days. Management wants to maintain a 95% cycle service level. "
        "Calculate the safety stock and reorder point."
    ),
    "4. Fixed-Period Review": (
        "A retailer uses a fixed-period review system and reviews inventory every 14 days with a lead time "
        "of 3 days. Average demand is 80 units per day with a standard deviation of 15 units. "
        "Current inventory on hand is 320 units. For a 98% service level, what is the target inventory "
        "level and how many units should be ordered?"
    ),
    "5. Newsvendor": (
        "A store sells a seasonal jacket for a selling price of $120 per unit and buys them for a unit cost "
        "of $70. Unsold jackets at the end of the season are salvaged for $30. Demand is normally distributed "
        "with a mean demand of 500 units and a standard deviation of 100 units. Using the newsvendor model, "
        "what is the optimal order quantity and expected profit?"
    ),
}

SCENARIO_DESCRIPTIONS: Dict[str, str] = {
    "1. Economic Order Quantity (EOQ)": (
        "Evaluates the optimal batch order quantity (Q*) that minimizes total relevant ordering "
        "and holding costs under deterministic, constant demand."
    ),
    "2. Quantity Discount": (
        "Evaluates all-units price break schedules to select the order quantity that minimizes "
        "total annual logistics cost (ordering + holding + purchase expense)."
    ),
    "3. Safety Stock and Reorder Point": (
        "Continuous review policy (s, Q) determining buffer safety stock and reorder trigger point (ROP) "
        "to satisfy a target cycle service level under demand/lead-time variability."
    ),
    "4. Fixed-Period Review": (
        "Periodic review policy (R, S / P-system) calculating order-up-to target inventory level "
        "over the protection period (review period + lead time) and net order quantity."
    ),
    "5. Newsvendor": (
        "Single-period stochastic model for perishable or seasonal products balancing the marginal cost "
        "of underage (lost profit) against overage (liquidation loss) via the critical ratio."
    ),
}


# =====================================================================
# Core CSV Data Processing & Analysis
# =====================================================================
def process_and_analyze_inventory_csv(file_or_buffer: Any) -> Dict[str, Any]:
    """
    Read an uploaded CSV file, validate required columns, and perform pandas-based
    inventory data analysis:
    1. Read CSV into pandas DataFrame
    2. Validate required columns: Product, Category, Units, Unit Cost
    3. Calculate total Units
    4. Calculate average Unit Cost
    5. Identify Product with highest Units
    6. Group by Category and calculate total Units
    7. Calculate new column: Inventory Value = Units * Unit Cost
    8. Calculate total Inventory Value across all products
    """
    df = pd.read_csv(file_or_buffer)

    # Clean whitespace in column names
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return {
            "valid": False,
            "missing_columns": missing_cols,
            "df": df,
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "columns": list(df.columns),
        }

    clean_df = df.copy()

    # Ensure numeric types for calculations (handles clean numbers as well as currency formatted strings)
    clean_df["Units"] = pd.to_numeric(
        clean_df["Units"].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    ).fillna(0.0)

    clean_df["Unit Cost"] = pd.to_numeric(
        clean_df["Unit Cost"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    ).fillna(0.0)

    # 1. Total Units
    total_units = float(clean_df["Units"].sum())

    # 2. Average Unit Cost
    avg_unit_cost = float(clean_df["Unit Cost"].mean()) if not clean_df.empty else 0.0

    # 3. Product with highest Units
    if not clean_df.empty:
        max_idx = clean_df["Units"].idxmax()
        highest_units_product = str(clean_df.loc[max_idx, "Product"])
        highest_units_count = float(clean_df.loc[max_idx, "Units"])
    else:
        highest_units_product = "N/A"
        highest_units_count = 0.0

    # 4. Group by Category and calculate total Units for each Category
    category_units = (
        clean_df.groupby("Category")["Units"]
        .sum()
        .reset_index()
        .rename(columns={"Units": "Total Units"})
        .sort_values(by="Total Units", ascending=False)
    )

    # 5. Inventory Value = Units * Unit Cost
    clean_df["Inventory Value"] = clean_df["Units"] * clean_df["Unit Cost"]

    # 6. Total Inventory Value across all products
    total_inventory_value = float(clean_df["Inventory Value"].sum())

    return {
        "valid": True,
        "missing_columns": [],
        "df": clean_df,
        "num_rows": len(clean_df),
        "num_cols": len(clean_df.columns),
        "columns": list(clean_df.columns),
        "total_units": total_units,
        "avg_unit_cost": avg_unit_cost,
        "highest_units_product": highest_units_product,
        "highest_units_count": highest_units_count,
        "category_units": category_units,
        "total_inventory_value": total_inventory_value,
    }


# =====================================================================
# UI Component Rendering Helpers (Theme-Adaptive & Polished)
# =====================================================================
def inject_custom_styles() -> None:
    """Inject subtle, professional CSS styling that respects both light and dark themes."""
    import streamlit as st

    st.markdown(
        """
        <style>
        /* Professional Business Dashboard Typography & Accents */
        .main-header {
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 0.95rem;
            color: #64748B;
            margin-bottom: 1.2rem;
        }
        /* Executive Decision Callout Card */
        .executive-banner {
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            margin: 1.0rem 0 1.25rem 0;
            border: 1px solid rgba(37, 99, 235, 0.35);
            border-left: 6px solid #2563EB;
            background: rgba(37, 99, 235, 0.06);
        }
        .executive-tag {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #2563EB;
            margin-bottom: 0.35rem;
        }
        .executive-recommendation {
            font-size: 1.20rem;
            font-weight: 600;
            line-height: 1.45;
        }
        /* Step badge styling */
        .pipeline-pill {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 9999px;
            background: rgba(37, 99, 235, 0.12);
            color: #2563EB;
            margin-right: 0.4rem;
        }
        /* Metric card container styling */
        div[data-testid="stMetric"] {
            border-radius: 8px;
            padding: 0.6rem 0.8rem;
            background: rgba(128, 128, 128, 0.04);
            border: 1px solid rgba(128, 128, 128, 0.15);
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.80rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.40rem !important;
            font-weight: 700 !important;
        }
        /* Expander headers */
        div[data-testid="stExpander"] {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 8px !important;
            margin-bottom: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_badges(response: Any) -> None:
    """Render the 7-step decision pipeline as professional status pills."""
    import streamlit as st

    stages = [
        ("1. Question", "Received"),
        ("2. Identify Model", response.model_display_name),
        ("3. Extract Inputs", f"{len(response.extracted_inputs)} params"),
        ("4. Select Tool", f"{response.selected_tool_name}()"),
        ("5. Calculate", "Deterministic"),
        ("6. Verify", "Validated" if response.verification.get("is_valid") else "Check Flagged"),
        ("7. Explain", "Generated"),
    ]
    cols = st.columns(7)
    for col, (title, sub) in zip(cols, stages):
        with col:
            st.markdown(
                f"<div style='border: 1px solid rgba(37, 99, 235, 0.25); border-radius: 6px; padding: 6px 8px; text-align: center; background: rgba(37, 99, 235, 0.05);'>"
                f"<div style='font-size: 0.72rem; font-weight: 700; color: #2563EB;'>{title}</div>"
                f"<div style='font-size: 0.80rem; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{sub}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_executive_summary_dashboard(response: Any) -> None:
    """
    Render Section 2 & 3: Executive Decision Summary with prominent recommendation
    and model-specific KPI cards for all five supported inventory models.
    """
    import streamlit as st

    model_name = response.identified_model
    calc = response.calculation_results
    final = calc.get("final_answer", {})
    inter = calc.get("intermediate_calculations", {})
    inputs_ext = calc.get("inputs_extracted", {})

    # 1. Prominent Executive Recommendation Banner
    recommendation_text = final.get(
        "recommendation",
        "Deterministic calculation completed successfully. Review detailed metrics and verification below.",
    )

    st.markdown(
        f"""
        <div class="executive-banner">
            <div class="executive-tag">🎯 Executive Decision Recommendation • {response.model_display_name}</div>
            <div class="executive-recommendation">{recommendation_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Model-Specific KPI Dashboard
    st.markdown("#### 📊 Key Performance Indicators (KPIs)")

    if model_name == "eoq":
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        eoq_val = final.get("eoq")
        eoq_round = final.get("eoq_rounded")
        k1.metric("Optimal Batch (Q*)", f"{eoq_val:,.1f} units" if eoq_val else "N/A", f"≈ {eoq_round:,} units" if eoq_round else None)

        n_orders = final.get("orders_per_year")
        k2.metric("Order Frequency (N)", f"{n_orders:,.2f} /yr" if n_orders else "N/A")

        cycle_days = final.get("cycle_time_days")
        k3.metric("Cycle Time", f"{cycle_days:,.1f} days" if cycle_days is not None else "N/A", help="Working days between consecutive orders")

        ann_order_cost = inter.get("annual_ordering_cost")
        k4.metric("Annual Ordering Cost", f"${ann_order_cost:,.2f}" if ann_order_cost is not None else "N/A")

        ann_hold_cost = inter.get("annual_holding_cost")
        k5.metric("Annual Holding Cost", f"${ann_hold_cost:,.2f}" if ann_hold_cost is not None else "N/A")

        tot_cost = final.get("total_annual_cost")
        purch_cost = inter.get("annual_purchase_cost")
        sub_text = f"Incl. ${purch_cost:,.2f} purch" if purch_cost else "Total Relevant Cost"
        k6.metric("Total Annual Cost", f"${tot_cost:,.2f}" if tot_cost is not None else "N/A", sub_text)

    elif model_name == "quantity_discount":
        k1, k2, k3, k4, k5 = st.columns(5)
        opt_q = final.get("optimal_order_quantity")
        k1.metric("Recommended Order Q*", f"{opt_q:,.0f} units" if opt_q is not None else "N/A")

        opt_tier = final.get("optimal_tier")
        k2.metric("Selected Price Tier", f"Tier {opt_tier}" if opt_tier is not None else "N/A")

        opt_price = final.get("optimal_unit_cost")
        k3.metric("Unit Purchase Price", f"${opt_price:,.2f}" if opt_price is not None else "N/A")

        # Extract annual purchase cost for optimal tier
        tiers_eval = inter.get("tier_evaluations", [])
        best_eval = next((t for t in tiers_eval if t.get("tier_index") == opt_tier), None)
        ann_purch = best_eval.get("annual_purchase_cost") if best_eval else None
        k4.metric("Annual Purchase Cost", f"${ann_purch:,.2f}" if ann_purch is not None else "N/A")

        min_tot_cost = final.get("min_total_annual_cost")
        k5.metric("Total Logistics Cost", f"${min_tot_cost:,.2f}" if min_tot_cost is not None else "N/A", "Ordering + Holding + Purchase")

    elif model_name in ("safety_stock_rop", "rop"):
        k1, k2, k3, k4, k5 = st.columns(5)
        ss_val = final.get("safety_stock")
        ss_round = final.get("safety_stock_rounded")
        k1.metric("Safety Stock (SS)", f"{ss_val:,.1f} units" if ss_val is not None else "N/A", f"≈ {ss_round:,} units" if ss_round else None)

        rop_val = final.get("reorder_point")
        rop_round = final.get("reorder_point_rounded")
        k2.metric("Reorder Point (ROP)", f"{rop_val:,.1f} units" if rop_val is not None else "N/A", f"≈ {rop_round:,} units" if rop_round else None)

        exp_d_lt = inter.get("expected_demand_during_lead_time")
        k3.metric("Lead-Time Demand", f"{exp_d_lt:,.1f} units" if exp_d_lt is not None else "N/A", help="Expected demand during replenishment (d * L)")

        sl_val = inputs_ext.get("service_level") or inter.get("service_level")
        z_score = inter.get("z_score")
        k4.metric("Service Level (CSL)", f"{sl_val * 100:.1f}%" if sl_val is not None else "N/A", f"z = {z_score:.2f}" if z_score is not None else None)

        sigma_lt = inter.get("sigma_lt")
        k5.metric("Lead-Time Std Dev (σLT)", f"{sigma_lt:,.2f} units" if sigma_lt is not None else "N/A")

    elif model_name in ("fixed_period_review", "periodic_review"):
        k1, k2, k3, k4, k5 = st.columns(5)
        ord_q = final.get("order_quantity")
        ord_round = final.get("order_quantity_rounded")
        k1.metric("Order Quantity (Q)", f"{ord_q:,.1f} units" if ord_q is not None else "N/A", f"≈ {ord_round:,} units" if ord_round else None)

        tgt_inv = final.get("target_inventory")
        tgt_round = final.get("target_inventory_rounded")
        k2.metric("Target Inventory (M)", f"{tgt_inv:,.1f} units" if tgt_inv is not None else "N/A", f"≈ {tgt_round:,} units" if tgt_round else None)

        rev_per = inputs_ext.get("review_period")
        k3.metric("Review Period (T)", f"{rev_per:,.0f} days" if rev_per is not None else "N/A")

        ss_per = final.get("safety_stock")
        k4.metric("Safety Stock", f"{ss_per:,.1f} units" if ss_per is not None else "N/A")

        inv_pos = inputs_ext.get("inventory_position") or inter.get("inventory_position") or final.get("inventory_position")
        k5.metric("Current Position (IP)", f"{inv_pos:,.1f} units" if inv_pos is not None else "N/A")

    elif model_name == "newsvendor":
        k1, k2, k3, k4, k5 = st.columns(5)
        q_star = final.get("recommended_order_quantity")
        q_round = final.get("recommended_order_quantity_rounded")
        k1.metric("Stocking Quantity (Q*)", f"{q_star:,.1f} units" if q_star is not None else "N/A", f"≈ {q_round:,} units" if q_round else None)

        cr = final.get("critical_ratio")
        k2.metric("Critical Ratio (CR)", f"{cr:.4f}" if cr is not None else "N/A", f"Fractile: {cr * 100:.1f}%" if cr else None)

        cu = final.get("underage_cost")
        k3.metric("Underage Cost (Cu)", f"${cu:,.2f}" if cu is not None else "N/A", "Marginal profit per unit")

        co = final.get("overage_cost")
        k4.metric("Overage Cost (Co)", f"${co:,.2f}" if co is not None else "N/A", "Marginal loss per unit")

        exp_prof = final.get("expected_profit")
        k5.metric("Expected Profit", f"${exp_prof:,.2f}" if exp_prof is not None else "N/A")


# =====================================================================
# Tab 1: Detailed Decision Analysis & Verification Trace
# =====================================================================
def render_decision_analysis_tab(response: Any) -> None:
    """Render structured pedagogical step-by-step decision analysis in expandable containers."""
    import streamlit as st

    calc = response.calculation_results

    st.markdown("### 🔍 Step-by-Step Mathematical Verification Trace")
    st.markdown(
        "This section documents the deterministic calculation steps, variable definitions, and "
        "optimality checks verified directly by the Python analytical engine."
    )

    with st.expander("📝 Executive Explanation & Problem Formulation", expanded=True):
        st.markdown(response.explanation)

    with st.expander("1. Extracted Input Parameters", expanded=True):
        inputs_data = calc.get("inputs_extracted") or response.extracted_inputs
        st.json(inputs_data)

    with st.expander("2. Variable Definitions & Domain Units", expanded=False):
        var_defs = calc.get("variable_definitions", {})
        if var_defs:
            for symbol, description in var_defs.items():
                st.markdown(f"- **`{symbol}`**: {description}")
        else:
            st.info("No variable definitions available.")

    with st.expander("3. Mathematical Model Formulas", expanded=False):
        formulas = calc.get("formula", {})
        if formulas:
            for name, form in formulas.items():
                st.markdown(f"- **{name.replace('_', ' ').title()}**: `{form}`")
        else:
            st.info("No mathematical formulas recorded.")

    with st.expander("4. Formulas with Substituted Numerical Values", expanded=True):
        substituted = calc.get("formula_with_substituted_values", {})
        if substituted:
            for name, expr in substituted.items():
                st.markdown(f"- **{name.replace('_', ' ').title()}**: {expr}")
        else:
            st.info("No formula substitutions available.")

    with st.expander("5. Intermediate Calculations (Deterministic Python)", expanded=False):
        inter = calc.get("intermediate_calculations", {})
        if "tier_evaluations" in inter:
            st.dataframe(pd.DataFrame(inter["tier_evaluations"]), use_container_width=True)
        elif inter:
            st.json(inter)
        else:
            st.info("No intermediate calculations recorded.")

    with st.expander("6. Final Answer Summary", expanded=True):
        st.json(calc.get("final_answer", {}))

    with st.expander("7. Verification & Optimality Checks", expanded=True):
        verification_data = response.verification
        checks = verification_data.get("checks", [])
        if checks:
            for chk in checks:
                status_icon = "✅" if chk.get("passed") else "❌"
                st.markdown(f"{status_icon} **{chk.get('check')}**: `{chk.get('detail')}`")
        else:
            is_valid = verification_data.get("is_valid", False)
            icon = "✅" if is_valid else "❌"
            st.markdown(f"{icon} **Global Model Validation**: `is_valid = {is_valid}`")

    with st.expander("8. Model Assumptions & Boundary Conditions", expanded=False):
        assumptions = calc.get("assumptions", [])
        if assumptions:
            for item in assumptions:
                st.markdown(f"- {item}")
        else:
            st.info("No assumptions documented.")


# =====================================================================
# Tab 2: Visual Analytics & Model Curves
# =====================================================================
def render_visual_analytics_tab(response: Any) -> None:
    """Generate professional, theme-adaptive Altair charts for the identified model."""
    import streamlit as st

    model_name = response.identified_model
    calc = response.calculation_results
    final = calc.get("final_answer", {})
    inter = calc.get("intermediate_calculations", {})
    inputs_ext = calc.get("inputs_extracted", {})

    st.markdown("### 📈 Visual Analytics & Trade-Off Curves")

    if model_name == "eoq":
        d_val = inputs_ext.get("annual_demand")
        s_val = inputs_ext.get("ordering_cost")
        h_val = inter.get("holding_cost_per_unit") or inputs_ext.get("holding_cost")
        q_star = final.get("eoq")

        if d_val and s_val and h_val and q_star:
            q_min = max(5.0, q_star * 0.25)
            q_max = q_star * 2.25
            q_points = np.linspace(q_min, q_max, 75)

            curve_df = pd.DataFrame({
                "Order Quantity (Q)": q_points,
                "Annual Ordering Cost (TOC)": (d_val / q_points) * s_val,
                "Annual Holding Cost (THC)": (q_points / 2.0) * h_val,
                "Total Relevant Cost (TRC)": (d_val / q_points) * s_val + (q_points / 2.0) * h_val,
            })
            melted = curve_df.melt("Order Quantity (Q)", var_name="Cost Component", value_name="Annual Cost ($)")

            color_scale = alt.Scale(
                domain=["Total Relevant Cost (TRC)", "Annual Holding Cost (THC)", "Annual Ordering Cost (TOC)"],
                range=["#1E3A8A", "#0D9488", "#D97706"],
            )

            base_lines = (
                alt.Chart(melted)
                .mark_line(strokeWidth=2.5)
                .encode(
                    x=alt.X("Order Quantity (Q):Q", title="Order Quantity (Q in units)"),
                    y=alt.Y("Annual Cost ($):Q", title="Annual Inventory Cost ($/year)"),
                    color=alt.Color("Cost Component:N", scale=color_scale),
                    tooltip=[
                        alt.Tooltip("Order Quantity (Q):Q", format=",.1f"),
                        alt.Tooltip("Annual Cost ($):Q", format="$,.2f"),
                        alt.Tooltip("Cost Component:N"),
                    ],
                )
            )

            # Mark Q* on the curve
            trc_min = final.get("total_relevant_cost", (d_val / q_star) * s_val + (q_star / 2.0) * h_val)
            q_point = pd.DataFrame([{"Q": q_star, "Cost": trc_min, "Label": f"Optimal EOQ: {q_star:,.1f}"}])
            point_mark = alt.Chart(q_point).mark_circle(size=120, color="#DC2626").encode(x="Q:Q", y="Cost:Q")
            rule_mark = (
                alt.Chart(q_point)
                .mark_rule(color="#DC2626", strokeDash=[4, 4], strokeWidth=1.5)
                .encode(x="Q:Q")
            )

            eoq_chart = (base_lines + point_mark + rule_mark).properties(
                height=380,
                title="Economic Order Quantity (EOQ) Cost Trade-Off Curve",
            )
            st.altair_chart(eoq_chart, use_container_width=True)
            st.caption(
                "💡 **Takeaway**: The Economic Order Quantity (Q*) occurs at the exact minimum of the Total Relevant Cost curve, "
                "where Annual Ordering Cost equals Annual Holding Cost ($TOC = THC$)."
            )
        else:
            st.info("Insufficient parameter details to plot EOQ curve.")

    elif model_name == "quantity_discount":
        tiers_eval = inter.get("tier_evaluations", [])
        if tiers_eval:
            tdf = pd.DataFrame(tiers_eval)
            opt_tier_idx = final.get("optimal_tier")
            tdf["Tier Label"] = tdf.apply(lambda r: f"Tier {r['tier_index']} (${r['unit_cost']:.2f})", axis=1)
            tdf["Is Optimal"] = tdf["tier_index"] == opt_tier_idx

            chart = (
                alt.Chart(tdf)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Tier Label:N", title="Discount Schedule Tiers", sort=None),
                    y=alt.Y("total_annual_cost:Q", title="Total Annual Cost ($/year)"),
                    color=alt.condition(alt.datum["Is Optimal"], alt.value("#0D9488"), alt.value("#64748B")),
                    tooltip=[
                        alt.Tooltip("Tier Label:N"),
                        alt.Tooltip("evaluated_order_qty:Q", title="Evaluated Q", format=",.0f"),
                        alt.Tooltip("unit_cost:Q", title="Unit Price", format="$,.2f"),
                        alt.Tooltip("total_annual_cost:Q", title="Total Cost", format="$,.2f"),
                        alt.Tooltip("is_feasible:N", title="Feasible"),
                    ],
                )
                .properties(height=340, title="Total Annual Cost Comparison Across Discount Tiers")
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("💡 **Takeaway**: Highlighted in teal is the lowest-cost feasible tier that minimizes overall total cost.")
        else:
            st.info("Tier evaluations not available to plot.")

    elif model_name in ("safety_stock_rop", "rop"):
        mu_lt = inter.get("expected_demand_during_lead_time")
        sigma_lt = inter.get("sigma_lt")
        rop_val = final.get("reorder_point")
        ss_val = final.get("safety_stock")

        if mu_lt and sigma_lt and sigma_lt > 0:
            x_vals = np.linspace(max(0, mu_lt - 3.5 * sigma_lt), mu_lt + 3.5 * sigma_lt, 100)
            y_vals = norm.pdf(x_vals, mu_lt, sigma_lt)
            dist_df = pd.DataFrame({"Lead-Time Demand": x_vals, "Probability Density": y_vals})

            area_chart = (
                alt.Chart(dist_df)
                .mark_area(opacity=0.35, color="#2563EB")
                .encode(
                    x=alt.X("Lead-Time Demand:Q", title="Lead-Time Demand (units)"),
                    y=alt.Y("Probability Density:Q", title="Probability Density"),
                    tooltip=[
                        alt.Tooltip("Lead-Time Demand:Q", format=",.1f"),
                        alt.Tooltip("Probability Density:Q", format=",.4f"),
                    ],
                )
            )

            # ROP and Mean reference lines
            ref_data = pd.DataFrame([
                {"x": mu_lt, "Label": "Expected Demand (d*L)"},
                {"x": rop_val, "Label": f"Reorder Point (ROP = {rop_val:,.1f})"},
            ])
            rules = (
                alt.Chart(ref_data)
                .mark_rule(strokeWidth=2)
                .encode(
                    x="x:Q",
                    color=alt.Color("Label:N", scale=alt.Scale(domain=["Expected Demand (d*L)", f"Reorder Point (ROP = {rop_val:,.1f})"], range=["#64748B", "#DC2626"])),
                    strokeDash=alt.condition(alt.datum.x == rop_val, alt.value([4, 4]), alt.value([1, 0])),
                )
            )

            combined = (area_chart + rules).properties(height=360, title="Lead-Time Demand Distribution & Safety Stock Buffer")
            st.altair_chart(combined, use_container_width=True)
            st.caption(
                f"💡 **Takeaway**: The distribution $\\mathcal{{N}}({mu_lt:,.1f}, {sigma_lt:,.2f}^2)$ reflects demand variability during lead time. "
                f"The safety stock buffer of **{ss_val:,.1f} units** shifts the reorder point to **{rop_val:,.1f} units**, guaranteeing the target service level."
            )
        else:
            st.info("Demand standard deviation or lead-time parameters not variable.")

    elif model_name in ("fixed_period_review", "periodic_review"):
        d_val = inputs_ext.get("avg_demand", 0.0)
        t_val = inputs_ext.get("review_period", 0.0)
        l_val = inputs_ext.get("lead_time", 0.0)
        ss_val = final.get("safety_stock", 0.0)
        tgt_val = final.get("target_inventory", 0.0)
        ip_val = final.get("inventory_position", 0.0)
        q_val = final.get("order_quantity", 0.0)

        comp_data = pd.DataFrame([
            {"Component": "1. Review Period Demand (d*T)", "Units": d_val * t_val},
            {"Component": "2. Lead-Time Demand (d*L)", "Units": d_val * l_val},
            {"Component": "3. Safety Stock (SS)", "Units": ss_val},
            {"Component": "4. Target Level (M = d*(T+L) + SS)", "Units": tgt_val},
            {"Component": "5. Current Position (IP)", "Units": ip_val},
            {"Component": "6. Net Order Quantity (Q)", "Units": q_val},
        ])

        chart = (
            alt.Chart(comp_data)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#2563EB")
            .encode(
                x=alt.X("Units:Q", title="Quantity (Units)"),
                y=alt.Y("Component:N", sort=None, title=""),
                tooltip=[alt.Tooltip("Component:N"), alt.Tooltip("Units:Q", format=",.1f")],
            )
            .properties(height=320, title="Fixed-Period Review Inventory Decomposition")
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption("💡 **Takeaway**: Order quantity is calculated dynamically as $Q = \\max(0, M - IP)$ to cover the full protection period $(T + L)$.")

    elif model_name == "newsvendor":
        mean_d = inter.get("mean_demand")
        std_d = inter.get("std_demand")
        q_star = final.get("recommended_order_quantity")

        if mean_d and std_d and q_star:
            x_vals = np.linspace(max(0, mean_d - 3.5 * std_d), mean_d + 3.5 * std_d, 100)
            y_vals = norm.pdf(x_vals, mean_d, std_d)
            nv_df = pd.DataFrame({"Demand": x_vals, "Density": y_vals})

            base_curve = (
                alt.Chart(nv_df)
                .mark_line(color="#1E3A8A", strokeWidth=2.5)
                .encode(x=alt.X("Demand:Q", title="Customer Demand (Units)"), y=alt.Y("Density:Q", title="Probability Density"))
            )
            q_rule = (
                alt.Chart(pd.DataFrame([{"Q": q_star}]))
                .mark_rule(color="#DC2626", strokeDash=[4, 4], strokeWidth=2)
                .encode(x="Q:Q")
            )
            st.altair_chart((base_curve + q_rule).properties(height=340, title="Newsvendor Demand PDF & Optimal Order Fractile Q*"), use_container_width=True)
            st.caption(
                f"💡 **Takeaway**: Order quantity **Q* = {q_star:,.1f}** satisfies the critical fractile $P(D \\le Q^*) = {final.get('critical_ratio', 0):.4f}$."
            )
        else:
            st.info("Continuous demand distribution parameters not specified.")


# =====================================================================
# Tab 3: Interactive Parameter Sensitivity Sandbox
# =====================================================================
def render_sensitivity_tab(response: Any) -> None:
    """Allow students and decision-makers to simulate what-if scenarios using deterministic tools."""
    import streamlit as st

    model_name = response.identified_model
    calc = response.calculation_results
    inputs_ext = calc.get("inputs_extracted", {})
    final = calc.get("final_answer", {})

    st.markdown("### 🧪 Interactive Parameter Sensitivity Sandbox")
    st.info(
        "💡 **Simulation Sandbox**: Adjust operational parameters below to evaluate real-time decision impacts. "
        "All calculations strictly execute the underlying deterministic Python tools."
    )

    if model_name == "eoq":
        base_d = inputs_ext.get("annual_demand", 10000.0)
        base_s = inputs_ext.get("ordering_cost", 75.0)
        base_h = calc.get("intermediate_calculations", {}).get("holding_cost_per_unit") or inputs_ext.get("holding_cost", 6.0)

        col1, col2, col3 = st.columns(3)
        sim_d = col1.slider("Simulated Demand (D)", min_value=float(base_d * 0.4), max_value=float(base_d * 2.0), value=float(base_d), step=float(base_d * 0.05))
        sim_s = col2.slider("Simulated Order Cost (S)", min_value=float(base_s * 0.3), max_value=float(base_s * 2.5), value=float(base_s), step=float(max(1.0, base_s * 0.05)))
        sim_h = col3.slider("Simulated Holding Cost (H)", min_value=float(max(0.5, base_h * 0.3)), max_value=float(base_h * 2.5), value=float(base_h), step=float(max(0.25, base_h * 0.05)))

        # Deterministic simulation execution
        sim_res = calculate_eoq(annual_demand=sim_d, ordering_cost=sim_s, holding_cost=sim_h)
        sim_q = sim_res["final_answer"]["eoq"]
        sim_trc = sim_res["final_answer"]["total_relevant_cost"]

        base_q = final.get("eoq", sim_q)
        base_trc = final.get("total_relevant_cost", sim_trc)

        m1, m2, m3 = st.columns(3)
        m1.metric("Simulated EOQ (Q*)", f"{sim_q:,.1f} units", f"{((sim_q - base_q) / base_q) * 100:+.1f}% vs baseline")
        m2.metric("Simulated Total Cost", f"${sim_trc:,.2f}", f"{((sim_trc - base_trc) / base_trc) * 100:+.1f}% vs baseline")
        m3.metric("Simulated Orders/Year", f"{sim_res['final_answer']['orders_per_year']:,.2f}")

    elif model_name in ("safety_stock_rop", "rop"):
        base_d = inputs_ext.get("avg_demand", 150.0)
        base_lt = inputs_ext.get("lead_time", 4.0)
        base_sd = inputs_ext.get("std_demand", 25.0)
        base_slt = inputs_ext.get("std_lead_time", 0.0)
        base_csl = inputs_ext.get("service_level", 0.95)

        col1, col2 = st.columns(2)
        sim_csl = col1.slider("Target Service Level (CSL)", min_value=0.80, max_value=0.999, value=float(base_csl), step=0.005, format="%.3f")
        sim_lt = col2.slider("Replenishment Lead Time (L)", min_value=1.0, max_value=float(base_lt * 2.5), value=float(base_lt), step=1.0)

        sim_res = calculate_rop(
            avg_demand=base_d,
            lead_time=sim_lt,
            std_demand=base_sd,
            std_lead_time=base_slt,
            service_level=sim_csl,
        )

        sim_ss = sim_res["final_answer"]["safety_stock"]
        sim_rop = sim_res["final_answer"]["reorder_point"]
        base_ss = final.get("safety_stock", sim_ss)
        base_rop = final.get("reorder_point", sim_rop)

        m1, m2, m3 = st.columns(3)
        m1.metric("Simulated Safety Stock", f"{sim_ss:,.1f} units", f"{((sim_ss - base_ss) / max(0.1, base_ss)) * 100:+.1f}%")
        m2.metric("Simulated Reorder Point", f"{sim_rop:,.1f} units", f"{((sim_rop - base_rop) / base_rop) * 100:+.1f}%")
        m3.metric("Normal z-score", f"{sim_res['intermediate_calculations']['z_score']:.4f}")

    elif model_name == "newsvendor":
        p_val = inputs_ext.get("selling_price", 120.0)
        c_val = inputs_ext.get("unit_cost", 70.0)
        s_val = inputs_ext.get("salvage_value", 30.0)
        mean_d = inputs_ext.get("mean_demand", 500.0)
        std_d = inputs_ext.get("std_demand", 100.0)

        col1, col2 = st.columns(2)
        sim_p = col1.slider("Selling Price (p)", min_value=float(c_val + 5.0), max_value=float(p_val * 2.0), value=float(p_val), step=5.0)
        sim_s = col2.slider("Salvage Value (s)", min_value=0.0, max_value=float(c_val - 1.0), value=float(s_val), step=2.0)

        sim_res = calculate_newsvendor(
            mean_demand=mean_d,
            std_demand=std_d,
            selling_price=sim_p,
            unit_cost=c_val,
            salvage_value=sim_s,
        )
        sim_q = sim_res["final_answer"]["recommended_order_quantity"]
        sim_cr = sim_res["final_answer"]["critical_ratio"]
        base_q = final.get("recommended_order_quantity", sim_q)

        m1, m2, m3 = st.columns(3)
        m1.metric("Simulated Stocking (Q*)", f"{sim_q:,.1f} units", f"{((sim_q - base_q) / base_q) * 100:+.1f}%")
        m2.metric("Critical Ratio", f"{sim_cr:.4f}")
        m3.metric("Underage Cost (Cu)", f"${sim_res['final_answer']['underage_cost']:,.2f}")

    elif model_name in ("fixed_period_review", "periodic_review"):
        base_d = inputs_ext.get("avg_demand", 80.0)
        base_sd = inputs_ext.get("std_demand", 15.0)
        base_t = inputs_ext.get("review_period", 14.0)
        base_l = inputs_ext.get("lead_time", 3.0)
        base_ip = inputs_ext.get("inventory_position", 320.0)
        base_csl = inputs_ext.get("service_level", 0.98)

        col1, col2 = st.columns(2)
        sim_t = col1.slider("Review Cycle (T in days)", min_value=1.0, max_value=float(base_t * 2.0), value=float(base_t), step=1.0)
        sim_csl = col2.slider("Service Level", min_value=0.85, max_value=0.999, value=float(base_csl), step=0.005)

        sim_res = calculate_periodic_review(
            avg_demand=base_d,
            std_demand=base_sd,
            review_period=sim_t,
            lead_time=base_l,
            inventory_position=base_ip,
            service_level=sim_csl,
        )
        sim_q = sim_res["final_answer"]["order_quantity"]
        sim_tgt = sim_res["final_answer"]["target_inventory"]

        m1, m2 = st.columns(2)
        m1.metric("Simulated Order (Q)", f"{sim_q:,.1f} units")
        m2.metric("Simulated Target (M)", f"{sim_tgt:,.1f} units")
    else:
        st.info("Interactive parameter sensitivity is available for EOQ, ROP, Periodic Review, and Newsvendor.")


# =====================================================================
# Multicriteria Inventory Classification UI Section (BAN 517)
# =====================================================================
def render_multicriteria_classification_section(clean_df: pd.DataFrame) -> None:
    """
    Render the comprehensive multicriteria inventory classification dashboard (BAN 517).
    Includes:
    - Boundary convention selector (strict, straddle, rounded)
    - Theoretical framework callout (Pareto thresholds, scoring, weighted formula)
    - KPI metric summary cards
    - Required Classification Summary Table (Final Class, Number of Items, % of Items, Total Sales $, % of Total Sales)
    - 4 Altair analytical charts:
        1. Number of Items by Final Class
        2. Total Sales ($) by Final Class
        3. Multicriteria Cross-Tabulation Matrix Heatmap (Sales Class vs Hits Class)
        4. Cumulative Sales (%) Pareto Curve with threshold reference lines
    - Full 15-column interactive data table with filters (Class, Search) and CSV download
    """
    import streamlit as st

    st.markdown("## 🏷️ Multicriteria Inventory Classification")
    st.markdown(
        "<div style='font-size: 0.95rem; color: #4B5563; margin-bottom: 1.25rem;'>"
        "Implementation of the BAN 517 multicriteria inventory classification methodology based on independent "
        "Pareto analysis of <b>Annual Sales ($)</b> and <b>Point-of-Sale (POS) Hits</b>, weighted categorical scoring, "
        "and matrix-based final classification."
        "</div>",
        unsafe_allow_html=True,
    )

    # Finalized Methodology & Continuous Ranges Callout
    with st.expander("📖 View Finalized Classification Methodology & Scoring Rules", expanded=True):
        f_col1, f_col2 = st.columns([1.2, 1.8])
        with f_col1:
            st.markdown(
                "**Continuous Pareto Cumulative Ranges**:\n"
                "- **Class A**: 0% through 60%, inclusive ($\\le 60\\%$)\n"
                "- **Class B**: Greater than 60% through 80%, inclusive ($>60\\% \\text{ to } \\le 80\\%$)\n"
                "- **Class C**: Greater than 80% through 90%, inclusive ($>80\\% \\text{ to } \\le 90\\%$)\n"
                "- **Class D**: Greater than 90% through 100%, inclusive ($>90\\% \\text{ to } \\le 100\\%$)\n\n"
                "*Classifications are assigned based on the cumulative percentage after including each ranked item.*"
            )
        with f_col2:
            st.markdown(
                "**Scoring & Multicriteria Classification**:\n"
                "- **Numerical Scores**: $A = 4,\\, B = 3,\\, C = 2,\\, D = 1$\n"
                "- **Combined Weighted Score**: $0.65 \\times \\text{Sales Score} + 0.35 \\times \\text{Hits Score}$\n"
                "- **Final Thresholds**: **A** [3.25 – 4.00], **B** [2.50 – 3.24], **C** [1.75 – 2.49], **D** [< 1.75]\n"
                "- *Professor Example*: Sales A (4) & Hits B (3) $\\rightarrow 0.65(4) + 0.35(3) = 3.65 \\rightarrow \\text{Final Class A}$"
            )

    # Perform deterministic multicriteria classification using finalized continuous ranges
    results = classify_inventory_multicriteria(clean_df)
    classified_df = results["classified_df"]
    summary_df = results["summary_df"]
    reconciliation = results["reconciliation"]
    score_matrix = results["score_matrix"]

    # 1. Dataset Summary KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Items", f"{results['total_items']:,}")
    m2.metric("Total Annual Sales", f"${results['total_sales']:,.2f}")
    m3.metric("Total POS Hits", f"{results['total_hits']:,}")
    m4.metric(
        "Data Reconciliation",
        "✅ 100% Balanced" if reconciliation["is_reconciled"] else "⚠️ Unbalanced",
        f"{reconciliation['reconciled_items']} items accounted for",
    )

    st.markdown("---")

    # 2. Required Classification Summary Table
    st.markdown("### 📋 Required Classification Summary Table")
    st.caption("Aggregated distribution of inventory items and annual sales volume across Final Inventory Classes.")

    st.dataframe(
        summary_df,
        column_config={
            "Final Class": st.column_config.TextColumn("Final Class"),
            "Number of Items": st.column_config.NumberColumn("Number of Items", format="%d"),
            "Percentage of Items (%)": st.column_config.NumberColumn("Percentage of Items (%)", format="%.2f%%"),
            "Total Sales ($)": st.column_config.NumberColumn("Total Sales ($)", format="$%,.2f"),
            "Percentage of Total Sales (%)": st.column_config.NumberColumn("Percentage of Total Sales (%)", format="%.2f%%"),
        },
        use_container_width=True,
        hide_index=True,
    )

    # Summary table reconciliation verification badges
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.caption(
            f"✓ **Items Total**: **{summary_df['Number of Items'].sum():,} items** "
            f"({summary_df['Percentage of Items (%)'].sum():.2f}%)"
        )
    with v_col2:
        st.caption(
            f"✓ **Sales Total**: **${summary_df['Total Sales ($)'].sum():,.2f}** "
            f"({summary_df['Percentage of Total Sales (%)'].sum():.2f}%)"
        )

    st.markdown("---")

    # 3. Four Visualizations
    st.markdown("### 📊 Multicriteria Visual Analytics")

    v_tab1, v_tab2, v_tab3 = st.tabs([
        "📊 Category Distribution (Items & Sales)",
        "🔲 Multicriteria Cross-Tabulation Matrix",
        "📈 Cumulative Sales Pareto Curve",
    ])

    class_color_scale = alt.Scale(
        domain=["A", "B", "C", "D"],
        range=["#1E3A8A", "#0D9488", "#D97706", "#6B7280"],
    )

    with v_tab1:
        c1_col, c2_col = st.columns(2)
        with c1_col:
            st.markdown("##### 📦 Number of Items by Final Class")
            chart_items = (
                alt.Chart(summary_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Final Class:N", sort=["A", "B", "C", "D"], title="Final Inventory Class"),
                    y=alt.Y("Number of Items:Q", title="Item Count (SKUs)"),
                    color=alt.Color("Final Class:N", scale=class_color_scale, legend=None),
                    tooltip=[
                        alt.Tooltip("Final Class:N", title="Class"),
                        alt.Tooltip("Number of Items:Q", title="Items", format=","),
                        alt.Tooltip("Percentage of Items (%):Q", title="Share (%)", format=".2f"),
                    ],
                )
                .properties(height=320)
            )
            text_items = chart_items.mark_text(dy=-10, fontWeight="bold", fontSize=12).encode(
                text=alt.Text("Number of Items:Q", format=",")
            )
            st.altair_chart(chart_items + text_items, use_container_width=True)

        with c2_col:
            st.markdown("##### 💰 Total Sales ($) by Final Class")
            chart_sales = (
                alt.Chart(summary_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Final Class:N", sort=["A", "B", "C", "D"], title="Final Inventory Class"),
                    y=alt.Y("Total Sales ($):Q", title="Total Sales ($)"),
                    color=alt.Color("Final Class:N", scale=class_color_scale, legend=None),
                    tooltip=[
                        alt.Tooltip("Final Class:N", title="Class"),
                        alt.Tooltip("Total Sales ($):Q", title="Total Sales", format="$,.2f"),
                        alt.Tooltip("Percentage of Total Sales (%):Q", title="Sales Share (%)", format=".2f"),
                    ],
                )
                .properties(height=320)
            )
            text_sales = chart_sales.mark_text(dy=-10, fontWeight="bold", fontSize=11).encode(
                text=alt.Text("Total Sales ($):Q", format="$,.0f")
            )
            st.altair_chart(chart_sales + text_sales, use_container_width=True)

    with v_tab2:
        st.markdown("##### 🔲 Multicriteria Co-Occurrence Matrix (Sales Class vs POS Hits Class)")
        st.caption(
            "Matrix cross-tabulating item distribution across independent Sales ($) and POS Hits classifications. "
            "Cells show number of items and their Combined Weighted Score."
        )

        matrix_long = (
            score_matrix.reset_index()
            .melt(id_vars="Sales Class", var_name="Hits Class", value_name="Item Count")
        )

        matrix_long["Score"] = matrix_long.apply(
            lambda row: round(0.65 * SCORE_MAP[row["Sales Class"]] + 0.35 * SCORE_MAP[row["Hits Class"]], 2),
            axis=1,
        )
        matrix_long["Result Class"] = matrix_long["Score"].apply(assign_final_class_from_score)

        heat_base = (
            alt.Chart(matrix_long)
            .mark_rect(cornerRadius=4)
            .encode(
                x=alt.X("Hits Class:N", sort=["A", "B", "C", "D"], title="POS Hits Class (Volume Frequency)"),
                y=alt.Y("Sales Class:N", sort=["A", "B", "C", "D"], title="Sales ($) Class (Revenue Value)"),
                color=alt.Color(
                    "Item Count:Q",
                    scale=alt.Scale(scheme="blues"),
                    title="Item Count",
                ),
                tooltip=[
                    alt.Tooltip("Sales Class:N", title="Sales Class"),
                    alt.Tooltip("Hits Class:N", title="Hits Class"),
                    alt.Tooltip("Item Count:Q", title="SKU Count"),
                    alt.Tooltip("Score:Q", title="Weighted Score", format=".2f"),
                    alt.Tooltip("Result Class:N", title="Final Class"),
                ],
            )
            .properties(height=360)
        )
        heat_text = heat_base.mark_text(fontSize=11, fontWeight="bold").encode(
            text=alt.Text("Item Count:Q", format="d"),
            color=alt.condition(alt.datum["Item Count"] > 30, alt.value("white"), alt.value("#1E293B")),
        )
        st.altair_chart(heat_base + heat_text, use_container_width=True)

    with v_tab3:
        st.markdown("##### 📈 Cumulative Sales Pareto Curve (Lorenz Curve)")
        st.caption("Visualizing the Pareto concentration (80/20 rule) of annual revenue across ranked items.")

        sorted_items = classified_df.sort_values(by="Sales ($)", ascending=False).reset_index(drop=True)
        sorted_items["Rank"] = range(1, len(sorted_items) + 1)
        sorted_items["Rank Percentage"] = (sorted_items["Rank"] / len(sorted_items)) * 100.0

        pareto_area = (
            alt.Chart(sorted_items)
            .mark_area(opacity=0.25, color="#1E3A8A")
            .encode(
                x=alt.X("Rank:Q", title="Item Rank (Sorted Descending by Sales)"),
                y=alt.Y("Cumulative Sales Percentage:Q", title="Cumulative Sales Percentage (%)", scale=alt.Scale(domain=[0, 100])),
            )
        )
        pareto_line = (
            alt.Chart(sorted_items)
            .mark_line(strokeWidth=2.5, color="#1E3A8A")
            .encode(
                x=alt.X("Rank:Q"),
                y=alt.Y("Cumulative Sales Percentage:Q"),
                tooltip=[
                    alt.Tooltip("Rank:Q", title="Rank"),
                    alt.Tooltip("Item Name:N", title="Item"),
                    alt.Tooltip("Sales ($):Q", title="Sales", format="$,.2f"),
                    alt.Tooltip("Cumulative Sales Percentage:Q", title="Cum Sales %", format=".2f"),
                    alt.Tooltip("Final Inventory Class:N", title="Final Class"),
                ],
            )
        )

        thresholds = pd.DataFrame([
            {"Threshold": 60, "Class": "Class A Threshold (60%)", "Color": "#DC2626"},
            {"Threshold": 80, "Class": "Class B Threshold (80%)", "Color": "#D97706"},
            {"Threshold": 90, "Class": "Class C Threshold (90%)", "Color": "#0D9488"},
        ])
        thresh_lines = (
            alt.Chart(thresholds)
            .mark_rule(strokeDash=[6, 4], strokeWidth=1.5)
            .encode(
                y=alt.Y("Threshold:Q"),
                color=alt.Color("Class:N", scale=alt.Scale(
                    domain=["Class A Threshold (60%)", "Class B Threshold (80%)", "Class C Threshold (90%)"],
                    range=["#DC2626", "#D97706", "#0D9488"],
                ), title="Pareto Boundary Lines"),
            )
        )

        pareto_chart = (pareto_area + pareto_line + thresh_lines).properties(height=360)
        st.altair_chart(pareto_chart, use_container_width=True)

    st.markdown("---")

    # 4. Interactive 15-Column Data Table with Search, Filter & CSV Export
    st.markdown("### 📑 Full Multicriteria Classified Inventory Dataset (15 Required Columns)")
    st.caption("Complete item-level classification records with intermediate percentages, scores, and final classifications.")

    filter_col, search_col, dl_col = st.columns([1.5, 2.5, 1.5])
    with filter_col:
        selected_class = st.selectbox(
            "Filter by Final Class:",
            options=["All", "A", "B", "C", "D"],
            help="Filter items displayed in the interactive table below by their Final Inventory Class.",
        )
    with search_col:
        search_query = st.text_input(
            "🔍 Search Item Number or Item Name:",
            value="",
            placeholder="e.g. 10025 or Drill...",
        )
    with dl_col:
        export_csv_bytes = classified_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=export_csv_bytes,
            file_name="multicriteria_inventory_classification.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download the complete 15-column classified inventory dataset as a CSV file.",
        )

    # Filter data table
    display_df = classified_df.copy()
    if selected_class != "All":
        display_df = display_df[display_df["Final Inventory Class"] == selected_class]
    if search_query.strip():
        q_clean = search_query.strip().lower()
        display_df = display_df[
            display_df["Item Number"].astype(str).str.lower().str.contains(q_clean)
            | display_df["Item Name"].astype(str).str.lower().str.contains(q_clean)
        ]

    st.caption(f"Displaying **{len(display_df):,}** of **{len(classified_df):,}** records.")

    st.dataframe(
        display_df,
        column_config={
            "Item Number": st.column_config.TextColumn("Item Number"),
            "Item Name": st.column_config.TextColumn("Item Name"),
            "UoM": st.column_config.TextColumn("UoM"),
            "Sales ($)": st.column_config.NumberColumn("Sales ($)", format="$%,.2f"),
            "Hits": st.column_config.NumberColumn("Hits", format="%d"),
            "Sales Percentage": st.column_config.NumberColumn("Sales %", format="%.2f%%"),
            "Cumulative Sales Percentage": st.column_config.NumberColumn("Cum Sales %", format="%.2f%%"),
            "Sales Class": st.column_config.TextColumn("Sales Class"),
            "Hits Percentage": st.column_config.NumberColumn("Hits %", format="%.2f%%"),
            "Cumulative Hits Percentage": st.column_config.NumberColumn("Cum Hits %", format="%.2f%%"),
            "Hits Class": st.column_config.TextColumn("Hits Class"),
            "Sales Score": st.column_config.NumberColumn("Sales Score", format="%d"),
            "Hits Score": st.column_config.NumberColumn("Hits Score", format="%d"),
            "Combined Weighted Score": st.column_config.NumberColumn("Weighted Score", format="%.2f"),
            "Final Inventory Class": st.column_config.TextColumn("Final Class"),
        },
        use_container_width=True,
        hide_index=True,
    )


# =====================================================================
# Main Streamlit Application Entrypoint
# =====================================================================
def run_streamlit_app() -> None:
    """Launch the modernized Streamlit analytical dashboard."""
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Analytics • Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_styles()

    # Application Header
    st.markdown("<div class='main-header'>📦 Supply Chain Analytics Inventory Decision Agent</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Graduate-level Decision Support System for inventory optimization, stochastic safety stock, "
        "and multi-echelon replenishment analytics.</div>",
        unsafe_allow_html=True,
    )

    agent = InventoryDecisionAgent()

    # -------------------------------------------------------------
    # Section 5: Modernized Sidebar Navigation
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 📦 Inventory Decision Agent")
        st.caption("Graduate-Level Business Analytics Decision Support System")

        with st.expander("ℹ️ Supported Inventory Models", expanded=False):
            for _, meta in MODEL_METADATA.items():
                st.markdown(f"• **{meta['display_name']}**\n`{meta['tool_name']}()`")

        with st.expander("🔄 7-Stage Decision Pipeline", expanded=False):
            st.markdown(
                "1. **Customer Question**\n"
                "2. **Identify Model**\n"
                "3. **Extract Inputs**\n"
                "4. **Select Python Tool**\n"
                "5. **Calculate** *(Python)*\n"
                "6. **Verify** *(Optimality)*\n"
                "7. **Explain** *(Pedagogical)*"
            )

        st.divider()

        # Section 2: Quick-Start Scenarios
        st.markdown("### ⚡ Quick-Start Scenarios")
        selected_sample = st.selectbox(
            "Select an industry scenario:",
            options=["Custom Question"] + list(SAMPLE_QUESTIONS.keys()),
            help="Choose a pre-configured supply chain problem to load into the decision agent.",
        )

        if selected_sample != "Custom Question":
            st.caption(f"📌 {SCENARIO_DESCRIPTIONS.get(selected_sample, '')}")

        st.divider()

        # Section 3: Data Upload (At the Bottom)
        st.markdown("### 📁 Data Upload")
        st.caption("Upload an inventory CSV for immediate batch analysis or multicriteria classification.")
        uploaded_csv = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="File must contain either: 1) BAN 517 Multicriteria Dataset (Item Number, Item Name, UoM, Sales ($), Hits) or 2) Standard Inventory Dataset (Product, Category, Units, Unit Cost)",
        )
        load_course_data = st.checkbox(
            "📂 Load Course Project Dataset (ProjectDataV2_UTF8.csv)",
            value=False,
            help="Instantly load the BAN 517 course project dataset (205 items) to inspect multicriteria classification.",
        )
        st.caption("Supports BAN 517 multicriteria dataset or general inventory dataset.")

    # Populate question box
    default_text = (
        SAMPLE_QUESTIONS[selected_sample]
        if selected_sample != "Custom Question"
        else SAMPLE_QUESTIONS["1. Economic Order Quantity (EOQ)"]
    )

    # -------------------------------------------------------------
    # Natural-Language Decision Agent Input
    # -------------------------------------------------------------
    with st.container(border=True):
        st.markdown("##### 💬 Natural-Language Inventory Question")
        question = st.text_area(
            "Enter your supply chain inventory question:",
            value=default_text,
            height=110,
            label_visibility="collapsed",
            help="State your inventory question naturally. The agent extracts parameters and selects the optimal analytical model.",
        )

        btn_col, info_col = st.columns([1, 3])
        with btn_col:
            run_clicked = st.button("🚀 Run Inventory Decision Agent", type="primary", use_container_width=True)
        with info_col:
            st.caption("Executes natural-language parsing, deterministic Python calculation, and verification checks.")

    # -------------------------------------------------------------
    # Calculation Pipeline Execution
    # -------------------------------------------------------------
    if run_clicked:
        if not question.strip():
            st.warning("Please enter an inventory question.")
            return

        try:
            with st.spinner("Executing deterministic inventory model and verification pipeline..."):
                response = agent.ask(question)

            st.divider()

            # Render 7-Stage Visual Badges
            render_pipeline_badges(response)

            # Render Section 2 & 3: Executive Decision Summary & Model-Specific KPIs
            render_executive_summary_dashboard(response)

            st.divider()

            # Section 4: Three Organized Tabs
            tab_decision, tab_analytics, tab_sensitivity = st.tabs([
                "📋 Decision Analysis & Verification",
                "📈 Visual Analytics & Curves",
                "🧪 Parameter Sensitivity Analysis",
            ])

            with tab_decision:
                render_decision_analysis_tab(response)

            with tab_analytics:
                render_visual_analytics_tab(response)

            with tab_sensitivity:
                render_sensitivity_tab(response)

        except Exception as exc:
            st.error(f"❌ Error processing question: {exc}")

    # -------------------------------------------------------------
    # Section 6: CSV Data Upload and Analysis Section
    # -------------------------------------------------------------
    active_csv_file = None
    data_source_label = ""
    if uploaded_csv is not None:
        active_csv_file = uploaded_csv
        data_source_label = getattr(uploaded_csv, "name", "Uploaded CSV")
    elif load_course_data:
        course_file_path = "ProjectDataV2_UTF8.csv"
        if os.path.exists(course_file_path):
            with open(course_file_path, "rb") as f:
                active_csv_file = io.BytesIO(f.read())
            data_source_label = "ProjectDataV2_UTF8.csv (Course Dataset)"
        else:
            st.warning("⚠️ Course dataset 'ProjectDataV2_UTF8.csv' not found in workspace directory.")

    if active_csv_file is not None:
        st.divider()
        try:
            if hasattr(active_csv_file, "seek"):
                active_csv_file.seek(0)
            df_raw = pd.read_csv(active_csv_file)
            if hasattr(active_csv_file, "seek"):
                active_csv_file.seek(0)

            is_mc, missing_mc, clean_mc_df = validate_classification_dataframe(df_raw)
            clean_headers = [str(c).strip() for c in df_raw.columns]
            is_gen = all(col in clean_headers for col in REQUIRED_COLUMNS)

            if is_mc:
                render_multicriteria_classification_section(clean_mc_df)
            elif is_gen:
                st.markdown(f"## 📊 Uploaded Inventory Data Analysis ({data_source_label})")
                csv_results = process_and_analyze_inventory_csv(active_csv_file)

                if not csv_results["valid"]:
                    st.error(
                        f"❌ Missing required column(s): **{', '.join(csv_results['missing_columns'])}**.\n\n"
                        f"Required schema columns: `{'`, `'.join(REQUIRED_COLUMNS)}`."
                    )
                    st.subheader("Uploaded Raw Dataframe Preview")
                    st.dataframe(csv_results["df"], use_container_width=True)
                    st.markdown(f"- **Number of Rows:** {csv_results['num_rows']:,}")
                    st.markdown(f"- **Number of Columns:** {csv_results['num_cols']:,}")
                    st.markdown(f"- **Column Names:** `{', '.join(csv_results['columns'])}`")
                else:
                    # Dataset Summary KPI Cards
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total Records (Rows)", f"{csv_results['num_rows']:,}")
                    k2.metric("Total Units in Stock", f"{csv_results['total_units']:,.0f}")
                    k3.metric("Average Unit Cost", f"${csv_results['avg_unit_cost']:,.2f}")
                    k4.metric("Total Inventory Value", f"${csv_results['total_inventory_value']:,.2f}")

                    # Top product banner
                    st.info(
                        f"🏆 **Top Stocked Item**: **{csv_results['highest_units_product']}** with "
                        f"**{csv_results['highest_units_count']:,.0f} units**."
                    )

                    # Charts and Breakdowns
                    c_tab1, c_tab2, c_tab3 = st.tabs([
                        "📊 Category Units Breakdown",
                        "🏆 Top Products by Value",
                        "📋 Full Dataset (with Inventory Value)",
                    ])

                    with c_tab1:
                        cat_df = csv_results["category_units"]
                        st.dataframe(
                            cat_df,
                            column_config={"Total Units": st.column_config.NumberColumn(format="%d")},
                            use_container_width=True,
                            hide_index=True,
                        )
                        if not cat_df.empty:
                            bar_chart = (
                                alt.Chart(cat_df)
                                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#2563EB")
                                .encode(
                                    x=alt.X("Total Units:Q", title="Total Units"),
                                    y=alt.Y("Category:N", sort="-x", title="Category"),
                                    tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Total Units:Q", format=",.0f")],
                                )
                                .properties(height=260, title="Units Distribution by Category")
                            )
                            st.altair_chart(bar_chart, use_container_width=True)

                    with c_tab2:
                        clean_df = csv_results["df"]
                        top10 = clean_df.sort_values(by="Inventory Value", ascending=False).head(10)
                        st.dataframe(
                            top10[["Product", "Category", "Units", "Unit Cost", "Inventory Value"]],
                            column_config={
                                "Unit Cost": st.column_config.NumberColumn(format="$%.2f"),
                                "Inventory Value": st.column_config.NumberColumn(format="$%.2f"),
                                "Units": st.column_config.NumberColumn(format="%d"),
                            },
                            use_container_width=True,
                            hide_index=True,
                        )
                        if not top10.empty:
                            val_chart = (
                                alt.Chart(top10)
                                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#0D9488")
                                .encode(
                                    x=alt.X("Inventory Value:Q", title="Total Inventory Value ($)"),
                                    y=alt.Y("Product:N", sort="-x", title="Product"),
                                    tooltip=[
                                        alt.Tooltip("Product:N"),
                                        alt.Tooltip("Units:Q", format=",.0f"),
                                        alt.Tooltip("Unit Cost:Q", format="$,.2f"),
                                        alt.Tooltip("Inventory Value:Q", format="$,.2f"),
                                    ],
                                )
                                .properties(height=280, title="Top 10 Products by Inventory Valuation ($)")
                            )
                            st.altair_chart(val_chart, use_container_width=True)

                    with c_tab3:
                        st.dataframe(
                            csv_results["df"],
                            column_config={
                                "Unit Cost": st.column_config.NumberColumn(format="$%.2f"),
                                "Inventory Value": st.column_config.NumberColumn(format="$%.2f"),
                                "Units": st.column_config.NumberColumn(format="%d"),
                            },
                            use_container_width=True,
                        )
            else:
                st.error(
                    f"❌ Uploaded dataset does not match any recognized schema.\n\n"
                    f"- For **Multicriteria Inventory Classification (BAN 517)**, missing columns: `{', '.join(missing_mc)}`\n"
                    f"- For **General Inventory Analysis**, missing columns: `{[c for c in REQUIRED_COLUMNS if c not in clean_headers]}`"
                )
                st.subheader("Uploaded Raw Dataframe Preview")
                st.dataframe(df_raw, use_container_width=True)
                st.markdown(f"- **Number of Rows:** {len(df_raw):,}")
                st.markdown(f"- **Number of Columns:** {len(df_raw.columns):,}")
                st.markdown(f"- **Column Names:** `{', '.join(df_raw.columns)}`")
        except Exception as exc:
            st.error(f"❌ Error processing uploaded CSV file: {exc}")


# =====================================================================
# Terminal CLI Demonstration
# =====================================================================
def run_cli_demo() -> None:
    """Run all 5 sample inventory questions in the terminal if executed directly via python app.py."""
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    agent = InventoryDecisionAgent()
    print("=" * 80)
    print("SUPPLY CHAIN ANALYTICS INVENTORY DECISION AGENT - CLI DEMO")
    print("Architecture: Question -> Identify Model -> Extract Inputs -> Select Tool -> Calculate -> Verify -> Explain")
    print("=" * 80)

    for label, question in SAMPLE_QUESTIONS.items():
        print(f"\n[{label}]")
        print(f"Question: {question}\n")
        resp = agent.ask(question)
        print(f"  -> Identified Model : {resp.identified_model} ({resp.model_display_name})")
        print(f"  -> Selected Tool    : {resp.selected_tool_name}()")
        print(f"  -> Extracted Inputs : {json.dumps(resp.extracted_inputs)}")
        print(f"  -> Final Answer     : {json.dumps(resp.calculation_results['final_answer'])}")
        print(f"  -> Verification     : {'PASSED' if resp.verification['is_valid'] else 'FAILED'}")
        print("-" * 80)
        try:
            print(resp.explanation)
        except UnicodeEncodeError:
            print(resp.explanation.encode("ascii", errors="replace").decode("ascii"))
        print("=" * 80)


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            run_streamlit_app()
        else:
            run_cli_demo()
    except ImportError:
        run_cli_demo()
