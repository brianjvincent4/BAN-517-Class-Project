"""
inventory_classification.py

Deterministic Multicriteria Inventory Classification Module for BAN 517.
Methodology:
1. Classify Sales ($) and POS Hits independently using Pareto thresholds (A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%).
2. Convert categorical classes to numerical scores (A=4, B=3, C=2, D=1).
3. Compute Combined Weighted Score = 0.65 * Sales Score + 0.35 * Hits Score.
4. Assign Final Inventory Class based on weighted score thresholds:
   - A: 3.25 to 4.00
   - B: 2.50 to 3.24
   - C: 1.75 to 2.49
   - D: < 1.75
5. Reconcile total items, category sales, and generate summary statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

CLASSIFICATION_REQUIRED_COLUMNS: List[str] = [
    "Item Number",
    "Item Name",
    "UoM",
    "Sales ($)",
    "Hits",
]

SCORE_MAP: Dict[str, int] = {
    "A": 4,
    "B": 3,
    "C": 2,
    "D": 1,
}

WEIGHT_SALES: float = 0.65
WEIGHT_HITS: float = 0.35


def validate_classification_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    Validate that the uploaded dataframe contains the required columns for
    multicriteria inventory classification. Normalizes whitespace and standardizes column names.
    """
    clean_df = df.copy()
    clean_df.columns = [str(c).strip() for c in clean_df.columns]

    # Map possible synonyms if user uploads alternative header cases
    header_map = {}
    for col in clean_df.columns:
        c_low = col.lower().replace("_", " ").replace("-", " ").strip()
        if c_low in ("item number", "item no", "item #", "item_num", "itemnum"):
            header_map[col] = "Item Number"
        elif c_low in ("item name", "name", "description", "item description"):
            header_map[col] = "Item Name"
        elif c_low in ("uom", "unit of measure", "unit"):
            header_map[col] = "UoM"
        elif c_low in ("sales ($)", "sales", "sales revenue", "total sales", "annual sales ($)", "annual sales"):
            header_map[col] = "Sales ($)"
        elif c_low in ("hits", "pos hits", "point of sale hits", "frequency", "pos_hits"):
            header_map[col] = "Hits"

    clean_df = clean_df.rename(columns=header_map)
    missing = [c for c in CLASSIFICATION_REQUIRED_COLUMNS if c not in clean_df.columns]
    return (len(missing) == 0, missing, clean_df)


def assign_pareto_class(cum_pct: float, rank_idx: int = 0, boundary_rule: str = "continuous") -> str:
    """
    Assign Pareto class based on continuous cumulative percentage ranges:
    - Class A: 0% through 60%, inclusive (cum_pct <= 60.0 or rank_idx == 0)
    - Class B: Greater than 60% through 80%, inclusive (60.0 < cum_pct <= 80.0)
    - Class C: Greater than 80% through 90%, inclusive (80.0 < cum_pct <= 90.0)
    - Class D: Greater than 90% through 100%, inclusive (90.0 < cum_pct <= 100.0)

    Eliminates numerical gaps in the classification table while ensuring all items are assigned.
    """
    if boundary_rule == "rounded":
        val = round(cum_pct)
        if val <= 60 or rank_idx == 0:
            return "A"
        elif val <= 80:
            return "B"
        elif val <= 90:
            return "C"
        else:
            return "D"
    else:  # 'continuous' / standard upper-bound
        if cum_pct <= 60.0 or rank_idx == 0:
            return "A"
        elif cum_pct <= 80.0:
            return "B"
        elif cum_pct <= 90.0:
            return "C"
        else:
            return "D"


def assign_final_class_from_score(score: float) -> str:
    """
    Assign final inventory class based on Combined Weighted Score:
    - A: 3.25 to 4.00 (score >= 3.25)
    - B: 2.50 to 3.24 (2.50 <= score < 3.25)
    - C: 1.75 to 2.49 (1.75 <= score < 2.50)
    - D: Below 1.75   (score < 1.75)
    """
    # Use full numeric precision comparison
    if score >= 3.25:
        return "A"
    elif score >= 2.50:
        return "B"
    elif score >= 1.75:
        return "C"
    else:
        return "D"


def classify_inventory_multicriteria(
    df: pd.DataFrame,
    boundary_rule: str = "continuous",
) -> Dict[str, Any]:
    """
    Execute the complete multicriteria inventory classification procedure.

    Returns:
    - 'classified_df': DataFrame with all 15 required columns
    - 'summary_df': Category summary table (Class, Items, % Items, Sales $, % Sales)
    - 'score_matrix': Cross-tabulation of Sales Class vs Hits Class
    - 'reconciliation': Validation checks ensuring 100% data reconciliation
    - 'boundary_rule_used': The boundary convention applied
    """
    is_valid, missing, clean_df = validate_classification_dataframe(df)
    if not is_valid:
        raise ValueError(f"Missing required columns for classification: {missing}")

    # Work on a copy and clean numeric columns
    work_df = clean_df.copy()

    # Clean Sales ($)
    if work_df["Sales ($)"].dtype == object or str(work_df["Sales ($)"].dtype).startswith("string"):
        work_df["Sales ($)"] = pd.to_numeric(
            work_df["Sales ($)"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce",
        ).fillna(0.0)
    else:
        work_df["Sales ($)"] = pd.to_numeric(work_df["Sales ($)"], errors="coerce").fillna(0.0)

    # Clean Hits
    if work_df["Hits"].dtype == object or str(work_df["Hits"].dtype).startswith("string"):
        work_df["Hits"] = pd.to_numeric(
            work_df["Hits"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce",
        ).fillna(0.0)
    else:
        work_df["Hits"] = pd.to_numeric(work_df["Hits"], errors="coerce").fillna(0.0)

    total_sales = float(work_df["Sales ($)"].sum())
    total_hits = float(work_df["Hits"].sum())

    if total_sales <= 0:
        raise ValueError("Total Sales ($) must be strictly positive.")
    if total_hits <= 0:
        raise ValueError("Total POS Hits must be strictly positive.")

    # -------------------------------------------------------------
    # Step A1: Sales Classification (Rank items highest to lowest)
    # -------------------------------------------------------------
    sales_sorted = work_df[["Item Number", "Sales ($)"]].sort_values(
        by="Sales ($)", ascending=False
    ).reset_index(drop=True)

    sales_sorted["Sales Percentage"] = (sales_sorted["Sales ($)"] / total_sales) * 100.0
    sales_sorted["Cumulative Sales Percentage"] = sales_sorted["Sales Percentage"].cumsum()

    if boundary_rule == "straddle":
        sales_sorted["prev_cum"] = sales_sorted["Cumulative Sales Percentage"].shift(1).fillna(0.0)
        sales_sorted["Sales Class"] = sales_sorted["prev_cum"].apply(
            lambda x: "A" if x < 60.0 else ("B" if x < 80.0 else ("C" if x < 90.0 else "D"))
        )
    else:
        sales_sorted["Sales Class"] = [
            assign_pareto_class(cum_val, idx, boundary_rule)
            for idx, cum_val in enumerate(sales_sorted["Cumulative Sales Percentage"])
        ]

    # -------------------------------------------------------------
    # Step A2: POS Hits Classification (Rank items highest to lowest)
    # -------------------------------------------------------------
    hits_sorted = work_df[["Item Number", "Hits"]].sort_values(
        by="Hits", ascending=False
    ).reset_index(drop=True)

    hits_sorted["Hits Percentage"] = (hits_sorted["Hits"] / total_hits) * 100.0
    hits_sorted["Cumulative Hits Percentage"] = hits_sorted["Hits Percentage"].cumsum()

    if boundary_rule == "straddle":
        hits_sorted["prev_cum"] = hits_sorted["Cumulative Hits Percentage"].shift(1).fillna(0.0)
        hits_sorted["Hits Class"] = hits_sorted["prev_cum"].apply(
            lambda x: "A" if x < 60.0 else ("B" if x < 80.0 else ("C" if x < 90.0 else "D"))
        )
    else:
        hits_sorted["Hits Class"] = [
            assign_pareto_class(cum_val, idx, boundary_rule)
            for idx, cum_val in enumerate(hits_sorted["Cumulative Hits Percentage"])
        ]

    # -------------------------------------------------------------
    # Step B & C: Merge, Convert to Scores & Combined Weighted Score
    # Combined Weighted Score = 0.65 * Sales Score + 0.35 * Hits Score
    # -------------------------------------------------------------
    merged = pd.merge(
        work_df,
        sales_sorted[["Item Number", "Sales Percentage", "Cumulative Sales Percentage", "Sales Class"]],
        on="Item Number",
    )
    merged = pd.merge(
        merged,
        hits_sorted[["Item Number", "Hits Percentage", "Cumulative Hits Percentage", "Hits Class"]],
        on="Item Number",
    )

    merged["Sales Score"] = merged["Sales Class"].map(SCORE_MAP)
    merged["Hits Score"] = merged["Hits Class"].map(SCORE_MAP)
    merged["Combined Weighted Score"] = (
        WEIGHT_SALES * merged["Sales Score"] + WEIGHT_HITS * merged["Hits Score"]
    ).round(4)

    # -------------------------------------------------------------
    # Step D: Final Inventory Classification
    # A: 3.25–4.00, B: 2.50–3.24, C: 1.75–2.49, D: < 1.75
    # -------------------------------------------------------------
    merged["Final Inventory Class"] = merged["Combined Weighted Score"].apply(assign_final_class_from_score)

    # Required 15-column output order
    ordered_columns = [
        "Item Number",
        "Item Name",
        "UoM",
        "Sales ($)",
        "Hits",
        "Sales Percentage",
        "Cumulative Sales Percentage",
        "Sales Class",
        "Hits Percentage",
        "Cumulative Hits Percentage",
        "Hits Class",
        "Sales Score",
        "Hits Score",
        "Combined Weighted Score",
        "Final Inventory Class",
    ]
    classified_df = merged[ordered_columns].sort_values(by="Sales ($)", ascending=False).reset_index(drop=True)

    # -------------------------------------------------------------
    # Step 5: Required Classification Summary Table
    # Final Class | Number of Items | % of Items | Total Sales ($) | % of Total Sales
    # -------------------------------------------------------------
    categories = ["A", "B", "C", "D"]
    total_items_count = len(classified_df)

    summary_rows = []
    for cat in categories:
        cat_df = classified_df[classified_df["Final Inventory Class"] == cat]
        items_count = int(len(cat_df))
        items_pct = (items_count / total_items_count * 100.0) if total_items_count > 0 else 0.0
        cat_sales = float(cat_df["Sales ($)"].sum())
        cat_sales_pct = (cat_sales / total_sales * 100.0) if total_sales > 0 else 0.0

        summary_rows.append({
            "Final Class": cat,
            "Number of Items": items_count,
            "Percentage of Items (%)": round(items_pct, 2),
            "Total Sales ($)": round(cat_sales, 2),
            "Percentage of Total Sales (%)": round(cat_sales_pct, 2),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Reconciliation Checks
    reconciled_items = int(summary_df["Number of Items"].sum())
    reconciled_sales = float(summary_df["Total Sales ($)"].sum())
    unclassified_count = int(classified_df["Final Inventory Class"].isna().sum())

    is_reconciled = (
        reconciled_items == total_items_count
        and np.isclose(reconciled_sales, total_sales, atol=1e-2)
        and unclassified_count == 0
    )

    reconciliation = {
        "total_items": total_items_count,
        "total_sales": total_sales,
        "reconciled_items": reconciled_items,
        "reconciled_sales": reconciled_sales,
        "unclassified_count": unclassified_count,
        "is_reconciled": bool(is_reconciled),
    }

    # Cross-tabulation score matrix: Sales Class vs Hits Class
    score_matrix = pd.crosstab(
        classified_df["Sales Class"],
        classified_df["Hits Class"],
        rownames=["Sales Class"],
        colnames=["Hits Class"],
    ).reindex(index=categories, columns=categories, fill_value=0)

    return {
        "classified_df": classified_df,
        "summary_df": summary_df,
        "score_matrix": score_matrix,
        "reconciliation": reconciliation,
        "boundary_rule_used": boundary_rule,
        "total_sales": total_sales,
        "total_hits": total_hits,
        "total_items": total_items_count,
    }
