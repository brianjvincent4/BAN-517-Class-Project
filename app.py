"""
app.py

Interactive Streamlit UI and CLI interface for the Supply Chain Analytics
Inventory Decision Agent.

Architecture:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
"""

from __future__ import annotations

import json
import sys
from typing import Dict

from inventory_agent import MODEL_METADATA, InventoryDecisionAgent

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


def run_streamlit_app() -> None:
    """Launch the Streamlit web interface."""
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
    )

    st.title("📦 Supply Chain Analytics Inventory Decision Agent")
    st.markdown(
        "**Architecture Pipeline:** "
        "`Customer Question` → `Identify Model` → `Extract Inputs` → "
        "`Select Python Tool` → `Calculate` → `Verify` → `Explain`"
    )

    agent = InventoryDecisionAgent()

    with st.sidebar:
        st.header("Supported Inventory Models")
        for key, meta in MODEL_METADATA.items():
            st.markdown(f"- **{meta['display_name']}** (`{meta['tool_name']}`)")
        st.divider()
        st.subheader("Sample Questions")
        selected_sample = st.selectbox(
            "Load an example scenario:",
            options=["Custom Question"] + list(SAMPLE_QUESTIONS.keys()),
        )

    default_text = (
        SAMPLE_QUESTIONS[selected_sample]
        if selected_sample != "Custom Question"
        else SAMPLE_QUESTIONS["1. Economic Order Quantity (EOQ)"]
    )

    question = st.text_area(
        "Ask a Natural-Language Inventory Question:",
        value=default_text,
        height=120,
    )

    if st.button("Run Inventory Decision Agent", type="primary"):
        if not question.strip():
            st.warning("Please enter an inventory question.")
            return

        try:
            response = agent.ask(question)

            # Pipeline Progress Banner
            cols = st.columns(7)
            stage_names = [
                "1. Question",
                "2. Identify Model",
                "3. Extract Inputs",
                "4. Select Tool",
                "5. Calculate",
                "6. Verify",
                "7. Explain",
            ]
            for col, s_name in zip(cols, stage_names):
                col.success(s_name)

            st.divider()

            # Executive Explanation & Key Metrics
            left_col, right_col = st.columns([3, 2])

            with left_col:
                st.markdown(response.explanation)

            with right_col:
                st.subheader("🔍 Pipeline Trace")
                st.markdown(f"**Identified Model:** `{response.model_display_name}`")
                st.markdown(f"**Selected Python Tool:** `inventory_tools.{response.selected_tool_name}()`")

                with st.expander("Step 3: Extracted Inputs (JSON)", expanded=True):
                    st.json(response.extracted_inputs)

                with st.expander("Step 5: Python Calculation Output (JSON)", expanded=False):
                    calc_display = {
                        k: v for k, v in response.calculation_results.items() if k != "verification"
                    }
                    st.json(calc_display)

                with st.expander("Step 6: Verification Checks", expanded=True):
                    for chk in response.verification.get("checks", []):
                        icon = "✅" if chk["passed"] else "❌"
                        st.markdown(f"{icon} **{chk['check']}**: `{chk['detail']}`")

        except Exception as exc:
            st.error(f"Error processing question: {exc}")


def run_cli_demo() -> None:
    """Run all 5 sample inventory questions in the terminal if executed directly via python app.py."""
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
        print(f"  -> Selected Tool    : {resp.selected_tool_name}")
        print(f"  -> Extracted Inputs : {json.dumps(resp.extracted_inputs)}")
        print(f"  -> Verification     : {'PASSED' if resp.verification['is_valid'] else 'FAILED'}")
        print("-" * 80)
        print(resp.explanation)
        print("=" * 80)


if __name__ == "__main__":
    # Detect whether launched via `streamlit run app.py` or `python app.py`
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            run_streamlit_app()
        else:
            run_cli_demo()
    except ImportError:
        run_cli_demo()
