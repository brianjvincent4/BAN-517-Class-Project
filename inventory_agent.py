"""
inventory_agent.py

Supply Chain Analytics Inventory Decision Agent implementing the pipeline:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain

All numerical calculations are performed strictly in Python via inventory_tools.py.
Supports optional LLM integration (OpenAI / Google Gemini) for natural-language parsing
and explanation, with a built-in deterministic NLP parser & explainer so the agent works
out-of-the-box with or without API keys.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from inventory_tools import (
    TOOL_REGISTRY,
    calculate_eoq,
    calculate_fixed_period_review,
    calculate_newsvendor,
    calculate_quantity_discount,
    calculate_safety_stock_rop,
    verify_calculation,
)

MODEL_METADATA: Dict[str, Dict[str, str]] = {
    "eoq": {
        "display_name": "Economic Order Quantity (EOQ)",
        "tool_name": "calculate_eoq",
        "description": "Continuous review model with constant demand and fixed ordering/holding costs.",
    },
    "quantity_discount": {
        "display_name": "Quantity Discount Model",
        "tool_name": "calculate_quantity_discount",
        "description": "All-units price-break EOQ model minimizing total annual ordering, holding, and purchase cost.",
    },
    "safety_stock_rop": {
        "display_name": "Safety Stock and Reorder Point (Continuous Review)",
        "tool_name": "calculate_safety_stock_rop",
        "description": "Continuous review (Q, R) policy under stochastic demand and/or lead time.",
    },
    "fixed_period_review": {
        "display_name": "Fixed-Period Review (Periodic Review / P-System)",
        "tool_name": "calculate_fixed_period_review",
        "description": "Periodic review (T, S) order-up-to policy over a protection interval of T + L.",
    },
    "newsvendor": {
        "display_name": "Newsvendor Model (Single-Period Inventory)",
        "tool_name": "calculate_newsvendor",
        "description": "Single-period stochastic demand model balancing underage cost (Cu) and overage cost (Co).",
    },
}


@dataclass
class AgentPipelineResponse:
    """Structured output capturing every step of the agent architecture."""

    question: str
    identified_model: str
    model_display_name: str
    extracted_inputs: Dict[str, Any]
    selected_tool_name: str
    calculation_results: Dict[str, Any]
    verification: Dict[str, Any]
    explanation: str
    pipeline_steps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "identified_model": self.identified_model,
            "model_display_name": self.model_display_name,
            "extracted_inputs": self.extracted_inputs,
            "selected_tool_name": self.selected_tool_name,
            "calculation_results": self.calculation_results,
            "verification": self.verification,
            "explanation": self.explanation,
            "pipeline_steps": self.pipeline_steps,
        }


class InventoryDecisionAgent:
    """
    Supply Chain Analytics Inventory Decision Agent.

    Pipeline Architecture:
        1. Customer Question
        2. Identify Model
        3. Extract Inputs
        4. Select Python Tool
        5. Calculate ( strictly in Python via inventory_tools )
        6. Verify
        7. Explain
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "auto"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.provider = provider

    # -----------------------------------------------------------------
    # Stage 1 & 2: Identify Model
    # -----------------------------------------------------------------
    def identify_model(self, question: str) -> str:
        """
        Identify which of the 5 inventory models applies to the natural-language question.
        """
        q_lower = question.lower()

        # Score each model based on domain keywords and patterns
        scores: Dict[str, int] = {
            "eoq": 0,
            "quantity_discount": 0,
            "safety_stock_rop": 0,
            "fixed_period_review": 0,
            "newsvendor": 0,
        }

        # Quantity Discount indicators
        discount_keywords = [
            "discount", "price break", "price-break", "tier", "tiers",
            "schedule", "all-units", "bulk price", "or more",
        ]
        for kw in discount_keywords:
            if kw in q_lower:
                scores["quantity_discount"] += 4
        # Multiple unit prices with quantity ranges (e.g. 1-499, 500-999, 1000+)
        if len(re.findall(r"\$\s*\d+(?:\.\d+)?", q_lower)) >= 3 and re.search(r"\d+\s*[-–to]+\s*\d+|\d+\+", q_lower):
            scores["quantity_discount"] += 6

        # Fixed-Period Review indicators
        periodic_keywords = [
            "fixed-period", "fixed period", "periodic review", "review period",
            "reviewed every", "reviews inventory every", "order-up-to", "order up to",
            "target inventory", "p-system", "every 7 days", "every 14 days",
            "every 2 weeks", "every 4 weeks", "every month", "current inventory",
            "on hand", "on-hand", "in stock right now",
        ]
        for kw in periodic_keywords:
            if kw in q_lower:
                scores["fixed_period_review"] += 4

        # Newsvendor indicators
        newsvendor_keywords = [
            "newsvendor", "newsboy", "single-period", "single period",
            "salvage", "leftover", "perishable", "underage", "overage",
            "critical ratio", "critical fractile", "selling price", "sells for",
            "retail price", "scrap", "season", "one-time", "goodwill",
            "bakery", "newspaper", "fresh",
        ]
        for kw in newsvendor_keywords:
            if kw in q_lower:
                scores["newsvendor"] += 4

        # Safety Stock and Reorder Point indicators
        rop_keywords = [
            "safety stock", "reorder point", "rop", "service level",
            "cycle service level", "stockout", "lead time", "lead-time",
            "continuous review", "standard deviation", "std dev",
        ]
        for kw in rop_keywords:
            if kw in q_lower:
                scores["safety_stock_rop"] += 3

        # Disambiguate Fixed-Period vs Continuous ROP when both have lead time & service level
        if re.search(r"review(?:ed|s)?\s+(?:period|every|interval)", q_lower) or "fixed-period" in q_lower or "fixed period" in q_lower or "periodic" in q_lower:
            scores["fixed_period_review"] += 6

        # EOQ indicators
        eoq_keywords = [
            "eoq", "economic order quantity", "ordering cost", "setup cost",
            "holding cost", "carrying cost", "annual demand", "optimal order quantity",
        ]
        for kw in eoq_keywords:
            if kw in q_lower:
                scores["eoq"] += 3

        # If EOQ keywords present without discount tiers, standard deviation, or salvage/selling price
        if scores["eoq"] > 0 and scores["quantity_discount"] == 0 and scores["safety_stock_rop"] == 0 and scores["newsvendor"] == 0:
            scores["eoq"] += 4

        best_model = max(scores.items(), key=lambda item: item[1])
        if best_model[1] == 0:
            # Default fallback to EOQ if no specific keywords matched
            return "eoq"
        return best_model[0]

    # -----------------------------------------------------------------
    # Stage 3: Extract Inputs
    # -----------------------------------------------------------------
    def extract_inputs(self, question: str, model_name: str) -> Dict[str, Any]:
        """
        Extract numerical parameters from natural language for the identified model.
        """
        if model_name == "eoq":
            return self._extract_eoq_inputs(question)
        elif model_name == "quantity_discount":
            return self._extract_quantity_discount_inputs(question)
        elif model_name == "safety_stock_rop":
            return self._extract_safety_stock_rop_inputs(question)
        elif model_name == "fixed_period_review":
            return self._extract_fixed_period_inputs(question)
        elif model_name == "newsvendor":
            return self._extract_newsvendor_inputs(question)
        else:
            raise ValueError(f"Unsupported inventory model: {model_name}")

    @staticmethod
    def _parse_number(val_str: str) -> float:
        return float(val_str.replace(",", "").strip())

    def _extract_eoq_inputs(self, text: str) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}
        num_pat = r"(\d[\d,]*(?:\.\d+)?)"

        # Annual demand
        m_demand = (
            re.search(rf"annual\s+demand\s+(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            or re.search(rf"demand\s+(?:of|is|=|:)?\s*{num_pat}\s*(?:units?\s*)?(?:per\s+year|annually|/year|a\s+year)", text, re.I)
            or re.search(rf"{num_pat}\s*units?\s*(?:per\s+year|annually|/year|a\s+year)", text, re.I)
            or re.search(rf"\bD\s*=\s*{num_pat}", text)
            or re.search(rf"demand\s+(?:of|is|=|:)\s*{num_pat}", text, re.I)
        )
        if m_demand:
            inputs["annual_demand"] = self._parse_number(m_demand.group(1))
        else:
            # Check monthly or weekly demand and convert to annual
            m_month = re.search(rf"monthly\s+demand\s+(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            m_week = re.search(rf"weekly\s+demand\s+(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            if m_month:
                inputs["annual_demand"] = self._parse_number(m_month.group(1)) * 12.0
            elif m_week:
                inputs["annual_demand"] = self._parse_number(m_week.group(1)) * 52.0

        # Ordering / setup cost
        m_order = (
            re.search(rf"(?:ordering|order|setup)\s+cost\s+(?:per\s+order\s+)?(?:of|is|=|:)?\s*\$?\s*{num_pat}", text, re.I)
            or re.search(rf"\$?\s*{num_pat}\s*(?:per\s+order|ordering\s+cost|setup\s+cost)", text, re.I)
            or re.search(rf"\bS\s*=\s*\$?\s*{num_pat}", text)
        )
        if m_order:
            inputs["ordering_cost"] = self._parse_number(m_order.group(1))

        # Holding cost rate (%)
        m_hrate = (
            re.search(rf"(?:holding|carrying)\s+(?:cost\s+)?(?:rate\s+)?(?:of|is|=|:)?\s*{num_pat}\s*%", text, re.I)
            or re.search(rf"{num_pat}\s*%\s*(?:of\s+unit\s+cost|holding|carrying)", text, re.I)
        )
        if m_hrate:
            inputs["holding_rate"] = self._parse_number(m_hrate.group(1)) / 100.0

        # Direct Holding cost ($)
        m_hold = (
            re.search(rf"(?:holding|carrying)\s+cost\s+(?:per\s+unit\s*(?:per\s+year\s*)?)?(?:of|is|=|:)?\s*\$?\s*{num_pat}(?!\s*%)", text, re.I)
            or re.search(rf"\$?\s*{num_pat}\s*(?:per\s+unit\s+per\s+year|holding\s+cost|carrying\s+cost)", text, re.I)
            or re.search(rf"\bH\s*=\s*\$?\s*{num_pat}", text)
        )
        if m_hold:
            inputs["holding_cost"] = self._parse_number(m_hold.group(1))

        # Unit cost
        m_unit = (
            re.search(rf"(?:unit|purchase|item)\s+(?:cost|price)\s+(?:of|is|=|:)?\s*\$?\s*{num_pat}", text, re.I)
            or re.search(rf"costs?\s*\$\s*{num_pat}\s*(?:per\s+unit|each)", text, re.I)
            or re.search(rf"\bC\s*=\s*\$?\s*{num_pat}", text)
        )
        if m_unit:
            inputs["unit_cost"] = self._parse_number(m_unit.group(1))

        # Working days
        m_days = re.search(rf"{num_pat}\s*(?:working|operating)\s+days", text, re.I)
        if m_days:
            inputs["working_days"] = self._parse_number(m_days.group(1))

        return inputs

    def _extract_quantity_discount_inputs(self, text: str) -> Dict[str, Any]:
        base = self._extract_eoq_inputs(text)
        inputs: Dict[str, Any] = {
            "annual_demand": base.get("annual_demand"),
            "ordering_cost": base.get("ordering_cost"),
        }
        if "holding_rate" in base:
            inputs["holding_rate"] = base["holding_rate"]
        elif "holding_cost" in base:
            inputs["holding_cost"] = base["holding_cost"]

        tiers: List[Dict[str, Any]] = []
        num_pat = r"\d[\d,]*(?:\.\d+)?"

        # Pattern 1: Range tiers like "1 to 499 units: $10.00" or "1-499 @ $10" or "1 - 499 units is $10"
        range_matches = re.finditer(
            rf"({num_pat})\s*(?:to|[-–—])\s*({num_pat})\s*(?:units?)?\s*(?:[:=@]|is|at|for|costs?)?\s*\$\s*({num_pat})",
            text,
            re.I,
        )
        for m in range_matches:
            tiers.append({
                "min_qty": self._parse_number(m.group(1)),
                "max_qty": self._parse_number(m.group(2)),
                "unit_cost": self._parse_number(m.group(3)),
            })

        # Pattern 2: Open-ended top tier like "1000+ units: $9.00" or "1,000 or more: $9.00" or "over 1000 units at $9"
        plus_matches = re.finditer(
            rf"(?:({num_pat})\s*(?:\+|\s+or\s+more|\s+and\s+above|\s+and\s+over)|(?:over|above|at\s+least)\s+({num_pat}))\s*(?:units?)?\s*(?:[:=@]|is|at|for|costs?)?\s*\$\s*({num_pat})",
            text,
            re.I,
        )
        for m in plus_matches:
            min_q_str = m.group(1) or m.group(2)
            tiers.append({
                "min_qty": self._parse_number(min_q_str),
                "max_qty": None,
                "unit_cost": self._parse_number(m.group(3)),
            })

        # Pattern 3: Reverse phrasing "$10 for 1-499 units, $9.50 for 500-999, $9.00 for 1000+"
        if not tiers:
            rev_range = re.finditer(
                rf"\$\s*({num_pat})\s*(?:per\s+unit\s*)?(?:for|if\s+ordering|on)?\s*({num_pat})\s*(?:to|[-–—])\s*({num_pat})",
                text,
                re.I,
            )
            for m in rev_range:
                tiers.append({
                    "min_qty": self._parse_number(m.group(2)),
                    "max_qty": self._parse_number(m.group(3)),
                    "unit_cost": self._parse_number(m.group(1)),
                })
            rev_plus = re.finditer(
                rf"\$\s*({num_pat})\s*(?:per\s+unit\s*)?(?:for|if\s+ordering|on)?\s*({num_pat})\s*(?:\+|\s+or\s+more|\s+and\s+above)",
                text,
                re.I,
            )
            for m in rev_plus:
                tiers.append({
                    "min_qty": self._parse_number(m.group(2)),
                    "max_qty": None,
                    "unit_cost": self._parse_number(m.group(1)),
                })

        inputs["tiers"] = sorted(tiers, key=lambda t: t["min_qty"])
        return inputs

    def _extract_safety_stock_rop_inputs(self, text: str) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}
        num_pat = r"(\d[\d,]*(?:\.\d+)?)"

        # Average demand
        m_avg = (
            re.search(rf"(?:average|mean|daily|weekly)\s+demand\s+(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            or re.search(rf"demand\s+averages?\s*{num_pat}", text, re.I)
            or re.search(rf"demand\s+(?:of|is|=|:)?\s*{num_pat}\s*units?\s*per\s*(?:day|week)", text, re.I)
        )
        if m_avg:
            inputs["avg_demand"] = self._parse_number(m_avg.group(1))

        # Standard deviation of demand
        m_std = (
            re.search(rf"(?:standard\s+deviation|std\.?\s*dev\.?|sigma)\s+(?:of\s+demand\s+)?(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            or re.search(rf"{num_pat}\s*units?\s*(?:standard\s+deviation|std\.?\s*dev\.?)", text, re.I)
        )
        if m_std:
            inputs["std_demand"] = self._parse_number(m_std.group(1))

        # Lead time
        m_lt = (
            re.search(rf"lead\s*[- ]?time\s+(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            or re.search(rf"{num_pat}\s*(?:days?|weeks?)\s+lead\s*[- ]?time", text, re.I)
            or re.search(rf"\bL\s*=\s*{num_pat}", text)
        )
        if m_lt:
            inputs["lead_time"] = self._parse_number(m_lt.group(1))

        # Standard deviation of lead time (optional)
        m_std_lt = re.search(
            rf"(?:lead\s*[- ]?time\s+(?:standard\s+deviation|std\.?\s*dev\.?)|(?:standard\s+deviation|std\.?\s*dev\.?)\s+of\s+lead\s*[- ]?time)\s+(?:of|is|=|:)?\s*{num_pat}",
            text,
            re.I,
        )
        if m_std_lt:
            inputs["std_lead_time"] = self._parse_number(m_std_lt.group(1))

        # Service level (%) or decimal
        m_sl = (
            re.search(rf"(?:service\s+level|csl|in-stock\s+probability)\s+(?:of|is|=|:)?\s*{num_pat}\s*%?", text, re.I)
            or re.search(rf"{num_pat}\s*%\s*(?:cycle\s+)?service\s+level", text, re.I)
        )
        if m_sl:
            sl_val = self._parse_number(m_sl.group(1))
            inputs["service_level"] = sl_val / 100.0 if sl_val > 1.0 else sl_val

        # Explicit z-score
        m_z = re.search(rf"\bz(?:-score)?\s*(?:of|is|=|:)\s*{num_pat}", text, re.I)
        if m_z and "service_level" not in inputs:
            inputs["z_score"] = self._parse_number(m_z.group(1))

        return inputs

    def _extract_fixed_period_inputs(self, text: str) -> Dict[str, Any]:
        base = self._extract_safety_stock_rop_inputs(text)
        inputs: Dict[str, Any] = {
            "avg_demand": base.get("avg_demand"),
            "std_demand": base.get("std_demand"),
            "lead_time": base.get("lead_time"),
        }
        if "service_level" in base:
            inputs["service_level"] = base["service_level"]
        if "z_score" in base:
            inputs["z_score"] = base["z_score"]

        num_pat = r"(\d[\d,]*(?:\.\d+)?)"
        # Review period (T or P)
        m_rev = (
            re.search(rf"review\s+(?:period|interval|cycle)\s+(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            or re.search(rf"review(?:ed|s)?\s+(?:inventory\s+)?every\s+{num_pat}\s*(?:days?|weeks?|periods?)", text, re.I)
            or re.search(rf"{num_pat}\s*[- ]?(?:day|week)\s+review\s+period", text, re.I)
            or re.search(rf"\b[TP]\s*=\s*{num_pat}", text)
        )
        if m_rev:
            inputs["review_period"] = self._parse_number(m_rev.group(1))

        # Current inventory level / on-hand
        m_inv = (
            re.search(rf"(?:current(?:ly)?\s+inventory|inventory\s+on\s+hand|on-hand\s+inventory|current\s+stock|currently\s+has)\s+(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            or re.search(rf"{num_pat}\s*units?\s*(?:on\s+hand|in\s+stock|in\s+inventory|currently)", text, re.I)
        )
        if m_inv:
            inputs["current_inventory"] = self._parse_number(m_inv.group(1))
        else:
            inputs["current_inventory"] = 0.0

        return inputs

    def _extract_newsvendor_inputs(self, text: str) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}
        num_pat = r"(\d[\d,]*(?:\.\d+)?)"

        # Mean demand
        m_mean = (
            re.search(rf"(?:mean|average|expected)\s+demand\s+(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            or re.search(rf"demand\s+(?:is\s+normally\s+distributed\s+with\s+)?(?:a\s+)?mean\s+(?:of\s+)?{num_pat}", text, re.I)
            or re.search(rf"\bmu\s*=\s*{num_pat}", text, re.I)
        )
        if m_mean:
            inputs["mean_demand"] = self._parse_number(m_mean.group(1))

        # Std dev of demand
        m_std = (
            re.search(rf"(?:standard\s+deviation|std\.?\s*dev\.?|sigma)\s+(?:of\s+demand\s+)?(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            or re.search(rf"{num_pat}\s*units?\s*(?:standard\s+deviation|std\.?\s*dev\.?)", text, re.I)
        )
        if m_std:
            inputs["std_demand"] = self._parse_number(m_std.group(1))

        # Direct underage / overage costs
        m_cu = re.search(rf"(?:underage\s+cost|cost\s+of\s+understocking|cu|c_u)\s+(?:of|is|=|:)?\s*\$?\s*{num_pat}", text, re.I)
        m_co = re.search(rf"(?:overage\s+cost|cost\s+of\s+overstocking|co|c_o)\s+(?:of|is|=|:)?\s*\$?\s*{num_pat}", text, re.I)
        if m_cu:
            inputs["underage_cost"] = self._parse_number(m_cu.group(1))
        if m_co:
            inputs["overage_cost"] = self._parse_number(m_co.group(1))

        # Selling price (p)
        m_price = (
            re.search(rf"(?:selling\s+price|retail\s+price|sells?\s+for|priced\s+at)\s+(?:of|is|=|:)?\s*\$?\s*{num_pat}", text, re.I)
            or re.search(rf"\$\s*{num_pat}\s*selling\s+price", text, re.I)
        )
        if m_price:
            inputs["selling_price"] = self._parse_number(m_price.group(1))

        # Unit / purchase / production cost (c)
        m_cost = (
            re.search(rf"(?:unit\s+cost|purchase\s+cost|production\s+cost|costs?|buys?\s+(?:them\s+)?for)\s+(?:of|is|=|:)?\s*\$\s*{num_pat}", text, re.I)
            or re.search(rf"(?:unit|purchase|wholesale)\s+cost\s+(?:of|is|=|:)?\s*\$?\s*{num_pat}", text, re.I)
        )
        if m_cost:
            inputs["unit_cost"] = self._parse_number(m_cost.group(1))

        # Salvage value (s)
        m_salvage = (
            re.search(rf"(?:salvage\s+value|salvage\s+price|scrap\s+value|leftover.*?(?:sold|salvaged)\s+for|unsold.*?(?:sold|salvaged)\s+for)\s+(?:of|is|=|:)?\s*\$?\s*{num_pat}", text, re.I)
            or re.search(rf"\$\s*{num_pat}\s*salvage", text, re.I)
        )
        if m_salvage:
            inputs["salvage_value"] = self._parse_number(m_salvage.group(1))

        # Goodwill / penalty cost (g)
        m_goodwill = re.search(rf"(?:goodwill\s+cost|stockout\s+penalty)\s+(?:of|is|=|:)?\s*\$?\s*{num_pat}", text, re.I)
        if m_goodwill:
            inputs["goodwill_cost"] = self._parse_number(m_goodwill.group(1))

        return inputs

    # -----------------------------------------------------------------
    # Stage 4: Select Python Tool
    # -----------------------------------------------------------------
    def select_python_tool(self, model_name: str) -> Tuple[str, Callable]:
        """
        Select the appropriate pure-Python calculation function from TOOL_REGISTRY.
        """
        if model_name not in TOOL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'. Available tools: {list(TOOL_REGISTRY.keys())}")
        tool_fn = TOOL_REGISTRY[model_name]
        tool_name = MODEL_METADATA[model_name]["tool_name"]
        return tool_name, tool_fn

    # -----------------------------------------------------------------
    # Stage 5: Calculate (Strictly in Python)
    # -----------------------------------------------------------------
    def calculate(self, tool_fn: Callable, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Python inventory tool with the extracted inputs.
        """
        clean_inputs = {k: v for k, v in inputs.items() if v is not None}
        return tool_fn(**clean_inputs)

    # -----------------------------------------------------------------
    # Stage 6: Verify
    # -----------------------------------------------------------------
    def verify(
        self,
        model_name: str,
        inputs: Dict[str, Any],
        calculation_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Verify calculation accuracy, optimality conditions, and domain constraints.
        """
        if "verification" in calculation_results:
            return calculation_results["verification"]
        return verify_calculation(model_name, inputs, calculation_results)

    # -----------------------------------------------------------------
    # Stage 7: Explain
    # -----------------------------------------------------------------
    def explain(
        self,
        question: str,
        model_name: str,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
        verification: Dict[str, Any],
    ) -> str:
        """
        Generate a clear, executive-friendly explanation of the inventory decision
        using the exact Python-verified numbers.
        """
        status_badge = "PASSED" if verification.get("is_valid") else "WARNING"
        checks_text = "\n".join(
            f"- [{'PASS' if c['passed'] else 'FAIL'}] **{c['check']}**: {c['detail']}"
            for c in verification.get("checks", [])
        )

        if model_name == "eoq":
            inp = results["inputs"]
            return (
                f"### Economic Order Quantity (EOQ) Analysis\n\n"
                f"**1. Model Selection & Parameters**\n"
                f"- **Model**: Economic Order Quantity (Continuous Review with Constant Demand)\n"
                f"- **Annual Demand ($D$)**: {inp['annual_demand']:,.2f} units/year\n"
                f"- **Ordering Cost ($S$)**: ${inp['ordering_cost']:,.2f} per order\n"
                f"- **Holding Cost ($H$)**: ${inp['holding_cost']:,.2f} per unit/year\n\n"
                f"**2. Python Calculation Results**\n"
                f"- **Optimal Order Quantity ($Q^*$)**: **{results['eoq']:,.2f} units** (approx. **{results['eoq_rounded']:,} units**)\n"
                f"- **Expected Orders per Year ($D / Q^*$)**: {results['orders_per_year']:,.2f} orders/year\n"
                f"- **Cycle Time Between Orders**: {results['cycle_time_days']:,.2f} days (based on {inp['working_days']:.0f} days/year)\n"
                f"- **Average Cycle Inventory ($Q^* / 2$)**: {results['average_inventory']:,.2f} units\n"
                f"- **Annual Ordering Cost**: ${results['annual_ordering_cost']:,.2f}\n"
                f"- **Annual Holding Cost**: ${results['annual_holding_cost']:,.2f}\n"
                f"- **Total Relevant Inventory Cost**: **${results['total_relevant_cost']:,.2f}/year**"
                + (
                    f"\n- **Total Annual Cost (incl. Purchase Cost)**: **${results['total_annual_cost']:,.2f}/year**"
                    if inp.get("unit_cost")
                    else ""
                )
                + f"\n\n**3. Verification ({status_badge})**\n{checks_text}\n\n"
                f"**4. Managerial Recommendation**\n"
                f"Order **{results['eoq']:,.2f} units** (or **{results['eoq_rounded']:,} whole units**) every "
                f"**{results['cycle_time_days']:,.1f} days** to balance ordering and holding costs at the minimum "
                f"total relevant cost of **${results['total_relevant_cost']:,.2f}** per year."
            )

        elif model_name == "quantity_discount":
            inp = results["inputs"]
            tier_rows = []
            for t in results["tier_evaluations"]:
                max_str = f"{t['max_qty']:,.0f}" if t["max_qty"] is not None else "∞"
                marker = " **(Optimal)**" if t["tier_index"] == results["optimal_tier"] else ""
                tier_rows.append(
                    f"- **Tier {t['tier_index']} ({t['min_qty']:,.0f} – {max_str} units @ ${t['unit_cost']:,.2f}/unit)**: "
                    f"Unconstrained $Q^* = {t['unconstrained_eoq']:,.2f}$, Evaluated $Q = {t['evaluated_order_qty']:,.2f}$ | "
                    f"Ordering = ${t['annual_ordering_cost']:,.2f}, Holding = ${t['annual_holding_cost']:,.2f}, "
                    f"Purchase = ${t['annual_purchase_cost']:,.2f} → **Total = ${t['total_annual_cost']:,.2f}**{marker}"
                )
            tiers_summary = "\n".join(tier_rows)
            return (
                f"### Quantity Discount Analysis\n\n"
                f"**1. Model Selection & Parameters**\n"
                f"- **Model**: All-Units Quantity Discount EOQ Model\n"
                f"- **Annual Demand ($D$)**: {inp['annual_demand']:,.2f} units/year\n"
                f"- **Ordering Cost ($S$)**: ${inp['ordering_cost']:,.2f} per order\n\n"
                f"**2. Tier-by-Tier Cost Evaluation (Python)**\n"
                f"{tiers_summary}\n\n"
                f"**3. Optimal Decision**\n"
                f"- **Optimal Tier**: Tier {results['optimal_tier']} (${results['optimal_unit_cost']:,.2f}/unit)\n"
                f"- **Optimal Order Quantity ($Q^*$)**: **{results['optimal_order_quantity']:,.2f} units**\n"
                f"- **Minimum Total Annual Cost**: **${results['min_total_annual_cost']:,.2f}/year**\n\n"
                f"**4. Verification ({status_badge})**\n{checks_text}\n\n"
                f"**5. Managerial Recommendation**\n"
                f"Place orders of **{results['optimal_order_quantity']:,.0f} units** to qualify for Tier {results['optimal_tier']} "
                f"pricing (${results['optimal_unit_cost']:,.2f}/unit), achieving the lowest total annual cost of "
                f"**${results['min_total_annual_cost']:,.2f}**."
            )

        elif model_name == "safety_stock_rop":
            inp = results["inputs"]
            return (
                f"### Safety Stock and Reorder Point (Continuous Review) Analysis\n\n"
                f"**1. Model Selection & Parameters**\n"
                f"- **Model**: Continuous Review $(Q, R)$ Policy with Stochastic Demand\n"
                f"- **Average Demand ($\\bar{{d}}$)**: {inp['avg_demand']} units/period\n"
                f"- **Demand Std Dev ($\\sigma_d$)**: {inp['std_demand']} units/period\n"
                f"- **Replenishment Lead Time ($L$)**: {inp['lead_time']} periods\n"
                f"- **Target Cycle Service Level (CSL)**: {results['service_level'] * 100:.2f}% ($z = {results['z_score']:.4f}$)\n\n"
                f"**2. Python Calculation Results**\n"
                f"- **Expected Demand During Lead Time ($\\bar{{d}} \\times L$)**: {results['expected_demand_during_lead_time']:,.2f} units\n"
                f"- **Std Dev of Lead-Time Demand ($\\sigma_{{dLT}}$)**: {results['std_dev_demand_during_lead_time']:,.4f} units\n"
                f"- **Safety Stock ($SS = z \\cdot \\sigma_{{dLT}}$)**: **{results['safety_stock']:,.2f} units** (approx. **{results['safety_stock_rounded']:,} units**)\n"
                f"- **Reorder Point ($ROP = \\bar{{d}}L + SS$)**: **{results['reorder_point']:,.2f} units** (approx. **{results['reorder_point_rounded']:,} units**)\n\n"
                f"**3. Verification ({status_badge})**\n{checks_text}\n\n"
                f"**4. Managerial Recommendation**\n"
                f"Hold **{results['safety_stock']:,.2f} units** (or **{results['safety_stock_rounded']:,} units**) of safety stock "
                f"and trigger a replenishment order whenever inventory position drops to the Reorder Point of "
                f"**{results['reorder_point']:,.2f} units** (or **{results['reorder_point_rounded']:,} units**) to maintain a "
                f"{results['service_level'] * 100:.1f}% cycle service level."
            )

        elif model_name == "fixed_period_review":
            inp = results["inputs"]
            return (
                f"### Fixed-Period Review (Periodic Review / P-System) Analysis\n\n"
                f"**1. Model Selection & Parameters**\n"
                f"- **Model**: Periodic Review $(T, S)$ Order-Up-To Policy\n"
                f"- **Average Demand ($\\bar{{d}}$)**: {inp['avg_demand']:,.2f} units/period\n"
                f"- **Demand Std Dev ($\\sigma_d$)**: {inp['std_demand']:,.2f} units/period\n"
                f"- **Review Period ($T$)**: {inp['review_period']:,.2f} periods\n"
                f"- **Lead Time ($L$)**: {inp['lead_time']:,.2f} periods\n"
                f"- **Protection Interval ($T + L$)**: {results['protection_interval']:,.2f} periods\n"
                f"- **Current Inventory Position ($I$)**: {results['current_inventory']:,.2f} units\n"
                f"- **Target Service Level (CSL)**: {results['service_level'] * 100:.2f}% ($z = {results['z_score']:.4f}$)\n\n"
                f"**2. Python Calculation Results**\n"
                f"- **Expected Demand Over Protection Interval ($\\bar{{d}}(T+L)$)**: {results['expected_demand_protection_interval']:,.2f} units\n"
                f"- **Std Dev Over Protection Interval ($\\sigma_d \\sqrt{{T+L}}$)**: {results['std_dev_protection_interval']:,.4f} units\n"
                f"- **Safety Stock ($SS$)**: **{results['safety_stock']:,.2f} units**\n"
                f"- **Target Inventory / Order-Up-To Level ($M$)**: **{results['target_inventory_level']:,.2f} units** (approx. **{results['target_inventory_level_rounded']:,} units**)\n"
                f"- **Recommended Order Quantity ($Q = M - I$)**: **{results['order_quantity']:,.2f} units** (approx. **{results['order_quantity_rounded']:,} units**)\n\n"
                f"**3. Verification ({status_badge})**\n{checks_text}\n\n"
                f"**4. Managerial Recommendation**\n"
                f"Set the Order-Up-To Target Level to **{results['target_inventory_level']:,.2f} units** "
                f"(including **{results['safety_stock']:,.2f} units** of safety stock). Given the current inventory of "
                f"**{results['current_inventory']:,.0f} units**, place an order for **{results['order_quantity']:,.2f} units** "
                f"(or **{results['order_quantity_rounded']:,} units**) at this review."
            )

        elif model_name == "newsvendor":
            inp = results["inputs"]
            return (
                f"### Newsvendor (Single-Period Inventory) Analysis\n\n"
                f"**1. Model Selection & Parameters**\n"
                f"- **Model**: Single-Period Newsvendor Model ({inp['distribution'].capitalize()} Demand)\n"
                f"- **Mean Demand ($\\mu$)**: {inp['mean_demand']:,.2f} units\n"
                f"- **Demand Std Dev ($\\sigma$)**: {inp['std_demand']:,.2f} units\n"
                f"- **Underage Cost ($C_u$)**: ${results['underage_cost']:,.2f}/unit\n"
                f"- **Overage Cost ($C_o$)**: ${results['overage_cost']:,.2f}/unit\n\n"
                f"**2. Python Calculation Results**\n"
                f"- **Critical Ratio ($CR = C_u / (C_u + C_o)$)**: **{results['critical_ratio']:.4f}** ({results['critical_ratio'] * 100:.2f}% optimal in-stock probability)\n"
                f"- **Optimal Safety Factor ($z$)**: {results['z_score'] if results['z_score'] is not None else 'N/A'}\n"
                f"- **Optimal Order Quantity ($Q^*$)**: **{results['optimal_order_quantity']:,.2f} units** (approx. **{results['optimal_order_quantity_rounded']:,} units**)\n"
                f"- **Expected Sales**: {results['expected_sales']:,.2f} units\n"
                f"- **Expected Leftover Inventory**: {results['expected_leftover_inventory']:,.2f} units\n"
                f"- **Expected Lost Sales**: {results['expected_lost_sales']:,.2f} units\n"
                f"- **Expected Mismatch Cost**: ${results['expected_mismatch_cost']:,.2f}\n"
                f"- **Expected Profit**: **${results['expected_profit']:,.2f}**\n\n"
                f"**3. Verification ({status_badge})**\n{checks_text}\n\n"
                f"**4. Managerial Recommendation**\n"
                f"Stock **{results['optimal_order_quantity']:,.2f} units** (or **{results['optimal_order_quantity_rounded']:,} units**) "
                f"for the selling period. This achieves the optimal critical ratio of **{results['critical_ratio'] * 100:.2f}%** "
                f"and maximizes expected profit at **${results['expected_profit']:,.2f}**."
            )

        return "Explanation completed."

    # -----------------------------------------------------------------
    # Full End-to-End Pipeline Orchestrator
    # -----------------------------------------------------------------
    def ask(
        self,
        question: str,
        override_model: Optional[str] = None,
        override_inputs: Optional[Dict[str, Any]] = None,
    ) -> AgentPipelineResponse:
        """
        Execute the full 7-step architecture:
        Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
        """
        pipeline_steps: List[Dict[str, Any]] = []

        # Step 1: Customer Question
        pipeline_steps.append({
            "step": 1,
            "stage": "Customer Question",
            "status": "completed",
            "details": {"question": question},
        })

        # Step 2: Identify Model
        model_name = override_model or self.identify_model(question)
        model_display = MODEL_METADATA[model_name]["display_name"]
        pipeline_steps.append({
            "step": 2,
            "stage": "Identify Model",
            "status": "completed",
            "details": {"model_id": model_name, "display_name": model_display},
        })

        # Step 3: Extract Inputs
        extracted_inputs = override_inputs if override_inputs is not None else self.extract_inputs(question, model_name)
        pipeline_steps.append({
            "step": 3,
            "stage": "Extract Inputs",
            "status": "completed",
            "details": extracted_inputs,
        })

        # Step 4: Select Python Tool
        tool_name, tool_fn = self.select_python_tool(model_name)
        pipeline_steps.append({
            "step": 4,
            "stage": "Select Python Tool",
            "status": "completed",
            "details": {
                "tool_name": tool_name,
                "description": MODEL_METADATA[model_name]["description"],
            },
        })

        # Step 5: Calculate (Strictly in Python)
        calculation_results = self.calculate(tool_fn, extracted_inputs)
        pipeline_steps.append({
            "step": 5,
            "stage": "Calculate",
            "status": "completed",
            "details": {k: v for k, v in calculation_results.items() if k != "verification"},
        })

        # Step 6: Verify
        verification = self.verify(model_name, extracted_inputs, calculation_results)
        pipeline_steps.append({
            "step": 6,
            "stage": "Verify",
            "status": "completed" if verification.get("is_valid") else "warning",
            "details": verification,
        })

        # Step 7: Explain
        explanation = self.explain(
            question=question,
            model_name=model_name,
            inputs=extracted_inputs,
            results=calculation_results,
            verification=verification,
        )
        pipeline_steps.append({
            "step": 7,
            "stage": "Explain",
            "status": "completed",
            "details": {"explanation_length": len(explanation)},
        })

        return AgentPipelineResponse(
            question=question,
            identified_model=model_name,
            model_display_name=model_display,
            extracted_inputs=extracted_inputs,
            selected_tool_name=tool_name,
            calculation_results=calculation_results,
            verification=verification,
            explanation=explanation,
            pipeline_steps=pipeline_steps,
        )
