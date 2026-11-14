"""
inventory_agent.py

Supply Chain Analytics Inventory Decision Agent implementing the pipeline:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain

All numerical calculations are performed strictly in Python via inventory_tools.py.
Features robust natural-language extraction recognizing common inventory terminology and synonyms.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from inventory_tools import (
    TOOL_REGISTRY,
    calculate_eoq,
    calculate_newsvendor,
    calculate_periodic_review,
    calculate_quantity_discount,
    calculate_rop,
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
        "tool_name": "calculate_rop",
        "description": "Continuous review (Q, R) policy supporting constant/variable demand and constant/variable lead time.",
    },
    "fixed_period_review": {
        "display_name": "Fixed-Period Review (Periodic Review / P-System)",
        "tool_name": "calculate_periodic_review",
        "description": "Periodic review (T, S) order-up-to policy over a protection period of review_period + lead_time.",
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
        3. Extract Inputs (synonym-aware & non-inventive)
        4. Select Python Tool
        5. Calculate (strictly in Python via inventory_tools)
        6. Verify
        7. Explain
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "auto"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.provider = provider

    # -----------------------------------------------------------------
    # Helper Parsing Utilities
    # -----------------------------------------------------------------
    @staticmethod
    def _parse_number(val_str: str) -> float:
        """Parse numbers handling commas, currency signs, and common suffixes (k, m)."""
        s = val_str.replace(",", "").replace("$", "").strip()
        if s.lower().endswith("k"):
            return float(s[:-1]) * 1000.0
        if s.lower().endswith("m"):
            return float(s[:-1]) * 1000000.0
        return float(s)

    # -----------------------------------------------------------------
    # Stage 1 & 2: Identify Model
    # -----------------------------------------------------------------
    def identify_model(self, question: str) -> str:
        """
        Identify which of the 5 inventory models applies to the natural-language question.
        Examines the entire question using comprehensive domain terminology.
        """
        q_lower = question.lower()

        scores: Dict[str, int] = {
            "eoq": 0,
            "quantity_discount": 0,
            "safety_stock_rop": 0,
            "fixed_period_review": 0,
            "newsvendor": 0,
        }

        # Quantity Discount indicators
        discount_keywords = [
            "discount", "price break", "price-break", "quantity break", "tier", "tiers",
            "schedule", "all-units", "bulk price", "or more", "and above", "at least",
            "price breaks", "quantity breaks",
        ]
        for kw in discount_keywords:
            if kw in q_lower:
                scores["quantity_discount"] += 4
        if len(re.findall(r"\$\s*\d+(?:\.\d+)?", q_lower)) >= 3 and (
            re.search(r"\d+\s*(?:to|[-–—]|and)\s*\d+|\d+\+", q_lower)
            or re.search(r"(?:below|under|at least|or more)\s*[\d,]+", q_lower)
        ):
            scores["quantity_discount"] += 8

        # Fixed-Period Review indicators
        periodic_keywords = [
            "fixed-period", "fixed period", "periodic review", "review period",
            "reviewed every", "reviews inventory every", "review cycle", "review interval",
            "order-up-to", "order up to", "target inventory", "p-system", "protection period",
            "inventory position", "on-hand inventory", "on-order inventory", "on hand",
            "on order", "backorders", "pipeline inventory",
        ]
        for kw in periodic_keywords:
            if kw in q_lower:
                scores["fixed_period_review"] += 4

        # Newsvendor indicators
        newsvendor_keywords = [
            "newsvendor", "newsboy", "single-period", "single period",
            "salvage", "leftover", "perishable", "underage", "shortage cost",
            "overage", "excess cost", "critical ratio", "critical fractile",
            "selling price", "retail price", "purchase cost", "scrap", "season",
            "one-time", "goodwill",
        ]
        for kw in newsvendor_keywords:
            if kw in q_lower:
                scores["newsvendor"] += 4

        # Safety Stock and Reorder Point indicators
        rop_keywords = [
            "safety stock", "reorder point", "rop", "service level",
            "cycle service level", "stockout", "lead time", "lead-time",
            "continuous review", "lead-time standard deviation",
            "demand standard deviation", "std dev",
        ]
        for kw in rop_keywords:
            if kw in q_lower:
                scores["safety_stock_rop"] += 3

        # Disambiguate Fixed-Period vs Continuous ROP when both mention lead time or service level
        if (
            re.search(r"review(?:ed|s)?\s+(?:period|every|interval|cycle)", q_lower)
            or "fixed-period" in q_lower
            or "fixed period" in q_lower
            or "periodic" in q_lower
            or "inventory position" in q_lower
            or "on-order" in q_lower
            or "backorder" in q_lower
        ):
            scores["fixed_period_review"] += 6

        # EOQ indicators (including all ordering cost synonyms)
        eoq_keywords = [
            "eoq", "economic order quantity", "ordering cost", "order cost",
            "setup cost", "setup expense", "cost per order", "cost of placing an order",
            "cost to place an order", "ordering/setup cost", "holding cost", "carrying cost",
            "annual demand", "annual usage", "optimal order quantity", "optimal batch",
            "optimal order size", "batch order size", "total annual cost", "costs to place",
        ]
        for kw in eoq_keywords:
            if kw in q_lower:
                scores["eoq"] += 3

        if (
            scores["eoq"] > 0
            and scores["quantity_discount"] == 0
            and scores["safety_stock_rop"] == 0
            and scores["fixed_period_review"] == 0
            and scores["newsvendor"] == 0
        ):
            scores["eoq"] += 4

        best_model = max(scores.items(), key=lambda item: item[1])
        return best_model[0] if best_model[1] > 0 else "eoq"

    # -----------------------------------------------------------------
    # Stage 3: Extract Inputs (Synonym-Aware & Non-Inventive)
    # -----------------------------------------------------------------
    def extract_inputs(self, question: str, model_name: str) -> Dict[str, Any]:
        """
        Extract numerical parameters from natural language for the identified model.
        Examines the entire question before deciding that information is missing.
        Never invents missing numerical inputs.
        """
        if model_name == "eoq":
            return self._extract_eoq_inputs(question)
        elif model_name == "quantity_discount":
            return self._extract_quantity_discount_inputs(question)
        elif model_name in ("safety_stock_rop", "rop"):
            return self._extract_safety_stock_rop_inputs(question)
        elif model_name in ("fixed_period_review", "periodic_review"):
            return self._extract_fixed_period_inputs(question)
        elif model_name == "newsvendor":
            return self._extract_newsvendor_inputs(question)
        else:
            raise ValueError(f"Unsupported inventory model: {model_name}")

    def _extract_eoq_inputs(self, text: str) -> Dict[str, Any]:
        """
        Extract EOQ inputs.
        Recognizes all ordering cost synonyms:
        - ordering cost, order cost, setup cost, setup expense, cost per order,
          cost of placing an order, ordering/setup cost, cost to place an order, etc.
        """
        inputs: Dict[str, Any] = {}
        num_pat = r"((?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[kKmM])?)"

        # 0. Working days (if specified in prompt)
        m_days = (
            re.search(rf"{num_pat}\s*(?:working|business|operating)\s*days(?:\s*(?:per|a)\s*year|\s*annually)?", text, re.I)
            or re.search(rf"(?:open|operates?|operating)\s*(?:for\s*)?{num_pat}\s*days(?:\s*(?:per|a)\s*year|\s*annually)?", text, re.I)
        )
        working_days = self._parse_number(m_days.group(1)) if m_days else None
        if working_days is not None:
            inputs["working_days"] = working_days

        # 1. Demand (D) extraction with time unit recognition & annualization
        # Prioritize matching demand with explicit time period (month, week, day, year)
        # to ensure proper annual basis conversion before calculation.
        raw_demand_val: Optional[float] = None
        demand_period: Optional[str] = None

        # Check 1: Prefix period: "monthly demand is 1,000", "annual demand is 10,000", "daily demand of 50"
        m_prefix = re.search(
            rf"\b(annual|yearly|monthly|weekly|daily|quarterly)\s+(?:demand|usage|sales|requirement|volume|orders?)(?:\s+for\s+[^,\.;\n\d]+?)?\s*(?:averages?|is\s+on\s+average|of|is|=|:)?\s*{num_pat}",
            text,
            re.I,
        )
        if m_prefix:
            demand_period = m_prefix.group(1).lower().strip()
            raw_demand_val = self._parse_number(m_prefix.group(2))

        # Check 2: Suffix period with demand keyword: "demand is 1,000 units per month", "sells 1,200 units per year"
        if raw_demand_val is None:
            m_suffix_kw = re.search(
                rf"(?:demand|sales|usage|requirement|volume|requires?|sells?|needs?|uses?)\s*(?:for\s+[^,\.;\n\d]+?)?\s*(?:averages?|is\s+on\s+average|of|is|=|:)?\s*{num_pat}\s*(?:units?|items?|components?|pieces?|cases?|boxes?|widgets?|gallons?|pounds?|lbs?|tons?)?\s*(?:per|\/|a|each)\s*(year|annum|month|week|day|quarter)",
                text,
                re.I,
            )
            if m_suffix_kw:
                raw_demand_val = self._parse_number(m_suffix_kw.group(1))
                demand_period = m_suffix_kw.group(2).lower().strip()

        # Check 3: Suffix period with units without demand keyword: "1,000 units per month", "1,200 units a year"
        if raw_demand_val is None:
            m_suffix_unit = re.search(
                rf"{num_pat}\s*(?:units?|items?|components?|pieces?|cases?|boxes?|widgets?|gallons?|pounds?|lbs?|tons?)\s*(?:per|\/|a|each)\s*(year|annum|month|week|day|quarter)",
                text,
                re.I,
            )
            if m_suffix_unit:
                raw_demand_val = self._parse_number(m_suffix_unit.group(1))
                demand_period = m_suffix_unit.group(2).lower().strip()

        # Check 4: Adverbs: "1,000 units monthly", "sells 10,000 units annually"
        if raw_demand_val is None:
            m_adverb = re.search(
                rf"(?:demand|sales|usage|requirement|volume|requires?|sells?|needs?|uses?)?\s*(?:for\s+[^,\.;\n\d]+?)?\s*(?:averages?|is\s+on\s+average|of|is|=|:)?\s*{num_pat}\s*(?:units?|items?|components?|pieces?)?\s*(annually|monthly|weekly|daily|quarterly)",
                text,
                re.I,
            )
            if m_adverb and m_adverb.group(2):
                raw_demand_val = self._parse_number(m_adverb.group(1))
                demand_period = m_adverb.group(2).lower().strip()

        # Check 5: Standalone D = number
        if raw_demand_val is None:
            m_d = re.search(rf"\bD\s*=\s*{num_pat}", text)
            if m_d:
                raw_demand_val = self._parse_number(m_d.group(1))
                demand_period = "year"

        # Check 6: Generic demand without specified period: "demand is 10,000", "demand of 5,000"
        if raw_demand_val is None:
            m_gen = re.search(rf"(?:demand|usage|sales|requirement)\s+(?:of|is|=|:)\s*{num_pat}", text, re.I)
            if m_gen:
                raw_demand_val = self._parse_number(m_gen.group(1))
                demand_period = "year"

        # Normalization to annual demand:
        if raw_demand_val is not None:
            p_clean = (demand_period or "year").lower().strip()
            if p_clean in ("year", "yearly", "annual", "annually", "annum"):
                conversion_factor = 1.0
                norm_period = "year"
            elif p_clean in ("month", "monthly"):
                conversion_factor = 12.0
                norm_period = "month"
            elif p_clean in ("week", "weekly"):
                conversion_factor = 52.0
                norm_period = "week"
            elif p_clean in ("quarter", "quarterly"):
                conversion_factor = 4.0
                norm_period = "quarter"
            elif p_clean in ("day", "daily"):
                conversion_factor = working_days if working_days else 365.0
                norm_period = "day"
            else:
                conversion_factor = 1.0
                norm_period = "year"

            inputs["annual_demand"] = raw_demand_val * conversion_factor
            inputs["stated_demand"] = raw_demand_val
            inputs["demand_period"] = norm_period
            inputs["demand_conversion_factor"] = conversion_factor

        # 2. Ordering cost (S) - MUST support all required synonyms and natural expressions:
        # - Subject is order: "Each order costs $50 to place", "An order costs $50 to place", "Every order costs $50", "Orders cost $50 to place"
        # - Verb / impersonal phrasing: "It costs $50 per order", "Costs $50 to place", "Costs $50 to place an order", "Incurs $50 per order"
        # - Noun phrase: "The ordering cost is $50", "ordering cost of $50", "order cost is $50", "setup expense of $50", "cost per order is $50"
        # - Number first: "$50 per order", "$50 to place", "$50 to place an order", "$50 ordering cost", "$50 setup expense"
        # - Variable notation: "S = $50", "S = 50"
        ordering_synonyms_pattern = (
            r"(?:the|our|an?|its)?\s*(?:"
            r"ordering\s*(?:/|and|\s+)?setup\s+cost|"
            r"ordering\s+cost|"
            r"order\s+cost|"
            r"setup\s+cost|"
            r"setup\s+expense|"
            r"cost\s+per\s+order|"
            r"cost\s+of\s+placing\s+(?:an?|each|every)?\s*order|"
            r"cost\s+to\s+place\s+(?:an?|each|every)?\s*order|"
            r"cost\s+of\s+(?:an?|each|every)?\s*order|"
            r"expense\s+per\s+order|"
            r"ordering\s+expense|"
            r"order\s+expense|"
            r"order\s+placement\s+cost|"
            r"order\s+processing\s+cost|"
            r"cost\s+per\s+setup|"
            r"fee\s+per\s+order|"
            r"order\s+fee)"
        )

        m_order = (
            # 1. Subject is order: e.g. "Each order costs $50 to place", "An order costs $50 to place", "Each order costs $50"
            re.search(
                rf"(?:each|an?|every|our)?\s*orders?\s*(?:placement\s+)?costs?\s*(?:about|approx\.?|approximately)?\s*\$?\s*{num_pat}(?:\s*(?:to\s+(?:place|process|submit)(?:\s+(?:an?|each|every)?\s*order)?|per\s+order|each))?",
                text,
                re.I,
            )
            # 2. Verb / impersonal phrasing: e.g. "It costs $50 per order", "Costs $50 to place", "Costs $50 to place an order", "Incurs $50 per order"
            or re.search(
                rf"(?:it\s+)?(?:costs?|incurs?|pays?)\s+(?:about|approx\.?|approximately)?\s*\$?\s*{num_pat}\s*(?:per\s+order|each\s+order|per\s+setup|to\s+(?:place|process|submit)(?:\s+(?:an?|each|every)?\s*order)?|in\s+(?:setup|ordering)\s+(?:costs?|expenses?))",
                text,
                re.I,
            )
            # 3. Noun phrase: e.g. "The ordering cost is $50", "ordering cost of $50", "setup expense of $50"
            or re.search(
                rf"{ordering_synonyms_pattern}\s+(?:per\s+order\s+)?(?:of|is|=|:|at|\$)?\s*\$?\s*{num_pat}",
                text,
                re.I,
            )
            # 4. Number followed by ordering synonym: e.g. "$50 per order", "$50 to place", "$50 to place an order", "$50 ordering cost", "$50 setup expense"
            or re.search(
                rf"\$?\s*{num_pat}\s*(?:per\s+order|per\s+setup|to\s+place(?:\s+(?:an?|each|every)?\s*order)?|cost\s+per\s+order|ordering\s+cost|order\s+cost|setup\s+cost|setup\s+expense)",
                text,
                re.I,
            )
            # 5. Variable notation: "S = $50", "S = 50"
            or re.search(rf"\bS\s*=\s*\$?\s*{num_pat}", text)
        )
        if m_order:
            inputs["ordering_cost"] = self._parse_number(m_order.group(1))

        # 3. Holding cost rate (%) - percentage holding cost
        m_hrate = (
            re.search(rf"(?:holding|carrying)\s+(?:cost\s+)?(?:rate\s+)?(?:of|is|=|:)?\s*{num_pat}\s*%", text, re.I)
            or re.search(rf"{num_pat}\s*%\s*(?:of\s+(?:the\s+)?(?:unit\s+)?(?:cost|price|value)|holding|carrying)", text, re.I)
            or re.search(rf"(?:annual\s+)?(?:holding|carrying)\s+cost\s+is\s+{num_pat}\s*%", text, re.I)
            or re.search(rf"carrying\s+charge\s+(?:of|is|=|:)?\s*{num_pat}\s*%", text, re.I)
            or re.search(rf"\bI\s*=\s*{num_pat}\s*%", text)
        )
        if m_hrate:
            inputs["holding_rate"] = self._parse_number(m_hrate.group(1)) / 100.0

        # 4. Direct fixed holding cost ($)
        m_hold = (
            re.search(rf"(?:the|our|an?|its)?\s*(?:inventory\s+)?(?:holding|carrying)\s+cost\s+(?:per\s+unit\s*(?:per\s+year\s*)?)?(?:of|is|=|:)?\s*\$?\s*{num_pat}(?!\s*%)", text, re.I)
            or re.search(rf"(?:holding|carrying)\s+expense\s+(?:of|is|=|:)?\s*\$?\s*{num_pat}(?!\s*%)", text, re.I)
            or re.search(rf"cost\s+to\s+hold\s+(?:is|of)?\s*\$?\s*{num_pat}(?!\s*%)", text, re.I)
            or re.search(rf"\$?\s*{num_pat}\s*(?:per\s+unit\s+per\s+year|per\s+unit\s+annually|holding\s+cost|carrying\s+cost)(?!\s*%)", text, re.I)
            or re.search(rf"\bH\s*=\s*\$?\s*{num_pat}(?!\s*%)", text)
        )
        if m_hold:
            inputs["holding_cost"] = self._parse_number(m_hold.group(1))

        # 5. Unit purchase price / unit cost (C)
        m_unit = (
            re.search(rf"(?:unit\s+purchase\s+price|purchase\s+price|unit\s+price|unit\s+cost|purchase\s+cost|item\s+cost|wholesale\s+price)\s+(?:of|is|=|:)?\s*\$?\s*{num_pat}", text, re.I)
            or re.search(rf"costs?\s*\$\s*{num_pat}\s*(?:per\s+unit|each|a\s+unit)", text, re.I)
            or re.search(rf"bought\s+for\s*\$\s*{num_pat}\s*(?:each|per\s+unit)?", text, re.I)
            or re.search(rf"\bC\s*=\s*\$?\s*{num_pat}", text)
        )
        if m_unit:
            inputs["unit_cost"] = self._parse_number(m_unit.group(1))

        # 6. Working days
        m_days = re.search(rf"{num_pat}\s*(?:working|operating)\s+days", text, re.I) or re.search(rf"open\s+{num_pat}\s+days\s*(?:per\s+year|a\s+year)", text, re.I)
        if m_days:
            inputs["working_days"] = self._parse_number(m_days.group(1))

        return inputs

    def _extract_quantity_discount_inputs(self, text: str) -> Dict[str, Any]:
        """
        Extract Quantity Discount inputs.
        Distinguishes:
        - quantity break
        - unit price
        - ordering cost (including all EOQ synonyms)
        - fixed holding cost
        - percentage holding cost
        - unit purchase price
        """
        base = self._extract_eoq_inputs(text)
        inputs: Dict[str, Any] = {
            "annual_demand": base.get("annual_demand"),
            "ordering_cost": base.get("ordering_cost"),
        }
        if "holding_rate" in base:
            inputs["holding_rate"] = base["holding_rate"]
        if "holding_cost" in base:
            inputs["holding_cost"] = base["holding_cost"]
        if "stated_demand" in base:
            inputs["stated_demand"] = base["stated_demand"]
        if "demand_period" in base:
            inputs["demand_period"] = base["demand_period"]
        if "demand_conversion_factor" in base:
            inputs["demand_conversion_factor"] = base["demand_conversion_factor"]

        tiers: List[Dict[str, Any]] = []
        num_pat = r"\d[\d,]*(?:\.\d+)?"

        # Pattern 1a: Below/under range first: "orders below 500 units at $12", "under 500 units: $12"
        below_matches = re.finditer(
            rf"(?:orders\s+(?:of\s+)?)?(?:below|under|less\s+than|up\s+to)\s*({num_pat})\s*(?:units?)?\s*(?:[:=@]|is|at|for|costs?|unit\s+price\s+is)?\s*\$\s*({num_pat})",
            text,
            re.I,
        )
        for m in below_matches:
            cutoff = self._parse_number(m.group(1))
            tiers.append({
                "min_qty": 1.0,
                "max_qty": (cutoff - 1.0) if cutoff > 1.0 else cutoff,
                "unit_cost": self._parse_number(m.group(2)),
            })

        # Pattern 1b: Standard range tiers: "1 to 999 units at $5.00" / "500-999 units at $11.50"
        range_matches = re.finditer(
            rf"(?:orders\s+(?:of\s+)?)?(?:between\s+)?({num_pat})\s*(?:to|[-–—]|and)\s*({num_pat})\s*(?:units?)?\s*(?:[:=@]|is|at|for|costs?|unit\s+price\s+is)?\s*\$\s*({num_pat})",
            text,
            re.I,
        )
        for m in range_matches:
            tiers.append({
                "min_qty": self._parse_number(m.group(1)),
                "max_qty": self._parse_number(m.group(2)),
                "unit_cost": self._parse_number(m.group(3)),
            })

        # Pattern 2: Open-ended top tier: "2,000+ units at $4.75" / "2,000 or more: $4.75" / "at least 2,000 units: $4.75"
        plus_matches = re.finditer(
            rf"(?:(?:orders\s+(?:of\s+)?)?({num_pat})\s*(?:units?)?\s*(?:\+|\s+or\s+more|\s+and\s+above|\s+and\s+over)|(?:orders\s+(?:of\s+)?)?(?:over|above|at\s+least)\s+({num_pat}))\s*(?:units?)?\s*(?:[:=@]|is|at|for|costs?|unit\s+price\s+is)?\s*\$\s*({num_pat})",
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

        # Pattern 3: Reverse phrasing: price first, then quantity
        # e.g. "$12 per unit for orders below 500 units, $11.50 for 500-999 units, and $11 for orders of 1,000 units or more"
        if not tiers:
            rev_below = re.finditer(
                rf"\$\s*({num_pat})\s*(?:per\s+unit\s*)?(?:for\s+(?:orders\s+(?:of\s+)?)?|if\s+ordering|on|for\s+)?(?:orders\s+(?:of\s+)?)?(?:below|under|less\s+than|up\s+to)\s*({num_pat})\s*(?:units?)?",
                text,
                re.I,
            )
            for m in rev_below:
                cutoff = self._parse_number(m.group(2))
                tiers.append({
                    "min_qty": 1.0,
                    "max_qty": (cutoff - 1.0) if cutoff > 1.0 else cutoff,
                    "unit_cost": self._parse_number(m.group(1)),
                })
            rev_range = re.finditer(
                rf"\$\s*({num_pat})\s*(?:per\s+unit\s*)?(?:for\s+(?:orders\s+(?:of\s+)?)?|if\s+ordering|on|for\s+)?(?:orders\s+(?:of\s+)?)?(?:between\s+)?({num_pat})\s*(?:to|[-–—]|and)\s*({num_pat})\s*(?:units?)?",
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
                rf"\$\s*({num_pat})\s*(?:per\s+unit\s*)?(?:for\s+(?:orders\s+(?:of\s+)?)?|if\s+ordering|on|for\s+)?(?:orders\s+(?:of\s+)?)?(?:at\s+least|over|above)?\s*({num_pat})\s*(?:units?)?\s*(?:\+|\s+or\s+more|\s+and\s+above|\s+and\s+over)",
                text,
                re.I,
            )
            for m in rev_plus:
                tiers.append({
                    "min_qty": self._parse_number(m.group(2)),
                    "max_qty": None,
                    "unit_cost": self._parse_number(m.group(1)),
                })

        # Pattern 4: Explicit "quantity break" phrasing:
        # e.g. "Quantity break at 1,000 units with unit price $4.80, quantity break at 2,000 units with unit price $4.75, base price $5.00"
        if not tiers:
            qb_matches = list(
                re.finditer(
                    rf"(?:quantity\s+break|price\s+break)\s+(?:at|of)\s+({num_pat})\s*(?:units?)?\s*(?:with\s+)?(?:unit\s+price|price)?\s*(?:of|is|=|:)?\s*\$\s*({num_pat})",
                    text,
                    re.I,
                )
            )
            base_price_m = re.search(
                rf"(?:base\s+(?:unit\s+)?(?:price|cost)|under\s+{num_pat}\s*(?:is|at)?)\s*\$?\s*({num_pat})",
                text,
                re.I,
            )
            if qb_matches:
                sorted_breaks = sorted(
                    [(self._parse_number(m.group(1)), self._parse_number(m.group(2))) for m in qb_matches],
                    key=lambda x: x[0],
                )
                base_price = (
                    self._parse_number(base_price_m.group(1) or base_price_m.group(2))
                    if base_price_m
                    else (base.get("unit_cost") or 0.0)
                )
                if base_price > 0:
                    tiers.append({"min_qty": 1.0, "max_qty": sorted_breaks[0][0] - 1.0, "unit_cost": base_price})
                for idx, (b_qty, b_price) in enumerate(sorted_breaks):
                    next_max = (sorted_breaks[idx + 1][0] - 1.0) if idx + 1 < len(sorted_breaks) else None
                    tiers.append({"min_qty": b_qty, "max_qty": next_max, "unit_cost": b_price})

        inputs["tiers"] = sorted(tiers, key=lambda t: t["min_qty"]) if tiers else None
        return inputs

    def _extract_safety_stock_rop_inputs(self, text: str) -> Dict[str, Any]:
        """
        Extract Safety Stock and Reorder Point inputs.
        Distinguishes:
        - average demand (and time unit: daily, weekly, etc.)
        - demand standard deviation
        - average lead time (and time unit: days vs. weeks)
        - lead-time standard deviation (and time unit)
        - service level
        - operating days per week (defaulting to 7 days, with 5 business days documented)
        """
        inputs: Dict[str, Any] = {}
        num_pat = r"(\d[\d,]*(?:\.\d+)?)"

        # Operating days per week assumption (default 7 days/week, detect if 5 business/work days specified)
        m_op_days = re.search(
            r"(\d+)\s*(?:operating\s+days|business\s+days|working\s+days|days)\s*(?:per|a)\s*week",
            text,
            re.I,
        ) or re.search(r"(\d+)[- ]day\s*(?:work\s*week|business\s*week|working\s*week)", text, re.I)
        days_per_week = self._parse_number(m_op_days.group(1)) if m_op_days else 7.0

        # Demand time unit detection
        demand_unit = "day"  # Default assumption for inventory ROP unless specified
        if re.search(r"(?:per\s+day|daily|a\s+day|day['s]?\s+demand|units?\s*/\s*day)", text, re.I):
            demand_unit = "day"
        elif re.search(r"(?:per\s+week|weekly|a\s+week|week['s]?\s+demand|units?\s*/\s*week)", text, re.I):
            demand_unit = "week"
        elif re.search(r"(?:per\s+month|monthly|a\s+month|month['s]?\s+demand|units?\s*/\s*month)", text, re.I):
            demand_unit = "month"
        elif re.search(r"(?:per\s+year|annually|annual|yearly|a\s+year|year['s]?\s+demand|units?\s*/\s*year)", text, re.I):
            demand_unit = "year"

        # 1. Lead time (L) and Lead-time standard deviation (sigma_L)
        stated_lead_time = None
        lead_time_unit = None
        stated_std_lead_time = None
        std_lead_time_unit = None

        # Pattern A: Combined sentence: e.g. "Average lead time is 3 weeks with a standard deviation of 0.5 weeks"
        # or "Lead time averages 3 weeks with std dev 0.5 weeks", "Average lead time is 3 weeks and standard deviation of 0.5 weeks"
        m_lt_combined = re.search(
            rf"(?:average|mean|replenishment)?\s*lead\s*[- ]?time\s+(?:is|of|=|:|averages?)\s*{num_pat}\s*(days?|weeks?|months?)?\s*(?:,|with|and)?\s*(?:a\s+)?(?:standard\s+deviation|std\.?\s*dev\.?|sigma)\s+(?:of|is|=|:)?\s*{num_pat}\s*(days?|weeks?|months?)?",
            text,
            re.I,
        )
        if m_lt_combined:
            stated_lead_time = self._parse_number(m_lt_combined.group(1))
            lead_time_unit = m_lt_combined.group(2).lower() if m_lt_combined.group(2) else None
            stated_std_lead_time = self._parse_number(m_lt_combined.group(3))
            std_lead_time_unit = m_lt_combined.group(4).lower() if m_lt_combined.group(4) else lead_time_unit

        # Pattern B: Standalone lead-time standard deviation (if not captured above)
        if stated_std_lead_time is None:
            m_std_lt = (
                re.search(
                    rf"(?:lead\s*[- ]?time\s+(?:standard\s+deviation|std\.?\s*dev\.?|sigma)|(?:standard\s+deviation|std\.?\s*dev\.?|sigma)\s+(?:of|in)\s+lead\s*[- ]?time)\s+(?:of|is|=|:)?\s*{num_pat}\s*(days?|weeks?|months?)?",
                    text,
                    re.I,
                )
                or re.search(
                    rf"lead\s*[- ]?time\s+(?:(?:is|of|=|:|averages?)\s+[\d,]+(?:\.\d+)?\s*(?:days?|weeks?|months?)?\s+)?(?:with|and)\s+(?:a\s+)?(?:standard\s+deviation|std\.?\s*dev\.?|sigma)\s+(?:of|is|=|:)?\s*{num_pat}\s*(days?|weeks?|months?)?",
                    text,
                    re.I,
                )
                or re.search(
                    rf"{num_pat}\s*(days?|weeks?|months?)\s*(?:standard\s+deviation|std\.?\s*dev\.?)\s+(?:of|in|for)\s+lead\s*[- ]?time",
                    text,
                    re.I,
                )
                or re.search(rf"\bsigma_?[lL]\s*=\s*{num_pat}\s*(days?|weeks?|months?)?", text)
            )
            if m_std_lt:
                stated_std_lead_time = self._parse_number(m_std_lt.group(1))
                if len(m_std_lt.groups()) > 1 and m_std_lt.group(2):
                    std_lead_time_unit = m_std_lt.group(2).lower()

        # Pattern C: Standalone average lead time (if not captured above)
        if stated_lead_time is None:
            m_lt = (
                re.search(
                    rf"(?:average|mean|replenishment|supplier)?\s*lead\s*[- ]?time\s+(?:of|is|=|:|averages?)\s*{num_pat}\s*(days?|weeks?|months?)?",
                    text,
                    re.I,
                )
                or re.search(
                    rf"{num_pat}\s*(days?|weeks?|months?)\s*(?:replenishment\s+)?lead\s*[- ]?time",
                    text,
                    re.I,
                )
                or re.search(
                    rf"(?:takes|requires|arrives\s+in)\s*{num_pat}\s*(days?|weeks?|months?)\s*(?:to\s+arrive|lead\s*[- ]?time)?",
                    text,
                    re.I,
                )
                or re.search(rf"\bL\s*=\s*{num_pat}\s*(days?|weeks?|months?)?", text)
            )
            if m_lt:
                stated_lead_time = self._parse_number(m_lt.group(1))
                if len(m_lt.groups()) > 1 and m_lt.group(2):
                    lead_time_unit = m_lt.group(2).lower()

        if lead_time_unit and not std_lead_time_unit:
            std_lead_time_unit = lead_time_unit

        # Unit normalization: convert lead time and lead time std dev to days if demand is daily and lead time is in weeks
        normalized_lead_time = stated_lead_time
        normalized_std_lead_time = stated_std_lead_time
        if stated_lead_time is not None:
            if demand_unit == "day" and lead_time_unit in ("week", "weeks"):
                normalized_lead_time = stated_lead_time * days_per_week
                if stated_std_lead_time is not None:
                    normalized_std_lead_time = stated_std_lead_time * days_per_week

            inputs["lead_time"] = normalized_lead_time
            inputs["stated_lead_time"] = stated_lead_time
            if lead_time_unit:
                inputs["lead_time_unit"] = lead_time_unit

        if normalized_std_lead_time is not None:
            inputs["std_lead_time"] = normalized_std_lead_time
            inputs["stated_std_lead_time"] = stated_std_lead_time
            if std_lead_time_unit:
                inputs["std_lead_time_unit"] = std_lead_time_unit
        elif stated_std_lead_time is None and (
            re.search(r"constant\s+lead\s*[- ]?time|lead\s*[- ]?time\s+is\s+constant", text, re.I)
        ):
            inputs["std_lead_time"] = 0.0

        inputs["demand_unit"] = demand_unit
        inputs["days_per_week"] = days_per_week

        # 2. Demand standard deviation (sigma_d)
        m_std_d = (
            re.search(rf"(?:demand\s+(?:standard\s+deviation|std\.?\s*dev\.?|sigma)|(?:standard\s+deviation|std\.?\s*dev\.?|sigma)\s+(?:of|in)\s+demand)\s+(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            or re.search(rf"demand(?:\s+(?:for|of)\s+[^,\.;\n]+?)?\s+(?:averages?\s+[\d,]+(?:\.\d+)?\s*(?:units?)?\s+)?with\s+(?:a\s+)?(?:standard\s+deviation|std\.?\s*dev\.?)\s+(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            or re.search(rf"{num_pat}\s*units?\s*(?:standard\s+deviation|std\.?\s*dev\.?)\s+(?:of|in|for)\s+demand", text, re.I)
            or re.search(rf"(?:standard\s+deviation|std\.?\s*dev\.?)\s+(?:of|is|=|:)?\s*{num_pat}\s*units?", text, re.I)
            or re.search(rf"\bsigma_?[dD]\s*=\s*{num_pat}", text)
        )
        if m_std_d:
            inputs["std_demand"] = self._parse_number(m_std_d.group(1))
        elif re.search(r"constant\s+demand|demand\s+is\s+constant", text, re.I):
            inputs["std_demand"] = 0.0

        # If lead time variability was not mentioned, but demand variability was, default std_lead_time to 0.0
        if "std_lead_time" not in inputs and "std_demand" in inputs:
            inputs["std_lead_time"] = 0.0

        # 3. Average demand (d_bar)
        # Supports business language variations:
        # - "daily demand averages 150 units", "daily demand for a warehouse item averages 150 units"
        # - "average daily demand is 150 units", "mean demand of 150 units", "average demand is 150 units"
        # - "daily demand of 150 units", "weekly demand is 150 units"
        # - "demand of 150 units per day", "demand averages 150 units per day"
        # - "constant demand of 150 units", "d = 150"
        m_avg = (
            # 1. Subject-first with 'averages': e.g. "daily demand for a warehouse item averages 150 units", "daily demand averages 150 units", "demand averages 150 units"
            re.search(
                rf"(?:daily|weekly|monthly|period|annual|yearly)?\s*demand(?:\s+for\s+[^,\.;\n\d]+?)?\s+(?:averages?|is\s+on\s+average)\s+(?:of|is|=|:|at)?\s*{num_pat}",
                text,
                re.I,
            )
            # 2. Adjective-first: e.g. "average daily demand is 150 units", "average demand of 150 units", "mean daily demand is 150"
            or re.search(
                rf"(?:average|mean|expected)\s+(?:daily|weekly|monthly|period|annual|yearly)?\s*demand(?:\s+for\s+[^,\.;\n\d]+?)?\s+(?:of|is|=|:|at|\$)?\s*{num_pat}(?!\s*%)",
                text,
                re.I,
            )
            # 3. Direct period demand: e.g. "daily demand is 150 units", "daily demand of 150 units", "weekly demand: 150"
            or re.search(
                rf"(?:daily|weekly|monthly|period)\s+demand(?:\s+for\s+[^,\.;\n\d]+?)?\s+(?:of|is|=|:|at)?\s*{num_pat}(?!\s*%)",
                text,
                re.I,
            )
            # 4. Rate-based phrasings: e.g. "demand of 150 units per day", "demand is 150 units/day", "demand averages 150 units a day"
            or re.search(
                rf"demand(?:\s+for\s+[^,\.;\n\d]+?)?\s+(?:of|is|=|:|at|averages?)\s*{num_pat}\s*units?\s*(?:per\s+(?:day|week|month|period)|a\s+(?:day|week|month)|daily|weekly)",
                text,
                re.I,
            )
            # 5. Constant demand phrasings: e.g. "constant demand of 150", "steady demand of 150"
            or re.search(
                rf"(?:constant|steady)\s+demand(?:\s+for\s+[^,\.;\n\d]+?)?\s+(?:of|is|=|:)?\s*{num_pat}",
                text,
                re.I,
            )
            # 6. Standalone verb: e.g. "averages 150 units per day", "averages 150 units daily"
            or re.search(
                rf"averages?\s+{num_pat}\s*units?\s*(?:per\s+(?:day|week|month|period)|a\s+(?:day|week|month)|daily|weekly)",
                text,
                re.I,
            )
            # 7. Variable notation: d = 150, d_bar = 150, D = 150
            or re.search(rf"\b(?:d_?bar|[dD])\s*=\s*{num_pat}", text)
        )
        if m_avg:
            inputs["avg_demand"] = self._parse_number(m_avg.group(1))

        # 5. Service level (%) or explicit z-score
        m_sl = (
            re.search(rf"(?:cycle\s+)?service\s+level\s+(?:of|is|=|:)?\s*{num_pat}\s*%?", text, re.I)
            or re.search(rf"{num_pat}\s*%\s*(?:cycle\s+)?service\s+level", text, re.I)
            or re.search(rf"(?:in-stock|fill)\s+(?:rate|probability)\s+(?:of|is|=|:)?\s*{num_pat}\s*%?", text, re.I)
            or re.search(rf"\bCSL\s*=\s*{num_pat}\s*%?", text)
        )
        if m_sl:
            sl_val = self._parse_number(m_sl.group(1))
            inputs["service_level"] = sl_val / 100.0 if sl_val > 1.0 else sl_val

        m_z = re.search(rf"\bz(?:-score)?\s*(?:of|is|=|:)\s*{num_pat}", text, re.I)
        if m_z and "service_level" not in inputs:
            inputs["z_score"] = self._parse_number(m_z.group(1))

        # 6. Direct safety stock if explicitly provided
        m_ss = (
            re.search(rf"safety\s+stock\s+(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            or re.search(rf"maintain(?:s)?\s*(?:a\s+)?(?:safety\s+stock\s+of\s+)?{num_pat}\s*units?\s*(?:of\s+safety\s+stock)?", text, re.I)
        )
        if m_ss:
            inputs["safety_stock"] = self._parse_number(m_ss.group(1))

        return inputs

    def _extract_fixed_period_inputs(self, text: str) -> Dict[str, Any]:
        """
        Extract Fixed-Period Review inputs.
        Distinguishes:
        - demand
        - review period
        - lead time
        - safety stock
        - inventory position
        - on-hand inventory
        - on-order inventory
        - backorders
        """
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
        if "safety_stock" in base:
            inputs["safety_stock"] = base["safety_stock"]

        num_pat = r"(\d[\d,]*(?:\.\d+)?)"

        # 1. Review period (T or P)
        m_rev = (
            re.search(rf"review\s+(?:period|interval|cycle)\s+(?:of|is|=|:)?\s*{num_pat}\s*(days?|weeks?|months?|periods?)?", text, re.I)
            or re.search(rf"reviews?\s+(?:inventory\s+)?every\s+{num_pat}\s*(days?|weeks?|months?|periods?)", text, re.I)
            or re.search(rf"{num_pat}\s*[- ]?(day|week|month)\s+review\s+period", text, re.I)
            or re.search(rf"order\s+(?:interval|cycle)\s+(?:of|is|=|:)?\s*{num_pat}\s*(days?|weeks?|months?|periods?)?", text, re.I)
            or re.search(rf"\b[TP]\s*=\s*{num_pat}\s*(days?|weeks?|months?|periods?)?", text)
        )
        if m_rev:
            rp_val = self._parse_number(m_rev.group(1))
            rp_unit = m_rev.group(2).lower() if len(m_rev.groups()) > 1 and m_rev.group(2) else None
            if rp_unit in ("week", "weeks") and base.get("demand_unit") == "day":
                rp_val = rp_val * base.get("days_per_week", 7.0)
            inputs["review_period"] = rp_val

        # 2. Inventory Position components: on-hand, on-order, backorders
        # On-hand inventory
        m_on_hand = (
            re.search(rf"(?:on-hand\s+inventory|on\s+hand\s+inventory|inventory\s+on\s+hand|current\s+inventory|current\s+stock|currently\s+in\s+stock|currently\s+has)\s+(?:of|is|are|=|:|at)?\s*{num_pat}", text, re.I)
            or re.search(rf"{num_pat}\s*units?\s*(?:on\s+hand|in\s+stock|currently\s+available)", text, re.I)
        )
        if m_on_hand:
            inputs["on_hand_inventory"] = self._parse_number(m_on_hand.group(1))

        # On-order inventory (scheduled receipts / pipeline)
        m_on_order = (
            re.search(rf"(?:on-order\s+inventory|on\s+order\s+inventory|inventory\s+on\s+order|scheduled\s+receipts?|pipeline\s+inventory)\s+(?:of|is|are|=|:|at)?\s*{num_pat}", text, re.I)
            or re.search(rf"{num_pat}\s*units?\s*(?:on\s+order|in\s+transit|pipeline|scheduled\s+to\s+arrive)", text, re.I)
            or re.search(rf"(?:currently\s+)?on\s+order\s+(?:of|is|are|=|:|at)?\s*{num_pat}", text, re.I)
        )
        if m_on_order:
            inputs["on_order_inventory"] = self._parse_number(m_on_order.group(1))

        # Backorders
        m_bo = (
            re.search(rf"(?:backorders?|backlog|backordered\s+units?|units?\s+on\s+backorder)\s+(?:of|is|are|=|:|at)?\s*{num_pat}", text, re.I)
            or re.search(rf"{num_pat}\s*units?\s*(?:on\s+backorder|backordered|in\s+backorders?)", text, re.I)
        )
        if m_bo:
            inputs["backorders"] = self._parse_number(m_bo.group(1))

        # Direct inventory position
        m_inv_pos = (
            re.search(rf"inventory\s+position\s+(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            or re.search(rf"current\s+inventory\s+position\s+(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            or re.search(rf"{num_pat}\s*units?\s*(?:inventory\s+position)", text, re.I)
        )
        if m_inv_pos:
            inputs["inventory_position"] = self._parse_number(m_inv_pos.group(1))
        elif "on_hand_inventory" in inputs:
            # Inventory Position = On-Hand + On-Order - Backorders
            on_h = inputs["on_hand_inventory"]
            on_o = inputs.get("on_order_inventory", 0.0)
            bo = inputs.get("backorders", 0.0)
            inputs["inventory_position"] = on_h + on_o - bo

        return inputs

    def _extract_newsvendor_inputs(self, text: str) -> Dict[str, Any]:
        """
        Extract Newsvendor inputs.
        Distinguishes:
        - selling price
        - purchase cost
        - salvage value
        - underage / shortage cost
        - overage / excess cost
        - demand distribution
        - mean demand
        - demand standard deviation
        """
        inputs: Dict[str, Any] = {}
        num_pat = r"(\d[\d,]*(?:\.\d+)?)"

        # 1. Demand distribution
        if "uniform" in text.lower():
            inputs["distribution"] = "uniform"
            m_unif = re.search(rf"uniformly\s+distributed\s+between\s*{num_pat}\s*and\s*{num_pat}", text, re.I)
            if m_unif:
                inputs["min_demand"] = self._parse_number(m_unif.group(1))
                inputs["max_demand"] = self._parse_number(m_unif.group(2))
        else:
            inputs["distribution"] = "normal"

        # 2. Mean demand (mu)
        m_mean = (
            re.search(rf"(?:mean|average|expected)\s+demand\s+(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            or re.search(rf"demand\s+is\s+normally\s+distributed\s+with\s+(?:a\s+)?mean\s+(?:of\s+)?{num_pat}", text, re.I)
            or re.search(rf"\bmu\s*=\s*{num_pat}", text, re.I)
            or re.search(rf"demand\s+averages?\s*{num_pat}", text, re.I)
        )
        if m_mean:
            inputs["mean_demand"] = self._parse_number(m_mean.group(1))

        # 3. Demand standard deviation (sigma)
        m_std = (
            re.search(rf"(?:demand\s+(?:standard\s+deviation|std\.?\s*dev\.?|sigma)|(?:standard\s+deviation|std\.?\s*dev\.?|sigma)\s+(?:of|in)\s+demand)\s+(?:of|is|=|:)?\s*{num_pat}", text, re.I)
            or re.search(rf"(?:standard\s+deviation|std\.?\s*dev\.?)\s+(?:of|is|=|:)?\s*{num_pat}\s*units?", text, re.I)
            or re.search(rf"{num_pat}\s*units?\s*(?:standard\s+deviation|std\.?\s*dev\.?)", text, re.I)
            or re.search(rf"\bsigma\s*=\s*{num_pat}", text, re.I)
        )
        if m_std:
            inputs["std_demand"] = self._parse_number(m_std.group(1))

        # 4. Underage / shortage cost (Cu)
        m_cu = (
            re.search(rf"(?:underage\s+cost|shortage\s+cost|cost\s+of\s+underage|cost\s+of\s+understocking|cost\s+of\s+shortage|lost\s+profit\s+per\s+unit|profit\s+margin\s+per\s+unit)\s+(?:of|is|=|:)?\s*\$?\s*{num_pat}", text, re.I)
            or re.search(rf"\b(?:c_?u|Cu)\s*=\s*\$?\s*{num_pat}", text)
        )
        if m_cu:
            inputs["underage_cost"] = self._parse_number(m_cu.group(1))

        # 5. Overage / excess cost (Co)
        m_co = (
            re.search(rf"(?:overage\s+cost|excess\s+cost|cost\s+of\s+overage|cost\s+of\s+overstocking|cost\s+of\s+excess|loss\s+per\s+unsold\s+unit)\s+(?:of|is|=|:)?\s*\$?\s*{num_pat}", text, re.I)
            or re.search(rf"\b(?:c_?o|Co)\s*=\s*\$?\s*{num_pat}", text)
        )
        if m_co:
            inputs["overage_cost"] = self._parse_number(m_co.group(1))

        # 6. Selling price (p)
        m_price = (
            re.search(rf"(?:selling\s+price|retail\s+price|sales\s+price|revenue\s+per\s+unit|price\s+per\s+unit|price\s+per\s+item)\s+(?:of|is|=|:)?\s*\$?\s*{num_pat}", text, re.I)
            or re.search(rf"sells?\s+(?:for|at)\s*\$?\s*{num_pat}", text, re.I)
            or re.search(rf"priced\s+at\s*\$?\s*{num_pat}", text, re.I)
            or re.search(rf"\$\s*{num_pat}\s*(?:selling\s+price|retail\s+price)", text, re.I)
            or re.search(rf"\bp\s*=\s*\$?\s*{num_pat}", text)
        )
        if m_price:
            inputs["selling_price"] = self._parse_number(m_price.group(1))

        # 7. Purchase cost / unit cost (c)
        m_cost = (
            re.search(rf"(?:purchase\s+cost|unit\s+cost|purchase\s+price|unit\s+purchase\s+price|wholesale\s+price|wholesale\s+cost|production\s+cost|acquisition\s+cost|cost\s+to\s+purchase|cost\s+per\s+unit)\s+(?:of|is|=|:)?\s*\$?\s*{num_pat}", text, re.I)
            or re.search(rf"costs?\s*\$\s*{num_pat}\s*(?:to\s+(?:buy|purchase|produce)|per\s+unit|each)?", text, re.I)
            or re.search(rf"buys?\s+(?:them\s+)?for\s*\$\s*{num_pat}", text, re.I)
            or re.search(rf"purchased\s+for\s*\$\s*{num_pat}", text, re.I)
            or re.search(rf"\bc\s*=\s*\$?\s*{num_pat}", text)
        )
        if m_cost:
            inputs["unit_cost"] = self._parse_number(m_cost.group(1))

        # 8. Salvage value (s)
        m_salvage = (
            re.search(rf"(?:salvage\s+value|salvage\s+price|scrap\s+value|residual\s+value|clearance\s+price|markdown\s+price)\s+(?:of|is|=|:)?\s*\$?\s*{num_pat}", text, re.I)
            or re.search(rf"(?:unsold|leftover)\s+(?:units?|items?|jackets?|goods?|products?)?\s*(?:are\s+|can\s+be\s+)?(?:salvaged|sold|liquidated)\s+for\s*\$?\s*{num_pat}", text, re.I)
            or re.search(rf"salvaged?\s+(?:for|at)\s*\$?\s*{num_pat}", text, re.I)
            or re.search(rf"\$\s*{num_pat}\s*salvage", text, re.I)
            or re.search(rf"\bs\s*=\s*\$?\s*{num_pat}", text)
        )
        if m_salvage:
            inputs["salvage_value"] = self._parse_number(m_salvage.group(1))
        elif re.search(r"no\s+salvage|salvage\s+(?:value\s+)?is\s+zero|worthless|scrapped\s+(?:with\s+)?no\s+value|zero\s+salvage", text, re.I):
            inputs["salvage_value"] = 0.0

        # 9. Goodwill cost (g)
        m_goodwill = re.search(rf"(?:goodwill\s+cost|stockout\s+penalty|penalty\s+cost)\s+(?:of|is|=|:)?\s*\$?\s*{num_pat}", text, re.I)
        if m_goodwill:
            inputs["goodwill_cost"] = self._parse_number(m_goodwill.group(1))

        return inputs

    # -----------------------------------------------------------------
    # Parameter Verification / Missing Information Validation
    # -----------------------------------------------------------------
    def validate_extracted_inputs(self, model_name: str, inputs: Dict[str, Any]) -> List[str]:
        """
        Examine extracted inputs and identify genuinely missing parameters.
        Returns a list of missing parameter explanations.
        """
        missing: List[str] = []

        if model_name == "eoq":
            if inputs.get("annual_demand") is None:
                missing.append("annual_demand (synonyms: annual demand, yearly demand, annual usage, demand per year)")
            if inputs.get("ordering_cost") is None:
                missing.append("ordering_cost (synonyms: ordering cost, order cost, setup cost, setup expense, cost per order, cost of placing an order, ordering/setup cost)")
            has_holding = inputs.get("holding_cost") is not None or (
                inputs.get("unit_cost") is not None and inputs.get("holding_rate") is not None
            )
            if not has_holding:
                missing.append("holding_cost (synonyms: holding cost, carrying cost, cost to hold, or unit price with holding rate %)")

        elif model_name == "quantity_discount":
            if inputs.get("annual_demand") is None:
                missing.append("annual_demand (synonyms: annual demand, yearly demand, annual usage)")
            if inputs.get("ordering_cost") is None:
                missing.append("ordering_cost (synonyms: ordering cost, order cost, setup cost, setup expense, cost per order, cost of placing an order)")
            if not inputs.get("tiers"):
                missing.append("tiers (price break schedule or quantity breaks with unit prices)")
            if inputs.get("holding_rate") is None and inputs.get("holding_cost") is None:
                missing.append("holding_rate or holding_cost (fixed holding cost per unit or percentage holding rate)")

        elif model_name in ("safety_stock_rop", "rop"):
            if inputs.get("avg_demand") is None:
                missing.append("avg_demand (synonyms: average demand, daily demand, mean demand)")
            if inputs.get("lead_time") is None:
                missing.append("lead_time (synonyms: lead time, replenishment lead time, days to arrive)")
            if inputs.get("std_demand") is None and inputs.get("std_lead_time") is None and inputs.get("safety_stock") is None:
                missing.append("demand or lead-time variability (std_demand, std_lead_time, or constant demand/lead time)")
            if inputs.get("service_level") is None and inputs.get("z_score") is None and inputs.get("safety_stock") is None:
                missing.append("service_level, z_score, or explicit safety_stock")

        elif model_name in ("fixed_period_review", "periodic_review"):
            if inputs.get("avg_demand") is None:
                missing.append("avg_demand (synonyms: average demand, daily demand)")
            if inputs.get("review_period") is None:
                missing.append("review_period (synonyms: review period, reviews inventory every X days, review cycle)")
            if inputs.get("lead_time") is None:
                missing.append("lead_time (synonyms: lead time, replenishment lead time)")
            if inputs.get("inventory_position") is None and inputs.get("on_hand_inventory") is None:
                missing.append("inventory_position or on_hand_inventory (and optional on-order and backorders)")
            if inputs.get("service_level") is None and inputs.get("z_score") is None and inputs.get("safety_stock") is None:
                missing.append("service_level, z_score, or explicit safety_stock")

        elif model_name == "newsvendor":
            has_direct_costs = inputs.get("underage_cost") is not None and inputs.get("overage_cost") is not None
            has_price_cost_salvage = (
                inputs.get("selling_price") is not None
                and inputs.get("unit_cost") is not None
                and inputs.get("salvage_value") is not None
            )
            if not (has_direct_costs or has_price_cost_salvage):
                missing.append("cost parameters: provide either (underage_cost and overage_cost) or (selling_price, purchase_cost, and salvage_value)")
            if inputs.get("distribution") == "uniform":
                if inputs.get("min_demand") is None or inputs.get("max_demand") is None:
                    missing.append("uniform distribution demand bounds: min_demand and max_demand")
            else:
                if inputs.get("mean_demand") is None or inputs.get("std_demand") is None:
                    missing.append("normal distribution parameters: mean_demand and std_demand")

        return missing

    # -----------------------------------------------------------------
    # Stage 4: Select Python Tool
    # -----------------------------------------------------------------
    def select_python_tool(self, model_name: str) -> Tuple[str, Callable]:
        """
        Select the appropriate deterministic Python calculation function from TOOL_REGISTRY.
        """
        if model_name not in TOOL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'. Available tools: {list(TOOL_REGISTRY.keys())}")
        tool_fn = TOOL_REGISTRY[model_name]
        meta_key = "safety_stock_rop" if model_name == "rop" else ("fixed_period_review" if model_name == "periodic_review" else model_name)
        tool_name = MODEL_METADATA[meta_key]["tool_name"]
        return tool_name, tool_fn

    # -----------------------------------------------------------------
    # Stage 5: Calculate (Strictly in Python)
    # -----------------------------------------------------------------
    def calculate(self, tool_fn: Callable, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Python inventory tool with the extracted inputs.
        Cleans None values before calling tool function.
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
    # Stage 7: Explain (Full Pedagogical Verification Pipeline)
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
        Generate a comprehensive, step-by-step verification explanation for students
        directly from the deterministic Python tool output, covering:
        1. Inputs Extracted
        2. Variable Definitions
        3. Formula
        4. Formula with Substituted Values
        5. Intermediate Calculations
        6. Final Answer
        7. Verification Checks
        8. Assumptions
        """
        model_title = MODEL_METADATA.get(
            "safety_stock_rop" if model_name == "rop" else ("fixed_period_review" if model_name == "periodic_review" else model_name),
            {},
        ).get("display_name", model_name)

        status_badge = "PASSED" if verification.get("is_valid") else "WARNING"
        checks_text = "\n".join(
            f"- [{'PASS' if c['passed'] else 'FAIL'}] **{c['check']}**: {c['detail']}"
            for c in verification.get("checks", [])
        )
        assumptions_text = "\n".join(f"- {a}" for a in results.get("assumptions", []))

        # 1. Inputs Extracted
        inp = results.get("inputs_extracted") or results.get("inputs", {})
        inp_lines = []
        for k, v in inp.items():
            if v is not None:
                if isinstance(v, list):
                    inp_lines.append(f"- **{k}**: {len(v)} tiers provided")
                elif isinstance(v, float):
                    inp_lines.append(f"- **{k}**: {v:,.4g}")
                else:
                    inp_lines.append(f"- **{k}**: {v}")
        inputs_section = "\n".join(inp_lines)

        # 2. Variable Definitions
        var_defs = results.get("variable_definitions", {})
        var_lines = [f"- **{var}**: {desc}" for var, desc in var_defs.items()]
        var_section = "\n".join(var_lines)

        # 3. Formula
        formula_dict = results.get("formula", {})
        form_lines = [f"- **{name.replace('_', ' ').title()}**: `{eq}`" for name, eq in formula_dict.items()]
        formula_section = "\n".join(form_lines)

        # 4. Formula with Substituted Values
        sub_dict = results.get("formula_with_substituted_values", {})
        sub_lines = [f"- **{name.replace('_', ' ').title()}**: {eq}" for name, eq in sub_dict.items() if eq]
        sub_section = "\n".join(sub_lines)

        # 5. Intermediate Calculations
        inter = results.get("intermediate_calculations", {})
        inter_lines = []
        if "tier_evaluations" in inter:
            for t in inter["tier_evaluations"]:
                max_str = f"{t['max_qty']:,.0f}" if t["max_qty"] is not None else "and above"
                marker = " **[Selected Optimal]**" if t.get("tier_index") == results.get("optimal_tier") else ""
                range_label = f"{t['min_qty']:,.0f} - {max_str} units" if t["max_qty"] is not None else f"{t['min_qty']:,.0f}+ units"
                inter_lines.append(
                    f"- **Tier {t['tier_index']} ({range_label} @ ${t['unit_cost']:,.2f}/unit)**: "
                    f"Unconstrained Q* = {t['unconstrained_eoq']:,.2f}, Evaluated Q = {t['evaluated_order_qty']:,.2f} | "
                    f"Ordering = ${t['annual_ordering_cost']:,.2f}, Holding = ${t['annual_holding_cost']:,.2f}, "
                    f"Purchase = ${t['annual_purchase_cost']:,.2f} -> **Total = ${t['total_annual_cost']:,.2f}**{marker}"
                )
        else:
            for k, v in inter.items():
                if v is not None:
                    if k == "alternative_5day_work_week" and isinstance(v, dict):
                        inter_lines.append(
                            f"- **Alternative 5-Day Work Week**: L = {v.get('lead_time_days', 0):g} days, "
                            f"sigma_L = {v.get('std_lead_time_days', 0):g} days -> "
                            f"Expected LTD = {v.get('expected_demand_during_lead_time', 0):,.2f} units, "
                            f"sigma_LT = {v.get('sigma_lt', 0):,.2f} units, "
                            f"Safety Stock = {v.get('safety_stock', 0):,.2f} units (approx. {v.get('safety_stock_rounded', 0):,} units), "
                            f"ROP = {v.get('reorder_point', 0):,.2f} units (approx. {v.get('reorder_point_rounded', 0):,} units)"
                        )
                    elif isinstance(v, float):
                        inter_lines.append(f"- **{k.replace('_', ' ').title()}**: {v:,.4g}")
                    else:
                        inter_lines.append(f"- **{k.replace('_', ' ').title()}**: {v}")
        inter_section = "\n".join(inter_lines)

        # 6. Final Answer
        ans = results.get("final_answer", {})
        ans_lines = []
        for k, v in ans.items():
            if k == "recommendation":
                continue
            if isinstance(v, float):
                ans_lines.append(f"- **{k.replace('_', ' ').title()}**: **{v:,.4g}**")
            else:
                ans_lines.append(f"- **{k.replace('_', ' ').title()}**: **{v}**")
        rec_text = ans.get("recommendation", "")
        if rec_text:
            ans_lines.append(f"\n> **Decision Summary**: {rec_text}")
        final_answer_section = "\n".join(ans_lines)

        return (
            f"### {model_title} — Step-by-Step Verification Report\n\n"
            f"#### 1. Inputs Extracted\n{inputs_section}\n\n"
            f"#### 2. Variable Definitions\n{var_section}\n\n"
            f"#### 3. Formula\n{formula_section}\n\n"
            f"#### 4. Formula with Substituted Values\n{sub_section}\n\n"
            f"#### 5. Intermediate Calculations (Deterministic Python)\n{inter_section}\n\n"
            f"#### 6. Final Answer\n{final_answer_section}\n\n"
            f"#### 7. Verification Checks ({status_badge})\n{checks_text}\n\n"
            f"#### 8. Assumptions\n{assumptions_text}"
        )

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

        Examines the entire question before declaring information missing.
        Does not report a parameter as missing when it is already present in the question.
        """
        pipeline_steps: List[Dict[str, Any]] = []

        pipeline_steps.append({
            "step": 1,
            "stage": "Customer Question",
            "status": "completed",
            "details": {"question": question},
        })

        model_name = override_model or self.identify_model(question)
        meta_key = "safety_stock_rop" if model_name == "rop" else ("fixed_period_review" if model_name == "periodic_review" else model_name)
        model_display = MODEL_METADATA[meta_key]["display_name"]
        pipeline_steps.append({
            "step": 2,
            "stage": "Identify Model",
            "status": "completed",
            "details": {"model_id": model_name, "display_name": model_display},
        })

        extracted_inputs = override_inputs if override_inputs is not None else self.extract_inputs(question, model_name)
        pipeline_steps.append({
            "step": 3,
            "stage": "Extract Inputs",
            "status": "completed",
            "details": extracted_inputs,
        })

        # Check for missing required inputs after examining the entire question
        missing_params = self.validate_extracted_inputs(model_name, extracted_inputs)
        if missing_params:
            raise ValueError(
                f"Missing required parameter(s) for {model_display}: {'; '.join(missing_params)}. "
                f"Extracted inputs: {extracted_inputs}"
            )

        tool_name, tool_fn = self.select_python_tool(model_name)
        pipeline_steps.append({
            "step": 4,
            "stage": "Select Python Tool",
            "status": "completed",
            "details": {
                "tool_name": tool_name,
                "description": MODEL_METADATA[meta_key]["description"],
            },
        })

        calculation_results = self.calculate(tool_fn, extracted_inputs)
        pipeline_steps.append({
            "step": 5,
            "stage": "Calculate",
            "status": "completed",
            "details": {
                "intermediate_calculations": calculation_results.get("intermediate_calculations"),
                "final_answer": calculation_results.get("final_answer"),
                "assumptions": calculation_results.get("assumptions"),
            },
        })

        verification = self.verify(model_name, extracted_inputs, calculation_results)
        pipeline_steps.append({
            "step": 6,
            "stage": "Verify",
            "status": "completed" if verification.get("is_valid") else "warning",
            "details": verification,
        })

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
