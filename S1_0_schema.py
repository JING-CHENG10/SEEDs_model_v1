# -*- coding: utf-8 -*-
"""
S1.0_schema: Basic data structures; PROCESSES comes from dict_v3 rather than hardcoding.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class Node:
    # M49 code as primary key; country_name is for display only.
    country: str
    iso3: str
    year: int
    commodity: str
    country_name: str = ''
    m49: str = ''
    Q0: float = 0.0
    D0: float = 0.0
    P0: float = 1.0
    eps_supply: float = 0.0
    eps_demand: float = 0.0
    eps_pop_demand: float = 0.0
    eps_income_demand: float = 0.0
    epsD_row: Dict[str, float] = field(default_factory=dict)
    eps_supply_temp: float = 0.0
    eps_supply_yield: float = 0.0
    Tmult: float = 1.0
    Ymult: float = 1.0
    e0_by_proc: Dict[str, float] = field(default_factory=dict)
    tax_unit: float = 0.0
    region_label_new: Optional[str] = None
    meta: Dict[str, float] = field(default_factory=dict)

    def q0_with_ty(self) -> float:
        return float(self.Q0) * (self.Tmult ** self.eps_supply_temp) * (self.Ymult ** self.eps_supply_yield)

@dataclass
class Universe:
    countries: List[str]
    iso3_by_country: Dict[str, str]
    commodities: List[str]
    years: List[int]
    m49_by_country: Dict[str, str] = field(default_factory=dict)
    country_by_m49: Dict[str, str] = field(default_factory=dict)
    processes: List[str] = field(default_factory=list)
    process_meta: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # For scenarios and matching
    region_aggMC_by_country: Dict[str, str] = field(default_factory=dict)
    item_cat2_by_commodity: Dict[str, str] = field(default_factory=dict)
    ssp_region_by_country: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Build reverse M49->country lookup when not provided explicitly."""
        if not self.country_by_m49 and self.m49_by_country:
            reverse: Dict[str, str] = {}
            for country, code in self.m49_by_country.items():
                if code is None:
                    continue
                code_str = str(code).strip()
                if not code_str:
                    continue
                try:
                    lowered = code_str.lower()
                except AttributeError:
                    lowered = ''
                if lowered in {'nan', 'inf', '-inf'}:
                    continue
                code_key = ''
                code_work = code_str.strip()
                if code_work.startswith("'"):
                    code_work = code_work[1:]
                code_work = code_work.strip()
                if code_work:
                    if code_work.count('.') == 1:
                        left, right = code_work.split('.', 1)
                        if left.isdigit() and right.strip('0') == '':
                            code_work = left
                    if code_work.isdigit():
                        code_key = f"'{code_work.zfill(3)}"
                    else:
                        code_key = f"'{code_work}"
                if not code_key:
                    continue
                reverse.setdefault(code_key, country)
            self.country_by_m49 = reverse

@dataclass
class ScenarioConfig:
    years_hist_start: int = 2010
    years_hist_end: int = 2020
    years_future: List[int] = field(default_factory=lambda: [2080])  # Restore execution for 2080 only.
  # years_future: List[int] = field(default_factory=lambda: [2030, 2040, 2050, 2060, 2070, 2080])
    feed_efficiency: float = 1.0
    seed_rate_default: float = 0.05

@dataclass
class ScenarioData:
    nodes: List[Node]
    universe: Universe
    config: ScenarioConfig = field(default_factory=ScenarioConfig)
