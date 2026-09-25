# -*- coding: utf-8 -*-
"""Focused regression tests for the biomass-energy MVP."""

from __future__ import annotations

import tempfile
import inspect
from pathlib import Path

import pandas as pd
import pytest

from S1_0_schema import Node, ScenarioData, Universe
from S0_51_prepare_bioenergy_feedstock_bridge import prepare_historical_feedstocks
from S3_0_ds_linear_regional import (
    build_linear_regional_model,
    is_region_aggregation_enabled,
    set_region_aggregation,
    solve_linear_regional,
)
from S3_3_bioenergy import (
    _target_feasibility,
    build_baseline_reconciliation,
    build_bioenergy_bundle,
    build_bioenergy_postsolve_assessment,
)
from S3_6_scenarios import ScenarioEffect, apply_scenario_to_data
from S4_1_results import summarize_market


def _universe() -> Universe:
    return Universe(
        countries=["'001", "'002"],
        iso3_by_country={"'001": "AAA", "'002": "BBB"},
        commodities=["Maize (corn)", "Sugar cane"],
        years=[2020, 2030, 2055, 2080],
        m49_by_country={"Country A": "'001", "Country B": "'002"},
        country_by_m49={"'001": "Country A", "'002": "Country B"},
        processes=[],
    )


def test_bundle_and_residual_reconciliation() -> None:
    universe = _universe()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        params = pd.DataFrame(
            [
                {
                    "feedstock": "maize_ethanol",
                    "feedstock_category": "crop",
                    "model_commodity": "Maize (corn)",
                    "market_link": True,
                    "carrier": "biogasoline",
                    "lhv_gj_per_tdm": 20.0,
                    "conversion_efficiency": 0.5,
                    "dry_matter_fraction": 0.8,
                }
            ]
        )
        historical = pd.DataFrame(
            [
                {
                    "scenario": "historical",
                    "M49_Country_Code": "001",
                    "year": 2020,
                    "feedstock": "maize_ethanol",
                    "feedstock_demand_t": 100.0,
                },
                {
                    "scenario": "historical",
                    "M49_Country_Code": "002",
                    "year": 2020,
                    "feedstock": "maize_ethanol",
                    "feedstock_demand_t": 300.0,
                },
            ]
        )
        scenario = pd.DataFrame(
            [
                {
                    "scenario": "high",
                    "M49_Country_Code": "World",
                    "year": 2030,
                    "feedstock": "maize_ethanol",
                    "energy_target_tj": 4.0,
                },
                {
                    "scenario": "high",
                    "M49_Country_Code": "World",
                    "year": 2080,
                    "feedstock": "maize_ethanol",
                    "energy_target_tj": 8.0,
                },
            ]
        )
        params_path = root / "params.csv"
        historical_path = root / "historical.csv"
        scenario_path = root / "scenario.csv"
        params.to_csv(params_path, index=False)
        historical.to_csv(historical_path, index=False)
        scenario.to_csv(scenario_path, index=False)

        bundle = build_bioenergy_bundle(
            universe=universe,
            active_years=universe.years,
            hist_end_year=2020,
            scenario="high",
            historical_feedstock_path=str(historical_path),
            scenario_path=str(scenario_path),
            parameter_path=str(params_path),
        )

        assert bundle.historical_crop_base_by_country_comm[("'001", "Maize (corn)")] == 100.0
        assert bundle.historical_crop_base_by_country_comm[("'002", "Maize (corn)")] == 300.0
        assert abs(bundle.crop_demand_by_country_comm_year[("'001", "Maize (corn)", 2080)] - 250.0) < 1e-9
        assert abs(bundle.crop_demand_by_country_comm_year[("'002", "Maize (corn)", 2080)] - 750.0) < 1e-9
        assert abs(bundle.crop_demand_by_country_comm_year[("'001", "Maize (corn)", 2055)] - 187.5) < 1e-9
        assert abs(bundle.crop_demand_by_country_comm_year[("'002", "Maize (corn)", 2055)] - 562.5) < 1e-9

        nodes = [
            Node(country="'001", iso3="AAA", year=2020, commodity="Maize (corn)", D0=1000.0),
        ]
        fbs = pd.DataFrame(
            [
                {
                    "country": "'001",
                    "year": 2020,
                    "commodity": "Maize (corn)",
                    "food_t": 600.0,
                    "feed_t": 100.0,
                }
            ]
        )
        check = build_baseline_reconciliation(
            nodes=nodes,
            fbs_components=fbs,
            historical_crop_base_by_country_comm=bundle.historical_crop_base_by_country_comm,
            hist_end_year=2020,
        ).iloc[0]
        assert check["residual_before_bioenergy_t"] == 300.0
        assert check["residual_after_bioenergy_t"] == 200.0
        assert check["bioenergy_overdraw_t"] == 0.0


def test_scenario_profile_effect() -> None:
    universe = _universe()
    effect = ScenarioEffect(
        scenario_id="S_BIO",
        kind="bioenergy_scenario",
        unit="profile",
        value_2080="residues_first",
        country_sel="All",
        commodity_sel="All",
        process_sel="All",
    )
    effect.countries = list(universe.countries)
    effect.commodities = list(universe.commodities)
    effect.processes = []
    ctx = apply_scenario_to_data([effect], "S_BIO", universe, [])
    assert ctx["bioenergy_scenario"] == "residues_first"


def test_carrier_feedstock_bridge_fallback() -> None:
    history = pd.DataFrame(
        [
            {
                "M49_Country_Code": "'001",
                "country_name": "Country A",
                "year": 2020,
                "carrier": "Biogasoline",
                "final_consumption_tj": 10.0,
            }
        ]
    )
    bridge = pd.DataFrame(
        [
            {
                "M49_Country_Code": "World",
                "year": 2019,
                "carrier": "Biogasoline",
                "feedstock": "maize_ethanol",
                "feedstock_category": "crop",
                "model_commodity": "Maize (corn)",
                "market_link": True,
                "share": 0.6,
                "source": "test",
                "notes": "",
            },
            {
                "M49_Country_Code": "World",
                "year": 2019,
                "carrier": "Biogasoline",
                "feedstock": "sugarcane_ethanol",
                "feedstock_category": "crop",
                "model_commodity": "Sugar cane",
                "market_link": True,
                "share": 0.4,
                "source": "test",
                "notes": "",
            },
        ]
    )
    rows, diagnostics = prepare_historical_feedstocks(history, bridge)
    assert abs(rows["energy_target_tj"].sum() - 10.0) < 1e-9
    assert diagnostics.iloc[0]["mapping_method"] == "world_latest_prior"
    assert diagnostics.iloc[0]["unmapped_energy_tj"] == 0.0


def test_market_summary_separates_food_and_bioenergy() -> None:
    class _Value:
        def __init__(self, value: float):
            self.X = value

    universe = _universe()
    node = Node(
        country="'001",
        iso3="AAA",
        year=2080,
        commodity="Maize (corn)",
        m49="'001",
    )
    data = ScenarioData(nodes=[node], universe=universe)
    key = ("'001", "Maize (corn)", 2080)
    var = {
        "Qs": {key: _Value(100.0)},
        "Qd": {key: _Value(100.0)},
        "Pc": {key: _Value(2.0)},
    }
    bioenergy = pd.DataFrame(
        [
            {
                "country": "'001",
                "year": 2080,
                "commodity": "Maize (corn)",
                "bioenergy_demand_t": 50.0,
            }
        ]
    )
    result = summarize_market(
        None,
        var,
        universe,
        data=data,
        bioenergy_sim_df=bioenergy,
    ).iloc[0]
    assert result["Qd"] == 100.0
    assert result["bioenergy_demand_t"] == 50.0
    assert result["market_total_use_t"] == 150.0
    assert result["net_import_t"] == 50.0


def test_solver_source_contains_explicit_bioenergy_balance() -> None:
    source = inspect.getsource(build_linear_regional_model)
    assert "qs_var + mi_var == qd_var + bioenergy_scaled" in source
    assert "'bioenergy_crop_demand_map': bioenergy_crop_demand_map" in source
    assert "energy_crop_land_requirement_map" in source
    assert "cropland_demand_eff = cropland_demand_eff + energy_crop_land_req" in source


def test_resource_constraints_and_handoffs() -> None:
    universe = _universe()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        params = pd.DataFrame(
            [
                {
                    "feedstock": "rice_straw",
                    "feedstock_category": "crop_residue",
                    "market_link": False,
                    "carrier": "solid_biomass",
                    "lhv_gj_per_tdm": 15.0,
                    "conversion_efficiency": 0.8,
                    "dry_matter_fraction": 0.9,
                    "ghg_direct_kgco2e_per_tdm": 10.0,
                    "ghg_soil_kgco2e_per_tdm": 20.0,
                    "ghg_avoided_kgco2e_per_tdm": 5.0,
                    "fossil_displacement_kgco2e_per_tj": 1000.0,
                },
                {
                    "feedstock": "miscanthus",
                    "feedstock_category": "dedicated_energy_crop",
                    "market_link": False,
                    "carrier": "solid_biomass",
                    "lhv_gj_per_tdm": 18.0,
                    "conversion_efficiency": 0.9,
                    "dry_matter_fraction": 1.0,
                    "yield_tdm_per_ha": 10.0,
                },
            ]
        )
        scenario = pd.DataFrame(
            [
                {
                    "scenario": "p1",
                    "M49_Country_Code": "001",
                    "year": 2030,
                    "feedstock": "rice_straw",
                    "energy_target_tj": 1.2,
                    "feedstock_demand_tdm": 100.0,
                },
                {
                    "scenario": "p1",
                    "M49_Country_Code": "001",
                    "year": 2030,
                    "feedstock": "miscanthus",
                    "feedstock_demand_tdm": 200.0,
                },
            ]
        )
        resources = pd.DataFrame(
            [
                {
                    "scenario": "p1",
                    "M49_Country_Code": "001",
                    "year": 2030,
                    "feedstock": "rice_straw",
                    "parent_commodity": "Rice",
                    "resource_available_tdm": 80.0,
                    "competing_use_tdm": 10.0,
                    "sustainable_fraction": 0.5,
                    "residue_carbon_pct": 45.0,
                    "residue_nitrogen_pct": 1.0,
                    "residue_quality_n_obs": 12,
                },
                {
                    "scenario": "p1",
                    "M49_Country_Code": "001",
                    "year": 2030,
                    "feedstock": "miscanthus",
                    "eligible_land_area_ha": 15.0,
                },
            ]
        )
        params_path = root / "params.csv"
        scenario_path = root / "scenario.csv"
        resources_path = root / "resources.csv"
        params.to_csv(params_path, index=False)
        scenario.to_csv(scenario_path, index=False)
        resources.to_csv(resources_path, index=False)

        bundle = build_bioenergy_bundle(
            universe=universe,
            active_years=[2020, 2030],
            hist_end_year=2020,
            scenario="p1",
            historical_feedstock_path=None,
            scenario_path=str(scenario_path),
            parameter_path=str(params_path),
            resource_path=str(resources_path),
        )

        residue = bundle.resource_balance[
            bundle.resource_balance["feedstock"].eq("rice_straw")
        ].iloc[0]
        assert residue["resource_status"] == "resource_overdraw"
        assert abs(residue["sustainable_supply_tdm"] - 30.0) < 1e-9
        assert abs(residue["feasible_feedstock_demand_tdm"] - 30.0) < 1e-9
        assert abs(residue["resource_gap_tdm"] - 70.0) < 1e-9
        assert abs(residue["unmet_energy_tj"] - 0.84) < 1e-9

        target = bundle.target_feasibility
        residue_target = target[
            target["feedstock_category_group"].eq("crop_residue")
        ].iloc[0]
        assert abs(residue_target["target_energy_tj"] - 1.2) < 1e-9
        assert abs(residue_target["feasible_supplied_tj"] - 0.36) < 1e-9
        assert abs(residue_target["crop_residue_cap_unmet_tj"] - 0.84) < 1e-9
        assert residue_target["dominant_gap_reason"] == "crop_residue_cap"

        emis = bundle.emissions_handoff[
            bundle.emissions_handoff["feedstock"].eq("rice_straw")
        ].iloc[0]
        assert emis["handoff_status"] == "ready_for_emissions_integration"
        expected_net = (30.0 * 10.0 + 30.0 * 20.0 - 30.0 * 5.0 - 0.36 * 1000.0) / 1_000_000.0
        assert abs(emis["net_biomass_emissions_ktco2e"] - expected_net) < 1e-12

        residue_handoff = bundle.residue_management_handoff.iloc[0]
        assert residue_handoff["feedstock"] == "rice_straw"
        assert abs(residue_handoff["bioenergy_residue_removed_tdm"] - 30.0) < 1e-9
        assert abs(residue_handoff["residue_removed_fraction_total"] - 0.375) < 1e-9
        assert abs(residue_handoff["bioenergy_residue_c_removed_t"] - 13.5) < 1e-9
        assert abs(residue_handoff["bioenergy_residue_n_removed_t"] - 0.3) < 1e-9
        assert abs(residue_handoff["crop_residue_n2o_multiplier"] - 0.625) < 1e-9
        assert residue_handoff["feed_competition_status"] == "bioenergy_overdraw_after_feed_soil_screen"
        assert bundle.crop_residue_management_multiplier[("'001", "Rice", "Crop residues", 2030)] == 0.625
        assert bundle.crop_residue_management_multiplier[("'001", "Rice", "Burning crop residues", 2030)] == 0.625

        land = bundle.land_handoff[
            bundle.land_handoff["feedstock"].eq("miscanthus")
        ].iloc[0]
        energy_crop = bundle.resource_balance[
            bundle.resource_balance["feedstock"].eq("miscanthus")
        ].iloc[0]
        assert energy_crop["resource_status"] == "resource_overdraw"
        assert abs(energy_crop["resource_available_tdm"] - 150.0) < 1e-9
        assert abs(energy_crop["feasible_feedstock_demand_tdm"] - 150.0) < 1e-9
        assert land["handoff_status"] == "eligible_land_overdraw"
        assert abs(land["land_requirement_ha"] - 20.0) < 1e-9
        assert abs(land["feasible_land_requirement_ha"] - 15.0) < 1e-9
        assert abs(land["land_gap_ha"] - 5.0) < 1e-9
        assert bundle.energy_crop_land_requirement_by_country_year[("'001", 2030)] == 15.0

        energy_crop_target = target[
            target["feedstock_category_group"].eq("dedicated_energy_crop")
        ].iloc[0]
        assert abs(energy_crop_target["target_energy_tj"] - 3.24) < 1e-9
        assert abs(energy_crop_target["feasible_supplied_tj"] - 2.43) < 1e-9
        assert abs(energy_crop_target["eligible_land_cap_unmet_tj"] - 0.81) < 1e-9
        assert energy_crop_target["dominant_gap_reason"] == "eligible_land_cap"


def test_solver_micro_bioenergy_balance_solves() -> None:
    years = [2020, 2030]
    commodity = "Maize (corn)"
    nodes = [
        Node(
            country="'001",
            country_name="Country A",
            iso3="AAA",
            m49="'001",
            year=year,
            commodity=commodity,
            Q0=120.0,
            D0=100.0,
            P0=1.0,
            eps_supply=0.05,
            eps_demand=-0.05,
            e0_by_proc={},
            meta={"yield0": 10.0},
        )
        for year in years
    ]
    previous_region_aggregation = is_region_aggregation_enabled()
    try:
        set_region_aggregation(False)
        result = solve_linear_regional(
            nodes=nodes,
            commodities=[commodity],
            years=years,
            time_limit=60,
            cost_calculation_method="off",
            demand_method="elasticity",
            feed_crop_link_mode="off",
            population_by_country_year={("'001", 2020): 1.0, ("'001", 2030): 1.0},
            income_mult_by_country_year={("'001", 2020): 1.0, ("'001", 2030): 1.0},
            bioenergy_crop_demand_by_country_comm_year={("'001", commodity, 2030): 10.0},
            nutrition_residual_demand_by_country_comm_year={("'001", commodity, 2030): 0.0},
            market_clearing_mode="country_trade",
            trade_cap_ratio=None,
            land_area_limits={("'001", 2020): 900.0},
            base_cropland_by_region={"'001": 100.0},
            base_grassland_by_region={"'001": 100.0},
            base_forest_by_region={"'001": 800.0},
            land_soft_constraints_enabled=True,
            land_slack_penalty=1.0,
            future_last_only=True,
            enable_output_diagnostics=False,
            enable_verbose_logging=False,
        )
    finally:
        set_region_aggregation(previous_region_aggregation)
    assert result["status"] == 2
    assert result["status_name"] == "optimal"
    assert result["sol_count"] >= 1
    assert result["has_solution"] is True
    assert result["runtime_seconds"] >= 0.0
    assert result["land_slack"][("'001", 2030)] > 0.0
    key = ("'001", commodity, 2030)
    assert result["Qs"][key] + 1e-6 >= result["Qd"][key] + 10.0


def test_enabled_bioenergy_requires_matching_scenario_rows() -> None:
    universe = _universe()
    with tempfile.TemporaryDirectory() as tmp:
        scenario_path = Path(tmp) / "scenario.csv"
        pd.DataFrame(
            [
                {
                    "scenario": "different_scenario",
                    "M49_Country_Code": "001",
                    "year": 2030,
                    "feedstock": "maize_ethanol",
                    "energy_target_tj": 1.0,
                }
            ]
        ).to_csv(scenario_path, index=False)

        with pytest.raises(ValueError, match="no usable rows"):
            build_bioenergy_bundle(
                universe=universe,
                active_years=universe.years,
                hist_end_year=2020,
                scenario="medium_bioenergy",
                historical_feedstock_path=None,
                scenario_path=str(scenario_path),
                parameter_path=None,
                require_scenario_rows=True,
            )

        empty_bundle = build_bioenergy_bundle(
            universe=universe,
            active_years=universe.years,
            hist_end_year=2020,
            scenario="medium_bioenergy",
            historical_feedstock_path=None,
            scenario_path=str(scenario_path),
            parameter_path=None,
            require_scenario_rows=False,
        )
        assert empty_bundle.scenario_detail.empty
        assert not empty_bundle.enabled


def test_pre_solver_feasibility_keeps_unattributed_gap_out_of_solver_gap() -> None:
    detail = pd.DataFrame(
        [
            {
                "scenario": "test",
                "M49_Country_Code": "'001",
                "country_name": "Country A",
                "year": 2030,
                "carrier": "biogasoline",
                "feedstock": "maize_ethanol",
                "feedstock_category": "crop",
                "market_link": True,
                "resource_status": "resource_overdraw",
                "energy_target_tj": 100.0,
                "feasible_energy_supplied_tj": 80.0,
                "unmet_energy_tj": 0.0,
                "land_gap_ha": 0.0,
                "feedstock_demand_tdm": 1000.0,
                "lhv_gj_per_tdm": 20.0,
                "conversion_efficiency": 0.5,
            }
        ]
    )
    row = _target_feasibility(detail).iloc[0]
    assert row["diagnostic_stage"] == "pre_solver"
    assert row["market_land_solver_unmet_tj"] == 0.0
    assert row["unmet_energy_tj"] == 20.0
    assert row["unattributed_pre_solver_unmet_tj"] == 20.0
    assert row["attributed_unmet_tj"] == row["unmet_energy_tj"]
    assert row["attribution_residual_tj"] == 0.0
    assert row["dominant_gap_reason"] == "unattributed_pre_solver"


def test_postsolve_assessment_distinguishes_valid_stress_and_failed() -> None:
    feasible = pd.DataFrame(
        [
            {
                "scenario": "test",
                "year": 2030,
                "target_energy_tj": 10.0,
                "feasible_supplied_tj": 10.0,
                "unmet_energy_tj": 0.0,
                "non_crop_cap_unmet_tj": 0.0,
                "crop_residue_cap_unmet_tj": 0.0,
                "eligible_land_cap_unmet_tj": 0.0,
                "unattributed_pre_solver_unmet_tj": 0.0,
                "attribution_residual_tj": 0.0,
            }
        ]
    )
    valid = build_bioenergy_postsolve_assessment(
        feasible,
        {"status": 2, "Qs": {("'001", "Maize", 2030): 1.0}, "shortage": {}, "excess": {}},
    ).iloc[0]
    assert valid["assessment_status"] == "valid"
    assert bool(valid["target_feasible"])

    stress = build_bioenergy_postsolve_assessment(
        feasible,
        {
            "status": 2,
            "Qs": {("'001", "Maize", 2030): 1.0},
            "shortage": {("Maize", 2030): 2.0},
            "excess": {},
        },
    ).iloc[0]
    assert stress["assessment_status"] == "stress_only"
    assert "market_shortage_slack" in stress["assessment_reasons"]

    land_stress = build_bioenergy_postsolve_assessment(
        feasible,
        {
            "status": 2,
            "Qs": {("'001", "Maize", 2030): 1.0},
            "shortage": {},
            "excess": {},
            "land_slack": {("'001", 2030): 2.0},
        },
        land_tolerance_ha=1.0,
    ).iloc[0]
    assert land_stress["assessment_status"] == "stress_only"
    assert land_stress["land_constraint_slack_ha"] == 2.0
    assert "land_constraint_slack" in land_stress["assessment_reasons"]

    failed = build_bioenergy_postsolve_assessment(
        feasible,
        {"status": 3, "shortage": {}, "excess": {}},
    ).iloc[0]
    assert failed["assessment_status"] == "failed"
    assert not bool(failed["solver_has_solution"])


def main() -> None:
    test_bundle_and_residual_reconciliation()
    test_scenario_profile_effect()
    test_carrier_feedstock_bridge_fallback()
    test_market_summary_separates_food_and_bioenergy()
    test_solver_source_contains_explicit_bioenergy_balance()
    test_resource_constraints_and_handoffs()
    test_solver_micro_bioenergy_balance_solves()
    test_enabled_bioenergy_requires_matching_scenario_rows()
    test_pre_solver_feasibility_keeps_unattributed_gap_out_of_solver_gap()
    test_postsolve_assessment_distinguishes_valid_stress_and_failed()
    print("bioenergy MVP tests passed")


if __name__ == "__main__":
    main()
