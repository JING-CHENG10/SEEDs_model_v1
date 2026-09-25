# Code catalog: Net-zero food (`new_clear`)

Source snapshot: 22 September 2026.

Companion to the [English model guide](MODEL_GUIDE_en.md). This catalog covers every active top-level Python, Shell, SQL, and notebook source, with a separate inventory of historical audit code. Direct imports are statically detected, including imports inside functions; dynamic loading is not a complete call graph. File-reference strings are clues, not a guarantee that a file is read or written in every execution.

**Coverage:** 192 active source files; 18 archived source files. Full interfaces, imports, file references, configuration snapshots, and SHA-256 hashes are in [source_inventory.json](source_inventory.json).

## Shared utilities and cost-analysis tools

### `build_cost_database_comparison_report.py`

Builds the cost-database comparison report from analysis outputs; this is a reporting utility, not the runtime cost loader.

[Source](../../build_cost_database_comparison_report.py) · 727 lines.

Entry points / interfaces: [`build_artifact`](../../build_cost_database_comparison_report.py#L26).

Direct local imports: [config_paths.py](../../config_paths.py).

### `config_paths.py`

Locates the shared Code root and input, src, output, and LUH2 directories; supports NZF_* environment overrides across operating systems.

[Source](../../config_paths.py) · 80 lines.

Entry points / interfaces: [`get_code_root`](../../config_paths.py#L15), [`get_input_base`](../../config_paths.py#L35), [`get_src_base`](../../config_paths.py#L48), [`get_results_base`](../../config_paths.py#L61), [`get_luh2_data_base`](../../config_paths.py#L75).

Imported by: [build_cost_database_comparison_report.py](../../build_cost_database_comparison_report.py), [cost_database_comparison_analysis.py](../../cost_database_comparison_analysis.py), [gce_emissions_complete.py](../../gce_emissions_complete.py), [gle_emissions_complete.py](../../gle_emissions_complete.py), [luc_emission_module.py](../../luc_emission_module.py), [S0_03_feed_system_cal.py](../../S0_03_feed_system_cal.py), [S0_08_make_LUC_mask_WGS.py](../../S0_08_make_LUC_mask_WGS.py), [S0_12_compute_pasture_DMyield_by_country.py](../../S0_12_compute_pasture_DMyield_by_country.py), [S0_16_gce_crop_parameters.py](../../S0_16_gce_crop_parameters.py), [S0_17_LUH2_data_analysis.py](../../S0_17_LUH2_data_analysis.py), [S0_18_soil_drained_EF_refill.py](../../S0_18_soil_drained_EF_refill.py), [S0_19_historical_max_production.py](../../S0_19_historical_max_production.py); further references are in the inventory.

### `cost_database_comparison_analysis.ipynb`

Interactive companion to the cost-database comparison analysis; inspects generated comparison tables and diagnostics without defining the main food-system equations.

[Source](../../cost_database_comparison_analysis.ipynb) · 1,232 lines.

### `cost_database_comparison_analysis.py`

Analyzes cost-database versions and their country-strategy differences; creates comparison tables for the companion notebook and report workflow, outside the model solver.

[Source](../../cost_database_comparison_analysis.py) · 517 lines.

Entry points / interfaces: [`run_analysis`](../../cost_database_comparison_analysis.py#L59).

Direct local imports: [S2_0_load_data.py](../../S2_0_load_data.py), [config_paths.py](../../config_paths.py).

### `cost_database_comparison_report_source.sql`

SQL source queries for the cost-database comparison report; table paths and query inputs belong to the analysis/reporting workflow.

[Source](../../cost_database_comparison_report_source.sql) · 82 lines.

### `cost_database_v2.py`

Validates the versioned nine-strategy cost JSON, including schema, units, target year, country coverage, unique keys, strategy layers, and final costs; returns a solver-ready cost map with provenance.

[Source](../../cost_database_v2.py) · 732 lines.

Entry points / interfaces: [`CostDatabaseValidationError`](../../cost_database_v2.py#L89), [`MitigationCostDatabase`](../../cost_database_v2.py#L94), [`file_sha256`](../../cost_database_v2.py#L140), [`load_cost_database_v2`](../../cost_database_v2.py#L515).

Imported by: [S4_0_main.py](../../S4_0_main.py), [S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py](../../S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py).

### `emission_module_manifest.py`

Tracks required emissions modules, completion status, row counts, and errors; writes the per-run manifest used to reject incomplete totals.

[Source](../../emission_module_manifest.py) · 167 lines.

Entry points / interfaces: [`count_result_rows`](../../emission_module_manifest.py#L26), [`new_emission_module_records`](../../emission_module_manifest.py#L37), [`mark_emission_module`](../../emission_module_manifest.py#L55), [`emission_module_records_complete`](../../emission_module_manifest.py#L115), [`write_emission_module_manifest`](../../emission_module_manifest.py#L135).

Imported by: [S4_0_main.py](../../S4_0_main.py).

### `figure5_production_ef_report_source.sql`

DuckDB queries over production-EF audit and quartile outputs; distinguishes production emissions per production calorie from sampled EF multipliers. Relative paths assume the Code working directory.

[Source](../../figure5_production_ef_report_source.sql) · 57 lines.

### `market_balance_diagnostics.py`

Reconstructs country-commodity market balances from supply, demand, net imports, explicit bioenergy demand, and slack; produces diagnostics used by S4 and S5.

[Source](../../market_balance_diagnostics.py) · 462 lines.

Entry points / interfaces: [`build_market_balance_diagnostics`](../../market_balance_diagnostics.py#L132), [`summarize_market_balance_frame`](../../market_balance_diagnostics.py#L246), [`read_market_balance_summary`](../../market_balance_diagnostics.py#L416), [`validate_market_balance_gap`](../../market_balance_diagnostics.py#L437).

Imported by: [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [S4_0_main.py](../../S4_0_main.py), [S5_1_1_sensitivity_mc_variable_effect.py](../../S5_1_1_sensitivity_mc_variable_effect.py), [S5_3_1_sensitivity_panel_yield_ef.py](../../S5_3_1_sensitivity_panel_yield_ef.py), [S5_4_1_monte_carlo_full_variables.py](../../S5_4_1_monte_carlo_full_variables.py), [ST_bioenergy_full_run_smoke.py](../../ST_bioenergy_full_run_smoke.py).

### `mc_bound_utils.py`

Parses legacy Monte Carlo bounds, including Y2020 baseline multipliers, and preserves the distinction between change rates and direct multipliers.

[Source](../../mc_bound_utils.py) · 57 lines.

Entry points / interfaces: [`draw_mc_multiplier`](../../mc_bound_utils.py#L35).

Imported by: [S3_0_ds_emis_mc_full.py](../../S3_0_ds_emis_mc_full.py), [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [S3_6_scenarios.py](../../S3_6_scenarios.py).

### `mc_sample_utils.py`

Validates sampled driver and emission-factor maps before in-place solver updates, including finite values and the logarithmic path's positivity requirements.

[Source](../../mc_sample_utils.py) · 27 lines.

Entry points / interfaces: [`validated_multipliers`](../../mc_sample_utils.py#L8).

Imported by: [S3_0_ds_emis_mc_full.py](../../S3_0_ds_emis_mc_full.py), [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py).

### `model_run_status.py`

Normalizes solver outcomes, writes structured run status, and validates provenance and artifact integrity before resume; does not import Gurobi to read status records.

[Source](../../model_run_status.py) · 720 lines.

Entry points / interfaces: [`ResumeValidation`](../../model_run_status.py#L69), [`ArtifactIntegrityValidation`](../../model_run_status.py#L86), [`build_resume_fingerprint`](../../model_run_status.py#L98), [`solver_status_name`](../../model_run_status.py#L115), [`solver_status_has_solution`](../../model_run_status.py#L132), [`solver_result_is_extractable`](../../model_run_status.py#L160), [`normalize_solver_status`](../../model_run_status.py#L181), [`write_run_status`](../../model_run_status.py#L233). Additional interfaces are listed in the source inventory.

Imported by: [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [S3_3_bioenergy.py](../../S3_3_bioenergy.py), [S4_0_main.py](../../S4_0_main.py), [S5_0_1_sensitivity_mc_levels.py](../../S5_0_1_sensitivity_mc_levels.py), [S5_1_1_sensitivity_mc_variable_effect.py](../../S5_1_1_sensitivity_mc_variable_effect.py), [S5_3_1_sensitivity_panel_yield_ef.py](../../S5_3_1_sensitivity_panel_yield_ef.py), [S5_4_1_monte_carlo_full_variables.py](../../S5_4_1_monte_carlo_full_variables.py), [S5_6_1_max_emission_reduction_potential.py](../../S5_6_1_max_emission_reduction_potential.py), [ST_bioenergy_full_run_smoke.py](../../ST_bioenergy_full_run_smoke.py).

### `runtime_data_cache.py`

Caches CSV, Excel, and other tabular reads using file signatures and read arguments; exposes cache clearing for changed inputs.

[Source](../../runtime_data_cache.py) · 296 lines.

Entry points / interfaces: [`get_runtime_cache_dir`](../../runtime_data_cache.py#L69), [`read_excel_cached`](../../runtime_data_cache.py#L215), [`read_csv_cached`](../../runtime_data_cache.py#L247), [`read_pickle_cached`](../../runtime_data_cache.py#L256), [`read_excel_sidecar_cached`](../../runtime_data_cache.py#L265), [`read_tabular_cached`](../../runtime_data_cache.py#L281), [`clear_memory_caches`](../../runtime_data_cache.py#L293).

Imported by: [gce_emissions_complete.py](../../gce_emissions_complete.py), [gle_emissions_complete.py](../../gle_emissions_complete.py), [S0_50_prepare_bioenergy_history.py](../../S0_50_prepare_bioenergy_history.py), [S0_51_prepare_bioenergy_feedstock_bridge.py](../../S0_51_prepare_bioenergy_feedstock_bridge.py), [S2_0_load_data.py](../../S2_0_load_data.py), [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [S3_2_feed_demand.py](../../S3_2_feed_demand.py), [S3_3_bioenergy.py](../../S3_3_bioenergy.py), [S4_0_main.py](../../S4_0_main.py), [S4_1_results.py](../../S4_1_results.py).

## Physical emissions modules

### `gce_emissions_complete.py`

Main crop-emissions engine: uses production, harvested area, residues, rice, and fertilizer parameters to calculate process-specific greenhouse gases for S4 reporting.

[Source](../../gce_emissions_complete.py) · 1,344 lines.

Entry points / interfaces: [`print`](../../gce_emissions_complete.py#L51), [`CropEmissionsCalculator`](../../gce_emissions_complete.py#L213), [`run_crop_emissions`](../../gce_emissions_complete.py#L1302).

Direct local imports: [config_paths.py](../../config_paths.py), [runtime_data_cache.py](../../runtime_data_cache.py).

Imported by: [S4_0_main.py](../../S4_0_main.py).

### `gce_emissions_module_fao_legacy.py`

Legacy crop-process functions for residue N2O, residue burning, rice CH4, and fertilizer emissions; do not add these results to the complete engine indiscriminately.

[Source](../../gce_emissions_module_fao_legacy.py) · 551 lines.

Entry points / interfaces: [`load_params_wide`](../../gce_emissions_module_fao_legacy.py#L78), [`get_param_wide`](../../gce_emissions_module_fao_legacy.py#L96), [`compute_crop_residues_direct`](../../gce_emissions_module_fao_legacy.py#L241), [`compute_crop_residues_indirect`](../../gce_emissions_module_fao_legacy.py#L285), [`compute_burning`](../../gce_emissions_module_fao_legacy.py#L325), [`compute_synth_fert_direct`](../../gce_emissions_module_fao_legacy.py#L371), [`compute_synth_fert_indirect`](../../gce_emissions_module_fao_legacy.py#L397), [`compute_rice_ch4`](../../gce_emissions_module_fao_legacy.py#L430). Additional interfaces are listed in the source inventory.

### `gfe_emissions_module_fao_legacy.py`

Legacy forest-remaining-forest carbon-stock accounting. GFE here denotes forestry, not the GFISH aquaculture module, and is not the full land-transition engine.

[Source](../../gfe_emissions_module_fao_legacy.py) · 253 lines.

Entry points / interfaces: [`get_param`](../../gfe_emissions_module_fao_legacy.py#L77), [`compute_forest_fl_fl_biomass`](../../gfe_emissions_module_fao_legacy.py#L132), [`compute_forest_fl_fl_soil_tier1_zero`](../../gfe_emissions_module_fao_legacy.py#L182), [`run_gfe_forest_only`](../../gfe_emissions_module_fao_legacy.py#L208), [`run_gfe`](../../gfe_emissions_module_fao_legacy.py#L234).

### `gfire_emission_fixed_module.py`

Standardizes historical savanna and peatland fire emissions into a fixed reference series; does not predict future weather-driven fire occurrence.

[Source](../../gfire_emission_fixed_module.py) · 180 lines.

Entry points / interfaces: [`load_fixed_gfire_emissions`](../../gfire_emission_fixed_module.py#L92).

Imported by: [S4_0_main.py](../../S4_0_main.py).

### `gfish_emission_module_complete.py`

Reads historical fisheries/aquaculture panels and computes future emissions from aquaculture shares, production, area, and CH4/N2O parameters.

[Source](../../gfish_emission_module_complete.py) · 584 lines.

Entry points / interfaces: [`run_fish_emissions`](../../gfish_emission_module_complete.py#L342).

Imported by: [S4_0_main.py](../../S4_0_main.py).

### `gle_emissions_complete.py`

Main livestock-emissions engine: reconstructs stocks and producing animals from output and computes enteric fermentation, manure management, manure application, and pasture deposition emissions.

[Source](../../gle_emissions_complete.py) · 3,534 lines.

Entry points / interfaces: [`LivestockEmissionsCalculator`](../../gle_emissions_complete.py#L73), [`calculate_stock_from_optimized_production`](../../gle_emissions_complete.py#L3266), [`run_livestock_emissions`](../../gle_emissions_complete.py#L3465).

Direct local imports: [config_paths.py](../../config_paths.py), [runtime_data_cache.py](../../runtime_data_cache.py).

Imported by: [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [S4_0_main.py](../../S4_0_main.py).

### `gle_emissions_module_fao_legacy.py`

Deprecated livestock implementation retained for historical comparisons, including slaughter-stock regression and process functions; the main pipeline uses gle_emissions_complete.py.

[Source](../../gle_emissions_module_fao_legacy.py) · 880 lines.

Entry points / interfaces: [`load_params_wide`](../../gle_emissions_module_fao_legacy.py#L117), [`get_param_wide`](../../gle_emissions_module_fao_legacy.py#L173), [`prepare_regression_data_from_fao`](../../gle_emissions_module_fao_legacy.py#L243), [`fit_stock_from_slaughter`](../../gle_emissions_module_fao_legacy.py#L296), [`mc_animals_from_production`](../../gle_emissions_module_fao_legacy.py#L404), [`mc_predict_stock`](../../gle_emissions_module_fao_legacy.py#L491), [`compute_enteric_ch4`](../../gle_emissions_module_fao_legacy.py#L550), [`compute_prp_n2o`](../../gle_emissions_module_fao_legacy.py#L595). Additional interfaces are listed in the source inventory.

### `gos_emissions_module_fao_legacy.py`

Legacy drained-organic-soil CO2/N2O functions with climate classification and parameter fallback; separate from the current complete soil engine.

[Source](../../gos_emissions_module_fao_legacy.py) · 332 lines.

Entry points / interfaces: [`load_params_wide`](../../gos_emissions_module_fao_legacy.py#L50), [`get_param_wide`](../../gos_emissions_module_fao_legacy.py#L114), [`compute_gv_n2o`](../../gos_emissions_module_fao_legacy.py#L215), [`compute_gv_co2`](../../gos_emissions_module_fao_legacy.py#L263), [`run_gv`](../../gos_emissions_module_fao_legacy.py#L312).

### `gsoil_emission_complete.py`

Uses historical inventories for past years and cropland/pasture activity with soil parameters for future drained-organic-soil CO2 and N2O.

[Source](../../gsoil_emission_complete.py) · 619 lines.

Entry points / interfaces: [`normalize_m49`](../../gsoil_emission_complete.py#L20), [`DrainedOrganicSoilsEmissions`](../../gsoil_emission_complete.py#L173), [`run_drained_organic_soils_emissions`](../../gsoil_emission_complete.py#L594).

Imported by: [S0_48_history_emission_summary.py](../../S0_48_history_emission_summary.py), [S4_0_main.py](../../S4_0_main.py).

### `lme_manure_module_fao.py`

Standalone livestock manure nitrogen-flow and treatment-system calculations from excretion, allocation, volatilization, and leaching parameters; the current S4 livestock chain principally uses GLE.

[Source](../../lme_manure_module_fao.py) · 405 lines.

Entry points / interfaces: [`load_parameters_wide`](../../lme_manure_module_fao.py#L96), [`get_param`](../../lme_manure_module_fao.py#L172), [`get_share`](../../lme_manure_module_fao.py#L246), [`get_loss_frac`](../../lme_manure_module_fao.py#L251), [`get_frac_scalar`](../../lme_manure_module_fao.py#L256), [`compute_record`](../../lme_manure_module_fao.py#L265), [`run_lme_from_wide`](../../lme_manure_module_fao.py#L348).

### `luc_emission_module.py`

Land-carbon bookkeeping for land transitions, wood harvest, shifting cultivation, biomass, soil, and harvested-wood pools; accepts spatial historical inputs and future aggregated solver land states.

[Source](../../luc_emission_module.py) · 2,264 lines.

Entry points / interfaces: [`load_params_from_excel`](../../luc_emission_module.py#L101), [`estimate_area_ha`](../../luc_emission_module.py#L492), [`discover_transitions`](../../luc_emission_module.py#L574), [`allocate_coarse_transitions_for_year`](../../luc_emission_module.py#L613), [`allocate_roundwood_for_year`](../../luc_emission_module.py#L667), [`aggregate_country_year`](../../luc_emission_module.py#L709), [`run_luc_bookkeeping`](../../luc_emission_module.py#L794), [`normalize_m49`](../../luc_emission_module.py#L1113). Additional interfaces are listed in the source inventory.

Direct local imports: [config_paths.py](../../config_paths.py).

Imported by: [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [S4_0_main.py](../../S4_0_main.py).

### `luc_historical_module.py`

Loads and normalizes historical LULUCF processes, including forest, wood harvest, deforestation, and abandonment, for combination with future land-carbon results.

[Source](../../luc_historical_module.py) · 177 lines.

Entry points / interfaces: [`read_luc_historical_emissions`](../../luc_historical_module.py#L18).

Imported by: [S4_0_main.py](../../S4_0_main.py).

## Input preparation and historical processing

### `S0_01_elasticity_processor.py`

Maps raw elasticity observations to country-commodity tables, separating supply/demand, own/cross-price, temperature, and income responses.

[Source](../../S0_01_elasticity_processor.py) · 439 lines.

Entry points / interfaces: [`norm`](../../S0_01_elasticity_processor.py#L49), [`canon`](../../S0_01_elasticity_processor.py#L57), [`lc`](../../S0_01_elasticity_processor.py#L65), [`find_col`](../../S0_01_elasticity_processor.py#L69), [`build_maps_many_to_many`](../../S0_01_elasticity_processor.py#L81), [`classify_element_ds`](../../S0_01_elasticity_processor.py#L106), [`agg_stats`](../../S0_01_elasticity_processor.py#L240), [`fetch_non_cross`](../../S0_01_elasticity_processor.py#L257). Additional interfaces are listed in the source inventory.

### `S0_02_elasticity_region_fill.py`

Fills missing elasticities using regional and applicable commodity-category means; preserves other workbook sheets and treats cross-price responses separately.

[Source](../../S0_02_elasticity_region_fill.py) · 238 lines.

Entry points / interfaces: [`norm`](../../S0_02_elasticity_region_fill.py#L42), [`canon`](../../S0_02_elasticity_region_fill.py#L50), [`find_col`](../../S0_02_elasticity_region_fill.py#L54), [`load_mappings`](../../S0_02_elasticity_region_fill.py#L71), [`fill_non_cross_with_region_and_cat`](../../S0_02_elasticity_region_fill.py#L102), [`fill_cross_mean_with_region`](../../S0_02_elasticity_region_fill.py#L160).

### `S0_03_feed_system_cal.py`

Combines production, animal stocks, food balances, and GLEAM inputs into livestock feeding-system matrices, per-head requirements, and grass shares.

[Source](../../S0_03_feed_system_cal.py) · 706 lines.

Entry points / interfaces: [`to_head_df`](../../S0_03_feed_system_cal.py#L72), [`auto_region`](../../S0_03_feed_system_cal.py#L81), [`default_system_shares`](../../S0_03_feed_system_cal.py#L97), [`seed_rations`](../../S0_03_feed_system_cal.py#L139), [`aggregate_rations`](../../S0_03_feed_system_cal.py#L234), [`load_inputs`](../../S0_03_feed_system_cal.py#L254), [`prep_feed_matrices`](../../S0_03_feed_system_cal.py#L279), [`to_head_df_filter`](../../S0_03_feed_system_cal.py#L290). Additional interfaces are listed in the source inventory.

Direct local imports: [config_paths.py](../../config_paths.py).

### `S0_04_world_price_perUnit_cal.py`

Divides unit-normalized global production value by production to estimate world unit values; includes historical test-named input paths that require configuration review before use.

[Source](../../S0_04_world_price_perUnit_cal.py) · 139 lines.

Entry points / interfaces: [`keep_and_numeric`](../../S0_04_world_price_perUnit_cal.py#L45), [`parse_value_unit`](../../S0_04_world_price_perUnit_cal.py#L72), [`build_output_unit`](../../S0_04_world_price_perUnit_cal.py#L100).

### `S0_05_convert_price_to_usd.py`

Converts local producer prices using exchange rates and combines them with world-price fallbacks; retains platform-specific /mnt/data paths.

[Source](../../S0_05_convert_price_to_usd.py) · 243 lines.

Entry points / interfaces: [`norm`](../../S0_05_convert_price_to_usd.py#L12), [`fx_direction`](../../S0_05_convert_price_to_usd.py#L15), [`to_usd`](../../S0_05_convert_price_to_usd.py#L22), [`is_usd_per_tonne`](../../S0_05_convert_price_to_usd.py#L26), [`is_lcu_per_tonne`](../../S0_05_convert_price_to_usd.py#L30), [`is_slc_per_tonne`](../../S0_05_convert_price_to_usd.py#L34), [`to_long`](../../S0_05_convert_price_to_usd.py#L107), [`pick`](../../S0_05_convert_price_to_usd.py#L127).

### `S0_06_build_world_price_for_forestry.py`

Calculates export, import, and mean world unit values for roundwood/fuelwood from trade quantities and values; provides input/output command-line options.

[Source](../../S0_06_build_world_price_for_forestry.py) · 108 lines.

Entry points / interfaces: [`load_wide_and_melt`](../../S0_06_build_world_price_for_forestry.py#L38), [`build_world_uv`](../../S0_06_build_world_price_for_forestry.py#L63), [`main`](../../S0_06_build_world_price_for_forestry.py#L89).

### `S0_07_fill_fish_prices_with_regionAvg.py`

Fills annual fish-price gaps with hierarchical regional means and historical moving averages, then appends world reference rows.

[Source](../../S0_07_fill_fish_prices_with_regionAvg.py) · 157 lines.

Entry points / interfaces: [`ensure_year_cols`](../../S0_07_fill_fish_prices_with_regionAvg.py#L28), [`region_mean_fill`](../../S0_07_fill_fish_prices_with_regionAvg.py#L36), [`backward_moving_average`](../../S0_07_fill_fish_prices_with_regionAvg.py#L47), [`add_world_row`](../../S0_07_fill_fish_prices_with_regionAvg.py#L60), [`main`](../../S0_07_fill_fish_prices_with_regionAvg.py#L83).

### `S0_08_make_LUC_mask_WGS.py`

Rasterizes country identifiers onto the LUH2 latitude-longitude grid and writes the country mask used for national aggregation.

[Source](../../S0_08_make_LUC_mask_WGS.py) · 220 lines.

Entry points / interfaces: [`get_lat_lon`](../../S0_08_make_LUC_mask_WGS.py#L60), [`geotransform_from_centers`](../../S0_08_make_LUC_mask_WGS.py#L71), [`main`](../../S0_08_make_LUC_mask_WGS.py#L91).

Direct local imports: [config_paths.py](../../config_paths.py).

### `S0_09_make_pasture_yield_mask_WGS.py`

Rasterizes country boundaries to the pasture-yield raster's coordinate system and grid for national dry-matter-yield aggregation.

[Source](../../S0_09_make_pasture_yield_mask_WGS.py) · 213 lines.

Entry points / interfaces: [`main`](../../S0_09_make_pasture_yield_mask_WGS.py#L124).

### `S0_10_recal_ef_livestock_country_item.py`

Derives country-livestock process factors from historical emissions and stocks, fills temporal/regional/global gaps, and normalizes gases and units.

[Source](../../S0_10_recal_ef_livestock_country_item.py) · 316 lines.

Entry points / interfaces: [`pick_cols`](../../S0_10_recal_ef_livestock_country_item.py#L39), [`detect_year_cols`](../../S0_10_recal_ef_livestock_country_item.py#L43), [`mass_to_kg_factor`](../../S0_10_recal_ef_livestock_country_item.py#L47), [`n_to_kgN_factor`](../../S0_10_recal_ef_livestock_country_item.py#L62), [`stock_to_animal_factor`](../../S0_10_recal_ef_livestock_country_item.py#L65), [`convert_by_unit_groups`](../../S0_10_recal_ef_livestock_country_item.py#L76), [`build_wide_block`](../../S0_10_recal_ef_livestock_country_item.py#L89), [`ratio`](../../S0_10_recal_ef_livestock_country_item.py#L101). Additional interfaces are listed in the source inventory.

### `S0_11_FAO_production_livestock_dairy_producingHeadRatio_prepare.py`

Prepares dairy/non-dairy production and animal-stock records, producing-head counts and ratios, and missing country-species combinations.

[Source](../../S0_11_FAO_production_livestock_dairy_producingHeadRatio_prepare.py) · 455 lines.

Entry points / interfaces: [`detect_year_cols`](../../S0_11_FAO_production_livestock_dairy_producingHeadRatio_prepare.py#L45), [`normalize_m49`](../../S0_11_FAO_production_livestock_dairy_producingHeadRatio_prepare.py#L51), [`typical_unit`](../../S0_11_FAO_production_livestock_dairy_producingHeadRatio_prepare.py#L61), [`make_rows_from_wide`](../../S0_11_FAO_production_livestock_dairy_producingHeadRatio_prepare.py#L68), [`build_area_year_matrix`](../../S0_11_FAO_production_livestock_dairy_producingHeadRatio_prepare.py#L101), [`main`](../../S0_11_FAO_production_livestock_dairy_producingHeadRatio_prepare.py#L131).

### `S0_12_compute_pasture_DMyield_by_country.py`

Aggregates pasture biomass and area rasters with country masks to produce national pasture dry-matter yields.

[Source](../../S0_12_compute_pasture_DMyield_by_country.py) · 152 lines.

Entry points / interfaces: [`main`](../../S0_12_compute_pasture_DMyield_by_country.py#L141).

Direct local imports: [config_paths.py](../../config_paths.py).

### `S0_13_extract_crop_grass_feed_ratio.py`

Extracts GLEAM feeding-system compositions and combines country mappings and herd weights into crop-feed and grass-feed shares.

[Source](../../S0_13_extract_crop_grass_feed_ratio.py) · 423 lines.

Entry points / interfaces: [`extract_text_range`](../../S0_13_extract_crop_grass_feed_ratio.py#L47), [`parse_table_s10`](../../S0_13_extract_crop_grass_feed_ratio.py#L56), [`fill_regions_with_wrd`](../../S0_13_extract_crop_grass_feed_ratio.py#L124), [`parse_table_s4_mapping`](../../S0_13_extract_crop_grass_feed_ratio.py#L143), [`parse_table_s11`](../../S0_13_extract_crop_grass_feed_ratio.py#L214), [`build_schema_codebook`](../../S0_13_extract_crop_grass_feed_ratio.py#L265), [`wavg`](../../S0_13_extract_crop_grass_feed_ratio.py#L324).

### `S0_14_SSPDB_GDP_change_refill.py`

Calculates GDP change relative to 2020 and maps scenario-region trajectories to countries using dictionary and income-group fallbacks.

[Source](../../S0_14_SSPDB_GDP_change_refill.py) · 274 lines.

Entry points / interfaces: [`is_year_col`](../../S0_14_SSPDB_GDP_change_refill.py#L31), [`year_to_int`](../../S0_14_SSPDB_GDP_change_refill.py#L45).

### `S0_15_manure_treatment_ratio_refill.py`

Fills and reconciles manure-management, soil-application, and pasture-deposition shares using temporal, regional, and global information.

[Source](../../S0_15_manure_treatment_ratio_refill.py) · 170 lines.

Entry points / interfaces: [`sel`](../../S0_15_manure_treatment_ratio_refill.py#L33), [`safe_div`](../../S0_15_manure_treatment_ratio_refill.py#L47), [`fill_all_nan_rows`](../../S0_15_manure_treatment_ratio_refill.py#L68), [`to_rows`](../../S0_15_manure_treatment_ratio_refill.py#L131).

### `S0_15_manure_treatment_ratio_refill_2.py`

Alternative manure-share filling and normalization utility targeting Environment_LivestockManure_with_ratio.csv; not an obligatory second processing pass.

[Source](../../S0_15_manure_treatment_ratio_refill_2.py) · 263 lines.

Entry points / interfaces: [`main`](../../S0_15_manure_treatment_ratio_refill_2.py#L113).

### `S0_16_gce_crop_parameters.py`

Back-calculates residue nitrogen/dry matter, rice, and fertilizer parameters from historical crop emissions and production.

[Source](../../S0_16_gce_crop_parameters.py) · 573 lines.

Entry points / interfaces: [`ensure_exists`](../../S0_16_gce_crop_parameters.py#L55), [`main`](../../S0_16_gce_crop_parameters.py#L402).

Direct local imports: [config_paths.py](../../config_paths.py).

### `S0_17_LUH2_data_analysis.py`

Aggregates LUH2 states and transitions to country cropland, pasture, forest, and coarse land-transition summaries for 2010-2020.

[Source](../../S0_17_LUH2_data_analysis.py) · 301 lines.

Entry points / interfaces: [`estimate_area_ha`](../../S0_17_LUH2_data_analysis.py#L72), [`ensure_year_dim`](../../S0_17_LUH2_data_analysis.py#L90), [`load_mask_array`](../../S0_17_LUH2_data_analysis.py#L107), [`load_mask_lookup`](../../S0_17_LUH2_data_analysis.py#L119), [`reindex_years`](../../S0_17_LUH2_data_analysis.py#L147), [`aggregate_by_mask`](../../S0_17_LUH2_data_analysis.py#L152), [`compute_land_cover_tables`](../../S0_17_LUH2_data_analysis.py#L174), [`compute_transition_tables`](../../S0_17_LUH2_data_analysis.py#L204). Additional interfaces are listed in the source inventory.

Direct local imports: [config_paths.py](../../config_paths.py).

### `S0_18_soil_drained_EF_refill.py`

Derives and fills drained-organic-soil CO2/N2O factors and area ratios from historical emissions and land statistics.

[Source](../../S0_18_soil_drained_EF_refill.py) · 213 lines.

Entry points / interfaces: [`parse_args`](../../S0_18_soil_drained_EF_refill.py#L38), [`melt_years`](../../S0_18_soil_drained_EF_refill.py#L54), [`load_region_map`](../../S0_18_soil_drained_EF_refill.py#L65), [`load_emission_data`](../../S0_18_soil_drained_EF_refill.py#L72), [`compute_emission_factors`](../../S0_18_soil_drained_EF_refill.py#L82), [`compute_area_correlation`](../../S0_18_soil_drained_EF_refill.py#L115), [`fill_gaps`](../../S0_18_soil_drained_EF_refill.py#L155), [`trim_years`](../../S0_18_soil_drained_EF_refill.py#L174). Additional interfaces are listed in the source inventory.

Direct local imports: [config_paths.py](../../config_paths.py).

### `S0_19_historical_max_production.py`

Extracts historical maximum production for valid country-commodity mappings; supplies capacity anchors rather than scenario forecasts.

[Source](../../S0_19_historical_max_production.py) · 243 lines.

Entry points / interfaces: [`main`](../../S0_19_historical_max_production.py#L140).

Direct local imports: [config_paths.py](../../config_paths.py).

### `S0_20_analysis_maskID_missing_country.py`

Diagnoses country IDs and missing coverage in the LUH2 mask, with geographic lookup helpers; outside routine S4 solves.

[Source](../../S0_20_analysis_maskID_missing_country.py) · 99 lines.

Entry points / interfaces: [`get_location_name`](../../S0_20_analysis_maskID_missing_country.py#L29), [`main`](../../S0_20_analysis_maskID_missing_country.py#L44).

### `S0_21_Landuse_historical_refill.py`

Combines country LUH2 summaries and FAO land statistics into the harmonized historical/base-year land-cover workbook.

[Source](../../S0_21_Landuse_historical_refill.py) · 123 lines.

### `S0_22_Forest_EF_recal.py`

Derives country forest emission/sink factors from historical forest emissions and forest area for parameter preparation.

[Source](../../S0_22_Forest_EF_recal.py) · 142 lines.

Entry points / interfaces: [`recal_forest_ef`](../../S0_22_Forest_EF_recal.py#L10).

### `S0_23_MACC_build.py`

Matches raw agricultural mitigation technologies to processes and prepares legacy MACC curves, tables, and diagnostic plots.

[Source](../../S0_23_MACC_build.py) · 230 lines.

Entry points / interfaces: [`match_ag_tech`](../../S0_23_MACC_build.py#L83).

### `S0_24_Reduction_cost_2080.py`

Fills technology costs for the 2080 target year using neighboring-year priorities and exports a cost-analysis workbook.

[Source](../../S0_24_Reduction_cost_2080.py) · 188 lines.

Entry points / interfaces: [`match_ag_tech`](../../S0_24_Reduction_cost_2080.py#L47), [`get_species`](../../S0_24_Reduction_cost_2080.py#L55), [`fill_data_with_priority`](../../S0_24_Reduction_cost_2080.py#L74).

### `S0_24_Reduction_cost_allYear.py`

Maps agricultural technologies to processes/livestock and calculates weighted mitigation costs across years.

[Source](../../S0_24_Reduction_cost_allYear.py) · 159 lines.

Entry points / interfaces: [`match_ag_tech`](../../S0_24_Reduction_cost_allYear.py#L37), [`get_species`](../../S0_24_Reduction_cost_allYear.py#L45).

### `S0_24_Reduction_cost_country_process_2080.py`

Maps and fills 2080 country-process costs for the legacy XLSX interface; it is not the strict v2 JSON runtime loader.

[Source](../../S0_24_Reduction_cost_country_process_2080.py) · 311 lines.

Entry points / interfaces: [`read_csv_robust`](../../S0_24_Reduction_cost_country_process_2080.py#L49), [`match_ag_tech`](../../S0_24_Reduction_cost_country_process_2080.py#L84), [`process_data`](../../S0_24_Reduction_cost_country_process_2080.py#L95).

### `S0_25_Dairy_NonDairy_emis_split.py`

Splits historical enteric/manure emissions into dairy and non-dairy livestock categories using stock relationships.

[Source](../../S0_25_Dairy_NonDairy_emis_split.py) · 195 lines.

Entry points / interfaces: [`split_emissions`](../../S0_25_Dairy_NonDairy_emis_split.py#L111).

Direct local imports: [config_paths.py](../../config_paths.py).

### `S0_26_feed_info_refill.py`

Fills country-livestock per-head feed requirements with hierarchical regional mappings and exports the runtime feed workbook.

[Source](../../S0_26_feed_info_refill.py) · 216 lines.

Entry points / interfaces: [`clean_m49_code`](../../S0_26_feed_info_refill.py#L13), [`fill_hierarchy`](../../S0_26_feed_info_refill.py#L19), [`apply_proxies`](../../S0_26_feed_info_refill.py#L48).

### `S0_27_grassfeed_ratio_refill.py`

Fills livestock crop-feed and grass-feed shares while retaining country/species identities.

[Source](../../S0_27_grassfeed_ratio_refill.py) · 154 lines.

Entry points / interfaces: [`clean_m49_code`](../../S0_27_grassfeed_ratio_refill.py#L12), [`fill_crop_with_means`](../../S0_27_grassfeed_ratio_refill.py#L19), [`main`](../../S0_27_grassfeed_ratio_refill.py#L57).

### `S0_28_dairy_yield_refill.py`

Fills dairy yields for buffaloes, camels, goats, and sheep and exports the completed production input.

[Source](../../S0_28_dairy_yield_refill.py) · 228 lines.

Entry points / interfaces: [`clean_m49_to_string`](../../S0_28_dairy_yield_refill.py#L27), [`fill_with_means`](../../S0_28_dairy_yield_refill.py#L35), [`main`](../../S0_28_dairy_yield_refill.py#L67).

### `S0_29_production_baseYear_fill_placeholder.py`

Fills 2020 yield/carcass-weight gaps for target country-commodity pairs; despite its name, it is a baseline-data preparation utility.

[Source](../../S0_29_production_baseYear_fill_placeholder.py) · 185 lines.

Entry points / interfaces: [`format_m49`](../../S0_29_production_baseYear_fill_placeholder.py#L14), [`load_dict`](../../S0_29_production_baseYear_fill_placeholder.py#L28), [`build_combos`](../../S0_29_production_baseYear_fill_placeholder.py#L42), [`build_templates`](../../S0_29_production_baseYear_fill_placeholder.py#L54), [`create_row`](../../S0_29_production_baseYear_fill_placeholder.py#L64), [`ensure_production`](../../S0_29_production_baseYear_fill_placeholder.py#L73), [`fill_elements`](../../S0_29_production_baseYear_fill_placeholder.py#L102), [`main`](../../S0_29_production_baseYear_fill_placeholder.py#L165).

### `S0_30_Nutrition_base_build.py`

Builds country-commodity baseline nutrition profiles from food-balance and production statistics.

[Source](../../S0_30_Nutrition_base_build.py) · 254 lines.

Entry points / interfaces: [`run`](../../S0_30_Nutrition_base_build.py#L9).

### `S0_31_Nutrition_base_supp.py`

Adds targeted missing country/commodity nutrition records from a missing-data report; historical paths and intended inputs require review.

[Source](../../S0_31_Nutrition_base_supp.py) · 65 lines.

### `S0_32_Nutrition_base_rescale.py`

Rescales nutrition profiles to intake constraints and exports diagnostics; the energy anchor uses the larger of baseline intake and minimum requirements.

[Source](../../S0_32_Nutrition_base_rescale.py) · 236 lines.

Entry points / interfaces: [`run`](../../S0_32_Nutrition_base_rescale.py#L104).

### `S0_33_build_fish_seafood_country_panel.py`

Builds country-year capture/aquaculture production, shares, yields, and emission factors from already downloaded fisheries statistics.

[Source](../../S0_33_build_fish_seafood_country_panel.py) · 1,307 lines.

Entry points / interfaces: [`build_panel`](../../S0_33_build_fish_seafood_country_panel.py#L988), [`write_outputs`](../../S0_33_build_fish_seafood_country_panel.py#L1180), [`main`](../../S0_33_build_fish_seafood_country_panel.py#L1270).

### `S0_34_build_country_minimum_nutrition_requirement.py`

Combines valid-country mappings and nutrition statistics into minimum-intake requirements; includes download helpers and regional fallbacks.

[Source](../../S0_34_build_country_minimum_nutrition_requirement.py) · 503 lines.

Entry points / interfaces: [`download`](../../S0_34_build_country_minimum_nutrition_requirement.py#L53), [`download_first_working`](../../S0_34_build_country_minimum_nutrition_requirement.py#L62), [`clean_m49`](../../S0_34_build_country_minimum_nutrition_requirement.py#L72), [`safe_mean`](../../S0_34_build_country_minimum_nutrition_requirement.py#L80), [`pick_latest_3yr_window`](../../S0_34_build_country_minimum_nutrition_requirement.py#L87), [`standardize_region_df`](../../S0_34_build_country_minimum_nutrition_requirement.py#L100), [`M49Membership`](../../S0_34_build_country_minimum_nutrition_requirement.py#L142), [`load_m49_membership`](../../S0_34_build_country_minimum_nutrition_requirement.py#L192). Additional interfaces are listed in the source inventory.

### `S0_35_Elasticity_signs_revise.py`

Applies cross-price sign matrices to supply and demand elasticity sheets; rows identify quantity responses and columns identify price changes.

[Source](../../S0_35_Elasticity_signs_revise.py) · 295 lines.

Entry points / interfaces: [`read_sign_matrix`](../../S0_35_Elasticity_signs_revise.py#L53), [`build_sign_dict`](../../S0_35_Elasticity_signs_revise.py#L90), [`find_header_row_and_col_map`](../../S0_35_Elasticity_signs_revise.py#L104), [`revise_sheet_inplace`](../../S0_35_Elasticity_signs_revise.py#L152), [`main`](../../S0_35_Elasticity_signs_revise.py#L238).

### `S0_36_make_armington_from_gtap10.py`

Extracts GTAP10 substitution elasticities and maps them to model commodities for the regional Armington trade path.

[Source](../../S0_36_make_armington_from_gtap10.py) · 369 lines.

Entry points / interfaces: [`load_gtap_esub`](../../S0_36_make_armington_from_gtap10.py#L160), [`item_to_gtap_code`](../../S0_36_make_armington_from_gtap10.py#L179), [`choose_sigma`](../../S0_36_make_armington_from_gtap10.py#L254), [`main`](../../S0_36_make_armington_from_gtap10.py#L265).

### `S0_37_FBS_Demand_base_refill.py`

Fills baseline food-balance domestic supply and use categories using dictionary and production coverage.

[Source](../../S0_37_FBS_Demand_base_refill.py) · 195 lines.

Entry points / interfaces: [`main`](../../S0_37_FBS_Demand_base_refill.py#L120).

### `S0_38_Nutrition_profile_recalculated_fromD0.py`

Reconstructs per-capita nutrition from S2 demand, food balances, and population; also supplies food-use extraction helpers.

[Source](../../S0_38_Nutrition_profile_recalculated_fromD0.py) · 322 lines.

Entry points / interfaces: [`run`](../../S0_38_Nutrition_profile_recalculated_fromD0.py#L132).

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py).

Imported by: [S0_38_Nutrition_profile_recalculated_fromD0_food.py](../../S0_38_Nutrition_profile_recalculated_fromD0_food.py).

### `S0_38_Nutrition_profile_recalculated_fromD0_food.py`

Reuses the D0 reconstruction helpers but derives nutrition from food use, producing the historical food_demand workbook used by the main pipeline.

[Source](../../S0_38_Nutrition_profile_recalculated_fromD0_food.py) · 189 lines.

Entry points / interfaces: [`run`](../../S0_38_Nutrition_profile_recalculated_fromD0_food.py#L20).

Direct local imports: [S0_38_Nutrition_profile_recalculated_fromD0.py](../../S0_38_Nutrition_profile_recalculated_fromD0.py), [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py).

### `S0_39_calibrate_gce_synthetic_fert_ef_2020.py`

Calibrates selected Y2020 synthetic-fertilizer parameters against baseline fertilizer efficiency/emission intensity and writes a derived workbook.

[Source](../../S0_39_calibrate_gce_synthetic_fert_ef_2020.py) · 327 lines.

Entry points / interfaces: [`main`](../../S0_39_calibrate_gce_synthetic_fert_ef_2020.py#L185).

### `S0_40_LUCE_parameter_gen.py`

Matches countries to climate, ecological, soil, and crop classes to prepare Tier 1 land-carbon parameters; includes fixed paths and external lookup helpers.

[Source](../../S0_40_LUCE_parameter_gen.py) · 638 lines.

Entry points / interfaces: [`parse_idrisi_rdc`](../../S0_40_LUCE_parameter_gen.py#L82), [`load_adm0_geoms`](../../S0_40_LUCE_parameter_gen.py#L96), [`wdi_fetch_indicator`](../../S0_40_LUCE_parameter_gen.py#L134), [`wdi_latest_by_iso3`](../../S0_40_LUCE_parameter_gen.py#L156), [`DominantResult`](../../S0_40_LUCE_parameter_gen.py#L201), [`dominant_code_in_geom`](../../S0_40_LUCE_parameter_gen.py#L216), [`nearest_nonzero_code`](../../S0_40_LUCE_parameter_gen.py#L254), [`get_dominant_with_fallback`](../../S0_40_LUCE_parameter_gen.py#L292). Additional interfaces are listed in the source inventory.

### `S0_41_Scenario_multiplier_bestHisValue_extract.py`

Extracts historical ranges across production, demand, and emissions parameters for scenario bounds and Y2020-referenced interpretation.

[Source](../../S0_41_Scenario_multiplier_bestHisValue_extract.py) · 812 lines.

Entry points / interfaces: [`main`](../../S0_41_Scenario_multiplier_bestHisValue_extract.py#L699).

Direct local imports: [S2_0_load_data.py](../../S2_0_load_data.py), [config_paths.py](../../config_paths.py).

### `S0_42_Nutrition_scenario_gen.py`

Builds ruminant-share caps and alternative dietary profiles/columns in the historical food-demand workbook, including low_land_new variants.

[Source](../../S0_42_Nutrition_scenario_gen.py) · 918 lines.

Entry points / interfaces: [`main`](../../S0_42_Nutrition_scenario_gen.py#L825).

Direct local imports: [config_paths.py](../../config_paths.py).

### `S0_43_EAT_LANCET_nutrition_profile_gen.py`

Maps an EAT-Lancet 2,500 kcal/day dietary profile to model nutrition items and exports Excel, CSV, and JSON representations.

[Source](../../S0_43_EAT_LANCET_nutrition_profile_gen.py) · 260 lines.

Entry points / interfaces: [`main`](../../S0_43_EAT_LANCET_nutrition_profile_gen.py#L28).

### `S0_44_AR6_harmonize_baseyear.py`

Harmonizes scenario series to historical base-year targets and writes diagnostics; the configured dataset can be SCI despite the AR6 filename.

[Source](../../S0_44_AR6_harmonize_baseyear.py) · 461 lines.

Entry points / interfaces: [`harmonize_sheet`](../../S0_44_AR6_harmonize_baseyear.py#L236), [`main`](../../S0_44_AR6_harmonize_baseyear.py#L397).

Direct local imports: [config_paths.py](../../config_paths.py).

### `S0_45_slice_LUH2_subset.py`

Extracts 2010-2020 states/transitions from larger LUH2 NetCDF datasets to reduce later I/O.

[Source](../../S0_45_slice_LUH2_subset.py) · 82 lines.

Entry points / interfaces: [`main`](../../S0_45_slice_LUH2_subset.py#L55).

Direct local imports: [config_paths.py](../../config_paths.py).

### `S0_46_AR6_SCI_scenario_combine.py`

Compares harmonized AR6 and SCI scenario keys and exports AR6 rows missing from SCI, matching model, scenario, region, variable, and unit.

[Source](../../S0_46_AR6_SCI_scenario_combine.py) · 117 lines.

Entry points / interfaces: [`main`](../../S0_46_AR6_SCI_scenario_combine.py#L79).

Direct local imports: [config_paths.py](../../config_paths.py).

### `S0_47_SCI_harmonization_pooled_fallback_fix.py`

Revises pooled historical targets for SCI rows that used fallback harmonization; writes corrected series and diagnostics.

[Source](../../S0_47_SCI_harmonization_pooled_fallback_fix.py) · 262 lines.

Entry points / interfaces: [`main`](../../S0_47_SCI_harmonization_pooled_fallback_fix.py#L223).

Direct local imports: [config_paths.py](../../config_paths.py).

### `S0_48_history_emission_summary.py`

Combines historical agricultural, land, and fisheries emissions into a 1961-2020 summary used by structural figures and baseline comparisons.

[Source](../../S0_48_history_emission_summary.py) · 2,260 lines.

Entry points / interfaces: [`build_history_emission_summary`](../../S0_48_history_emission_summary.py#L2109), [`main`](../../S0_48_history_emission_summary.py#L2254).

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py), [S3_2_feed_demand.py](../../S3_2_feed_demand.py), [S4_1_results.py](../../S4_1_results.py), [config_paths.py](../../config_paths.py), [gsoil_emission_complete.py](../../gsoil_emission_complete.py).

### `S0_49_emission_result_summary.py`

Combines a run's detailed emissions, production, and nutrition/market results into 2080 emission and commodity indicators for decomposition figures.

[Source](../../S0_49_emission_result_summary.py) · 435 lines.

Entry points / interfaces: [`build_emission_result_summary`](../../S0_49_emission_result_summary.py#L389), [`main`](../../S0_49_emission_result_summary.py#L418).

Direct local imports: [S2_0_load_data.py](../../S2_0_load_data.py), [SP_M1a_Figure_pie_structure_pre.py](../../SP_M1a_Figure_pie_structure_pre.py), [config_paths.py](../../config_paths.py).

### `S0_50_prepare_bioenergy_history.py`

Standardizes historical bioenergy by country and carrier and exports allocation shares and coverage diagnostics.

[Source](../../S0_50_prepare_bioenergy_history.py) · 186 lines.

Entry points / interfaces: [`prepare_history`](../../S0_50_prepare_bioenergy_history.py#L43), [`build_country_profiles`](../../S0_50_prepare_bioenergy_history.py#L121), [`main`](../../S0_50_prepare_bioenergy_history.py#L151).

Direct local imports: [S2_0_load_data.py](../../S2_0_load_data.py), [S3_3_bioenergy.py](../../S3_3_bioenergy.py), [config_paths.py](../../config_paths.py), [runtime_data_cache.py](../../runtime_data_cache.py).

### `S0_51_prepare_bioenergy_feedstock_bridge.py`

Converts energy-carrier history into physical feedstock demand using carrier-feedstock shares, heating values, and conversion parameters.

[Source](../../S0_51_prepare_bioenergy_feedstock_bridge.py) · 364 lines.

Entry points / interfaces: [`prepare_historical_feedstocks`](../../S0_51_prepare_bioenergy_feedstock_bridge.py#L177), [`main`](../../S0_51_prepare_bioenergy_feedstock_bridge.py#L306).

Direct local imports: [S0_53_prepare_bioenergy_scenarios.py](../../S0_53_prepare_bioenergy_scenarios.py), [S2_0_load_data.py](../../S2_0_load_data.py), [S3_3_bioenergy.py](../../S3_3_bioenergy.py), [config_paths.py](../../config_paths.py), [runtime_data_cache.py](../../runtime_data_cache.py).

Imported by: [ST_bioenergy_mvp_test.py](../../ST_bioenergy_mvp_test.py).

### `S0_52_prepare_bioenergy_resource_constraints.py`

Combines residues, non-crop potentials, energy-crop yields, and eligible land into resource constraints and derived bioenergy parameters.

[Source](../../S0_52_prepare_bioenergy_resource_constraints.py) · 1,169 lines.

Entry points / interfaces: [`build_residue_quality_lookup`](../../S0_52_prepare_bioenergy_resource_constraints.py#L326), [`build_crop_residue_constraints`](../../S0_52_prepare_bioenergy_resource_constraints.py#L378), [`build_li2020_country_yields`](../../S0_52_prepare_bioenergy_resource_constraints.py#L508), [`build_energy_crop_constraints`](../../S0_52_prepare_bioenergy_resource_constraints.py#L653), [`build_noncrop_feedstock_constraints`](../../S0_52_prepare_bioenergy_resource_constraints.py#L780), [`write_outputs`](../../S0_52_prepare_bioenergy_resource_constraints.py#L972), [`main`](../../S0_52_prepare_bioenergy_resource_constraints.py#L1007).

Direct local imports: [S2_0_load_data.py](../../S2_0_load_data.py), [config_paths.py](../../config_paths.py).

Imported by: [S0_54_prepare_bioenergy_eligible_land_mask.py](../../S0_54_prepare_bioenergy_eligible_land_mask.py).

### `S0_53_prepare_bioenergy_scenarios.py`

Constructs low/medium/high bioenergy demand from scenario quantiles, historical national shares, and resource constraints; exports target and allocation audits.

[Source](../../S0_53_prepare_bioenergy_scenarios.py) · 1,599 lines.

Entry points / interfaces: [`FeedstockSpec`](../../S0_53_prepare_bioenergy_scenarios.py#L362), [`normalize_m49`](../../S0_53_prepare_bioenergy_scenarios.py#L547), [`parse_years`](../../S0_53_prepare_bioenergy_scenarios.py#L566), [`ramp_to_target`](../../S0_53_prepare_bioenergy_scenarios.py#L577), [`trajectory_value`](../../S0_53_prepare_bioenergy_scenarios.py#L593), [`trajectory_from_points`](../../S0_53_prepare_bioenergy_scenarios.py#L616), [`native_feedstock_split_targets`](../../S0_53_prepare_bioenergy_scenarios.py#L635), [`agricultural_bioenergy_demand_value`](../../S0_53_prepare_bioenergy_scenarios.py#L655). Additional interfaces are listed in the source inventory.

Imported by: [S0_51_prepare_bioenergy_feedstock_bridge.py](../../S0_51_prepare_bioenergy_feedstock_bridge.py), [S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py).

### `S0_54_prepare_bioenergy_eligible_land_mask.py`

Aligns WorldCover and optional protected/suitable-land data to LUH2 and derives eligible energy-crop land masks and national areas for S0.52.

[Source](../../S0_54_prepare_bioenergy_eligible_land_mask.py) · 508 lines.

Entry points / interfaces: [`build_mask`](../../S0_54_prepare_bioenergy_eligible_land_mask.py#L356), [`parse_args`](../../S0_54_prepare_bioenergy_eligible_land_mask.py#L474), [`main`](../../S0_54_prepare_bioenergy_eligible_land_mask.py#L499).

Direct local imports: [S0_52_prepare_bioenergy_resource_constraints.py](../../S0_52_prepare_bioenergy_resource_constraints.py), [S2_0_load_data.py](../../S2_0_load_data.py), [config_paths.py](../../config_paths.py).

### `S0_55_prepare_bioenergy_noncrop_availability.py`

Aggregates OMD non-crop residues/byproducts to national technical potential and conservative availability for S0.52 resource limits.

[Source](../../S0_55_prepare_bioenergy_noncrop_availability.py) · 499 lines.

Entry points / interfaces: [`build_availability`](../../S0_55_prepare_bioenergy_noncrop_availability.py#L375), [`parse_args`](../../S0_55_prepare_bioenergy_noncrop_availability.py#L473), [`main`](../../S0_55_prepare_bioenergy_noncrop_availability.py#L485).

Direct local imports: [config_paths.py](../../config_paths.py).

## Core model and single-run outputs

### `S1_0_schema.py`

Defines Node, Universe, ScenarioConfig, and ScenarioData: country-commodity-year state, mappings, model scope, and the default historical/future timeline.

[Source](../../S1_0_schema.py) · 101 lines.

Entry points / interfaces: [`Node`](../../S1_0_schema.py#L10), [`Universe`](../../S1_0_schema.py#L39), [`ScenarioConfig`](../../S1_0_schema.py#L89), [`ScenarioData`](../../S1_0_schema.py#L98).

Imported by: [S0_38_Nutrition_profile_recalculated_fromD0.py](../../S0_38_Nutrition_profile_recalculated_fromD0.py), [S0_38_Nutrition_profile_recalculated_fromD0_food.py](../../S0_38_Nutrition_profile_recalculated_fromD0_food.py), [S0_48_history_emission_summary.py](../../S0_48_history_emission_summary.py), [S2_0_load_data.py](../../S2_0_load_data.py), [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [S3_2_feed_demand.py](../../S3_2_feed_demand.py), [S3_3_bioenergy.py](../../S3_3_bioenergy.py), [S3_6_scenarios.py](../../S3_6_scenarios.py), [S4_0_main.py](../../S4_0_main.py), [S5_3_1_sensitivity_panel_yield_ef.py](../../S5_3_1_sensitivity_panel_yield_ef.py), [S5_4_1_monte_carlo_full_variables.py](../../S5_4_1_monte_carlo_full_variables.py), [S5_6_1_max_emission_reduction_potential.py](../../S5_6_1_max_emission_reduction_potential.py); further references are in the inventory.

### `S2_0_load_data.py`

Central input layer: DataPaths, dictionaries, production, food balances, drivers, prices, trade, nutrition, land, feed, and legacy cost readers; constructs model nodes and parameter maps.

[Source](../../S2_0_load_data.py) · 4,559 lines.

Entry points / interfaces: [`DataPaths`](../../S2_0_load_data.py#L68), [`load_process_cost_mapping`](../../S2_0_load_data.py#L446), [`load_unit_cost_data`](../../S2_0_load_data.py#L477), [`load_luh2_land_cover`](../../S2_0_load_data.py#L568), [`load_land_cover_from_excel`](../../S2_0_load_data.py#L707), [`load_forest_ef_from_excel`](../../S2_0_load_data.py#L856), [`load_roundwood_supply`](../../S2_0_load_data.py#L933), [`build_universe_from_dict_v3`](../../S2_0_load_data.py#L1014). Additional interfaces are listed in the source inventory.

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [config_paths.py](../../config_paths.py), [runtime_data_cache.py](../../runtime_data_cache.py).

Imported by: [cost_database_comparison_analysis.py](../../cost_database_comparison_analysis.py), [S0_38_Nutrition_profile_recalculated_fromD0.py](../../S0_38_Nutrition_profile_recalculated_fromD0.py), [S0_38_Nutrition_profile_recalculated_fromD0_food.py](../../S0_38_Nutrition_profile_recalculated_fromD0_food.py), [S0_41_Scenario_multiplier_bestHisValue_extract.py](../../S0_41_Scenario_multiplier_bestHisValue_extract.py), [S0_48_history_emission_summary.py](../../S0_48_history_emission_summary.py), [S0_49_emission_result_summary.py](../../S0_49_emission_result_summary.py), [S0_50_prepare_bioenergy_history.py](../../S0_50_prepare_bioenergy_history.py), [S0_51_prepare_bioenergy_feedstock_bridge.py](../../S0_51_prepare_bioenergy_feedstock_bridge.py), [S0_52_prepare_bioenergy_resource_constraints.py](../../S0_52_prepare_bioenergy_resource_constraints.py), [S0_54_prepare_bioenergy_eligible_land_mask.py](../../S0_54_prepare_bioenergy_eligible_land_mask.py), [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [S3_2_feed_demand.py](../../S3_2_feed_demand.py); further references are in the inventory.

### `S3_0_ds_emis_mc_full.py`

Alternative logarithmic/PWL formulation with emissions, MACC, and model-cache interfaces; rejects the unit_cost path and is not a feature-equivalent replacement for the linear solver.

[Source](../../S3_0_ds_emis_mc_full.py) · 1,250 lines.

Entry points / interfaces: [`build_model`](../../S3_0_ds_emis_mc_full.py#L174), [`SolveOpt`](../../S3_0_ds_emis_mc_full.py#L938), [`ModelCache`](../../S3_0_ds_emis_mc_full.py#L947), [`build_model_cache`](../../S3_0_ds_emis_mc_full.py#L980), [`apply_sample_updates`](../../S3_0_ds_emis_mc_full.py#L1026), [`run_mc`](../../S3_0_ds_emis_mc_full.py#L1157).

Direct local imports: [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [config_paths.py](../../config_paths.py), [mc_bound_utils.py](../../mc_bound_utils.py), [mc_sample_utils.py](../../mc_sample_utils.py).

Imported by: [S4_0_main.py](../../S4_0_main.py).

### `S3_0_ds_linear_regional.py`

Default linear solver for demand, production, trade, feed, land conversion, emissions approximations, and costs; includes iterative and cached-MC interfaces. Nutrition mode makes future supply demand/resource driven.

[Source](../../S3_0_ds_linear_regional.py) · 14,751 lines.

Entry points / interfaces: [`solve_with_luc_iteration`](../../S3_0_ds_linear_regional.py#L820), [`build_nutrition_demand_map`](../../S3_0_ds_linear_regional.py#L1865), [`LinearModelCache`](../../S3_0_ds_linear_regional.py#L2312), [`get_region`](../../S3_0_ds_linear_regional.py#L2470), [`get_all_regions`](../../S3_0_ds_linear_regional.py#L2501), [`reset_region_cache`](../../S3_0_ds_linear_regional.py#L2510), [`set_region_aggregation`](../../S3_0_ds_linear_regional.py#L2518), [`is_region_aggregation_enabled`](../../S3_0_ds_linear_regional.py#L2524). Additional interfaces are listed in the source inventory.

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py), [S3_2_feed_demand.py](../../S3_2_feed_demand.py), [S3_5_land_use_change.py](../../S3_5_land_use_change.py), [config_paths.py](../../config_paths.py), [gle_emissions_complete.py](../../gle_emissions_complete.py), [luc_emission_module.py](../../luc_emission_module.py), [market_balance_diagnostics.py](../../market_balance_diagnostics.py), [mc_bound_utils.py](../../mc_bound_utils.py), [mc_sample_utils.py](../../mc_sample_utils.py), [model_run_status.py](../../model_run_status.py), [runtime_data_cache.py](../../runtime_data_cache.py).

Imported by: [S2_0_load_data.py](../../S2_0_load_data.py), [S3_0_ds_emis_mc_full.py](../../S3_0_ds_emis_mc_full.py), [S4_0_main.py](../../S4_0_main.py), [S5_0_1_sensitivity_mc_levels.py](../../S5_0_1_sensitivity_mc_levels.py), [S5_1_1_sensitivity_mc_variable_effect.py](../../S5_1_1_sensitivity_mc_variable_effect.py), [ST_bioenergy_mvp_test.py](../../ST_bioenergy_mvp_test.py).

### `S3_1_emissions_orchestrator_fao.py`

Legacy FAO orchestration interface and output adapter, including placeholder outputs; cannot replace the complete physical emissions engines.

[Source](../../S3_1_emissions_orchestrator_fao.py) · 52 lines.

Entry points / interfaces: [`FAOPaths`](../../S3_1_emissions_orchestrator_fao.py#L12), [`EmissionsFAO`](../../S3_1_emissions_orchestrator_fao.py#L19).

Imported by: [S4_0_main.py](../../S4_0_main.py).

### `S3_2_feed_demand.py`

Converts animal stocks to per-head dry matter, grass/crop-feed requirements, commodity tonnes, and grassland demand; returns totals and species-level detail.

[Source](../../S3_2_feed_demand.py) · 609 lines.

Entry points / interfaces: [`FeedDemandOutputs`](../../S3_2_feed_demand.py#L42), [`build_feed_demand_from_stock`](../../S3_2_feed_demand.py#L48).

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py), [runtime_data_cache.py](../../runtime_data_cache.py).

Imported by: [S0_48_history_emission_summary.py](../../S0_48_history_emission_summary.py), [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [S4_0_main.py](../../S4_0_main.py).

### `S3_3_bioenergy.py`

Builds BioenergyBundle: energy/feedstock conversion, historical reconciliation, market uses, resource and dedicated-land limits, feed coproduct credits, residue interfaces, and feasibility diagnostics.

[Source](../../S3_3_bioenergy.py) · 2,497 lines.

Entry points / interfaces: [`BioenergyBundle`](../../S3_3_bioenergy.py#L240), [`normalize_m49`](../../S3_3_bioenergy.py#L267), [`load_feedstock_parameters`](../../S3_3_bioenergy.py#L421), [`build_bioenergy_postsolve_assessment`](../../S3_3_bioenergy.py#L1546), [`build_bioenergy_bundle`](../../S3_3_bioenergy.py#L2167), [`build_baseline_reconciliation`](../../S3_3_bioenergy.py#L2292), [`write_bioenergy_bundle`](../../S3_3_bioenergy.py#L2417).

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [model_run_status.py](../../model_run_status.py), [runtime_data_cache.py](../../runtime_data_cache.py).

Imported by: [S0_50_prepare_bioenergy_history.py](../../S0_50_prepare_bioenergy_history.py), [S0_51_prepare_bioenergy_feedstock_bridge.py](../../S0_51_prepare_bioenergy_feedstock_bridge.py), [S4_0_main.py](../../S4_0_main.py), [ST_bioenergy_full_run_smoke.py](../../ST_bioenergy_full_run_smoke.py), [ST_bioenergy_mvp_test.py](../../ST_bioenergy_mvp_test.py).

### `S3_5_land_use_change.py`

Derives land demand and changes from production, yields, grass needs, and baseline area; provides the compatibility/post-solve land path, while luc_emission_module handles carbon pools.

[Source](../../S3_5_land_use_change.py) · 400 lines.

Entry points / interfaces: [`LUCConfig`](../../S3_5_land_use_change.py#L16), [`compute_luc_areas`](../../S3_5_land_use_change.py#L32), [`get_luc_emis`](../../S3_5_land_use_change.py#L396).

Imported by: [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [S4_0_main.py](../../S4_0_main.py).

### `S3_6_scenarios.py`

Parses scenario and MC sheets, selectors, rates, multipliers, absolute values, and Y2020 references; creates ScenarioEffect objects and applies year-specific parameter changes.

[Source](../../S3_6_scenarios.py) · 1,090 lines.

Entry points / interfaces: [`ScenarioEffect`](../../S3_6_scenarios.py#L127), [`load_scenario_config`](../../S3_6_scenarios.py#L250), [`apply_scenario_to_data`](../../S3_6_scenarios.py#L290), [`load_scenarios`](../../S3_6_scenarios.py#L821), [`load_mc_specs`](../../S3_6_scenarios.py#L856), [`draw_mc_to_params`](../../S3_6_scenarios.py#L871), [`draw_mc_effects`](../../S3_6_scenarios.py#L1025).

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [config_paths.py](../../config_paths.py), [mc_bound_utils.py](../../mc_bound_utils.py).

Imported by: [S4_0_main.py](../../S4_0_main.py), [S5_1_1_sensitivity_mc_variable_effect.py](../../S5_1_1_sensitivity_mc_variable_effect.py), [S5_3_1_sensitivity_panel_yield_ef.py](../../S5_3_1_sensitivity_panel_yield_ef.py), [S5_3_1_sensitivity_panel_yield_ef_v2.py](../../S5_3_1_sensitivity_panel_yield_ef_v2.py), [ST_bioenergy_mvp_test.py](../../ST_bioenergy_mvp_test.py).

### `S4_0_main.py`

Full-model coordinator: baseline loading, scenarios, feed/land/bioenergy coupling, optimization, activity reconstruction, physical emissions, costs, diagnostics, and terminal status. S5 primarily reuses run_one_pipeline.

[Source](../../S4_0_main.py) · 22,553 lines.

Entry points / interfaces: [`MCPrecheckFailed`](../../S4_0_main.py#L823), [`refresh_runtime_caches`](../../S4_0_main.py#L1071), [`build_detailed_outputs`](../../S4_0_main.py#L6531), [`build_production_summary`](../../S4_0_main.py#L7313), [`apply_trade_baseline_to_nodes`](../../S4_0_main.py#L8881), [`fill_zero_trade_with_supply_demand`](../../S4_0_main.py#L8929), [`build_trade_base_by_region`](../../S4_0_main.py#L8985), [`build_trade_base_volume_by_market_region`](../../S4_0_main.py#L9063). Additional interfaces are listed in the source inventory.

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py), [S3_0_ds_emis_mc_full.py](../../S3_0_ds_emis_mc_full.py), [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [S3_1_emissions_orchestrator_fao.py](../../S3_1_emissions_orchestrator_fao.py), [S3_2_feed_demand.py](../../S3_2_feed_demand.py), [S3_3_bioenergy.py](../../S3_3_bioenergy.py), [S3_5_land_use_change.py](../../S3_5_land_use_change.py), [S3_6_scenarios.py](../../S3_6_scenarios.py), [S4_1_results.py](../../S4_1_results.py), [config_paths.py](../../config_paths.py), [cost_database_v2.py](../../cost_database_v2.py); further references are in the inventory.

Imported by: [S5_0_1_sensitivity_mc_levels.py](../../S5_0_1_sensitivity_mc_levels.py), [S5_1_1_sensitivity_mc_variable_effect.py](../../S5_1_1_sensitivity_mc_variable_effect.py), [S5_3_1_sensitivity_panel_yield_ef.py](../../S5_3_1_sensitivity_panel_yield_ef.py), [S5_4_1_monte_carlo_full_variables.py](../../S5_4_1_monte_carlo_full_variables.py), [S5_6_1_max_emission_reduction_potential.py](../../S5_6_1_max_emission_reduction_potential.py), [S5_6_4_max_reduction_impend.py](../../S5_6_4_max_reduction_impend.py), [ST_bioenergy_full_run_smoke.py](../../ST_bioenergy_full_run_smoke.py).

### `S4_1_results.py`

Normalizes market and emission outputs, mappings, gases, and units; produces detailed/aggregated results and physical versus priced abatement cost summaries.

[Source](../../S4_1_results.py) · 3,301 lines.

Entry points / interfaces: [`print`](../../S4_1_results.py#L87), [`filter_years_for_aggregation`](../../S4_1_results.py#L140), [`EmissionsAggregator`](../../S4_1_results.py#L167), [`summarize_emissions_from_detail`](../../S4_1_results.py#L781), [`summarize_saved_detail_long`](../../S4_1_results.py#L1623), [`write_summary_tables_from_detail_long`](../../S4_1_results.py#L1648), [`summarize_emissions`](../../S4_1_results.py#L1682), [`summarize_market`](../../S4_1_results.py#L2209). Additional interfaces are listed in the source inventory.

Direct local imports: [config_paths.py](../../config_paths.py), [runtime_data_cache.py](../../runtime_data_cache.py).

Imported by: [S0_48_history_emission_summary.py](../../S0_48_history_emission_summary.py), [S4_0_main.py](../../S4_0_main.py), [S4_3_results_summary_only.py](../../S4_3_results_summary_only.py), [S5_cost_summary_outputs.py](../../S5_cost_summary_outputs.py), [ST_bioenergy_mvp_test.py](../../ST_bioenergy_mvp_test.py).

### `S4_3_results_summary_only.py`

Rebuilds summary and fast-emission tables from saved emission detail without solving supply and demand again.

[Source](../../S4_3_results_summary_only.py) · 176 lines.

Entry points / interfaces: [`main`](../../S4_3_results_summary_only.py#L77).

Direct local imports: [S4_1_results.py](../../S4_1_results.py), [config_paths.py](../../config_paths.py).

## Experiments, batches, and attribution

### `S5_0_1_diagnose_small_variable_importance.py`

Investigates small variable-importance values using sampled variation, activation, and regression diagnostics; writes diagnostic workbooks.

[Source](../../S5_0_1_diagnose_small_variable_importance.py) · 500 lines.

Entry points / interfaces: [`main`](../../S5_0_1_diagnose_small_variable_importance.py#L451).

Direct local imports: [config_paths.py](../../config_paths.py).

### `S5_0_1_sensitivity_mc_levels.py`

Samples multiple scenario variables and repeatedly runs S4 to estimate importance around emission levels/targets; the current configured sample count is 20,000.

[Source](../../S5_0_1_sensitivity_mc_levels.py) · 2,279 lines.

Entry points / interfaces: [`SampleResult`](../../S5_0_1_sensitivity_mc_levels.py#L75), [`main`](../../S5_0_1_sensitivity_mc_levels.py#L1826).

Direct local imports: [S2_0_load_data.py](../../S2_0_load_data.py), [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [S4_0_main.py](../../S4_0_main.py), [S5_1_1_sensitivity_mc_variable_effect.py](../../S5_1_1_sensitivity_mc_variable_effect.py), [S5_cost_summary_outputs.py](../../S5_cost_summary_outputs.py), [config_paths.py](../../config_paths.py), [model_run_status.py](../../model_run_status.py).

Imported by: [S5_0_1_sensitivity_mc_levels_batches.py](../../S5_0_1_sensitivity_mc_levels_batches.py), [S5_0_2_merge_sensitivity_mc_levels_batches.py](../../S5_0_2_merge_sensitivity_mc_levels_batches.py).

### `S5_0_1_sensitivity_mc_levels_batches.py`

Assigns a deterministic subset of S5.0 samples to one batch using batch environment settings while preserving global sample identities.

[Source](../../S5_0_1_sensitivity_mc_levels_batches.py) · 103 lines.

Entry points / interfaces: [`main`](../../S5_0_1_sensitivity_mc_levels_batches.py#L75).

Direct local imports: [S5_0_1_sensitivity_mc_levels.py](../../S5_0_1_sensitivity_mc_levels.py).

### `S5_0_2_merge_sensitivity_mc_levels_batches.py`

Merges S5.0 samples and rebuilds full-ensemble importance statistics using the core statistical routines.

[Source](../../S5_0_2_merge_sensitivity_mc_levels_batches.py) · 144 lines.

Entry points / interfaces: [`main`](../../S5_0_2_merge_sensitivity_mc_levels_batches.py#L104).

Direct local imports: [S5_0_1_sensitivity_mc_levels.py](../../S5_0_1_sensitivity_mc_levels.py), [config_paths.py](../../config_paths.py).

### `S5_1_1_sensitivity_mc_variable_effect.py`

Runs conditional Monte Carlo experiments at specified yield, EF, ruminant-share, and other variable levels; also supplies sampling and bound helpers reused by other S5 workflows.

[Source](../../S5_1_1_sensitivity_mc_variable_effect.py) · 3,668 lines.

Entry points / interfaces: [`resolve_mc_effect_sheet`](../../S5_1_1_sensitivity_mc_variable_effect.py#L79), [`main`](../../S5_1_1_sensitivity_mc_variable_effect.py#L3267).

Direct local imports: [S2_0_load_data.py](../../S2_0_load_data.py), [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [S3_6_scenarios.py](../../S3_6_scenarios.py), [S4_0_main.py](../../S4_0_main.py), [S5_cost_summary_outputs.py](../../S5_cost_summary_outputs.py), [config_paths.py](../../config_paths.py), [market_balance_diagnostics.py](../../market_balance_diagnostics.py), [model_run_status.py](../../model_run_status.py).

Imported by: [S5_0_1_sensitivity_mc_levels.py](../../S5_0_1_sensitivity_mc_levels.py), [S5_1_1_sensitivity_mc_variable_effect_batches.py](../../S5_1_1_sensitivity_mc_variable_effect_batches.py), [S5_1_3_merge_variable_effect_batches.py](../../S5_1_3_merge_variable_effect_batches.py), [S5_3_1_sensitivity_panel_yield_ef.py](../../S5_3_1_sensitivity_panel_yield_ef.py), [S5_4_1_monte_carlo_full_variables.py](../../S5_4_1_monte_carlo_full_variables.py).

### `S5_1_1_sensitivity_mc_variable_effect_batches.py`

Wraps one S5.1 conditional-MC batch, handling batch indices, total batch count, and resume configuration.

[Source](../../S5_1_1_sensitivity_mc_variable_effect_batches.py) · 107 lines.

Entry points / interfaces: [`main`](../../S5_1_1_sensitivity_mc_variable_effect_batches.py#L78).

Direct local imports: [S5_1_1_sensitivity_mc_variable_effect.py](../../S5_1_1_sensitivity_mc_variable_effect.py).

### `S5_1_2_rebuild_fig_effect_sensivity_data.py`

Rebuilds legacy Fig5 sample, histogram, summary, target, and audit tables from existing S5.1 outputs without new model solves.

[Source](../../S5_1_2_rebuild_fig_effect_sensivity_data.py) · 483 lines.

Entry points / interfaces: [`main`](../../S5_1_2_rebuild_fig_effect_sensivity_data.py#L274).

Direct local imports: [config_paths.py](../../config_paths.py).

### `S5_1_3_merge_variable_effect_batches.py`

Validates and merges S5.1 samples, statuses, and metadata, then rebuilds grouped outputs and experiment-level cost summaries.

[Source](../../S5_1_3_merge_variable_effect_batches.py) · 485 lines.

Entry points / interfaces: [`main`](../../S5_1_3_merge_variable_effect_batches.py#L203).

Direct local imports: [S5_1_1_sensitivity_mc_variable_effect.py](../../S5_1_1_sensitivity_mc_variable_effect.py), [S5_cost_summary_outputs.py](../../S5_cost_summary_outputs.py), [config_paths.py](../../config_paths.py).

### `S5_3_1_forest_area_fixed_sensitivity.py`

Scans forest targets with yield, EF, and ruminant caps fixed, reusing the panel engine and exporting forest-sensitivity/process-emission results.

[Source](../../S5_3_1_forest_area_fixed_sensitivity.py) · 105 lines.

Entry points / interfaces: [`parse_args`](../../S5_3_1_forest_area_fixed_sensitivity.py#L34), [`main`](../../S5_3_1_forest_area_fixed_sensitivity.py#L77).

Direct local imports: [S5_3_1_sensitivity_panel_yield_ef.py](../../S5_3_1_sensitivity_panel_yield_ef.py).

### `S5_3_1_sensitivity_panel_yield_ef.py`

Runs yield-by-EF grids under forest and ruminant-share settings, recording emissions, realized nutrition/land indicators, and feasibility diagnostics for contour plots.

[Source](../../S5_3_1_sensitivity_panel_yield_ef.py) · 2,044 lines.

Entry points / interfaces: [`main`](../../S5_3_1_sensitivity_panel_yield_ef.py#L1344).

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py), [S3_6_scenarios.py](../../S3_6_scenarios.py), [S4_0_main.py](../../S4_0_main.py), [S5_1_1_sensitivity_mc_variable_effect.py](../../S5_1_1_sensitivity_mc_variable_effect.py), [S5_cost_summary_outputs.py](../../S5_cost_summary_outputs.py), [config_paths.py](../../config_paths.py), [market_balance_diagnostics.py](../../market_balance_diagnostics.py), [model_run_status.py](../../model_run_status.py).

Imported by: [S5_3_1_forest_area_fixed_sensitivity.py](../../S5_3_1_forest_area_fixed_sensitivity.py), [S5_3_1_sensitivity_panel_yield_ef_batches.py](../../S5_3_1_sensitivity_panel_yield_ef_batches.py), [S5_3_1_sensitivity_panel_yield_ef_v2.py](../../S5_3_1_sensitivity_panel_yield_ef_v2.py), [S5_3_3_merge_panel_yield_ef_batches.py](../../S5_3_3_merge_panel_yield_ef_batches.py).

### `S5_3_1_sensitivity_panel_yield_ef_batches.py`

Partitions the S5.3 scenario grid into batches and passes configuration overrides to the panel runner.

[Source](../../S5_3_1_sensitivity_panel_yield_ef_batches.py) · 156 lines.

Entry points / interfaces: [`main`](../../S5_3_1_sensitivity_panel_yield_ef_batches.py#L123).

Direct local imports: [S5_3_1_sensitivity_panel_yield_ef.py](../../S5_3_1_sensitivity_panel_yield_ef.py).

Imported by: [S5_3_1_sensitivity_panel_yield_ef_batches_v2.py](../../S5_3_1_sensitivity_panel_yield_ef_batches_v2.py).

### `S5_3_1_sensitivity_panel_yield_ef_batches_v2.py`

Combines the original batch interface with the v2 linked-axis configuration and output layout.

[Source](../../S5_3_1_sensitivity_panel_yield_ef_batches_v2.py) · 44 lines.

Entry points / interfaces: [`main`](../../S5_3_1_sensitivity_panel_yield_ef_batches_v2.py#L9).

Direct local imports: [S5_3_1_sensitivity_panel_yield_ef_batches.py](../../S5_3_1_sensitivity_panel_yield_ef_batches.py), [S5_3_1_sensitivity_panel_yield_ef_v2.py](../../S5_3_1_sensitivity_panel_yield_ef_v2.py).

### `S5_3_1_sensitivity_panel_yield_ef_v2.py`

Defines coupled panel axes: E/L coordinates move EF/fertilizer/crop-soil factors together and manure oppositely; the L/A-labelled axis links yield and inverse feed intensity. Axis meanings differ from v1.

[Source](../../S5_3_1_sensitivity_panel_yield_ef_v2.py) · 372 lines.

Entry points / interfaces: [`apply_v2_axis_aliases`](../../S5_3_1_sensitivity_panel_yield_ef_v2.py#L228), [`postprocess_v2_outputs`](../../S5_3_1_sensitivity_panel_yield_ef_v2.py#L318), [`main`](../../S5_3_1_sensitivity_panel_yield_ef_v2.py#L339).

Direct local imports: [S3_6_scenarios.py](../../S3_6_scenarios.py), [S5_3_1_sensitivity_panel_yield_ef.py](../../S5_3_1_sensitivity_panel_yield_ef.py).

Imported by: [S5_3_1_sensitivity_panel_yield_ef_batches_v2.py](../../S5_3_1_sensitivity_panel_yield_ef_batches_v2.py), [S5_3_2_panel_data_summary_v2.py](../../S5_3_2_panel_data_summary_v2.py), [S5_3_3_merge_panel_yield_ef_batches_v2.py](../../S5_3_3_merge_panel_yield_ef_batches_v2.py).

### `S5_3_2_panel_data_summary.py`

Reads merged panel tables, reconstructs plotting fields, and reports missing points and emission detail; does not copy individual run directories.

[Source](../../S5_3_2_panel_data_summary.py) · 1,424 lines.

Entry points / interfaces: [`main`](../../S5_3_2_panel_data_summary.py#L1153).

Direct local imports: [config_paths.py](../../config_paths.py).

Imported by: [S5_3_2_panel_data_summary_v2.py](../../S5_3_2_panel_data_summary_v2.py).

### `S5_3_2_panel_data_summary_v2.py`

Applies panel summarization to v2 outputs and axis definitions, producing plot-ready data and missing-point diagnostics.

[Source](../../S5_3_2_panel_data_summary_v2.py) · 202 lines.

Entry points / interfaces: [`main`](../../S5_3_2_panel_data_summary_v2.py#L168).

Direct local imports: [S5_3_1_sensitivity_panel_yield_ef_v2.py](../../S5_3_1_sensitivity_panel_yield_ef_v2.py), [S5_3_2_panel_data_summary.py](../../S5_3_2_panel_data_summary.py).

### `S5_3_3_merge_panel_yield_ef_batches.py`

Checks expected batch/scenario coverage, unique IDs, and terminal statuses before merging panel CSVs and cost summaries; keeps original run directories in place.

[Source](../../S5_3_3_merge_panel_yield_ef_batches.py) · 729 lines.

Entry points / interfaces: [`main`](../../S5_3_3_merge_panel_yield_ef_batches.py#L432).

Direct local imports: [S5_3_1_sensitivity_panel_yield_ef.py](../../S5_3_1_sensitivity_panel_yield_ef.py), [S5_cost_summary_outputs.py](../../S5_cost_summary_outputs.py).

Imported by: [S5_3_3_merge_panel_yield_ef_batches_v2.py](../../S5_3_3_merge_panel_yield_ef_batches_v2.py).

### `S5_3_3_merge_panel_yield_ef_batches_v2.py`

Reuses the strict panel merger with the v2 module and output-root settings.

[Source](../../S5_3_3_merge_panel_yield_ef_batches_v2.py) · 47 lines.

Entry points / interfaces: [`main`](../../S5_3_3_merge_panel_yield_ef_batches_v2.py#L24).

Direct local imports: [S5_3_1_sensitivity_panel_yield_ef_v2.py](../../S5_3_1_sensitivity_panel_yield_ef_v2.py), [S5_3_3_merge_panel_yield_ef_batches.py](../../S5_3_3_merge_panel_yield_ef_batches.py).

### `S5_4_1_monte_carlo_full_variables.py`

Full-variable Monte Carlo through S4; records draws, statuses, fast totals, global process emissions, weighted parameters, realized diets, and land balance.

[Source](../../S5_4_1_monte_carlo_full_variables.py) · 2,698 lines.

Entry points / interfaces: [`main`](../../S5_4_1_monte_carlo_full_variables.py#L1825).

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py), [S4_0_main.py](../../S4_0_main.py), [S5_1_1_sensitivity_mc_variable_effect.py](../../S5_1_1_sensitivity_mc_variable_effect.py), [S5_cost_summary_outputs.py](../../S5_cost_summary_outputs.py), [config_paths.py](../../config_paths.py), [market_balance_diagnostics.py](../../market_balance_diagnostics.py), [model_run_status.py](../../model_run_status.py).

Imported by: [S5_4_1_monte_carlo_full_variables_batches.py](../../S5_4_1_monte_carlo_full_variables_batches.py), [S5_4_3_extreme_robustness_test.py](../../S5_4_3_extreme_robustness_test.py), [S5_5_1_Region-Item-Process_Importance_Gen.py](../../S5_5_1_Region-Item-Process_Importance_Gen.py), [S5_6_1_max_emission_reduction_potential.py](../../S5_6_1_max_emission_reduction_potential.py), [S5_6_4_max_reduction_impend.py](../../S5_6_4_max_reduction_impend.py), [S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py](../../S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py).

### `S5_4_1_monte_carlo_full_variables_batches.py`

Deterministically divides S5.4 samples into batches and handles resume and run-directory cleanup settings.

[Source](../../S5_4_1_monte_carlo_full_variables_batches.py) · 173 lines.

Entry points / interfaces: [`main`](../../S5_4_1_monte_carlo_full_variables_batches.py#L140).

Direct local imports: [S5_4_1_monte_carlo_full_variables.py](../../S5_4_1_monte_carlo_full_variables.py).

### `S5_4_2_merge_batches.py`

Validates batches and sample provenance before merging successful emissions, variables, diets, land indicators, and costs; excludes failed runs from successful result tables.

[Source](../../S5_4_2_merge_batches.py) · 592 lines.

Entry points / interfaces: [`main`](../../S5_4_2_merge_batches.py#L476).

Direct local imports: [S5_cost_summary_outputs.py](../../S5_cost_summary_outputs.py), [config_paths.py](../../config_paths.py).

### `S5_4_3_extreme_robustness_test.py`

Replaces random draws with five fixed endpoint/stress configurations to examine extreme-case feasibility; retained as a scientific robustness-analysis entry point.

[Source](../../S5_4_3_extreme_robustness_test.py) · 343 lines.

Entry points / interfaces: [`build_extreme_unit_matrix`](../../S5_4_3_extreme_robustness_test.py#L117), [`parse_args`](../../S5_4_3_extreme_robustness_test.py#L299), [`main`](../../S5_4_3_extreme_robustness_test.py#L317).

Direct local imports: [S5_4_1_monte_carlo_full_variables.py](../../S5_4_1_monte_carlo_full_variables.py), [config_paths.py](../../config_paths.py).

### `S5_5_0_Region-Item-Process_Importance_extract_previous.py`

Audits older S5.1/S5.3/S5.4 outputs for sufficient detailed emissions and reconstructs structural data where possible; global totals cannot recover missing country detail.

[Source](../../S5_5_0_Region-Item-Process_Importance_extract_previous.py) · 1,378 lines.

Entry points / interfaces: [`build_region_item_process_importance`](../../S5_5_0_Region-Item-Process_Importance_extract_previous.py#L1263), [`main`](../../S5_5_0_Region-Item-Process_Importance_extract_previous.py#L1371).

Direct local imports: [SP_M1a_Figure_pie_structure_pre.py](../../SP_M1a_Figure_pie_structure_pre.py), [config_paths.py](../../config_paths.py).

### `S5_5_1_Region-Item-Process_Importance_Gen.py`

Reuses full-variable MC while retaining detailed emissions to build region/item/process totals, shares, and distribution tables for structural analysis.

[Source](../../S5_5_1_Region-Item-Process_Importance_Gen.py) · 1,496 lines.

Entry points / interfaces: [`main`](../../S5_5_1_Region-Item-Process_Importance_Gen.py#L1474).

Direct local imports: [S5_4_1_monte_carlo_full_variables.py](../../S5_4_1_monte_carlo_full_variables.py), [S5_cost_summary_outputs.py](../../S5_cost_summary_outputs.py), [SP_M1a_Figure_pie_structure_pre.py](../../SP_M1a_Figure_pie_structure_pre.py), [config_paths.py](../../config_paths.py).

### `S5_5_2_Region-Item-Process_Importance_Gen_batches.py`

Loads the hyphenated S5.5 script dynamically and runs one consistent MC/structural-processing batch.

[Source](../../S5_5_2_Region-Item-Process_Importance_Gen_batches.py) · 157 lines.

Entry points / interfaces: [`main`](../../S5_5_2_Region-Item-Process_Importance_Gen_batches.py#L133).

### `S5_5_3_Region-Item-Process_Importance_merge_batches.py`

Validates S5.5 batches and creates merged_structure tables from detailed batch results without copying the original output trees.

[Source](../../S5_5_3_Region-Item-Process_Importance_merge_batches.py) · 114 lines.

Entry points / interfaces: [`main`](../../S5_5_3_Region-Item-Process_Importance_merge_batches.py#L92).

### `S5_6_1_max_emission_reduction_potential.py`

Runs baseline, global packages/single measures, and country-action endpoints to compare tested reduction potentials; does not prove an optimum over all policy choices.

[Source](../../S5_6_1_max_emission_reduction_potential.py) · 1,057 lines.

Entry points / interfaces: [`ScenarioPlan`](../../S5_6_1_max_emission_reduction_potential.py#L183), [`parse_args`](../../S5_6_1_max_emission_reduction_potential.py#L918), [`main`](../../S5_6_1_max_emission_reduction_potential.py#L985).

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py), [S4_0_main.py](../../S4_0_main.py), [S5_4_1_monte_carlo_full_variables.py](../../S5_4_1_monte_carlo_full_variables.py), [S5_cost_summary_outputs.py](../../S5_cost_summary_outputs.py), [config_paths.py](../../config_paths.py), [model_run_status.py](../../model_run_status.py).

Imported by: [S5_6_1_max_emission_reduction_potential_batches.py](../../S5_6_1_max_emission_reduction_potential_batches.py), [S5_6_2_merge_batches.py](../../S5_6_2_merge_batches.py), [S5_6_4_max_reduction_impend.py](../../S5_6_4_max_reduction_impend.py), [S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py](../../S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py), [S5_8_1_country_strategy_map_sensitivity.py](../../S5_8_1_country_strategy_map_sensitivity.py).

### `S5_6_1_max_emission_reduction_potential_batches.py`

Partitions country endpoints; the first batch additionally runs shared reference/global scenarios.

[Source](../../S5_6_1_max_emission_reduction_potential_batches.py) · 262 lines.

Entry points / interfaces: [`main`](../../S5_6_1_max_emission_reduction_potential_batches.py#L233).

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py), [S5_6_1_max_emission_reduction_potential.py](../../S5_6_1_max_emission_reduction_potential.py).

### `S5_6_2_merge_batches.py`

Merges S5.6 designs and statuses and recalculates national/global reductions against the common merged reference.

[Source](../../S5_6_2_merge_batches.py) · 301 lines.

Entry points / interfaces: [`main`](../../S5_6_2_merge_batches.py#L217).

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py), [S5_6_1_max_emission_reduction_potential.py](../../S5_6_1_max_emission_reduction_potential.py).

### `S5_6_4_max_reduction_impend.py`

Relaxes measures in a prescribed order until a strong endpoint becomes feasible; records the first feasible combination as a greedy search result.

[Source](../../S5_6_4_max_reduction_impend.py) · 921 lines.

Entry points / interfaces: [`parse_args`](../../S5_6_4_max_reduction_impend.py#L737), [`main`](../../S5_6_4_max_reduction_impend.py#L778).

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py), [S4_0_main.py](../../S4_0_main.py), [S5_4_1_monte_carlo_full_variables.py](../../S5_4_1_monte_carlo_full_variables.py), [S5_6_1_max_emission_reduction_potential.py](../../S5_6_1_max_emission_reduction_potential.py), [S5_cost_summary_outputs.py](../../S5_cost_summary_outputs.py).

### `S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py`

Maps nine strategy groups through the strategy JSON, reruns single/combined measures, and constructs normalized or exact Shapley contributions and MACC outputs.

[Source](../../S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py) · 2,527 lines.

Entry points / interfaces: [`parse_args`](../../S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py#L1302), [`main`](../../S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py#L2409).

Direct local imports: [S5_4_1_monte_carlo_full_variables.py](../../S5_4_1_monte_carlo_full_variables.py), [S5_6_1_max_emission_reduction_potential.py](../../S5_6_1_max_emission_reduction_potential.py), [config_paths.py](../../config_paths.py), [cost_database_v2.py](../../cost_database_v2.py).

Imported by: [S5_7_1_strategy_endpoint_rerun_max_reduction_potential_batches.py](../../S5_7_1_strategy_endpoint_rerun_max_reduction_potential_batches.py), [S5_7_2_strategy_endpoint_rerun_merge_batches.py](../../S5_7_2_strategy_endpoint_rerun_merge_batches.py), [S5_7_3_strategy_macc_cdr_price.py](../../S5_7_3_strategy_macc_cdr_price.py), [S5_8_1_country_strategy_map_sensitivity.py](../../S5_8_1_country_strategy_map_sensitivity.py), [S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py), [S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves_batches.py](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves_batches.py).

### `S5_7_1_strategy_endpoint_rerun_max_reduction_potential_batches.py`

Batches strategy coalitions with reference alignment; nine measures imply 512 unique subsets, not a country-based partition.

[Source](../../S5_7_1_strategy_endpoint_rerun_max_reduction_potential_batches.py) · 260 lines.

Entry points / interfaces: [`main`](../../S5_7_1_strategy_endpoint_rerun_max_reduction_potential_batches.py#L229).

Direct local imports: [S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py](../../S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py).

### `S5_7_2_strategy_endpoint_rerun_merge_batches.py`

Merges and deduplicates coalition results, costs, and provenance, then reconstructs normalized or exact Shapley attribution and MACC tables.

[Source](../../S5_7_2_strategy_endpoint_rerun_merge_batches.py) · 518 lines.

Entry points / interfaces: [`run_merge`](../../S5_7_2_strategy_endpoint_rerun_merge_batches.py#L367), [`main`](../../S5_7_2_strategy_endpoint_rerun_merge_batches.py#L512).

Direct local imports: [S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py](../../S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py).

Imported by: [S5_9_2_bioenergy_scenario_marginal_abatement_cost_curves_merge_batches.py](../../S5_9_2_bioenergy_scenario_marginal_abatement_cost_curves_merge_batches.py).

### `S5_7_3_strategy_macc_cdr_price.py`

Reruns measure portfolios at specified implementation levels and scans/refines land-carbon prices to approach an emission target.

[Source](../../S5_7_3_strategy_macc_cdr_price.py) · 619 lines.

Entry points / interfaces: [`parse_args`](../../S5_7_3_strategy_macc_cdr_price.py#L476), [`main`](../../S5_7_3_strategy_macc_cdr_price.py#L517).

Direct local imports: [S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py](../../S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py), [S5_cost_summary_outputs.py](../../S5_cost_summary_outputs.py).

Imported by: [S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py).

### `S5_8_1_country_strategy_map_sensitivity.py`

Applies each of nine measures independently in each country while other countries remain at reference settings; records national/global effects and priced costs.

[Source](../../S5_8_1_country_strategy_map_sensitivity.py) · 1,138 lines.

Entry points / interfaces: [`ensure_output_child`](../../S5_8_1_country_strategy_map_sensitivity.py#L61), [`main`](../../S5_8_1_country_strategy_map_sensitivity.py#L991).

Direct local imports: [S5_6_1_max_emission_reduction_potential.py](../../S5_6_1_max_emission_reduction_potential.py), [S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py](../../S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py), [S5_8_3_prepare_country_dominant_mitigation_intervention.py](../../S5_8_3_prepare_country_dominant_mitigation_intervention.py), [S5_cost_summary_outputs.py](../../S5_cost_summary_outputs.py), [config_paths.py](../../config_paths.py).

Imported by: [S5_8_1_country_strategy_map_sensitivity_batches.py](../../S5_8_1_country_strategy_map_sensitivity_batches.py), [S5_8_2_country_strategy_map_merge_batches.py](../../S5_8_2_country_strategy_map_merge_batches.py).

### `S5_8_1_country_strategy_map_sensitivity_batches.py`

Partitions countries and builds an independent detailed reference in each batch for matched national intervention comparisons.

[Source](../../S5_8_1_country_strategy_map_sensitivity_batches.py) · 239 lines.

Entry points / interfaces: [`main`](../../S5_8_1_country_strategy_map_sensitivity_batches.py#L213).

Direct local imports: [S5_8_1_country_strategy_map_sensitivity.py](../../S5_8_1_country_strategy_map_sensitivity.py).

### `S5_8_2_country_strategy_map_merge_batches.py`

Checks country-measure coverage, statuses, and reference consistency, then merges results, rankings, and countries without valid candidates.

[Source](../../S5_8_2_country_strategy_map_merge_batches.py) · 700 lines.

Entry points / interfaces: [`main`](../../S5_8_2_country_strategy_map_merge_batches.py#L373).

Direct local imports: [S5_8_1_country_strategy_map_sensitivity.py](../../S5_8_1_country_strategy_map_sensitivity.py), [S5_8_3_prepare_country_dominant_mitigation_intervention.py](../../S5_8_3_prepare_country_dominant_mitigation_intervention.py), [S5_cost_summary_outputs.py](../../S5_cost_summary_outputs.py).

### `S5_8_3_prepare_country_dominant_mitigation_intervention.py`

Converts S5.8 results into traceable Figure 3d categories with explicit ranking metrics, tie/near-tie handling, and coverage audits.

[Source](../../S5_8_3_prepare_country_dominant_mitigation_intervention.py) · 974 lines.

Entry points / interfaces: [`Figure3dSettings`](../../S5_8_3_prepare_country_dominant_mitigation_intervention.py#L102), [`ensure_output_child`](../../S5_8_3_prepare_country_dominant_mitigation_intervention.py#L148), [`rank_country_interventions`](../../S5_8_3_prepare_country_dominant_mitigation_intervention.py#L271), [`build_validation_table`](../../S5_8_3_prepare_country_dominant_mitigation_intervention.py#L508), [`write_figure3d_outputs`](../../S5_8_3_prepare_country_dominant_mitigation_intervention.py#L692), [`main`](../../S5_8_3_prepare_country_dominant_mitigation_intervention.py#L923).

Direct local imports: [SP_M3d_Figure_country_dominant_mitigation_intervention.py](../../SP_M3d_Figure_country_dominant_mitigation_intervention.py).

Imported by: [S5_8_1_country_strategy_map_sensitivity.py](../../S5_8_1_country_strategy_map_sensitivity.py), [S5_8_2_country_strategy_map_merge_batches.py](../../S5_8_2_country_strategy_map_merge_batches.py), [SP_M3d_Figure_country_dominant_mitigation_intervention.py](../../SP_M3d_Figure_country_dominant_mitigation_intervention.py).

### `S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py`

Coordinates low/medium/high bioenergy MACCs using S5.7; supports preflight, quick, and exact modes, cost-database hash checks, feasibility checks, and figure outputs.

[Source](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py) · 992 lines.

Entry points / interfaces: [`PanelCase`](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py#L78), [`resolve_output_root`](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py#L126), [`parse_cases`](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py#L138), [`build_arg_parser`](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py#L796), [`run_full_chain`](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py#L828), [`main`](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py#L986).

Direct local imports: [S0_53_prepare_bioenergy_scenarios.py](../../S0_53_prepare_bioenergy_scenarios.py), [S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py](../../S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py), [S5_7_3_strategy_macc_cdr_price.py](../../S5_7_3_strategy_macc_cdr_price.py), [ST_bioenergy_full_run_smoke.py](../../ST_bioenergy_full_run_smoke.py).

Imported by: [S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves_batches.py](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves_batches.py), [S5_9_2_bioenergy_scenario_marginal_abatement_cost_curves_merge_batches.py](../../S5_9_2_bioenergy_scenario_marginal_abatement_cost_curves_merge_batches.py).

### `S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves_batches.py`

Runs the corresponding S5.7 batch within each bioenergy panel and enforces persistent output placement under Code/output.

[Source](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves_batches.py) · 302 lines.

Entry points / interfaces: [`BatchSettings`](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves_batches.py#L39), [`build_arg_parser`](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves_batches.py#L93), [`resolve_settings`](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves_batches.py#L113), [`run_batch`](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves_batches.py#L226), [`main`](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves_batches.py#L284).

Direct local imports: [S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py](../../S5_7_1_strategy_endpoint_rerun_max_reduction_potential.py), [S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py).

Imported by: [S5_9_2_bioenergy_scenario_marginal_abatement_cost_curves_merge_batches.py](../../S5_9_2_bioenergy_scenario_marginal_abatement_cost_curves_merge_batches.py).

### `S5_9_2_bioenergy_scenario_marginal_abatement_cost_curves_merge_batches.py`

Strictly merges each bioenergy panel through S5.7, checks cost provenance, combines panel workbooks, and regenerates PNG/SVG figures.

[Source](../../S5_9_2_bioenergy_scenario_marginal_abatement_cost_curves_merge_batches.py) · 345 lines.

Entry points / interfaces: [`build_arg_parser`](../../S5_9_2_bioenergy_scenario_marginal_abatement_cost_curves_merge_batches.py#L48), [`run_merge`](../../S5_9_2_bioenergy_scenario_marginal_abatement_cost_curves_merge_batches.py#L207), [`main`](../../S5_9_2_bioenergy_scenario_marginal_abatement_cost_curves_merge_batches.py#L339).

Direct local imports: [S5_7_2_strategy_endpoint_rerun_merge_batches.py](../../S5_7_2_strategy_endpoint_rerun_merge_batches.py), [S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py), [S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves_batches.py](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves_batches.py).

### `S5_cost_summary_outputs.py`

Collects country/global measure costs across runs and emits experiment-level summaries and unpriced/missing-cost audits.

[Source](../../S5_cost_summary_outputs.py) · 312 lines.

Entry points / interfaces: [`write_sensitivity_cost_summaries`](../../S5_cost_summary_outputs.py#L181).

Direct local imports: [S4_1_results.py](../../S4_1_results.py).

Imported by: [S5_0_1_sensitivity_mc_levels.py](../../S5_0_1_sensitivity_mc_levels.py), [S5_1_1_sensitivity_mc_variable_effect.py](../../S5_1_1_sensitivity_mc_variable_effect.py), [S5_1_3_merge_variable_effect_batches.py](../../S5_1_3_merge_variable_effect_batches.py), [S5_3_1_sensitivity_panel_yield_ef.py](../../S5_3_1_sensitivity_panel_yield_ef.py), [S5_3_3_merge_panel_yield_ef_batches.py](../../S5_3_3_merge_panel_yield_ef_batches.py), [S5_4_1_monte_carlo_full_variables.py](../../S5_4_1_monte_carlo_full_variables.py), [S5_4_2_merge_batches.py](../../S5_4_2_merge_batches.py), [S5_5_1_Region-Item-Process_Importance_Gen.py](../../S5_5_1_Region-Item-Process_Importance_Gen.py), [S5_6_1_max_emission_reduction_potential.py](../../S5_6_1_max_emission_reduction_potential.py), [S5_6_4_max_reduction_impend.py](../../S5_6_4_max_reduction_impend.py), [S5_7_3_strategy_macc_cdr_price.py](../../S5_7_3_strategy_macc_cdr_price.py), [S5_8_1_country_strategy_map_sensitivity.py](../../S5_8_1_country_strategy_map_sensitivity.py); further references are in the inventory.

## Analysis and diagnostics

### `SA0_1_Population_analysis.py`

Maps population time series to model countries and regional groups for interpreting future demand drivers.

[Source](../../SA0_1_Population_analysis.py) · 143 lines.

Entry points / interfaces: [`build_population_growth_table`](../../SA0_1_Population_analysis.py#L38), [`parse_args`](../../SA0_1_Population_analysis.py#L91), [`main`](../../SA0_1_Population_analysis.py#L123).

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py), [config_paths.py](../../config_paths.py).

### `SA1_1_GHG_emis_map.py`

Combines baseline emissions, population, and country geometry to map 2020 total and per-capita greenhouse gases.

[Source](../../SA1_1_GHG_emis_map.py) · 194 lines.

Entry points / interfaces: [`get_optimized_norm_cmap`](../../SA1_1_GHG_emis_map.py#L97), [`plot_map`](../../SA1_1_GHG_emis_map.py#L146).

### `SC0_1_Nutrition_check.py`

Compares model demand, nutrient factors, and minimum intake at country-year level to diagnose baseline coverage or unit differences.

[Source](../../SC0_1_Nutrition_check.py) · 210 lines.

Entry points / interfaces: [`parse_args`](../../SC0_1_Nutrition_check.py#L128), [`main`](../../SC0_1_Nutrition_check.py#L154).

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py), [config_paths.py](../../config_paths.py).

### `SC0_2_Nutrition_production_check.py`

Compares baseline production with nutrition-profile quantities and supply in 2020.

[Source](../../SC0_2_Nutrition_production_check.py) · 122 lines.

Entry points / interfaces: [`main`](../../SC0_2_Nutrition_production_check.py#L74).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SC0_3_Item_Demand_Category.py`

Maps food-balance uses to model commodities and summarizes food, feed, seed, processing, losses, and residual shares.

[Source](../../SC0_3_Item_Demand_Category.py) · 254 lines.

Entry points / interfaces: [`main`](../../SC0_3_Item_Demand_Category.py#L118).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SC0_4_Item_Demand_Supply_Trade.py`

Aggregates 2010-2020 demand, supply, imports, exports, and use categories by M49 and commodity for trade/balance checks.

[Source](../../SC0_4_Item_Demand_Supply_Trade.py) · 253 lines.

Entry points / interfaces: [`main`](../../SC0_4_Item_Demand_Supply_Trade.py#L179).

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py), [config_paths.py](../../config_paths.py).

### `SC1_1_regression_output_compare.py`

Compares tabular outputs from two scenario directories by keys and numerical tolerances, producing semantic differences rather than log-text comparisons.

[Source](../../SC1_1_regression_output_compare.py) · 348 lines.

Entry points / interfaces: [`TableCompareResult`](../../SC1_1_regression_output_compare.py#L31), [`main`](../../SC1_1_regression_output_compare.py#L261).

Direct local imports: [config_paths.py](../../config_paths.py).

## Manuscript and supplementary figures

### `SP_M1a_Figure_pie_structure_plot.py`

Plots baseline emission composition from the 2020 structure workbook and exports PNG/SVG figures.

[Source](../../SP_M1a_Figure_pie_structure_plot.py) · 296 lines.

Entry points / interfaces: [`save_process_legend_svg`](../../SP_M1a_Figure_pie_structure_plot.py#L203), [`plot_y2020_emis_structure`](../../SP_M1a_Figure_pie_structure_plot.py#L225), [`main`](../../SP_M1a_Figure_pie_structure_plot.py#L287).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M1a_Figure_pie_structure_plot_v2.1.py`

Reuses v2 plotting with the v2 structure workbook for a particular layout; its dotted filename is commonly loaded through runpy.

[Source](../../SP_M1a_Figure_pie_structure_plot_v2.1.py) · 198 lines.

Entry points / interfaces: [`plot_y2020_emis_structure_v2_1`](../../SP_M1a_Figure_pie_structure_plot_v2.1.py#L141), [`main`](../../SP_M1a_Figure_pie_structure_plot_v2.1.py#L191).

Direct local imports: [SP_M1a_Figure_pie_structure_plot_v2.py](../../SP_M1a_Figure_pie_structure_plot_v2.py).

### `SP_M1a_Figure_pie_structure_plot_v2.py`

Alternative structure-figure layout and colors, also exposing plotting/configuration functions reused by later scripts.

[Source](../../SP_M1a_Figure_pie_structure_plot_v2.py) · 476 lines.

Entry points / interfaces: [`plot_y2020_emis_structure_v2`](../../SP_M1a_Figure_pie_structure_plot_v2.py#L423), [`main`](../../SP_M1a_Figure_pie_structure_plot_v2.py#L469).

Direct local imports: [config_paths.py](../../config_paths.py).

Imported by: [SP_M1a_Figure_pie_structure_plot_v2.1.py](../../SP_M1a_Figure_pie_structure_plot_v2.1.py), [SP_SI3_Figure_emis_reduction_disaggregate.py](../../SP_SI3_Figure_emis_reduction_disaggregate.py), [SP_SI4_Figure_emission_per_kcal_item.py](../../SP_SI4_Figure_emission_per_kcal_item.py).

### `SP_M1a_Figure_pie_structure_pre.py`

Maps historical emissions into figure-specific region/item/process groups and builds the shared 2020 structure workbook.

[Source](../../SP_M1a_Figure_pie_structure_pre.py) · 462 lines.

Entry points / interfaces: [`build_y2020_emis_structure_summary`](../../SP_M1a_Figure_pie_structure_pre.py#L434), [`main`](../../SP_M1a_Figure_pie_structure_pre.py#L456).

Direct local imports: [config_paths.py](../../config_paths.py).

Imported by: [S0_49_emission_result_summary.py](../../S0_49_emission_result_summary.py), [S5_5_0_Region-Item-Process_Importance_extract_previous.py](../../S5_5_0_Region-Item-Process_Importance_extract_previous.py), [S5_5_1_Region-Item-Process_Importance_Gen.py](../../S5_5_1_Region-Item-Process_Importance_Gen.py), [SP_M1a_Figure_pie_structure_pre_v2.py](../../SP_M1a_Figure_pie_structure_pre_v2.py).

### `SP_M1a_Figure_pie_structure_pre_v2.py`

Reuses structure preparation with revised forest-net-process aggregation and writes the v2 workbook.

[Source](../../SP_M1a_Figure_pie_structure_pre_v2.py) · 52 lines.

Entry points / interfaces: [`build_y2020_emis_structure_summary_v2`](../../SP_M1a_Figure_pie_structure_pre_v2.py#L23), [`main`](../../SP_M1a_Figure_pie_structure_pre_v2.py#L46).

Direct local imports: [SP_M1a_Figure_pie_structure_pre.py](../../SP_M1a_Figure_pie_structure_pre.py).

### `SP_M1aSI_Figure_bar2_structure_plot.py`

Second supplementary structure-bar layout with the same missing legacy color/process-module import.

[Source](../../SP_M1aSI_Figure_bar2_structure_plot.py) · 263 lines.

Entry points / interfaces: [`plot_y2020_emis_structure_bar2`](../../SP_M1aSI_Figure_bar2_structure_plot.py#L182), [`main`](../../SP_M1aSI_Figure_bar2_structure_plot.py#L256).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M1aSI_Figure_bar_structure_plot.py`

Supplementary historical-structure bar plots; imports constants from the currently missing legacy module SP1_8_2_Figure_pie_structure_plot.

[Source](../../SP_M1aSI_Figure_bar_structure_plot.py) · 218 lines.

Entry points / interfaces: [`plot_y2020_emis_structure_bar`](../../SP_M1aSI_Figure_bar_structure_plot.py#L164), [`main`](../../SP_M1aSI_Figure_bar_structure_plot.py#L211).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M1b_Figure_AR6_scenario_plot.py`

Plots prepared AR6 scenario trajectories using the original layout and configured comparison inputs.

[Source](../../SP_M1b_Figure_AR6_scenario_plot.py) · 440 lines.

Entry points / interfaces: [`main`](../../SP_M1b_Figure_AR6_scenario_plot.py#L424).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M1b_Figure_AR6_scenario_plot_2.py`

Alternative layout for the AR6 scenario comparison; retain its own input and aggregation conventions.

[Source](../../SP_M1b_Figure_AR6_scenario_plot_2.py) · 318 lines.

Entry points / interfaces: [`main`](../../SP_M1b_Figure_AR6_scenario_plot_2.py#L264).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M1b_Figure_AR6_scenario_plot_v2.py`

Updated AR6 comparison layout and rendering workflow using prepared scenario data.

[Source](../../SP_M1b_Figure_AR6_scenario_plot_v2.py) · 526 lines.

Entry points / interfaces: [`main`](../../SP_M1b_Figure_AR6_scenario_plot_v2.py#L510).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M1b_Figure_AR6_scenario_prep.py`

Prepares AR6 scenario series and categories for historical/future emission comparison plots.

[Source](../../SP_M1b_Figure_AR6_scenario_prep.py) · 420 lines.

Entry points / interfaces: [`main`](../../SP_M1b_Figure_AR6_scenario_prep.py#L362).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M1b_Figure_SCI_scenario_plot_v2.py`

Plots prepared SCI scenario comparisons with the v2 layout and figure-specific settings.

[Source](../../SP_M1b_Figure_SCI_scenario_plot_v2.py) · 999 lines.

Entry points / interfaces: [`main`](../../SP_M1b_Figure_SCI_scenario_plot_v2.py#L928).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M1b_Figure_SCI_scenario_prep.py`

Prepares SCI scenario trajectories, groups, and reference data for the SCI comparison figures.

[Source](../../SP_M1b_Figure_SCI_scenario_prep.py) · 536 lines.

Entry points / interfaces: [`main`](../../SP_M1b_Figure_SCI_scenario_prep.py#L482).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M2_Figure_contour.py`

Plots emission contours/heat maps from the original S5.3 yield-EF panel and exports actual plotting data and diagnostics.

[Source](../../SP_M2_Figure_contour.py) · 1,830 lines.

Entry points / interfaces: [`main`](../../SP_M2_Figure_contour.py#L1737).

Direct local imports: [config_paths.py](../../config_paths.py).

Imported by: [SP_M2_Figure_contour_v2.py](../../SP_M2_Figure_contour_v2.py).

### `SP_M2_Figure_contour_v2.py`

Uses realized P, A/P, L/A, LUC/L, and agricultural-emissions/A factors for a PALE decomposition view; axes are not simply the original sampled yield and EF multipliers.

[Source](../../SP_M2_Figure_contour_v2.py) · 441 lines.

Entry points / interfaces: [`plot_pale_figure`](../../SP_M2_Figure_contour_v2.py#L273), [`main`](../../SP_M2_Figure_contour_v2.py#L416).

Direct local imports: [SP_M2_Figure_contour.py](../../SP_M2_Figure_contour.py), [config_paths.py](../../config_paths.py).

### `SP_M3a_Figure_macc_stock.py`

Legacy MACC/remaining-emissions plot reading a global_macc sheet; retains historical Figure4 output naming.

[Source](../../SP_M3a_Figure_macc_stock.py) · 165 lines.

Entry points / interfaces: [`main`](../../SP_M3a_Figure_macc_stock.py#L85).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M3a_Figure_macc_stock_v2.py`

Second layout for the legacy single-sheet global_macc contract, distinct from the three-bioenergy-panel figures.

[Source](../../SP_M3a_Figure_macc_stock_v2.py) · 200 lines.

Entry points / interfaces: [`main`](../../SP_M3a_Figure_macc_stock_v2.py#L107).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M3a_Figure_macc_stock_v3.1.py`

Current S5.9 three-panel figure entry point with explicit workbook/output paths, corresponding to the Figure 3a-c workflow.

[Source](../../SP_M3a_Figure_macc_stock_v3.1.py) · 325 lines.

Entry points / interfaces: [`main`](../../SP_M3a_Figure_macc_stock_v3.1.py#L264).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M3a_Figure_macc_stock_v3.py`

Plots three bioenergy MACC/remaining-emission panels from low_Bio, medium_Bio, and high_Bio workbook sheets.

[Source](../../SP_M3a_Figure_macc_stock_v3.py) · 249 lines.

Entry points / interfaces: [`main`](../../SP_M3a_Figure_macc_stock_v3.py#L199).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M3b_Figure_map_reduction_cost_potential.py`

Legacy dominant-process reduction/cost map using historical Figure3 input formats; not the standardized Figure 3d intervention definition.

[Source](../../SP_M3b_Figure_map_reduction_cost_potential.py) · 258 lines.

Entry points / interfaces: [`main`](../../SP_M3b_Figure_map_reduction_cost_potential.py#L211).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M3b_Figure_map_reduction_cost_potential_v2.1.py`

Compatibility map variant with explicit inputs and missing-value controls; does not replace the nine-intervention Figure 3d ranking.

[Source](../../SP_M3b_Figure_map_reduction_cost_potential_v2.1.py) · 317 lines.

Entry points / interfaces: [`main`](../../SP_M3b_Figure_map_reduction_cost_potential_v2.1.py#L260).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M3b_Figure_map_reduction_cost_potential_v2.py`

Alternative legacy map layout with separate legend export and historical input contracts.

[Source](../../SP_M3b_Figure_map_reduction_cost_potential_v2.py) · 267 lines.

Entry points / interfaces: [`main`](../../SP_M3b_Figure_map_reduction_cost_potential_v2.py#L220).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M3d_Figure_country_dominant_mitigation_intervention.py`

Maps standardized S5.8.3 dominant-intervention categories; preserves missing geometry/rankings as missing and exports matching audits and a figure manifest.

[Source](../../SP_M3d_Figure_country_dominant_mitigation_intervention.py) · 481 lines.

Entry points / interfaces: [`prepare_geometry_join`](../../SP_M3d_Figure_country_dominant_mitigation_intervention.py#L154), [`plot_figure3d`](../../SP_M3d_Figure_country_dominant_mitigation_intervention.py#L307), [`main`](../../SP_M3d_Figure_country_dominant_mitigation_intervention.py#L420).

Direct local imports: [S5_8_3_prepare_country_dominant_mitigation_intervention.py](../../S5_8_3_prepare_country_dominant_mitigation_intervention.py), [config_paths.py](../../config_paths.py).

Imported by: [S5_8_3_prepare_country_dominant_mitigation_intervention.py](../../S5_8_3_prepare_country_dominant_mitigation_intervention.py).

### `SP_M4a_Figure_sensitivity_stock.py`

Rebuilds figure workbooks from S5.0 importance outputs and plots strategy-importance stacks; retains historical Figure2 naming.

[Source](../../SP_M4a_Figure_sensitivity_stock.py) · 449 lines.

Entry points / interfaces: [`main`](../../SP_M4a_Figure_sensitivity_stock.py#L430).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M4b_Figure_yield_ef_effect_line.py`

Plots legacy yield/EF sensitivity curves from S5.1 Fig5 samples, histograms, and target tables.

[Source](../../SP_M4b_Figure_yield_ef_effect_line.py) · 198 lines.

Entry points / interfaces: [`main`](../../SP_M4b_Figure_yield_ef_effect_line.py#L155).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M4b_Figure_yield_ef_effect_line_targetrange.py`

Builds variable panels for emission-target ranges from merged S5.4 successes, including sample, histogram, and mean tables.

[Source](../../SP_M4b_Figure_yield_ef_effect_line_targetrange.py) · 794 lines.

Entry points / interfaces: [`main`](../../SP_M4b_Figure_yield_ef_effect_line_targetrange.py#L732).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M4b_Figure_yield_ef_effect_line_targetrange_v2.py`

Current quartile-distribution figure workflow using realized yield, production emission intensity, ruminant intake, and land intensity; exports baseline and intensity audits.

[Source](../../SP_M4b_Figure_yield_ef_effect_line_targetrange_v2.py) · 2,817 lines.

Entry points / interfaces: [`main`](../../SP_M4b_Figure_yield_ef_effect_line_targetrange_v2.py#L2668).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M4b_Figure_yield_ef_effect_line_targetval.py`

Plots yield, EF, and ruminant-intake distributions/curves around selected emission targets using S5.1/legacy Fig5 summaries.

[Source](../../SP_M4b_Figure_yield_ef_effect_line_targetval.py) · 537 lines.

Entry points / interfaces: [`main`](../../SP_M4b_Figure_yield_ef_effect_line_targetval.py#L488).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M4e_Figure_sensitivity_Item-Region-Process_stock.py`

Builds region/item/process structure figures from S5.5 merged long tables and writes workbooks with historical Figure9 naming.

[Source](../../SP_M4e_Figure_sensitivity_Item-Region-Process_stock.py) · 696 lines.

Entry points / interfaces: [`main`](../../SP_M4e_Figure_sensitivity_Item-Region-Process_stock.py#L649).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M4f_Figure_structure_importance_targetrange_v1.py`

Early structural-figure entry point consuming precomputed shares and quartile distributions.

[Source](../../SP_M4f_Figure_structure_importance_targetrange_v1.py) · 753 lines.

Entry points / interfaces: [`main`](../../SP_M4f_Figure_structure_importance_targetrange_v1.py#L665).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_M4f_Figure_structure_importance_targetrange_v2.py`

Plots region/item/process emission shares from the Figure9 workbook; measures fractions of emissions, not v3 regression importance.

[Source](../../SP_M4f_Figure_structure_importance_targetrange_v2.py) · 769 lines.

Entry points / interfaces: [`main`](../../SP_M4f_Figure_structure_importance_targetrange_v2.py#L757).

Direct local imports: [config_paths.py](../../config_paths.py).

Imported by: [SP_M4f_Figure_structure_importance_targetrange_v3.py](../../SP_M4f_Figure_structure_importance_targetrange_v3.py).

### `SP_M4f_Figure_structure_importance_targetrange_v3.py`

Combines S5.5 structural emissions and S5.0 strategy importance into regression-contribution figures and auditable source tables for Figure 4a-d.

[Source](../../SP_M4f_Figure_structure_importance_targetrange_v3.py) · 1,881 lines.

Entry points / interfaces: [`main`](../../SP_M4f_Figure_structure_importance_targetrange_v3.py#L1822).

Direct local imports: [SP_M4f_Figure_structure_importance_targetrange_v2.py](../../SP_M4f_Figure_structure_importance_targetrange_v2.py), [config_paths.py](../../config_paths.py).

### `SP_SI1_Figure_prod_trend_region_fb.py`

Aggregates production summaries to region/item trends and exports source CSVs plus PNG/SVG figures.

[Source](../../SP_SI1_Figure_prod_trend_region_fb.py) · 289 lines.

Entry points / interfaces: [`plot_prod_trend_region_fb`](../../SP_SI1_Figure_prod_trend_region_fb.py#L194), [`main`](../../SP_SI1_Figure_prod_trend_region_fb.py#L272).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_SI2_Figure_ruminateIntakeRatio_map.py`

Calculates country ruminant-food intake shares from food uses and nutrition mappings and plots geographic patterns.

[Source](../../SP_SI2_Figure_ruminateIntakeRatio_map.py) · 308 lines.

Entry points / interfaces: [`build_or_load_cache`](../../SP_SI2_Figure_ruminateIntakeRatio_map.py#L136), [`plot_map_2020`](../../SP_SI2_Figure_ruminateIntakeRatio_map.py#L203), [`main`](../../SP_SI2_Figure_ruminateIntakeRatio_map.py#L300).

Direct local imports: [config_paths.py](../../config_paths.py).

### `SP_SI3_Figure_emis_reduction_disaggregate.py`

Uses S0.49's 2080 summaries to decompose reduction by process/item and produce waterfall-style figures with source data.

[Source](../../SP_SI3_Figure_emis_reduction_disaggregate.py) · 414 lines.

Entry points / interfaces: [`build_fig10_emission_reduction_waterfalls`](../../SP_SI3_Figure_emis_reduction_disaggregate.py#L364), [`main`](../../SP_SI3_Figure_emis_reduction_disaggregate.py#L407).

Direct local imports: [SP_M1a_Figure_pie_structure_plot_v2.py](../../SP_M1a_Figure_pie_structure_plot_v2.py), [config_paths.py](../../config_paths.py).

### `SP_SI4_Figure_emission_per_kcal_item.py`

Calculates and visualizes commodity emissions per calorie from 2080 emissions and energy indicators.

[Source](../../SP_SI4_Figure_emission_per_kcal_item.py) · 642 lines.

Entry points / interfaces: [`build_fig11_emission_per_kcal_item`](../../SP_SI4_Figure_emission_per_kcal_item.py#L609).

Direct local imports: [SP_M1a_Figure_pie_structure_plot_v2.py](../../SP_M1a_Figure_pie_structure_plot_v2.py), [config_paths.py](../../config_paths.py).

## Retained bioenergy validation dependencies

### `ST_bioenergy_full_run_smoke.py`

Layered bioenergy validation runner for unit, module integration, micro-solvers, and full regression; retained because S5.9 imports it.

[Source](../../ST_bioenergy_full_run_smoke.py) · 692 lines.

Entry points / interfaces: [`run_smoke`](../../ST_bioenergy_full_run_smoke.py#L627), [`parse_args`](../../ST_bioenergy_full_run_smoke.py#L659), [`main`](../../ST_bioenergy_full_run_smoke.py#L685).

Direct local imports: [S1_0_schema.py](../../S1_0_schema.py), [S2_0_load_data.py](../../S2_0_load_data.py), [S3_3_bioenergy.py](../../S3_3_bioenergy.py), [S4_0_main.py](../../S4_0_main.py), [ST_bioenergy_mvp_test.py](../../ST_bioenergy_mvp_test.py), [config_paths.py](../../config_paths.py), [market_balance_diagnostics.py](../../market_balance_diagnostics.py), [model_run_status.py](../../model_run_status.py).

Imported by: [S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py](../../S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves.py).

### `ST_bioenergy_mvp_test.py`

Minimal bioenergy cases and helper functions still imported by the smoke runner; retained as an active dependency despite its test-like name.

[Source](../../ST_bioenergy_mvp_test.py) · 622 lines.

Entry points / interfaces: [`test_bundle_and_residual_reconciliation`](../../ST_bioenergy_mvp_test.py#L43), [`test_scenario_profile_effect`](../../ST_bioenergy_mvp_test.py#L146), [`test_carrier_feedstock_bridge_fallback`](../../ST_bioenergy_mvp_test.py#L164), [`test_market_summary_separates_food_and_bioenergy`](../../ST_bioenergy_mvp_test.py#L210), [`test_solver_source_contains_explicit_bioenergy_balance`](../../ST_bioenergy_mvp_test.py#L253), [`test_resource_constraints_and_handoffs`](../../ST_bioenergy_mvp_test.py#L261), [`test_solver_micro_bioenergy_balance_solves`](../../ST_bioenergy_mvp_test.py#L413), [`test_enabled_bioenergy_requires_matching_scenario_rows`](../../ST_bioenergy_mvp_test.py#L473). Additional interfaces are listed in the source inventory.

Direct local imports: [S0_51_prepare_bioenergy_feedstock_bridge.py](../../S0_51_prepare_bioenergy_feedstock_bridge.py), [S1_0_schema.py](../../S1_0_schema.py), [S3_0_ds_linear_regional.py](../../S3_0_ds_linear_regional.py), [S3_3_bioenergy.py](../../S3_3_bioenergy.py), [S3_6_scenarios.py](../../S3_6_scenarios.py), [S4_1_results.py](../../S4_1_results.py).

Imported by: [ST_bioenergy_full_run_smoke.py](../../ST_bioenergy_full_run_smoke.py).

## Slurm submission wrappers

### `submit_sbatch_S5_0_mclevels.sh`

Submits S5.0 Slurm batches; current defaults are 20,000 samples and 100 batches. WORKDIR is fixed to the cluster's Code/bin/new path.

[Source](../../submit_sbatch_S5_0_mclevels.sh) · 70 lines.

### `submit_sbatch_S5_1_1_variable_effect.sh`

Submits S5.1 conditional-MC jobs, defaulting to 100 batches with a fixed Code/bin/new WORKDIR.

[Source](../../submit_sbatch_S5_1_1_variable_effect.sh) · 71 lines.

### `submit_sbatch_S5_3_1_panel_yield_ef.sh`

Submits the original S5.3 panel grid, defaulting to 200 batches and a fixed Code/bin/new WORKDIR; includes batch reuse/cleanup settings.

[Source](../../submit_sbatch_S5_3_1_panel_yield_ef.sh) · 186 lines.

### `submit_sbatch_S5_3_1_panel_yield_ef_v2.sh`

Submits v2 panel experiments, defaulting to 200 batches; supports environment overrides, but default WORKDIR remains Code/bin/new.

[Source](../../submit_sbatch_S5_3_1_panel_yield_ef_v2.sh) · 204 lines.

### `submit_sbatch_S5_4_fullmc.sh`

Submits full-variable MC batches, defaulting to 200 batches and a fixed Code/bin/new WORKDIR; passes resume and sample settings.

[Source](../../submit_sbatch_S5_4_fullmc.sh) · 80 lines.

### `submit_sbatch_S5_5_RIP_importance.sh`

Submits structural MC, defaulting to 20,000 samples and 200 batches, with a fixed Code/bin/new WORKDIR.

[Source](../../submit_sbatch_S5_5_RIP_importance.sh) · 82 lines.

### `submit_sbatch_S5_6_max_reduction.sh`

Submits country-endpoint experiments, defaulting to 10 batches; the first batch carries reference/global tasks. WORKDIR remains Code/bin/new.

[Source](../../submit_sbatch_S5_6_max_reduction.sh) · 115 lines.

### `submit_sbatch_S5_7_strategy_endpoint_rerun.sh`

Submits strategy-coalition reruns, defaulting to 32 batches and a fixed Code/bin/new WORKDIR.

[Source](../../submit_sbatch_S5_7_strategy_endpoint_rerun.sh) · 145 lines.

### `submit_sbatch_S5_8_country_strategy_map.sh`

Submits country-by-measure experiments and associated merge/preparation jobs; defaults to 32 batches and Code/bin/new.

[Source](../../submit_sbatch_S5_8_country_strategy_map.sh) · 139 lines.

### `submit_sbatch_S5_9_bioenergy_macc.sh`

Submits quick/exact bioenergy MACC panels and merge jobs; defaults to 32 batches with a fixed Code/bin/new WORKDIR.

[Source](../../submit_sbatch_S5_9_bioenergy_macc.sh) · 193 lines.

## Historical audit and repair sources

These files are retained provenance, probes, test fixtures, and earlier source snapshots. They are outside the active S4/S5 workflow. Repair and deployment scripts describe past changes and must not be treated as an installation sequence. Their old tests may reference standalone test files removed during cleanup.

| Archived source | Role |
|---|---|
| [_codex_historical_reproduction_20260918/compare_history.py](../../_codex_historical_reproduction_20260918/compare_history.py) | Historical audit, reproduction, or repair utility |
| [_codex_historical_reproduction_20260918/diagnose.py](../../_codex_historical_reproduction_20260918/diagnose.py) | Historical audit, reproduction, or repair utility |
| [_codex_historical_reproduction_20260918/prepare_replay.py](../../_codex_historical_reproduction_20260918/prepare_replay.py) | Historical audit, reproduction, or repair utility |
| [_codex_historical_reproduction_20260918/replay_driver.py](../../_codex_historical_reproduction_20260918/replay_driver.py) | Historical audit, reproduction, or repair utility |
| [_codex_reaudit_new_vs_new_TS_20260918/final_checks.py](../../_codex_reaudit_new_vs_new_TS_20260918/final_checks.py) | Historical audit, reproduction, or repair utility |
| [_codex_reaudit_new_vs_new_TS_20260918/inventory.py](../../_codex_reaudit_new_vs_new_TS_20260918/inventory.py) | Historical audit, reproduction, or repair utility |
| [_codex_reaudit_new_vs_new_TS_20260918/mc_cache_probe.py](../../_codex_reaudit_new_vs_new_TS_20260918/mc_cache_probe.py) | Historical audit, reproduction, or repair utility |
| [_codex_reaudit_new_vs_new_TS_20260918/new_TS/pytest_tmp_1789713834187298000/test_manifest_hashes_full_cfg_0/code/model.py](../../_codex_reaudit_new_vs_new_TS_20260918/new_TS/pytest_tmp_1789713834187298000/test_manifest_hashes_full_cfg_0/code/model.py) | Historical audit, reproduction, or repair utility |
| [_codex_reaudit_new_vs_new_TS_20260918/panel_probe.py](../../_codex_reaudit_new_vs_new_TS_20260918/panel_probe.py) | Historical audit, reproduction, or repair utility |
| [_codex_reaudit_new_vs_new_TS_20260918/pipeline_probe.py](../../_codex_reaudit_new_vs_new_TS_20260918/pipeline_probe.py) | Historical audit, reproduction, or repair utility |
| [_codex_reaudit_new_vs_new_TS_20260918/prepare_probes.py](../../_codex_reaudit_new_vs_new_TS_20260918/prepare_probes.py) | Historical audit, reproduction, or repair utility |
| [_codex_reaudit_new_vs_new_TS_20260918/probes.py](../../_codex_reaudit_new_vs_new_TS_20260918/probes.py) | Historical audit, reproduction, or repair utility |
| [_codex_reaudit_new_vs_new_TS_20260918/pwl_probe.py](../../_codex_reaudit_new_vs_new_TS_20260918/pwl_probe.py) | Historical audit, reproduction, or repair utility |
| [_codex_reaudit_new_vs_new_TS_20260918/run_pipeline_probes.py](../../_codex_reaudit_new_vs_new_TS_20260918/run_pipeline_probes.py) | Historical audit, reproduction, or repair utility |
| [_codex_reaudit_new_vs_new_TS_20260918/run_tests.py](../../_codex_reaudit_new_vs_new_TS_20260918/run_tests.py) | Historical audit, reproduction, or repair utility |
| [_codex_reaudit_new_vs_new_TS_20260918/solver_probe.py](../../_codex_reaudit_new_vs_new_TS_20260918/solver_probe.py) | Historical audit, reproduction, or repair utility |
| [_codex_reaudit_new_vs_new_TS_20260918/t_new_TS/test_manifest_hashes_full_cfg_0/code/model.py](../../_codex_reaudit_new_vs_new_TS_20260918/t_new_TS/test_manifest_hashes_full_cfg_0/code/model.py) | Synthetic audit fixture |
| [_codex_reaudit_new_vs_new_TS_20260918/workflow_checks.py](../../_codex_reaudit_new_vs_new_TS_20260918/workflow_checks.py) | Historical audit, reproduction, or repair utility |
