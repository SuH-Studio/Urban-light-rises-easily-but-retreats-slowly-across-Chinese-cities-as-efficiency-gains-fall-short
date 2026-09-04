from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import PUBLIC_DATA, normalize_code


BASELINE_YEAR = 2016
ANALYSIS_YEARS = list(range(2016, 2025))
RATE_GRID = np.arange(0.01, 0.0801, 0.005)


def quality_pass(frame: pd.DataFrame, threshold: float = 0.80) -> pd.Series:
    return (
        frame["radiance_stock"].gt(0)
        & frame["valid_area_km2"].gt(0)
        & frame["valid_pixel_fraction"].ge(threshold)
        & frame["valid_months"].ge(6)
    )


def balanced_sample(stock: pd.DataFrame) -> pd.DataFrame:
    main = stock[
        stock["method_id"].eq("main_median")
        & stock["year"].isin(ANALYSIS_YEARS)
        & stock["city_name_result1"].notna()
    ].copy()
    main["quality_pass"] = quality_pass(main)
    summary = main.groupby("city_code").agg(years=("year", "nunique"), passing=("quality_pass", "sum"))
    codes = summary.index[(summary["years"] == 9) & (summary["passing"] == 9)]
    return main[main["city_code"].isin(codes)].copy()


def accounting_rows(sample: pd.DataFrame, rate: float) -> pd.DataFrame:
    baseline = sample[sample["year"].eq(BASELINE_YEAR)][["city_code", "radiance_stock"]].rename(
        columns={"radiance_stock": "baseline_stock"}
    )
    rows = sample.merge(baseline, on="city_code", how="inner", validate="many_to_one")
    multiplier = (1 - rate) ** (rows["year"] - BASELINE_YEAR)
    rows["efficiency_rate"] = rate
    rows["frozen_demand_path"] = rows["baseline_stock"] * multiplier
    rows["observed_radiance_demand_path"] = rows["radiance_stock"] * multiplier
    rows["potential_dividend"] = rows["baseline_stock"] - rows["frozen_demand_path"]
    rows["signed_expansion_offset"] = rows["observed_radiance_demand_path"] - rows["frozen_demand_path"]
    return rows


def summarize(sample: pd.DataFrame) -> pd.DataFrame:
    records = []
    for rate in RATE_GRID:
        rows = accounting_rows(sample, float(rate))
        window = rows[rows["year"].between(2017, 2024)]
        potential = window["potential_dividend"].sum()
        offset = window["signed_expansion_offset"].sum()
        records.append({
            "efficiency_rate": rate,
            "cities": window["city_code"].nunique(),
            "city_years": len(window),
            "potential_dividend_stock_units": potential,
            "signed_expansion_offset_stock_units": offset,
            "signed_offset_ratio": offset / potential,
        })
    return pd.DataFrame(records)


def break_even_rate(summary: pd.DataFrame) -> float:
    rates = summary["efficiency_rate"].to_numpy(float)
    ratios = summary["signed_offset_ratio"].to_numpy(float)
    for index in range(len(rates) - 1):
        if (ratios[index] - 1) * (ratios[index + 1] - 1) <= 0:
            return float(
                rates[index]
                + (1 - ratios[index]) * (rates[index + 1] - rates[index]) / (ratios[index + 1] - ratios[index])
            )
    raise ValueError("No break-even point within the efficiency-rate grid")


def classify_2024(sample: pd.DataFrame) -> pd.DataFrame:
    rows = accounting_rows(sample, 0.04)
    target = rows[rows["year"].eq(2024)].copy()
    target["classification"] = np.select(
        [
            target["radiance_stock"].lt(target["baseline_stock"]),
            target["signed_expansion_offset"].le(target["potential_dividend"]),
        ],
        [
            "radiance demand contracted",
            "modeled efficiency dominates non-contracting expansion",
        ],
        default="demand expansion outweighs modeled efficiency",
    )
    return target[["city_code", "city_name_result1", "province_name", "classification"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalculate the VNP46A2-based Result 3 accounting.")
    parser.add_argument("--radiance-stock", type=Path, default=PUBLIC_DATA / "vnp46a2_city_year_2014_2024.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/result3"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stock = pd.read_csv(args.radiance_stock)
    stock["city_code"] = stock["city_code"].map(normalize_code)
    sample = balanced_sample(stock)
    summary = summarize(sample)
    classification = classify_2024(sample)
    summary.to_csv(args.output_dir / "efficiency_rate_sensitivity.csv", index=False)
    classification.to_csv(args.output_dir / "classification_2024.csv", index=False)
    rate_4 = summary.loc[np.isclose(summary["efficiency_rate"], 0.04)].iloc[0]
    crossing = break_even_rate(summary)
    print(
        f"Result 3: {sample.city_code.nunique()} cities; 4% ratio={rate_4.signed_offset_ratio:.3f}; "
        f"break-even={100 * crossing:.2f}%."
    )


if __name__ == "__main__":
    main()
