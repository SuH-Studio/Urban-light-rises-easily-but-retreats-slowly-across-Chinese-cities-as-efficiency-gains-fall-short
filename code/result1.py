from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import PUBLIC_DATA, fit_clustered_fixed_effects, load_functional_panel, normalize_code, result_row


ANALYSIS_YEARS = range(2014, 2020)
RANDOM_SEED = 20260726


def nearest_neighbours(metadata: pd.DataFrame, k: int = 5) -> dict[str, list[str]]:
    coordinates = metadata[["city_code", "lon", "lat"]].dropna().drop_duplicates("city_code")
    codes = coordinates["city_code"].tolist()
    longitude = np.deg2rad(coordinates["lon"].to_numpy(float))
    latitude = np.deg2rad(coordinates["lat"].to_numpy(float))
    longitude_difference = longitude[None, :] - longitude[:, None]
    latitude_difference = latitude[None, :] - latitude[:, None]
    haversine = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(latitude[:, None]) * np.cos(latitude[None, :]) * np.sin(longitude_difference / 2) ** 2
    )
    distances = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    np.fill_diagonal(distances, np.inf)
    return {code: [codes[index] for index in np.argsort(distances[row])[:k]] for row, code in enumerate(codes)}


def peer_values(peer_map: dict[str, list[str]], lookup: dict[tuple[str, int], float], city: str, year: int) -> list[float]:
    return [float(lookup[(peer, year)]) for peer in peer_map.get(city, []) if pd.notna(lookup.get((peer, year)))]


def prepare_panel(functional_panel: Path, city_metadata: Path, yicai_file: Path) -> pd.DataFrame:
    panel = load_functional_panel(functional_panel)
    panel = panel[panel["year"].isin(ANALYSIS_YEARS)].copy()
    metadata = pd.read_csv(city_metadata)
    metadata["city_code"] = metadata["city_key"].map(normalize_code)
    yicai = pd.read_csv(yicai_file)
    metadata = metadata.drop(columns=["yicai_tier", "yicai_overall_order"], errors="ignore")
    metadata = metadata.merge(
        yicai[["city_name_yicai", "yicai_tier", "yicai_overall_order"]],
        on="city_name_yicai",
        how="left",
    )
    panel = panel.merge(
        metadata[["city_code", "lon", "lat", "resource_excluded_flag", "yicai_tier", "yicai_overall_order"]],
        on="city_code",
        how="left",
        validate="many_to_one",
    ).sort_values(["city_code", "year"])
    panel["own_residual_lag"] = panel.groupby("city_code")["residual"].shift(1)
    panel["delta_residual"] = panel["residual"] - panel["own_residual_lag"]

    cities = panel.drop_duplicates("city_code")
    province_peers_all = {
        row.city_code: cities.loc[
            cities["province_name"].eq(row.province_name) & ~cities["city_code"].eq(row.city_code), "city_code"
        ].tolist()
        for row in cities.itertuples()
    }
    analysis_codes = {city for city, peers in province_peers_all.items() if peers}
    analysis_cities = cities[cities["city_code"].isin(analysis_codes)].copy()
    geographic_peers = nearest_neighbours(analysis_cities)
    province_peers = {
        city: [peer for peer in province_peers_all[city] if peer in analysis_codes]
        for city in cities["city_code"]
    }
    tier_peers = {
        row.city_code: analysis_cities.loc[
            analysis_cities["yicai_tier"].eq(row.yicai_tier)
            & ~analysis_cities["city_code"].eq(row.city_code),
            "city_code",
        ].tolist()
        for row in analysis_cities.dropna(subset=["yicai_tier"]).itertuples()
    }
    lookup = panel.set_index(["city_code", "year"])["residual"].to_dict()

    def mean_lag(peer_map: dict[str, list[str]], city: str, year: int) -> float:
        values = peer_values(peer_map, lookup, city, year - 1)
        return float(np.mean(values)) if values else np.nan

    panel["geo_lag"] = [mean_lag(geographic_peers, city, int(year)) for city, year in zip(panel.city_code, panel.year)]
    panel["province_lag"] = [mean_lag(province_peers, city, int(year)) for city, year in zip(panel.city_code, panel.year)]
    panel["same_tier_mean_lag"] = [mean_lag(tier_peers, city, int(year)) for city, year in zip(panel.city_code, panel.year)]
    def frontier_lag(city: str, year: int) -> float:
        values = sorted(peer_values(tier_peers, lookup, city, year - 1), reverse=True)
        count = max(1, int(np.ceil(len(values) * 0.25)))
        return float(np.mean(values[:count])) if values else np.nan

    panel["same_tier_frontier_lag"] = [
        frontier_lag(city, int(year)) for city, year in zip(panel.city_code, panel.year)
    ]
    return panel


def estimate_models(panel: pd.DataFrame) -> pd.DataFrame:
    controls = ["own_residual_lag", "geo_lag", "province_lag"]
    matched = panel.dropna(subset=["yicai_tier"]).copy()
    samples = {
        "full": matched,
        "excluding_resource_energy": matched[~matched["resource_excluded_flag"].astype(bool)],
    }
    rows = []
    for sample_name, sample in samples.items():
        for exposure in ("same_tier_mean_lag", "same_tier_frontier_lag"):
            model = fit_clustered_fixed_effects(
                sample, "delta_residual", [exposure, *controls], ["province_name", "year"], "city_code"
            )
            rows.append(result_row(model, exposure, sample_name))
    return pd.DataFrame(rows)


def red_queen_summary(panel: pd.DataFrame) -> pd.DataFrame:
    ranked = panel.dropna(subset=["yicai_tier"]).copy()
    ranked["tier_rank_percentile"] = ranked.groupby(["year", "yicai_tier"])["residual"].rank(pct=True)
    ranked = ranked.sort_values(["city_code", "year"])
    ranked["rank_change"] = ranked.groupby("city_code")["tier_rank_percentile"].diff()
    valid = ranked.dropna(subset=["delta_residual", "rank_change"])
    rows = []
    for threshold in (0.00, 0.01, 0.02, 0.05, 0.10):
        positive = valid[valid["delta_residual"] > 0]
        flag = (valid["delta_residual"] > 0) & (valid["rank_change"] <= threshold)
        rows.append({
            "threshold": threshold,
            "city_years": len(valid),
            "count": int(flag.sum()),
            "share_all": float(flag.mean()),
            "share_positive_growth": float((positive["rank_change"] <= threshold).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalculate the main Result 1 statistics.")
    parser.add_argument("--functional-panel", type=Path, required=True)
    parser.add_argument("--city-metadata", type=Path, required=True)
    parser.add_argument("--yicai", type=Path, default=PUBLIC_DATA / "yicai_2017_city_tiers.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/result1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel(args.functional_panel, args.city_metadata, args.yicai)
    models = estimate_models(panel)
    red_queen = red_queen_summary(panel)
    models.to_csv(args.output_dir / "coefficients.csv", index=False)
    red_queen.to_csv(args.output_dir / "red_queen_summary.csv", index=False)
    headline = models[(models.model == "full") & (models.term == "same_tier_mean_lag")].iloc[0]
    print(f"Result 1: beta={headline.coefficient:.3f}, P={headline.p_value:.3f}, N={int(headline.observations)}")


if __name__ == "__main__":
    main()
