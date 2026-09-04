from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import fit_clustered_fixed_effects, load_functional_panel, result_row


STATES = ["L", "LM", "UM", "H"]
STATE_RANK = {state: index for index, state in enumerate(STATES)}
INTERVALS = [(2014, 2016), (2016, 2019), (2019, 2021), (2021, 2024)]


def transition_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint_years = sorted({year for interval in INTERVALS for year in interval})
    endpoints = panel[panel["year"].isin(endpoint_years)].copy()
    q25, q50, q75 = endpoints["residual"].quantile([0.25, 0.50, 0.75])
    endpoints["state"] = pd.cut(
        endpoints["residual"],
        [-np.inf, q25, q50, q75, np.inf],
        labels=STATES,
        include_lowest=True,
    )
    cells = []
    summaries = []
    for start, end in INTERVALS:
        wide = endpoints[endpoints["year"].isin([start, end])].pivot(index="city_id", columns="year", values="state").dropna()
        counts = pd.crosstab(wide[start], wide[end]).reindex(index=STATES, columns=STATES, fill_value=0)
        upward = persistence = downward = 0
        for origin in STATES:
            for destination in STATES:
                count = int(counts.loc[origin, destination])
                difference = STATE_RANK[destination] - STATE_RANK[origin]
                upward += count if difference > 0 else 0
                downward += count if difference < 0 else 0
                persistence += count if difference == 0 else 0
                cells.append({
                    "transition": f"{start}-{end}",
                    "origin_state": origin,
                    "destination_state": destination,
                    "count": count,
                    "row_share": count / counts.loc[origin].sum(),
                })
        total = int(counts.to_numpy().sum())
        summaries.append({
            "transition": f"{start}-{end}",
            "cities": total,
            "upward": upward / total,
            "persistence": persistence / total,
            "downward": downward / total,
        })
    return pd.DataFrame(cells), pd.DataFrame(summaries)


def directional_adjustment(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.sort_values(["city_id", "year"]).copy()
    data["residual_lag"] = data.groupby("city_id")["residual"].shift(1)
    data["delta_residual"] = data["residual"] - data["residual_lag"]
    data["loo_province_median"] = np.nan
    for _, indices in data.groupby(["province_name", "year"]).groups.items():
        indices = list(indices)
        values = data.loc[indices, "residual_lag"].to_numpy(float)
        medians = []
        for position in range(len(values)):
            peers = np.delete(values, position)
            peers = peers[np.isfinite(peers)]
            medians.append(float(np.median(peers)) if len(peers) else np.nan)
        data.loc[indices, "loo_province_median"] = medians
    gap = data["residual_lag"] - data["loo_province_median"]
    data["below_reference"] = (-gap).where(gap < 0, 0)
    data["above_reference"] = gap.where(gap > 0, 0)
    model = fit_clustered_fixed_effects(
        data,
        "delta_residual",
        ["below_reference", "above_reference"],
        ["city_id", "year"],
        "city_id",
    )
    rows = [
        result_row(model, "below_reference", "upward_catch_up"),
        result_row(model, "above_reference", "downward_correction"),
    ]
    estimate, standard_error, p_value = model.contrast({"below_reference": 1, "above_reference": 1})
    rows.append({
        "model": "directional_symmetry_test",
        "term": "below_reference + above_reference",
        "coefficient": estimate,
        "standard_error": standard_error,
        "ci_low": estimate - 1.96 * standard_error,
        "ci_high": estimate + 1.96 * standard_error,
        "p_value": p_value,
        "observations": model.n,
        "cities": model.clusters,
    })
    result = pd.DataFrame(rows)
    result["relative_downward_strength"] = abs(result.loc[1, "coefficient"]) / result.loc[0, "coefficient"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalculate the main Result 2 transition and asymmetry statistics.")
    parser.add_argument("--functional-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/result2"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_functional_panel(args.functional_panel)
    cells, directions = transition_tables(panel)
    asymmetry = directional_adjustment(panel)
    cells.to_csv(args.output_dir / "transition_cells.csv", index=False)
    directions.to_csv(args.output_dir / "transition_directions.csv", index=False)
    asymmetry.to_csv(args.output_dir / "directional_adjustment.csv", index=False)
    print(
        f"Result 2: upward={asymmetry.loc[0, 'coefficient']:.3f}, "
        f"downward={asymmetry.loc[1, 'coefficient']:.3f}, "
        f"relative strength={asymmetry.loc[0, 'relative_downward_strength']:.1%}."
    )


if __name__ == "__main__":
    main()
