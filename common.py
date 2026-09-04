from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA = REPOSITORY_ROOT / "data" / "public"


def normalize_code(value: object) -> str:
    return str(value).replace(".0", "").zfill(6)


def normalize_city(value: object) -> str:
    text = str(value).strip()
    for suffix in ("特别行政区", "自治州", "地区", "盟", "市"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def read_table(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    return pd.read_csv(path)


def load_functional_panel(path: Path) -> pd.DataFrame:
    frame = read_table(path)
    frame = frame[frame["main_sample"].astype(str).str.lower().eq("true")].copy()
    crossfit = pd.to_numeric(frame["residual_crossfit"], errors="coerce")
    in_sample = pd.to_numeric(frame["residual_in_sample"], errors="coerce")
    frame["residual"] = crossfit.combine_first(in_sample)
    frame["city_code"] = frame["city_code"].map(normalize_code)
    frame["city_id"] = frame["province_name"].astype(str) + "|" + frame["city_name"].map(normalize_city)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    return frame.dropna(subset=["residual", "year"]).copy()


@dataclass
class RegressionResult:
    names: list[str]
    coefficients: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    p_values: np.ndarray
    n: int
    clusters: int

    def index(self, term: str) -> int:
        return self.names.index(term)

    def coefficient(self, term: str) -> float:
        return float(self.coefficients[self.index(term)])

    def standard_error(self, term: str) -> float:
        return float(self.standard_errors[self.index(term)])

    def p_value(self, term: str) -> float:
        return float(self.p_values[self.index(term)])

    def contrast(self, weights: dict[str, float]) -> tuple[float, float, float]:
        vector = np.zeros(len(self.names))
        for term, weight in weights.items():
            vector[self.index(term)] = weight
        estimate = float(vector @ self.coefficients)
        standard_error = float(np.sqrt(vector @ self.covariance @ vector))
        p_value = math.erfc(abs(estimate / standard_error) / math.sqrt(2))
        return estimate, standard_error, p_value


def fit_clustered_fixed_effects(
    data: pd.DataFrame,
    outcome: str,
    terms: list[str],
    fixed_effects: list[str],
    cluster: str,
) -> RegressionResult:
    columns = list(dict.fromkeys([outcome, *terms, *fixed_effects, cluster]))
    model_data = data[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    design_parts = [pd.Series(1.0, index=model_data.index, name="Intercept")]
    names = ["Intercept"]
    for term in terms:
        design_parts.append(model_data[term].astype(float).rename(term))
        names.append(term)
    for fixed_effect in fixed_effects:
        dummies = pd.get_dummies(
            model_data[fixed_effect].astype(str),
            prefix=fixed_effect,
            drop_first=True,
            dtype=float,
        )
        design_parts.append(dummies)
        names.extend(dummies.columns.tolist())

    design = pd.concat(design_parts, axis=1).to_numpy(float)
    response = model_data[outcome].to_numpy(float)
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ coefficients
    inverse_cross_product = np.linalg.pinv(design.T @ design)
    groups = model_data[cluster].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    meat = np.zeros((design.shape[1], design.shape[1]))
    for group in unique_groups:
        rows = groups == group
        score = design[rows].T @ residuals[rows]
        meat += np.outer(score, score)
    n = design.shape[0]
    rank = np.linalg.matrix_rank(design)
    correction = (len(unique_groups) / (len(unique_groups) - 1)) * ((n - 1) / (n - rank))
    covariance = correction * inverse_cross_product @ meat @ inverse_cross_product
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    statistics = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(coefficients, np.nan),
        where=standard_errors > 0,
    )
    p_values = np.asarray([math.erfc(abs(value) / math.sqrt(2)) for value in statistics])
    return RegressionResult(names, coefficients, covariance, standard_errors, p_values, n, len(unique_groups))


def result_row(result: RegressionResult, term: str, label: str) -> dict[str, object]:
    estimate = result.coefficient(term)
    standard_error = result.standard_error(term)
    return {
        "model": label,
        "term": term,
        "coefficient": estimate,
        "standard_error": standard_error,
        "ci_low": estimate - 1.96 * standard_error,
        "ci_high": estimate + 1.96 * standard_error,
        "p_value": result.p_value(term),
        "observations": result.n,
        "cities": result.clusters,
    }
