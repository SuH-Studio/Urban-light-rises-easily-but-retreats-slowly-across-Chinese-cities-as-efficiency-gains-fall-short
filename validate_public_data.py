from __future__ import annotations

import pandas as pd

from common import PUBLIC_DATA


def main() -> None:
    radiance = pd.read_csv(PUBLIC_DATA / "vnp46a2_city_year_2014_2024.csv")
    tiers = pd.read_csv(PUBLIC_DATA / "yicai_2017_city_tiers.csv")
    assert set(radiance["collection_id"].dropna()) == {"NASA/VIIRS/002/VNP46A2"}
    assert radiance["year"].between(2014, 2024).all()
    assert tiers["city_name_yicai"].nunique() == 338
    restricted_markers = {"gdp", "population", "procurement", "electricity", "tourism", "leader"}
    public_columns = {column.lower() for column in radiance.columns} | {column.lower() for column in tiers.columns}
    assert not any(any(marker in column for marker in restricted_markers) for column in public_columns)
    print(f"VNP46A2 records: {len(radiance):,}; city-tier records: {len(tiers):,}.")


if __name__ == "__main__":
    main()
