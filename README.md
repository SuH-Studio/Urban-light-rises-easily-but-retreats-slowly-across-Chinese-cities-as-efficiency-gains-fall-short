# Urban lighting demand in Chinese cities

This repository contains the public data and Python code used for the three main results. Figure files and panel-specific source-data tables are not included.

## Contents

- `data/public/vnp46a2_city_year_2014_2024.csv`: annual city-level radiance data derived from NASA VIIRS Black Marble VNP46A2 Collection 2.
- `data/public/yicai_2017_city_tiers.csv`: publicly available 2017 Yicai city-tier classification used to define peer groups.
- `code/result1.py`: peer-group adjustment and Red Queen calculations.
- `code/result2.py`: transition and directional-adjustment calculations.
- `code/result3.py`: lighting-expansion and efficiency-dividend accounting.
- `code/validate_public_data.py`: basic checks for the two included datasets.

The VNP46A2 product is described in the [Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP46A2) and archived under [DOI: 10.5067/VIIRS/VNP46A2.002](https://doi.org/10.5067/VIIRS/VNP46A2.002). The city-tier table is based on the publicly accessible [2017 China city ranking](https://fgw.wuhan.gov.cn/xwzx/mtgz/202001/t20200115_862179.html).

## Running the code

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python code/validate_public_data.py
python code/result3.py
```

Results 1 and 2 use the functional-lighting panel estimated with licensed socioeconomic inputs. After placing that panel outside the repository, run:

```bash
python code/result1.py --functional-panel /path/to/city_year_panel.csv \
  --city-metadata /path/to/city_metadata.csv
python code/result2.py --functional-panel /path/to/city_year_panel.csv
```

Generated tables are written to `outputs/`, which is not tracked by Git.

GDP, population, road infrastructure and other socioeconomic variables obtained from CEI Data are not redistributed. Purchased procurement records and restricted electricity data are also excluded. These exclusions follow the data providers' access conditions.
