# AML Client Risk Rating Model

Team Seeking Alpha work.

The goal is an anti-money laundering (AML) client risk rating model: it uses a customer's profile, products and transaction patterns to classify them as high-risk or non-high-risk.

**Status:** data exploration and transformation are done. The model is still to come.

## Repository

```
AML_prediction_model/
├── data/                       Dataset, not in this repo (see Data)
├── data_analysis/
│   ├── EDA_work.ipynb          Exploratory analysis and transformation decisions
│   └── lib.py                  eda() chart
├── report/                     LaTeX report
├── src/
│   └── Data_transformation.py  Cleans the data and exports it
├── pyproject.toml              Project dependencies
└── uv.lock                     Exact package versions
```

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs the right Python version (3.12) and all packages for you.

```
git clone https://github.com/MatMor1010/AML_prediction_model.git
cd AML_prediction_model
uv sync
```

## Data

The dataset comes from the course and isn't included in this repo. Create a `data` folder at the root of the repo and put the Excel file in it:

```
mkdir data
```

The file must be at `data/AML Assignment data_std.xlsx`.

## Run the transformation

Run this from the root of the repo:

```
uv run python src/Data_transformation.py
```

This creates `data/AML_data_transformed.pkl`, with 588 customers, 145 features and the target `rating` (1 = high risk, 0 = moderate or low risk).

The script:

1. Drops the customer ID, which isn't needed for prediction.
2. Converts the binary indicators back to 0 and 1, as categories (they were standardized in the original file).
3. Drops `tot_acct_num` and `tot_acct_num.1`, which are exact copies of `prod_prim_acct_cnt`.
4. Drops the three `geo_domicile_country_rate_*` columns, which are exact copies of the `geo_mail_addr_country_rate_*` columns in this dataset.


## Explore the data

Open `data_analysis/EDA_work.ipynb` and select the `.venv` environment as the kernel. The notebook reads the original Excel file, so it needs the `data` folder too.
