# AML Client Risk Rating Model

Team Seeking Alpha work.

The goal is an anti-money laundering (AML) client risk rating model: it uses a customer's profile, products and transaction patterns to classify them as high-risk or non-high-risk.

**Status:** done. Data transformation, feature selection, model and report.

## Repository

```
AML_prediction_model/
├── data/                           Dataset, not in this repo (see Data)
├── data_analysis/
│   ├── EDA_work.ipynb              Exploratory analysis and transformation decisions
│   ├── feature_selection.ipynb     Look at the selected features, saves the EDA plots of the report
│   └── lib.py                      eda() chart
├── report/                         LaTeX report (answers the 7 questions of the assignment)
├── results/
│   ├── eda/                        EDA plots used in the report
│   ├── gbm_feature_selection/      Outputs of gbm_feature_selection.py
│   └── modelling/                  Outputs of modelling.py
├── src/
│   ├── Data_transformation.py      Cleans the data and exports it
│   ├── gbm_feature_selection.py    Picks the features
│   ├── modelling.py                Trains, tests and explains the model
│   └── README_gbm_feature_selection.md   Details on the two scripts above
├── pyproject.toml                  Project dependencies
└── uv.lock                         Exact package versions
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

## Run everything

From the root of the repo, in this order:

```
uv run python src/Data_transformation.py
uv run python src/gbm_feature_selection.py
uv run python src/modelling.py
```

The selection is the slow one (several minutes). Each script prints its results, and the last two save them in `results/`. See [src/README_gbm_feature_selection.md](src/README_gbm_feature_selection.md) for how the selection and the model work.

## The transformation

`Data_transformation.py` creates `data/AML_data_transformed.pkl`, with 588 customers, 145 features and the target `rating` (1 = high risk, 0 = moderate or low risk).

The script:

1. Drops the customer ID, which isn't needed for prediction.
2. Converts the binary indicators back to 0 and 1, as categories (they were standardized in the original file).
3. Drops `tot_acct_num` and `tot_acct_num.1`, which are exact copies of `prod_prim_acct_cnt`.
4. Drops the three `geo_domicile_country_rate_*` columns, which are exact copies of the `geo_mail_addr_country_rate_*` columns in this dataset.
5. Puts the counts (`_cnt`, `_num`) back to whole numbers, since they were standardized too.

## Notebooks

Open them in `data_analysis/` and select the `.venv` environment as the kernel.

- `EDA_work.ipynb` reads the original Excel file and shows why each transformation is needed.
- `feature_selection.ipynb` reads the transformed data and the selected features, so run the scripts first. Its last cell saves the plots in `results/eda/`; plotly needs Chrome for that (if it's missing, run `uv run plotly_get_chrome`).

## Report

`report/AML_Model_Report_Team_Seeking_Alpha.tex` uses the figures in `results/`, so compile it from the `report` folder:

```
cd report
latexmk -pdf AML_Model_Report_Team_Seeking_Alpha.tex
```
