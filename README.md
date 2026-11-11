
# UK Online Retail Analysis

An end-to-end retail data science project using the UCI Online Retail
transaction dataset. It cleans transaction records, explores revenue and
customer behavior, evaluates a time-ordered weekly sales prediction against a
naive baseline, and writes charts and a findings report.

## Dataset

- **Source:** [UCI Machine Learning Repository - Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Coverage:** UK-based online retailer transactions from 1 December 2010 to
  9 December 2011.
- **Unit of analysis:** individual invoice line.
- **Currency:** prices and revenue are denominated in GBP.
- The analysis excludes cancelled invoices, non-positive quantities or prices,
  and rows without a valid invoice date. Revenue is therefore gross sales from
  retained lines, not net revenue after returns. Customer metrics use records
  with a customer ID.
- The source workbook is downloaded automatically to `data/raw/` the first
  time the analysis runs; it is not included in this project.

## Run it

Use Python 3.10 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python analyze_retail.py
```

The script downloads the source workbook if needed and writes:

- `outputs/analysis_summary.md` — measured findings, model comparison, and
  limitations.
- `outputs/monthly_revenue.png`
- `outputs/top_countries.png`
- `outputs/weekly_sales_prediction.png`
- `outputs/top_customers.png`

The final 20% of weekly observations are held out chronologically. A ridge
regression with a time trend and annual seasonal terms is compared with a
last-week-value forecast. The holdout is small and covers only one year, so
these predictions are an educational baseline, not a production forecast.

## Project structure

```text
.
├── analyze_retail.py
├── data/
│   └── raw/                 # downloaded source workbook (ignored by Git)
├── outputs/                 # charts and generated findings report
├── requirements.txt
└── tests/
    └── test_analysis.py
```
