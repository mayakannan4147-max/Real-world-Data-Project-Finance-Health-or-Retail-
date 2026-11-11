"""Explore UCI's Online Retail transactions and evaluate a simple sales model."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
ARCHIVE_PATH = RAW_DIR / "online-retail.zip"
WORKBOOK_PATH = RAW_DIR / "Online Retail.xlsx"


def ensure_dataset() -> Path:
    """Download and extract the public UCI workbook when it is not present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if WORKBOOK_PATH.exists():
        return WORKBOOK_PATH

    if not ARCHIVE_PATH.exists():
        print(f"Downloading dataset from {DATA_URL}")
        urlretrieve(DATA_URL, ARCHIVE_PATH)

    with ZipFile(ARCHIVE_PATH) as archive:
        workbook_member = next(
            (name for name in archive.namelist() if name.lower().endswith(".xlsx")),
            None,
        )
        if workbook_member is None:
            raise FileNotFoundError("The UCI archive does not contain an Excel workbook.")
        WORKBOOK_PATH.write_bytes(archive.read(workbook_member))

    return WORKBOOK_PATH


def load_transactions(workbook_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the source workbook and retain valid, non-cancelled sales lines."""
    transactions = pd.read_excel(workbook_path, engine="openpyxl")
    required = {
        "InvoiceNo",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    }
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    counts = {"source_rows": len(transactions)}
    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"], errors="coerce"
    )
    transactions["Quantity"] = pd.to_numeric(
        transactions["Quantity"], errors="coerce"
    )
    transactions["UnitPrice"] = pd.to_numeric(
        transactions["UnitPrice"], errors="coerce"
    )
    invoice_numbers = transactions["InvoiceNo"].astype("string")
    valid = (
        transactions["InvoiceDate"].notna()
        & transactions["Quantity"].gt(0)
        & transactions["UnitPrice"].gt(0)
        & ~invoice_numbers.str.startswith("C", na=False)
    )
    transactions = transactions.loc[valid].copy()
    transactions["Revenue"] = transactions["Quantity"] * transactions["UnitPrice"]
    counts["retained_rows"] = len(transactions)
    counts["excluded_rows"] = counts["source_rows"] - counts["retained_rows"]
    return transactions, counts


def weekly_features(
    dates: pd.Series | pd.DatetimeIndex, origin: pd.Timestamp
) -> pd.DataFrame:
    """Build leakage-free calendar features for a weekly time series."""
    dates = pd.Series(pd.to_datetime(dates))
    elapsed_days = (dates - origin).dt.total_seconds() / (24 * 60 * 60)
    annual_phase = 2 * np.pi * dates.dt.day_of_year / 365.25
    return pd.DataFrame(
        {
            "elapsed_years": elapsed_days / 365.25,
            "annual_sin": np.sin(annual_phase),
            "annual_cos": np.cos(annual_phase),
        },
        index=dates.index,
    )


def forecast_weekly_sales(weekly_revenue: pd.Series) -> dict[str, object]:
    """Evaluate a time-ordered ridge forecast and a last-week baseline."""
    weekly_revenue = weekly_revenue.astype(float).sort_index()
    if len(weekly_revenue) < 20:
        raise ValueError("At least 20 weekly observations are needed for evaluation.")

    split = int(len(weekly_revenue) * 0.8)
    split = min(max(split, 1), len(weekly_revenue) - 1)
    train_dates = weekly_revenue.index[:split]
    test_dates = weekly_revenue.index[split:]
    origin = weekly_revenue.index[0]
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(weekly_features(train_dates, origin), weekly_revenue.iloc[:split])

    actual = weekly_revenue.iloc[split:]
    predicted = pd.Series(
        model.predict(weekly_features(test_dates, origin)),
        index=test_dates,
        name="Model",
    )
    baseline = weekly_revenue.shift(1).iloc[split:]
    metrics = {
        "model_mae": mean_absolute_error(actual, predicted),
        "model_rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "model_wape": (
            float(np.abs(actual - predicted).sum() / np.abs(actual).sum())
            if np.abs(actual).sum()
            else float("nan")
        ),
        "baseline_mae": mean_absolute_error(actual, baseline),
        "baseline_rmse": np.sqrt(mean_squared_error(actual, baseline)),
        "baseline_wape": (
            float(np.abs(actual - baseline).sum() / np.abs(actual).sum())
            if np.abs(actual).sum()
            else float("nan")
        ),
        "train_weeks": len(train_dates),
        "test_weeks": len(test_dates),
    }

    final_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    final_model.fit(weekly_features(weekly_revenue.index, origin), weekly_revenue)
    future_dates = pd.date_range(
        weekly_revenue.index[-1] + pd.Timedelta(weeks=1),
        periods=4,
        freq="W-SUN",
    )
    future = pd.Series(
        final_model.predict(weekly_features(future_dates, origin)),
        index=future_dates,
        name="Forecast",
    ).clip(lower=0)

    return {
        "actual": actual,
        "predicted": predicted,
        "baseline": baseline,
        "metrics": metrics,
        "future": future,
    }


def customer_rfm(transactions: pd.DataFrame) -> pd.DataFrame:
    """Calculate recency, frequency, and monetary value for identified customers."""
    identified = transactions.dropna(subset=["CustomerID"]).copy()
    if identified.empty:
        return pd.DataFrame(
            columns=["CustomerID", "RecencyDays", "Frequency", "Monetary"]
        )

    snapshot = identified["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = (
        identified.groupby("CustomerID", as_index=False)
        .agg(
            LastPurchase=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
    )
    rfm["RecencyDays"] = (snapshot - rfm["LastPurchase"].dt.normalize()).dt.days
    return rfm


def save_charts(
    transactions: pd.DataFrame,
    weekly_revenue: pd.Series,
    forecast: dict[str, object],
    rfm: pd.DataFrame,
) -> dict[str, str]:
    """Create and save the project's four explanatory plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"blue": "#2878B5", "orange": "#F28E2B", "green": "#59A14F"}

    monthly = (
        transactions.set_index("InvoiceDate")["Revenue"]
        .resample("MS")
        .sum()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly.plot(kind="bar", ax=ax, color=colors["blue"])
    ax.set(title="Monthly gross sales", xlabel="Invoice month", ylabel="Revenue (GBP)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "monthly_revenue.png", dpi=160)
    plt.close(fig)

    countries = (
        transactions.groupby("Country")["Revenue"]
        .sum()
        .nlargest(10)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    countries.plot(kind="barh", ax=ax, color=colors["green"])
    ax.set(title="Top 10 countries by gross sales", xlabel="Revenue (GBP)", ylabel="")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_countries.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    weekly_revenue.plot(ax=ax, color="#B9C1C9", label="Observed")
    forecast["actual"].plot(ax=ax, color="#222222", label="Holdout actual", linewidth=2)
    forecast["predicted"].plot(
        ax=ax, color=colors["blue"], label="Ridge prediction", linewidth=2
    )
    forecast["baseline"].plot(
        ax=ax, color=colors["orange"], label="Last-week baseline", linestyle="--"
    )
    forecast["future"].plot(
        ax=ax, color=colors["green"], label="Next 4 weeks", marker="o"
    )
    ax.axvline(forecast["actual"].index[0], color="#777777", linestyle=":")
    ax.set(
        title="Weekly gross sales: chronological holdout and short-term forecast",
        xlabel="Week ending",
        ylabel="Revenue (GBP)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "weekly_sales_prediction.png", dpi=160)
    plt.close(fig)

    if not rfm.empty:
        top_customers = (
            rfm.nlargest(10, "Monetary")
            .sort_values("Monetary")
            .copy()
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(
            top_customers["CustomerID"].astype(str),
            top_customers["Monetary"],
            color=colors["orange"],
        )
        ax.set(
            title="Top 10 identified customers by gross sales",
            xlabel="Revenue (GBP)",
        )
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "top_customers.png", dpi=160)
        plt.close(fig)

    return {
        "monthly": "monthly_revenue.png",
        "countries": "top_countries.png",
        "forecast": "weekly_sales_prediction.png",
        "customers": "top_customers.png" if not rfm.empty else "",
    }


def write_report(
    transactions: pd.DataFrame,
    counts: dict[str, int],
    weekly_revenue: pd.Series,
    forecast: dict[str, object],
    rfm: pd.DataFrame,
    charts: dict[str, str],
) -> Path:
    """Write a concise report using results measured from the source data."""
    metrics = forecast["metrics"]
    country_sales = transactions.groupby("Country")["Revenue"].sum()
    top_country = country_sales.idxmax()
    country_revenue = country_sales.max()
    country_share = country_revenue / transactions["Revenue"].sum()
    peak_month = (
        transactions.set_index("InvoiceDate")["Revenue"].resample("MS").sum().idxmax()
    )
    model_wins = metrics["model_mae"] < metrics["baseline_mae"]
    customer_count = transactions["CustomerID"].nunique()
    top_customer_share = (
        rfm["Monetary"].nlargest(10).sum() / rfm["Monetary"].sum()
        if not rfm.empty and rfm["Monetary"].sum()
        else float("nan")
    )

    report_lines = [
        "# Retail transaction analysis",
        "",
        "## Dataset and cleaning",
        "",
        f"- Source rows: {counts['source_rows']:,}; retained sales lines: "
        f"{counts['retained_rows']:,}; excluded rows: {counts['excluded_rows']:,}.",
        f"- Date range: {transactions['InvoiceDate'].min():%d %b %Y} to "
        f"{transactions['InvoiceDate'].max():%d %b %Y}.",
        f"- Retained gross sales: {transactions['Revenue'].sum():,.2f}.",
        "- Prices and revenue are denominated in GBP.",
        "- Cancelled invoices and lines with invalid dates, non-positive "
        "quantities, or non-positive prices are excluded.",
        "",
        "## Findings",
        "",
        f"- {top_country} is the largest market by retained gross sales "
        f"({country_revenue:,.2f} GBP; {country_share:.1%} of retained sales).",
        f"- {peak_month:%B %Y} is the highest-revenue month in the observed data.",
        f"- The dataset contains {customer_count:,} distinct non-missing customer "
        f"IDs; {len(rfm):,} customers are included in RFM calculations.",
        f"- The 10 highest-spending identified customers account for "
        f"{top_customer_share:.1%} of identified-customer gross sales.",
        "- Customer recency is measured in days from the day after the last "
        "recorded transaction; frequency counts distinct invoices.",
        "",
        "## Weekly sales prediction",
        "",
        f"- Evaluation uses the first {metrics['train_weeks']} weeks for training "
        f"and the final {metrics['test_weeks']} weeks for chronological testing.",
        f"- Ridge model holdout MAE: {metrics['model_mae']:,.2f}; RMSE: "
        f"{metrics['model_rmse']:,.2f}; WAPE: {metrics['model_wape']:.1%}.",
        f"- Last-week baseline holdout MAE: {metrics['baseline_mae']:,.2f}; "
        f"RMSE: {metrics['baseline_rmse']:,.2f}; "
        f"WAPE: {metrics['baseline_wape']:.1%}.",
        (
            "- The ridge model has lower holdout MAE than the baseline."
            if model_wins
            else "- The ridge model does not improve on the last-week baseline "
            "by holdout MAE; prefer the baseline until more data and validation "
            "are available."
        ),
        "- The next-four-week values are exploratory point forecasts with no "
        "prediction intervals. The short, single-year history limits reliable "
        "seasonality estimates.",
        "",
        "## Visualizations",
        "",
        f"- [Monthly revenue](./{charts['monthly']})",
        f"- [Top countries](./{charts['countries']})",
        f"- [Weekly prediction](./{charts['forecast']})",
    ]
    if charts["customers"]:
        report_lines.append(f"- [Top customers](./{charts['customers']})")
    report_lines.extend(
        [
            "",
            "## Limitations",
            "",
            "This is observational transaction data from one retailer and one "
            "year. The cleaned revenue measure excludes returns/cancellations "
            "rather than netting them against sales; missing customer IDs "
            "remain in sales analysis but cannot contribute to customer "
            "metrics. Results should not be interpreted as causal or as "
            "representative of other retailers.",
            "",
        ]
    )
    report_path = OUTPUT_DIR / "analysis_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def main() -> None:
    workbook_path = ensure_dataset()
    transactions, counts = load_transactions(workbook_path)
    if transactions.empty:
        raise ValueError("No valid sales lines remain after cleaning.")

    weekly_revenue = (
        transactions.set_index("InvoiceDate")["Revenue"]
        .resample("W-SUN")
        .sum()
    )
    forecast = forecast_weekly_sales(weekly_revenue)
    rfm = customer_rfm(transactions)
    charts = save_charts(transactions, weekly_revenue, forecast, rfm)
    report_path = write_report(
        transactions, counts, weekly_revenue, forecast, rfm, charts
    )
    print(f"Analysis complete. Findings: {report_path}")
    print(
        f"Holdout MAE — ridge: {forecast['metrics']['model_mae']:,.2f}; "
        f"last-week baseline: {forecast['metrics']['baseline_mae']:,.2f}"
    )


if __name__ == "__main__":
    main()
