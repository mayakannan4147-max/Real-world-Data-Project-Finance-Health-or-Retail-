# Retail transaction analysis

## Dataset and cleaning

- Source rows: 541,909; retained sales lines: 530,104; excluded rows: 11,805.
- Date range: 01 Dec 2010 to 09 Dec 2011.
- Retained gross sales: 10,666,684.54.
- Prices and revenue are denominated in GBP.
- Cancelled invoices and lines with invalid dates, non-positive quantities, or non-positive prices are excluded.

## Findings

- United Kingdom is the largest market by retained gross sales (9,025,222.08 GBP; 84.6% of retained sales).
- November 2011 is the highest-revenue month in the observed data.
- The dataset contains 4,338 distinct non-missing customer IDs; 4,338 customers are included in RFM calculations.
- The 10 highest-spending identified customers account for 17.3% of identified-customer gross sales.
- Customer recency is measured in days from the day after the last recorded transaction; frequency counts distinct invoices.

## Weekly sales prediction

- Evaluation uses the first 43 weeks for training and the final 11 weeks for chronological testing.
- Ridge model holdout MAE: 115,595.88; RMSE: 140,771.72; WAPE: 36.3%.
- Last-week baseline holdout MAE: 70,775.08; RMSE: 86,891.55; WAPE: 22.2%.
- The ridge model does not improve on the last-week baseline by holdout MAE; prefer the baseline until more data and validation are available.
- The next-four-week values are exploratory point forecasts with no prediction intervals. The short, single-year history limits reliable seasonality estimates.

## Visualizations

- [Monthly revenue](./monthly_revenue.png)
- [Top countries](./top_countries.png)
- [Weekly prediction](./weekly_sales_prediction.png)
- [Top customers](./top_customers.png)

## Limitations

This is observational transaction data from one retailer and one year. The cleaned revenue measure excludes returns/cancellations rather than netting them against sales; missing customer IDs remain in sales analysis but cannot contribute to customer metrics. Results should not be interpreted as causal or as representative of other retailers.
