# House Price Prediction: Ames, Iowa

Predicting house sale prices from 79 features (size, quality, location, age) using regularized linear regression on a log-transformed target.

**Start with [`notebooks/01_analysis.ipynb`](notebooks/01_analysis.ipynb)** for the full story.

## Result

| Model | CV RMSE (log price) |
|---|---|
| Baseline (always predict the mean) | 0.3971 |
| Linear Regression | 0.1293 |
| Ridge (alpha=30) | 0.1134 |
| **Lasso (alpha=0.0005)** | **0.1134** |
| XGBoost (tuned: depth 2, lr 0.05) | 0.1184 |

**Final model: Lasso.** On a held-out test set (292 houses, used once):
- RMSE 0.116 on log price (close to CV, so it generalizes)
- Average miss about **$13.8k per house**, roughly 8-9% of the median price
- R² 0.92 on log price

## Approach

1. **Log target:** sale prices are right-skewed (skew 1.88). Log price is close to a bell curve (skew 0.12) and matches the competition metric.
2. **Cleaning:** removed 2 very large houses that sold far below the trend. Filled "NA" as "None" where it means the house lacks the feature (no pool, no garage). Treated MSSubClass as a category, not a number.
3. **No leakage:** split the data first; median and most-frequent filling happen inside a scikit-learn Pipeline, learned from training folds only.
4. **Model choice:** Ridge and Lasso tied on 5-fold CV. I chose Lasso because it keeps only 115 of the 311 model inputs (after one-hot encoding), so it's simpler to explain. The choice was made before touching the test set.

## Key findings

- **Living area** adds the most: about +500 sq ft means about +13% price.
- **Commercial zoning** lowers price the most (about -18%), though it's based on only a handful of houses.
- **Location matters:** the median price in the most expensive neighbourhood is more than 3x the cheapest.
- **Where it fails:** the model overprices the cheapest, lowest-quality houses (few examples, and regularization pulls extreme predictions toward the average).

## What I tried that didn't help

I set a rule in advance: keep a change only if it lowers CV RMSE by at least 0.002.
- **XGBoost:** 0.1184 even after tuning. Its best setting used the shallowest trees, which suggests price depends on features in mostly simple, additive ways.
- **Summed features** (total square footage, house age, total baths): 0.1135, no gain. A linear model already builds sums of columns by itself.
- **Logging skewed inputs:** 0.1124, even with alpha re-tuned. Right direction, but below the threshold.

## Next steps

- Fill LotFrontage with the neighbourhood median instead of the overall median.
- Extend the XGBoost search (depth 1 not tried); summed features may help trees.
- Flag very low-quality houses and abnormal sales for human review.

## How to run

1. Download `train.csv` from the [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and put it in a `data/` folder (not included here, per Kaggle's rules).
2. Install the libraries: `pip install -r requirements.txt`
3. Open `notebooks/01_analysis.ipynb` and run all cells.

## Project structure

```
house-price-regression/
├── data/                     # train.csv goes here (not in the repo)
├── notebooks/
│   ├── 00_exploration.ipynb  # working notes and experiments
│   └── 01_analysis.ipynb     # the clean story
├── requirements.txt
└── README.md
```

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib
