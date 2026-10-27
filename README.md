# Car Price Regression: What Drives the Price of a Car?

Predicting the price of 205 cars (26 columns: engine, size, weight, fuel economy, brand and more) with a regression model, and checking how far the result can be trusted on such a small dataset. This started as a class assignment in my Certified Data Scientist training (IABAC / AFRICDSA); I rebuilt it with proper validation. The write-up and code were produced with AI assistance and I reviewed the results.

## Results (5-fold CV repeated 5 times, so 25 test scores per model)

| Model | R² | MAE (price units) |
|---|---|---|
| Baseline: predict the average price | -0.04 | 5,932 |
| Linear regression | 0.86 | 1,837 |
| Ridge (on log price) | 0.89 | 1,563 |
| Lasso (on log price) | 0.80 | 1,959 |
| **Random forest** | **0.92** | **1,469** |

A random forest is wrong by about 1,500 on a typical car, against about 5,900 for guessing the average, so the features carry real signal.

## What I found
- **Engine size and weight dominate.** Engine size alone accounts for about 65% of the random forest's importance; curb weight adds about 21%. Their correlations with price are 0.87 and 0.84.
- **Brand adds little once the physical specs are known.** Dropping the brand column barely changes the random forest (R² 0.920 to 0.919) but hurts ridge (0.89 to 0.85).
- **Unseen brands are the honest test.** When whole brands are held out (so the model has never seen the make), R² falls to 0.56 to 0.74 depending on the model; the random forest is again best (0.74). That is the realistic expectation for a car brand outside the data, and it is much lower than the headline 0.92.
- **Data cleaning mattered.** The brand is hidden in the `CarName` text and has typos (`maxda`, `porcshce`, `toyouta`, `vokswagen`, `vw`). I merged them into 22 clean brands.

## Limitations
- Only 205 cars and 22 brands, so scores move with the fold split (I report the average of 25 splits).
- Prices are in the dataset's currency units (undocumented); the column set resembles the 1985 UCI Automobile dataset (I did not verify the provenance), so results say nothing about today's market.
- Lasso does worse than ridge here; I did not tune it further.

## Run it
```
pip install pandas numpy scikit-learn
python analysis.py              # reads data/CarPrice.csv
```
Results are written to `results.json`. The CSV is the Car Price Assignment dataset (205 rows) as given in my course materials, included in `data/`; I have not verified its original licence.
