# Price Disparity Analysis between Branded and Generic Drugs
### A log-linear regression approach to quantify the effect of "brand premium" 

---

## Overview

According to the Global Burden of Disease (2023), the global prevalence rate for 
all health conditions is nearly 96.9%, meaning that most of the world's population 
lives with at least one physical or mental condition. This translates into a massive 
reliance on healthcare resources, expressed through mechanisms like the WHO's annual 
List of Essential Medicines, which highlights the critical need for drug availability 
globally.


In Venezuela, NGOs like Convite contribute to this goal by monitoring drug availability 
for the six leading causes of morbidity in the country. Combining this framework with 
the WHO list and other research in the field, a target list of 15 essential drugs was
established based on their usage frequency and disease prevalence.


To analyze the price disparity between branded and generic drugs, a custom WebScraper was 
built to extract pricing data for 179 medicines from the three largest pharmaceutical 
retailers in the country (Farmatodo, Locatel and Farmacia SAAS) . Finally, this dataset 
was processed using an OLS log-linear regression to quantify how much the "brand premium" 
influences the final consumer price, yielding a highly robust model with an R² of 0.77

---

## Business Context

Understanding how drug prices are structured in the Venezuelan market 
is a sensitive issue for pharmacies. When evaluating whether to stock 
a new product, retailers need to assess if its price point aligns with 
what the local market can absorb — otherwise they risk holding inventory 
that becomes economically unviable before it sells.

---

## Hedonic Pricing Framework

This regression follows a hedonic pricing framework, decomposing drug price into five 
dimensions: physical attributes (dose, units per box), brand identity (Es_Marca), 
therapeutic category, fabrication origin segment, and a baseline constant. Each 
dimension isolates a distinct component of the consumer's perceived value

---

## Dataset
- **Source:** The websites of Farmatodo, Locatel and Farmacia SAAS.
- **Raw observations:** 182
- **Final sample:** 179 (after removing 3 misidentified drugs)
- **Target:** `Precio_USD` (market price in USD)

---

## Methodology

### Feature engineering
| Feature | Type | Rationale |
|---------|------|-----------|
| Dosis_Mg | Raw | Concentration of the active ingredient |
| Unidades_por_caja | Raw | Amount of pills in the box |
| Es_Marca | Engineered | Captures if a drug is branded or not |
| Categoria_antibiotico | Engineered | Identification of antibiotics |
| Categoria_antidepresivo | Engineered | Identification of antidepressants |
| Categoria_antifúngico | Engineered | Identification of antifungal drugs |
| Categoria_cardiovascular | Engineered | Identification of cardiovascular drugs |
| Categoria_gastrointestinal | Engineered | Identification of gastrointestinal drugs |
| Categoria_neurológico | Engineered | Identification of neurological drugs |
| Segmento_Alto | Engineered | Country of fabrication origin has an average price higher 
than 10 USD  |
| Segmento_Bajo | Engineered | Country of fabrication origin has an average price below 5 USD |






### Econometric decisions
- **Log transformation:** `log1p(Price_USD)` applied — compresses the right tail of the price 
distribution and stabilizes variance across drug categories with very different price levels.
- **Base categories:** The analgesics category was designated as the base category
due to its elevated usage frequency (more than 50% of people take it), and
the "Segmento_Medio" (Mid segment) also constituted it due to the amount of observances
that belonged to it (120).
- **Category removal:** The category of antihistamines was removed as a variable because it did 
not have enough statistic significance to differ from the base category, joining analgesics as 
a part of it.

---

## Results

| Metric | Value |
|--------|-------|
| R² | 0.77 |
| MAPE (test set) | 43.62% |

1) All eleven predictors are significant at the 5% level. 
2) Neurological category emerges as the most expensive one out of the 6 groups. 
3) A fabrication origin that belongs to the low segment is associated with lower prices
4)The "brand premium" effect increases the price of a drug by approximately 67.48% (calculated as (e^0.5157 − 1) × 100)

---

## Diagnostic tests

| Test | Statistic | p-value | Conclusion |
|------|-----------|---------|------------|
| Breusch-Pagan | — | 0.1192 | Homoscedasticity present |
| Breusch-Godfrey | 2.9358 | 0.086 | No autocorrelation |
| Durbin-Watson | 2.219 | — | No serial correlation |
| VIF (all features) | < 10 | — | No multicollinearity |
| Shapiro-Wilk | 0.9856 | 0.0625 | Normality present |


---

## Limitations
- The list of 15 drugs only enables the comprehension of the overall trends
of the market, but does not act as a highly accurate predictor of their price
by not including a broader amount of active ingredients and more variables such as 
"Brand perception".
- Although Venezuela's results manifest the confirmation of the findings made by
 Signaling theory (Kirimani and Rao, 2000), not retrieving information from other countries
of the region restricts the applicability of its results internationally.


---

## Tech stack
- **Language:** Python 3
- **Libraries:** pandas · numpy · statsmodels · scikit-learn · scipy · matplotlib
- **Output:** CSV with predicted values
---

## Files
| File | Description |
|------|-------------|
| `ModelodeRegresión.py` | Full pipeline: cleaning, modeling, diagnostics, export |
| `datos_farmacias_con_estimaciones.csv` | Output with predicted values |