

# Mars-Style Category Analytics – Instacart (Python-Only)

**Role this project targets:**  
_Analytics Specialist / Category Analytics Specialist – FMCG (e.g. Mars New Zealand)._

**Tech stack:** `Python`, `Pandas`, `Matplotlib`, `Scikit-learn`, `Jupyter`

---

## 1. Project Overview

This project simulates the work of a **Category Analytics Specialist** for a company like **Mars New Zealand** using the public **Instacart Online Grocery dataset**.

I treat Instacart categories as proxies for three Mars business segments:

- **Snacking** – chocolate, candy, chips, popcorn, etc.  
- **Pet** – pet food and pet supplies.  
- **Food** – core grocery & cooking food categories.  

All analysis is done in **Python only** (no BI tool), to prove I can:

1. Design a **category data model** (fact + dimensions).  
2. Build **segment-level KPIs** (size, penetration, daypart, loyalty).  
3. Train and interpret a **reorder prediction model**.  
4. Segment customers and explore **cross-sell potential**.

---

## 2. Data & Folder Structure

Dataset: **Instacart Online Grocery Shopping 2017** (Kaggle).

Project structure:

```text
mars_category_analytics/
│
├─ data_raw/                # Original Instacart CSVs
├─ data_processed/          # Cleaned & modeled tables (fact + dim)
├─ notebooks/
│   ├─ 00_environment_check.ipynb
│   ├─ 01_data_modeling_and_segmentation.ipynb
│   ├─ 02_ml_reorder_model.ipynb
│   
├─ src/                     # (optional) reusable functions
└─ reports/                 # Exported charts / PDF summaries
````

Core tables created in `data_processed/`:

* **`products_full.csv`** – product + aisle + department (product dimension)
* **`order_lines_full.csv`** – fact table (order × product)
* **`order_lines_full_segmented.csv`** – fact table + segment flag (Snacking/Pet/Food/Other)

---

## 3. Data Modeling

### 3.1 Dimension: Product

Merge 3 raw tables: `products`, `aisles`, `departments` → `products_full`:

* `product_id`, `product_name`
* `aisle_id`, `aisle`
* `department_id`, `department`

This mirrors real FMCG category trees: **Department → Aisle → Product**.

### 3.2 Fact: Order Lines

Merge:

1. `order_products__prior` (order–product lines)
2. `products_full` (product hierarchy)
3. `orders` (customer + time)

→ **`order_lines_full`** with ~32M rows.

Each row =

> “Customer X bought Product Y (department, aisle) in Order Z at day-of-week & hour-of-day.”

Add:

* `line_qty = 1` (each row = one unit)

---

## 4. Segment Mapping (Snacking / Pet / Food)

I classify each line into a **Mars-style business segment** using department/aisle keywords:

* **Pet** if department/aisle contains `"pet"`.
* **Snacking** if keywords like `"snacks"`, `"chips"`, `"candy"`, `"chocolate"`, `"popcorn"`.
* **Food** if general food categories (`"grocery"`, `"dairy"`, `"frozen"`, `"bakery"`, `"meat"`, `"seafood"`, …).
* **Other** as a safe fallback.

Output is **`order_lines_full_segmented.csv`** with a `segment` column.

---

## 5. Key Business KPIs

All KPIs calculated in `01_data_modeling_and_segmentation.ipynb`.

### 5.1 Segment Size (Units Sold)

* Aggregate `line_qty` by `segment`.
* Visualise as a bar chart.
![alt text](image.png)
<img width="590" height="390" alt="image" src="https://github.com/user-attachments/assets/0db61e28-733a-4340-a1a8-6b97c8a6fd6a" />

**Insight (example):**

* Food dominates total units.
* Snacking is meaningful but smaller.
* Pet is small in volume but strategically important.

### 5.2 Basket Penetration

For each `order_id`:

* Flag whether it contains Snacking / Pet / Food.
* **Basket penetration (%)** = % of orders where segment is present.
![alt text](image-3.png)
<img width="590" height="390" alt="image-3" src="https://github.com/user-attachments/assets/cc708050-2180-449e-8965-20fd1803dc65" />


**Example pattern (your numbers may vary slightly):**

* Food: ~90% of baskets
* Snacking: ~40–50%
* Pet: ~5–10%

This mirrors real-world behaviour: food is essential, snacking is high-frequency but not universal, pet is niche but loyal.

### 5.3 Daypart / Hour-of-day Profile

Group by `order_hour_of_day` × `segment`, sum `line_qty` → line chart.
![alt text](image-1.png)
<img width="790" height="490" alt="image-1" src="https://github.com/user-attachments/assets/730d8e15-6e4f-4a86-8169-fe2c64c40ae1" />

**Behavioural pattern:**

* **Food**: strong from morning through early evening.
* **Snacking**: afternoon / early evening peak (2–5pm).
* **Pet**: relatively flat and planned across the day.

This is the kind of daypart view used in activation & media planning.

### 5.4 Reorder & Loyalty

* **Reorder rate per segment**: mean of `reordered` per `segment`.
* **Customer-level loyalty**: for each user × segment, how many times segment was bought; “loyal” if ≥2 purchases.
![alt text](image-2.png)
<img width="590" height="390" alt="image-2" src="https://github.com/user-attachments/assets/aa6fcb2c-63ce-4638-9304-c87c5c01c220" />

Typical result:

* Pet shows the highest reorder and loyalty rates.
* Snacking and Food are more mixed, combining planned and impulse missions.

---

## 6. Reorder Prediction Model (ML)

Implemented in `02_ml_reorder_model.ipynb`.

### 6.1 Problem

> Predict the probability that a given line (customer buying a product) will be **reordered** in the future.

* Target: `reordered` (0/1).
* Unit: order–product line.

### 6.2 Feature Engineering

To keep runtime fast, I train on a random **3% sample** of lines (`SAMPLE_FRAC_MODEL = 0.03`).

Features are grouped into 3 levels and renamed to be business-friendly:

**Product-level**

* `product_orders_count` – number of orders containing the product
* `product_units_sold` – total units
* `product_avg_cart_position` – average position in cart (early = planned, late = impulse)
* `product_reorder_rate` – historical reorder rate of the product

**Customer-level**

* `customer_orders_count` – number of orders placed by customer
* `customer_units_bought` – total units bought
* `customer_avg_basket_size` – average lines per order
* `customer_reorder_rate` – overall reorder rate
* `customer_avg_days_between_orders` – purchase frequency

**Customer–Product interaction**

* `cust_prod_times_bought` – times this customer bought this product
* `cust_prod_order_span` – span between first and last purchase (loyalty over time)

### 6.3 Models

I use a **demo-friendly** setup:

```python
SAMPLE_FRAC_MODEL = 0.03
N_ESTIMATORS_GB = 50
```

**Baseline:** Logistic Regression

* ROC–AUC ≈ 0.89 on the test set.
* Coefficients confirm strong positive effect from `cust_prod_times_bought` and `product_reorder_rate`.

**Improved:** Gradient Boosting Classifier

* `n_estimators=50`, `max_depth=3`, `learning_rate=0.1`
* ROC–AUC typically improves slightly (≈ 0.90+ depending on sample).
* Tree feature importance ranks:

  * `cust_prod_times_bought`
  * `product_reorder_rate`
  * `customer_reorder_rate`
    as top drivers.

### 6.4 Calibration

Using `sklearn.calibration.calibration_curve` to compare predicted probs vs actual reorder rate:
![alt text](image-4.png)
<img width="490" height="490" alt="image-4" src="https://github.com/user-attachments/assets/4b94efaa-a5bc-45c1-ad6e-da79325b801a" />

* Gradient Boosting probabilities are reasonably well calibrated for decision-making in bands (e.g., low / medium / high likelihood).

---

## 7. Segment-specific Models (Snacking vs Pet vs Food)

To understand segment dynamics, I also train **separate Gradient Boosting models** for each segment:

* Filter `order_lines_full_segmented` by `segment`
* Train same feature set on each subset.

Typical pattern:

* **Pet**: highest ROC–AUC (reorder highly predictable; stable routines).
* **Snacking**: slightly lower AUC, reflecting more impulse behaviour.
* **Food**: mid-range.

This supports the narrative that pet buyers are more predictable and loyal, which is important for Mars’s pet portfolio.

---

## 8. Customer Clustering (KMeans)

Implemented in `03_customer_clustering_and_basket.ipynb`:

Using customer-level features:

* `orders_count`
* `total_units`
* `avg_basket_size`
* `reorder_rate`
* `avg_days_between_orders`

I standardise features and run **KMeans (k=4)** on a sampled subset of ~20k customers for speed.

Example cluster archetypes:

1. **Heavy Family Stockers** – many orders, large baskets, high reorder rate.
2. **Snack-Focused Top-Up Shoppers** – smaller baskets, moderate orders, higher snacking share.
3. **Occasional Browsers** – few orders, low reorder rate.
4. **Pet-Driven Loyalists** – strong reorder, frequent pet purchases.

These clusters are useful for tailored promo strategies and personalised offers.

---

## 9. Market Basket / Association Rules (optional)

Using a 1% sample of orders:

* Pivot to `order_id` × `product_id` matrix.
* Run `apriori` to find frequent itemsets.
* Generate rules with `lift > 1.1`.

This identifies product combinations that co-occur more than expected (e.g., certain pet foods frequently bought with specific treats or food items), which can inform **cross-promo bundles**.

---

## 10. How This Maps to the Mars Analytics Specialist Role

This project demonstrates that I can:

* Build a **category-level data mart** from raw transactional data.
* Translate product hierarchy into **business segments** (Snacking / Pet / Food).
* Deliver **practical category insight**:

  * segment size, penetration, daypart, loyalty.
* Build and interpret an **ML model** around repeat purchasing.
* Segment customers and explore **cross-sell** with association rules.
* Work entirely in **Python**, but the same logic can be deployed via Power BI/Tableau or into production pipelines.

---

## 11. How to Run (Demo-friendly)

1. Clone the repo and create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

(minimal: `pandas`, `matplotlib`, `scikit-learn`, `mlxtend`)

3. Download the Instacart dataset and place CSV files in `data_raw/`.
4. Run notebooks in this order:

   1. `01_data_modeling_and_segmentation.ipynb`
   2. `02_ml_reorder_model.ipynb`

Each notebook is written to be **demo-friendly**: samples and number of trees are tuned so they complete within ~30 minutes on a typical laptop.

````





## Executive Summary

Using the Instacart Online Grocery dataset as a proxy for the New Zealand grocery market, I built a Python-only analytics project that mirrors the work of a Category Analytics Specialist at Mars.

I constructed a full category data model (fact + dimensions), mapped products into three Mars-style business segments (Snacking, Pet, Food), engineered behavioural KPIs (penetration, daypart, loyalty) and trained a reorder prediction model with ROC–AUC ≈ 0.89–0.90. I also clustered customers into distinct shopper types and explored cross-sell opportunities via association rules.

The result is a reusable analytics blueprint that could be implemented with Power BI or Tableau in a Mars context to support category reviews, Perfect Store work, and customer activation.

## Key Insights (numbers illustrative)

- **Category size & reach**
  - Food accounts for ~X% of units sold and appears in ~Y% of baskets – the “everyday core”.
  - Snacking contributes ~A% of units and appears in ~B% of baskets, with strong afternoon/evening peaks.
  - Pet represents only ~C% of units but shows the highest reorder and loyalty rates.

- **Shopper behaviour**
  - Food purchases are spread across the day, with clear lunchtime and early-evening peaks.
  - Snacking peaks between around 14:00–17:00, consistent with “afternoon treat / top-up” missions.
  - Pet purchases are more evenly distributed, supporting the idea of planned, routine pet care missions.

- **Reorder dynamics (ML model)**
  - A Gradient Boosting model trained on a 3% sample reaches **ROC–AUC ≈ 0.90**, using only behavioural features.
  - The strongest drivers of reorder probability are:
    - how many times a customer has previously bought the product,
    - the product’s historical reorder rate,
    - the customer’s overall reorder tendencies.
  - Segment-specific models show Pet as the most predictable segment (highest AUC), reinforcing its role as a loyal, routine category.

- **Customer segments**
  - KMeans clustering reveals distinct shopper types such as:
    - heavy family stockers with large, highly repetitive baskets,
    - snack-focused top-up shoppers,
    - occasional browsers with low loyalty,
    - pet-driven loyalists.
  - These segments suggest different activation levers (e.g., pet loyalty rewards vs. snacking impulse promotions).

- **Cross-sell opportunities**
  - Association rules on a 1% order sample highlight product combinations with high lift, pointing to potential cross-category bundles and in-store/online “Buy Together” recommendations.

## Business Value

For a company like Mars New Zealand, this analytics framework can be used to:

- size and prioritise segments (Snacking, Pet, Food) by volume and reach,
- understand when and how shoppers buy each segment,
- identify high-value, loyal customers and products,
- support targeted promotions and cross-sell strategies,
- and provide a foundation for more advanced AI use cases (e.g., personalised offers or dynamic assortment optimisation).


