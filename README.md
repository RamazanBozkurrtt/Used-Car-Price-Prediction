# Used Car Price Prediction & Market Segmentation

### Project Overview
This project aims to predict the fair market price of used vehicles in the UK market based on technical specifications such as mileage, year, engine size, and transmission type.

The study adopts an **iterative data mining approach**, comparing two base models (**KNN Regressor** and **Decision Tree Regressor**) across multiple preprocessing scenarios. A key innovation of this project is the use of **Unsupervised Learning (Agglomerative Clustering)** to generate new features for Supervised Learning models.

### Key Techniques & Methodology

* **Data Cleaning:** * Handled currency symbols (`£`) and formatting issues using **Regex**.
    * Standardized column names and removed noise.
* **Advanced Imputation:** * Used **Decision Tree Classifier** to impute missing categorical data based on other patterns, rather than simple mode imputation.
* **Feature Engineering:** * **Clustering as a Feature:** Applied **Agglomerative Hierarchical Clustering** to discover hidden market segments (Luxury, Economy, etc.) and used these cluster labels as a new input feature for regression models.
    * Derived `Car Age` from production year.
* **A/B Testing (Preprocessing):** * Systematically compared model performance on **Raw Data** vs. **Scaled Data** vs. **Feature Selected Data**.

### 📊 Model Performance Results

We compared models based on **R2 Score** and **MAE (Mean Absolute Error)**.

| Model | Scenario | Scaling | Clustering Feature | R2 Score |
|-------|----------|---------|--------------------|----------|
| **KNN** | Best Perf. | ✅ Standard Scaler | ❌ No | **0.937** |
| **KNN** | Baseline | ❌ None | ❌ No | 0.368 |
| **Decision Tree** | Best Perf. | ❌ None | ❌ No | **0.889** |
| **Decision Tree** | Clustering | ❌ None | ✅ Yes | 0.887 |

**Key Insights:**
* **KNN** is highly sensitive to scaling; standardization improved accuracy from 0.36 to 0.93.
* **Decision Trees** are scale-invariant and performed robustly (~0.89) even on raw data.
* **Agglomerative Clustering** provided interpretability but did not significantly boost the R2 score, suggesting that the existing features (Year, Engine, Mileage) already capture the variance well (Redundancy).

### 📂 File Structure

* `veri_on_isleme.py`: Main data cleaning, regex fixing, and imputation pipeline.
* `clustering_feature.py`: Unsupervised learning module (Agglomerative Clustering) & Dendrogram visualization.
* `KNN_method.py`: KNN model training with Scale/No-Scale/Feature Selection scenarios.
* `DecisionTreeRegressor.py`: Decision Tree training with Pre-Pruning (max_depth=10) parameters.
* `data.csv`: Raw dataset.
* `cleaned_data.csv`: Processed dataset ready for modeling.

### 🚀 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

2.  **Run Preprocessing:**
    ```bash
    python veri_on_isleme.py
    ```

3.  **Generate Clusters (Optional Feature Engineering):**
    ```bash
    python clustering_feature.py
    ```

4.  **Run Models:**
    ```bash
    python KNN_method.py
    python DecisionTreeRegressor.py
    ```

### 👤 Author
**Ramazan Bozkurt**
* *Data Mining Project - Fall 2025*
