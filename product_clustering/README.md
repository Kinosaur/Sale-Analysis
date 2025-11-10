# Product Clustering with K-Means

This folder contains a complete product clustering pipeline using K-Means algorithm to group Maven Toys products based on pricing, profitability, and sales performance.

## 📁 Project Structure

```
product_clustering/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── product_clustering_analysis.ipynb  # Interactive Jupyter notebook
├── models/                            # Saved models (generated)
│   ├── kmeans_model.joblib
│   └── scaler.joblib
└── outputs/                           # Results (generated)
    └── products_with_clusters.csv
```

## 🎯 Project Goal

Group **35 Maven Toys products** into meaningful clusters based on:
- **Pricing attributes**: cost, price, margin, markup
- **Sales performance**: total units sold across all stores
- **Product categories**: Toys, Games, Art & Crafts, Electronics, Sports & Outdoors

This helps with pricing strategy, promotional planning, inventory management, and assortment decisions.

## 🚀 Quick Start

### Prerequisites
1. **Open this folder in VS Code**
   - In VS Code, go to `File > Open Folder...`
   - Navigate to and select the `product_clustering` folder
   - This ensures the terminal opens in the correct directory

### Setup Steps

```bash
# 1. Create virtual environment (if not already created)
python -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter notebook
jupyter notebook product_clustering_analysis.ipynb
```

**Then**: Run all cells sequentially to perform the complete clustering analysis with step-by-step visualizations and explanations.

**Note**: Make sure you see `(.venv)` prefix in your terminal prompt, indicating the virtual environment is active.

## 📊 What This Analysis Does

### 1. **Data Loading & Cleaning**
- Loads `products.csv` (35 products) and `sales.csv` (829K+ transactions)
- Cleans price/cost fields (removes `$`, handles whitespace)
- Aggregates sales by product to get total units sold

### 2. **Feature Engineering**
Creates meaningful features for clustering:
- `price`, `cost`: Base pricing information
- `margin = price - cost`: Absolute profitability
- `markup = (price - cost) / cost`: Relative profitability
- `total_units_sold`: Demand signal
- One-hot encoded categories: Product type indicators

**Why these features?**
- Capture both **absolute** (margin) and **relative** (markup) profitability
- Include **demand signal** (units sold) to distinguish popular vs. niche products
- Preserve **category information** to allow similar product types to cluster together

### 3. **Preprocessing**
- **StandardScaler**: Normalizes all features to mean=0, std=1
  - **Critical**: K-Means uses Euclidean distance, so features with larger scales would dominate
  - Without scaling, `total_units_sold` (100s-1000s) would overwhelm `markup` (0-1)
- Saves scaler for production use (to transform new products)

### 4. **Optimal k Selection**
Uses **two complementary methods**:
- **Elbow Method**: Plot inertia vs. k; look for the "elbow" where improvement slows
- **Silhouette Score**: Measures cluster separation (higher = better; range -1 to 1)

**Best practice**: Don't rely on just one method. Automated selection picks k with highest silhouette, but you can override based on business needs.

### 5. **K-Means Clustering**
- Fits K-Means with chosen k
- Uses `random_state=42` for reproducibility
- Uses `n_init=20` for stability (multiple random initializations)
- Assigns cluster labels to each product

### 6. **Evaluation & Validation**
Validates cluster quality with multiple metrics:
- **Silhouette Score**: Overall separation (0.25-0.5 = good, >0.5 = excellent)
- **Davies-Bouldin Index**: Lower is better (cluster compactness vs. separation)
- **Calinski-Harabasz Score**: Higher is better (variance ratio)

### 7. **Visualization**
- **PCA 2D plots**: Reduce high-dimensional features to 2D for visualization
  - Note: PCA is lossy; doesn't show true cluster structure, just a projection
- **Annotated plots**: Top products labeled for business interpretation
- **Correlation heatmap**: Shows feature relationships
- **Outlier detection**: Identifies extreme values using IQR method

### 8. **Business Interpretation**
- Summary statistics per cluster (avg price, margin, units sold)
- Top products per cluster
- Dominant category per cluster
- Actionable insights for business decisions

## 📈 Understanding K-Means

### What is K-Means?
K-Means is an **unsupervised learning** algorithm that groups data points into k clusters by:
1. Randomly initializing k cluster centers (centroids)
2. Assigning each point to the nearest centroid (using Euclidean distance)
3. Recalculating centroids as the mean of assigned points
4. Repeating steps 2-3 until convergence

**Objective**: Minimize within-cluster variance (inertia):
$$\min_{C} \sum_{i=1}^k \sum_{x \in C_i} \|x-\mu_i\|^2$$

### When to Use K-Means
✅ **Good for**:
- Numerical/continuous features
- Spherical/globular clusters
- Similar-sized clusters
- Large datasets (computationally efficient)
- Exploratory analysis

⚠️ **Not ideal for**:
- Non-convex clusters (use DBSCAN instead)
- Different cluster densities (use hierarchical clustering)
- Categorical-only data (use k-modes)
- Very noisy data with many outliers

### Key Assumptions & Limitations
1. **Must specify k in advance** (we use Elbow + Silhouette to find it)
2. **Sensitive to initial centroids** (we use n_init=20 to mitigate)
3. **Sensitive to outliers** (we detect and document them)
4. **Assumes spherical clusters** (acceptable for most business use cases)
5. **Requires scaled features** (we use StandardScaler)

## 🔧 Key Decisions & Best Practices

### ✅ What Makes This Implementation Strong

1. **Feature Scaling**: StandardScaler ensures all features contribute equally
2. **Reproducibility**: Random seed + n_init for consistent results
3. **Multiple Validation Metrics**: Not just one score—comprehensive evaluation
4. **Outlier Detection**: Identifies extreme values before clustering
5. **Correlation Analysis**: Checks for redundant features
6. **Model Persistence**: Saves scaler + model for production use
7. **Documentation**: Clear explanations of every decision

### 📝 Customization Options

**Change number of clusters:**
```python
# In notebook cell 6, modify K_RANGE
K_RANGE = range(2, 15)  # Test up to 14 clusters
```

**Add/remove features:**
```python
# In notebook cell 5, modify FEATURE_NUMERIC
FEATURE_NUMERIC = ['price', 'margin', 'total_units_sold']  # Simplified
```

**Handle outliers differently:**
```python
# After outlier detection, cap extreme values
Q3 = prod['total_units_sold'].quantile(0.75)
prod['total_units_sold'] = prod['total_units_sold'].clip(upper=Q3*1.5)
```

**Try different scaling:**
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()  # Scales to [0,1] instead of StandardScaler
```

## 📚 Files & Outputs

### Input Files (from parent `dataSet/` folder)
- `products.csv`: Product catalog with names, categories, prices, costs
- `sales.csv`: Transaction-level sales data (filtered to Product_ID and Units)

### Generated Files

**Models** (in `models/`):
- `kmeans_model.joblib`: Trained K-Means model
- `scaler.joblib`: Fitted StandardScaler

**Outputs** (in `outputs/`):
- `products_with_clusters.csv`: Products with assigned cluster IDs
- `elbow_inertia.png`: Elbow plot for k selection
- `silhouette_score.png`: Silhouette scores by k
- `clusters_pca.png`: PCA visualization of clusters

### Using Saved Models

```python
import joblib
import pandas as pd

# Load artifacts
km = joblib.load('models/kmeans_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# Prepare new product (must match training features)
new_product = pd.DataFrame({
    'price': [15.99], 'cost': [10.99], 'margin': [5.0],
    'markup': [0.45], 'total_units_sold': [0],
    'cat_Art & Crafts': [0], 'cat_Electronics': [0],
    'cat_Games': [1], 'cat_Sports & Outdoors': [0], 'cat_Toys': [0]
})

# Scale and predict
new_scaled = scaler.transform(new_product)
cluster_id = km.predict(new_scaled)[0]
print(f'Product assigned to cluster: {cluster_id}')
```

## 🎓 Learning Resources

### Understanding the Analysis
- **Elbow Method**: [Scikit-learn Docs](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
- **Silhouette Score**: Measures how similar an object is to its cluster vs. other clusters
- **PCA**: Principal Component Analysis for dimensionality reduction (visualization only)

### Business Applications
- **Pricing Strategy**: Products in same cluster should have similar pricing
- **Promotions**: Target clusters with specific campaigns (e.g., discount high-margin clusters)
- **Inventory**: Optimize stock levels by cluster patterns
- **Assortment**: Ensure balanced representation across clusters per store
- **New Products**: Place new items in appropriate clusters based on attributes

## 🚨 Common Issues & Solutions

**Issue**: Outlier detection references undefined `FEATURE_NUMERIC`
```python
# Solution: Run cell 5 (feature matrix building) before outlier detection
```

**Issue**: High correlation between price and cost
```python
# This is expected! Margin and markup are derived from these.
# Decision: Keep all for comprehensive view, or remove one pair if redundant.
```

**Issue**: Low silhouette score (<0.25)
```python
# Try: Fewer clusters, different features, or different algorithm (DBSCAN)
```

**Issue**: ImportError for sklearn/pandas
```python
# Solution: Activate venv and run: pip install -r requirements.txt
```

## 📊 Expected Results

- **Number of clusters**: Typically 3-5 for this dataset
- **Silhouette score**: 0.25-0.50 (acceptable to good cluster structure)
- **Cluster sizes**: Varies; some clusters may be small (niche products)
- **Interpretation examples**:
  - Cluster 0: Low-price, high-volume items (e.g., PlayDoh Can, Deck of Cards)
  - Cluster 1: Premium toys (e.g., Lego Bricks, high margin)
  - Cluster 2: Electronics with moderate pricing
  - Cluster 3: Sports & Outdoors with specific pricing strategy

## 🎯 Next Steps

After running the analysis:

1. **Label Clusters**: Give business-meaningful names (e.g., "Budget Bestsellers")
2. **Validate**: Share results with domain experts—do clusters make sense?
3. **Test Stability**: Run multiple times with different random_state values
4. **Apply Insights**: Use cluster assignments for pricing/promotion decisions
5. **Extend Analysis**: Add temporal features, store-level patterns, or try different algorithms

## 🤝 Contributing

Improvements welcome! Consider adding:
- Hierarchical clustering comparison
- DBSCAN for density-based clustering
- Time-series features (seasonality, trends)
- Store-level features (which stores sell products best)
- Product co-purchase patterns

## 📄 License

This project is part of the Maven Toys Sales Analysis repository.

---

**Note**: This is educational/analytical code for product segmentation. For production use, add error handling, logging, unit tests, and integration with your data pipeline.