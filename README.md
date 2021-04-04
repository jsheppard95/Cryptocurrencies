# Cryptocurrencies
Unsupervised machine learning model pipeline for the clustering of
cryptocurrencies including scaling, dimensionality reduction with principal
component analysis, grouping using a K-means cluster model, and finally
visualizing with scatter plots.

## Resources
- Data Source: [`crypto_data.csv`](Resources/crypto_data.csv)
- Software:
    - Python 3.7.6
    - pandas 1.0.1
    - hvPlot 0.7.1
    - Plotly 4.14.3
    - scikit-learn 0.22.1

## Pipeline
### Data Cleaning
We first read in the cryptocurrency data set as a pandas dataframe and clean
the raw data to remove any rows containing null values and keep only actively
traded currencies with working algorithms and coins already mined. We then
encode categorical variables using the function `pandas.get_dummies` and
scale numerical data using `sklearn.preprocessing.StandardScaler` resulting in
the dataframe `X_scaled`.

### Dimensionality Reduction: PCA
After cleaning the raw data, we reduce the 98 features of `X_scaled` to three
using principal component analysis with `sklearn.decomposition.PCA`. This
results in the input dataframe (`pcs_df`) for our K-means cluster model.

### K-means Cluster
We next determine the optimal number of clusters, `k`, by generating an
"Elbow Curve", a plot of inertia for a given K-means cluster model with `k`
clusters. We then see a clear change in the slope of this curve at `k = 4`,
indicating that four clusters are optimal. We then instantiate this model
using `sklearn.cluster.KMeans` and categorize our reduced data contained in
`pcs_df`, resulting in a new data frame `clustered_df` containing each
currency, its principal components, and the predicted class.

### Visualization
Finally, we visualize our model's clusters using `plotly.express.scatter_3d`.
Here we see four distinct classes as expected from the Elbow Curve, but
checking counts with `clustered_df["Class"].value_counts()`, we find that
class 2 only has one member. Our 3D scatter plot reveals this outlier is
BitTorrent (BTT) with `PC1 = 34.11`. Inspecting in `clustered_df`, we see that
BTT has `TotalCoinsMined = 9.9e+11`. We can the sort `clustered_df` by this
column using the interactive table generated from `hvplot.pandas.table`. We
thus find that BTT has the largest number of coins mined and is roughly nine
times that of the second largest. It is therefore clear that `clustered_df`
has four distinct groups, but the outlier BTT is less representative of the
data set as a whole. This is further confirmed by creating a 2D scatter plot
of the tradable currencies `TotalCoinSupply` and `TotalCoinsMined` scaled using
`sklearn.preprocessing.MinMaxScaler` and colored by class, where these scaled
features for BTT are both close to one.