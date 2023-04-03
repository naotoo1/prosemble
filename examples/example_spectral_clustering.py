"""Spectral Clustering example using a Time Series Data."""

import pandas as pd
import prosemble as ps
from sklearn.preprocessing import StandardScaler

# prepare the data
df = pd.read_csv('data1.csv')
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.iloc[:, 2:])

# Get number of features.
number_of_features = len(
    [column for column in df.columns if column not in df.columns[:2]]
)

# Setup the model
model = ps.models.SpectralClustering(
    data=df_scaled,
    input_dim=number_of_features,
    num_clusters=3,
    adjacency=None,
    method='fcm',
    plot_steps=True,
    ord=None,
    num_iter=1000
)

# Get the clustering results.
print(model.predict())

