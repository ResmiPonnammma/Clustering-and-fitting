# Clustering-and-fitting
import pandas as pd

# load the dataset for Forest area (% of land area)
df_area = pd.read_csv('API_AG.LND.FRST.ZS_DS2_en_csv_v2_5358376.csv', skiprows=4)

df_area.head() 


df_area.tail()

import pandas as pd

# load the dataset for Forest rents (% of GDP)
df_rents = pd.read_csv('API_NY.GDP.FRST.RT.ZS_DS2_en_csv_v2_5363484.csv', skiprows=4)



df_rents.tail()

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("API_NY.GDP.FRST.RT.ZS_DS2_en_csv_v2_5363484.csv", skiprows=4)

# Select relevant columns and drop missing values
df_cleaned = df[["Country Name", "2020"]].dropna()

# Rename columns
df_cleaned.columns = ["Country", "Forest rents (% of GDP)"]

# Set index to country name
df_cleaned.set_index("Country", inplace=True)

# Save the cleaned dataset to a new file
df_cleaned.to_csv("clustering_dataset.csv")


import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df_cleaned = pd.read_csv("clustering_dataset.csv")

# Filter the data for Russia, Brazil, USA, and Canada
countries = ['Russian Federation', 'Brazil', 'United States', 'Canada']
filtered_data = df_cleaned[df_cleaned['Country'].isin(countries)]

# Create a bar graph
plt.figure(figsize=(10, 6))
plt.bar(filtered_data['Country'], filtered_data['Forest rents (% of GDP)'])
plt.xlabel('Country')
plt.ylabel('Forest rents (% of GDP)')
plt.title('Forest Rents as a Percentage of GDP')
plt.xticks(rotation=45)
plt.tight_layout()

# Display the bar graph
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the cleaned dataset
df = pd.read_csv("clustering_dataset.csv")

# Extract the Forest rents column and normalize the data
X = df['Forest rents (% of GDP)'].values.reshape(-1, 1)
X_norm = (X - X.mean()) / X.std()

# Define the range of number of clusters to try
n_clusters_range = range(2, 11)

# Initialize an empty list to store the within-cluster sum of squares
wcss = []

# Iterate over the number of clusters and compute the within-cluster sum of squares
for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_norm)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(n_clusters_range, wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.title("Elbow Curve: Optimal Number of Clusters")
plt.show()









