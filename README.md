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

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv('clustering_dataset.csv')

# Extract Forest rents column and normalize the data
X = df['Forest rents (% of GDP)'].values.reshape(-1, 1)
X_norm = StandardScaler().fit_transform(X)

# Perform Gaussian Mixture Model clustering with n_clusters=4
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X_norm)
df['Cluster'] = gmm.predict(X_norm)

# Plot the results
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['red', 'green', 'blue', 'orange']  # Add more colors if needed
for i in range(4):
    cluster_data = df[df['Cluster'] == i]
    scatter = ax.scatter(cluster_data.index, cluster_data['Forest rents (% of GDP)'],
                         color=colors[i], label=f'Cluster {i+1}')
plt.xticks(np.arange(0, df.shape[0], 50), np.arange(0, df.shape[0], 50), fontsize=12)
plt.xlabel('Country Index', fontsize=14)
plt.ylabel('Forest rents (% of GDP)', fontsize=14)
plt.title('Gaussian Mixture Model Clustering plot', fontsize=16)
ax.legend(fontsize=12)

# Add annotation for the cluster centers
centers = gmm.means_
for i, center in enumerate(centers):
    ax.annotate(f'Cluster {i+1} center: {center[0]:,.2f}', xy=(i+1, center[0]), xytext=(6, 0),
                textcoords="offset points", ha='left', va='center', fontsize=12, color=colors[i])

plt.show()



# Extract Forest rents column and normalize the data
X = df['Forest rents (% of GDP)'].values.reshape(-1, 1)
X_norm = StandardScaler().fit_transform(X)

# Perform Gaussian Mixture Model clustering with n_clusters=4
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X_norm)
df['Cluster'] = gmm.predict(X_norm)

# Print the cluster members
for cluster in range(4):
    cluster_members = df[df['Cluster'] == cluster]['Country'].values
    print(f'Cluster {cluster+1} members:')
    print(', '.join(cluster_members))
    print()


import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('API_AG.LND.FRST.ZS_DS2_en_csv_v2_5358376.csv', skiprows=4)

# Select only the necessary data for fitting analysis
df = df[['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', *map(str, range(1990, 2021))]]  

# Rename columns to simpler names
df.columns = ['Country', 'Code', 'Indicator', 'IndicatorCode', *range(1990, 2021)] 

# Melt the DataFrame to transform the columns into rows
df_melted = pd.melt(df, id_vars=['Country', 'Code', 'Indicator', 'IndicatorCode'], var_name='Year', value_name='Value') 

# Drop rows with missing values
df_cleaned = df_melted.dropna()  

# Save the cleaned data to a new CSV file
df_cleaned.to_csv('fitting_data.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv('fitting_data.csv')

# Select a sample of four countries
countries = ['Russian Federation', 'Brazil', 'United States', 'Canada']
sample_data = df[df['Country'].isin(countries)]

# Pivot the data to have years as columns
pivot_data = sample_data.pivot(index='Country', columns='Year', values='Value')

# Plot the forest area for the sample countries
plt.figure(figsize=(12, 8))
for country in countries:
    plt.plot(pivot_data.columns, pivot_data.loc[country], label=country)

plt.xlabel('Year', fontsize=14)
plt.ylabel('Forest area (% of land area)', fontsize=14)
plt.title('Forest Area for Sample Countries', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv('fitting_data.csv')

# Filter data for Brazil
brazil_data = df[df['Country'] == 'Brazil']

# Extract the necessary columns
years = brazil_data['Year'].values
values = brazil_data['Value'].values

# Fit a polynomial curve to the data
coeffs = np.polyfit(years, values, deg=2)
poly_func = np.poly1d(coeffs)

# Calculate the residuals
residuals = values - poly_func(years)

# Calculate the standard deviation of the residuals
std_dev = np.std(residuals)

# Generate predictions for future years
future_years = np.arange(years.min(), years.max() + 21)  # Predict for 20 additional years
predicted_values = poly_func(future_years)

# Calculate confidence ranges
cov_matrix = np.cov(years, values, ddof=0)
sigma = np.sqrt(np.diag(cov_matrix))
lower_bound = predicted_values - 1.96 * std_dev
upper_bound = predicted_values + 1.96 * std_dev

# Plot the best fitting function and confidence range
plt.figure(figsize=(12, 8), dpi=80)  # Set fixed dimensions for the figure
plt.plot(years, values, 'ko', label='Actual Data')
plt.plot(future_years, predicted_values, 'r-', label='Best Fitting Function')
plt.fill_between(future_years, lower_bound, upper_bound, color='gray', alpha=0.4, label='Confidence Range')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Forest area (% of land area)', fontsize=14)
plt.title('Polynomial Model Fit for Brazil', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()












