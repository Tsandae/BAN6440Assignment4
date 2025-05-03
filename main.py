import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
file_path = r"C:\Users\User\Downloads\01-01-2022.csv"
df= pd.read_csv(file_path)
df = df[['Deaths', 'Confirmed']]
df.dropna(inplace=True)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_data)
df['Cluster'] = kmeans.labels_
print("Clustered Data Sample:")
print(df.head())

plt.figure(figsize=(8, 6))
plt.scatter(df['Confirmed'], df['Deaths'], c=df['Cluster'], cmap='viridis', s=50)
plt.xlabel("Confirmed Cases")
plt.ylabel("Deaths")
plt.title("K-Means Clustering on COVID-19 Data")
plt.grid(True)
plt.show()