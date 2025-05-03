import unittest
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
class TestKMeansApplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_path = r"C:\Users\User\Downloads\01-01-2022.csv"
        if not os.path.exists(cls.file_path):
            raise FileNotFoundError(f"Test data file not found at: {cls.file_path}")
        cls.df = pd.read_csv(cls.file_path)[['Deaths', 'Confirmed']].dropna()

    def test_data_loaded(self):
        """Test if the data is loaded and not empty."""
        self.assertFalse(self.df.empty, "DataFrame is empty after loading.")

    def test_kmeans_clustering(self):
        """Test if KMeans clustering creates expected output."""
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df)

        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(scaled_data)

        self.df['Cluster'] = kmeans.labels_
        self.assertIn('Cluster', self.df.columns, "Cluster column not found after KMeans.")

    def test_no_missing_values(self):
        """Ensure no NaN values remain in the dataset before clustering."""
        self.assertFalse(self.df.isnull().values.any(), "There are missing values in the data.")

    def test_invalid_file_handling(self):
        """Check if FileNotFoundError is raised for a wrong file path."""
        with self.assertRaises(FileNotFoundError):
            pd.read_csv("non_existent_file.csv")


if __name__ == '__main__':
    unittest.main()
