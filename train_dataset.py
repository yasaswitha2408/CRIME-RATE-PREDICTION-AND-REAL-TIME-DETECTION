import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder

# Load dataset
data = pd.read_csv("Dataset/crp.csv")  # Ensure this file exists in the correct path

# Print column names to verify
print(data.columns)

# Handling missing values (Fill with mean for numeric columns only)
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Convert categorical values (if necessary)
if 'City' in data.columns:
    # One-hot encode the 'City' column
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Use drop='first' to avoid multicollinearity
    city_encoded = encoder.fit_transform(data[['City']])
    city_encoded_df = pd.DataFrame(city_encoded, columns=encoder.get_feature_names_out(['City']))
    data = pd.concat([data, city_encoded_df], axis=1)

# Define features and target variable
# Use the encoded city columns and other features
X = data[['Year', 'Population (in Lakhs) (2011)+'] + list(city_encoded_df.columns)]  # Include encoded city columns
y = data['Murder']  # Ensure the target column exists in the dataset

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Save the model
with open("Model/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved successfully.")