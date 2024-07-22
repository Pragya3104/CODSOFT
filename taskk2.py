import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = r'C:\Users\HP\OneDrive\Desktop\codsoft\datasets/IMDb Movies India.csv'
movies_df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Data Cleaning
# Remove rows where 'Rating' is missing
movies_df = movies_df.dropna(subset=['Rating'])

# Convert 'Year' to integer
movies_df['Year'] = movies_df['Year'].str.extract('(\d{4})').astype(float).astype('Int64')

# Convert 'Votes' to integer
movies_df['Votes'] = movies_df['Votes'].str.replace(',', '').astype(float).astype('Int64')

# Extract numeric part of 'Duration' and convert to integer
movies_df['Duration'] = movies_df['Duration'].str.extract('(\d+)').astype(float).astype('Int64')

# Fill missing values in 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3' with 'Unknown'
movies_df['Genre'].fillna('Unknown', inplace=True)
movies_df['Director'].fillna('Unknown', inplace=True)
movies_df['Actor 1'].fillna('Unknown', inplace=True)
movies_df['Actor 2'].fillna('Unknown', inplace=True)
movies_df['Actor 3'].fillna('Unknown', inplace=True)

# Feature Engineering
# Define the features and target variable
X = movies_df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Year', 'Duration', 'Votes']]
y = movies_df['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']),
        ('num', 'passthrough', ['Year', 'Duration', 'Votes'])
    ]
)

# Create the pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f'Root Mean Squared Error: {rmse}')

# Save the model if needed
import joblib
joblib.dump(model_pipeline, 'movie_rating_predictor.pkl')
