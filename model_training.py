import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
import pickle
from car_data_prep import prepare_data

# Load and prepare the data
df = pd.read_csv('Car.csv')
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
df_train = prepare_data(df_train)

X_train = df_train.drop(columns=['Price'])
y_train = df_train['Price']

# Train the model
model = ElasticNetCV(cv=5, random_state=42, max_iter=10000, tol=1e-4, 
                     alphas=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0], 
                     l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9])
model.fit(X_train, y_train)

# Save the trained model
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)