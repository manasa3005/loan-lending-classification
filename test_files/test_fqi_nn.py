import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# --------------------------
# Load your preprocessed test dataset
# --------------------------
df_test = pd.read_csv("test_data.csv")
final_features = [
    'loan_amnt','int_rate','term_months','annual_inc','dti',
    'emp_length_yrs','grade','home_ownership','purpose',
    'verification_status','application_type','revol_util',
    'open_acc','total_acc','delinq_2yrs','inq_last_6mths',
    'pub_rec','credit_age_years'
]

categorical_cols = df_test.select_dtypes(include=['object']).columns.tolist()
numerical_cols = [c for c in final_features if c not in categorical_cols]

num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])

states = preprocessor.fit_transform(df_test[final_features]).astype(np.float32)
loan_amnt = df_test['loan_amnt'].values.astype(np.float32)
int_rate = df_test['int_rate'].astype(str).str.replace("%","").astype(float).astype(np.float32)/100.0

# --------------------------
# Load FQI-NN model
# --------------------------
with open("models/fqi_nn_model.pkl", "rb") as f:
    fqi_nn = pickle.load(f)

# --------------------------
# Evaluate policy on new data
# --------------------------
policy_value = fqi_nn.evaluate_policy(states, np.zeros(len(states)), loan_amnt, int_rate)
print("Estimated Policy Value (FQI-NN):", policy_value)

actions = fqi_nn.policy(states)
print("Predicted actions (0=deny,1=approve):")
print(actions[:20])  # show first 20 actions
