import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import TensorDataset, DataLoader

# --------------------------
# Load your preprocessed test dataset
# --------------------------
df_test = pd.read_csv("data/test_data.csv")  # path to new data
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

X_test = preprocessor.fit_transform(df_test[final_features])
X_tensor = torch.tensor(X_test, dtype=torch.float32)

# --------------------------
# Load MLP model
# --------------------------
class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

input_dim = X_tensor.shape[1]
model = MLP(input_dim)
model.load_state_dict(torch.load("models/mlp_model.pth"))
model.eval()

# --------------------------
# Predict
# --------------------------
with torch.no_grad():
    predictions = model(X_tensor).numpy()

print("Predictions (probabilities of default):")
print(predictions[:10])  # show first 10 predictions
