
# Loan Default Prediction & Offline Reinforcement Learning Agent

This repository contains the implementation of two models to predict and manage loan approval decisions using a dataset of loan applicants:

1. **Predictive Deep Learning Model (MLP)** – A supervised model that predicts the probability of loan default.  
2. **Offline Reinforcement Learning (Fitted Q Iteration)** – An RL agent that learns an optimal loan approval policy based on simulated rewards from historical data.

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Results](#results)
- [Future Steps](#future-steps)

---

## Dataset

The dataset contains loan applications with features such as:  

- Loan amount, interest rate, term, annual income, debt-to-income ratio  
- Employment length, credit grade, home ownership, purpose, verification status  
- Revolving utilization, number of open/total accounts, delinquencies, credit age, etc.  

The target variable is `loan_status`, which is binarized as:  
- `0`: Fully Paid  
- `1`: Defaulted (including "Charged Off")  

---

## Project Structure

├── data/ # Raw and processed datasets
├── models/ # Saved models (MLP and RL agent)
├── notebooks/ # Jupyter notebooks for EDA and model exploration
├── src/ # Python scripts
│ ├── preprocessing.py # Data cleaning & feature preprocessing
│ ├── train_mlp.py # Train MLP model
│ ├── train_rl.py # Train FQI-RL agent
│ └── evaluate.py # Model evaluation
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/yourusername/loan-default-rl.git
cd loan-default-rl
pip install -r requirements.txt
python src/train_mlp.py --data data/loan_data.csv --save_model models/mlp_default_risk.pth
python src/train_rl.py --data data/loan_data.csv --save_model models/fqi_rl_agent.pth

