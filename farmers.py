
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# def create_sample_data():
#     milk_records = {
#         'milk_record_ID': [1, 2, 3, 4, 5],
#         'farmer_ID': [101, 102, 103, 104, 105],
#         'co_op_ID': [201, 202, 203, 204, 205],
#         'milk_delivery_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),
#         'quantity': [100.50, 20.00, 150.75, 300.00, 250.00],
#         'sale': [200.00, 40.00, 300.00, 600.00, 500.00],
#         'total': [200.00, 40.00, 300.00, 600.00, 500.00]
#     }
#     transactions = {
#         'transaction_ID': [1, 2, 3, 4, 5],
#         'farmer_ID': [101, 102, 103, 104, 105],
#         'co_op_ID': [201, 202, 203, 204, 205],
#         'transaction_type': ['credit', 'debit', 'credit', 'debit', 'credit'],
#         'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),
#         'amount': [150.00, 0.00, 200.00, 100.00, 300.00],
#         'description': ['Milk Sale', 'Milk Purchase', 'Milk Sale', 'Milk Purchase', 'Milk Sale'],
#         'total': [150.00, 0.00, 200.00, 100.00, 300.00]
#     }
#     return pd.DataFrame(milk_records), pd.DataFrame(transactions)
# def calculate_creditworthiness_score(farmer_id, k=3, random_state=42):
#     milk_df, transaction_df = create_sample_data()
#     milk_summary = milk_df.groupby('farmer_ID').agg({
#         'quantity': 'sum',
#         'sale': 'sum',
#         'total': 'sum'
#     }).reset_index()
#     transaction_summary = transaction_df.groupby('farmer_ID').agg({
#         'amount': 'sum'
#     }).reset_index()
#     combined_data = pd.merge(milk_summary, transaction_summary, on='farmer_ID', how='inner')
#     combined_data.rename(columns={'amount': 'transaction_amount'}, inplace=True)
#     combined_data['creditworthiness'] = [1, 0, 1, 0, 1]
#     X = combined_data[['quantity', 'sale', 'total', 'transaction_amount']]
#     y = combined_data['creditworthiness']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train_scaled, y_train)
#     farmer_data = combined_data[combined_data['farmer_ID'] == farmer_id]
#     if farmer_data.empty:
#         return "Farmer ID not found."
#     farmer_data_scaled = scaler.transform(farmer_data[['quantity', 'sale', 'total', 'transaction_amount']])
#     score = knn.predict(farmer_data_scaled)
#     creditworthiness = "High Creditworthiness" if score[0] == 1 else "Low Creditworthiness"
#     return score[0], creditworthiness

# farmer_id = 103
# random_state = 42
# k = 3
# score, creditworthiness = calculate_creditworthiness_score(farmer_id, k, random_state)
# print(f"The creditworthiness score for Farmer ID {farmer_id} is: {score}")
# print(f"Creditworthiness: {creditworthiness}")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def create_sample_data():
    milk_records = {
        'milk_record_ID': [1, 2, 3, 4, 5],
        'farmer_ID': [101, 102, 103, 104, 105],
        'co_op_ID': [201, 202, 203, 204, 205],
        'milk_delivery_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),
        'quantity': [100.50, 20.00, 150.75, 300.00, 250.00],
        'sale': [200.00, 40.00, 300.00, 600.00, 500.00],
        'total': [200.00, 40.00, 300.00, 600.00, 500.00]
    }
    transactions = {
        'transaction_ID': [1, 2, 3, 4, 5],
        'farmer_ID': [101, 102, 103, 104, 105],
        'co_op_ID': [201, 202, 203, 204, 205],
        'transaction_type': ['credit', 'debit', 'credit', 'debit', 'credit'],
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),
        'amount': [150.00, 0.00, 200.00, 100.00, 300.00],
        'description': ['Milk Sale', 'Milk Purchase', 'Milk Sale', 'Milk Purchase', 'Milk Sale'],
        'total': [150.00, 0.00, 200.00, 100.00, 300.00]
    }
    return pd.DataFrame(milk_records), pd.DataFrame(transactions)

def calculate_creditworthiness_score(farmer_id, k=3, random_state=42):
    milk_df, transaction_df = create_sample_data()
    
    
    milk_summary = milk_df.groupby('farmer_ID').agg({
        'quantity': 'sum',
        'sale': 'sum',
        'total': 'sum'
    }).reset_index()
    
    transaction_summary = transaction_df.groupby('farmer_ID').agg({
        'amount': 'sum'
    }).reset_index()
    
    combined_data = pd.merge(milk_summary, transaction_summary, on='farmer_ID', how='inner')
    combined_data.rename(columns={'amount': 'transaction_amount'}, inplace=True)
    
    
    combined_data['creditworthiness'] = [1, 0, 1, 0, 1]
   
    X = combined_data[['quantity', 'sale', 'total', 'transaction_amount']]
    y = combined_data['creditworthiness']
    
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    
    farmer_data = combined_data[combined_data['farmer_ID'] == farmer_id]
    if farmer_data.empty:
        return "Farmer ID not found."
    
    farmer_data_scaled = scaler.transform(farmer_data[['quantity', 'sale', 'total', 'transaction_amount']])
    score = knn.predict(farmer_data_scaled)
    
    
    loan_ranges = {
        1: "5000 - 10000",
        0: "1000 - 5000"
    }
    
    creditworthiness = "High Creditworthiness" if score[0] == 1 else "Low Creditworthiness"
    loan_range = loan_ranges[score[0]]
    
    return score[0], creditworthiness, loan_range

# Example usage
farmer_id = 103
random_state = 42
k = 3
score, creditworthiness, loan_range = calculate_creditworthiness_score(farmer_id, k, random_state)

print(f"The creditworthiness score for Farmer ID {farmer_id} is: {score}")
print(f"Creditworthiness: {creditworthiness}")
print(f"Loan Range: {loan_range}")