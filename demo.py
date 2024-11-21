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
#     # Summarize milk and transaction data
#     milk_summary = milk_df.groupby('farmer_ID').agg({
#         'quantity': 'sum',
#         'sale': 'sum',
#         'total': 'sum'
#     }).reset_index()
#     transaction_summary = transaction_df.groupby('farmer_ID').agg({
#         'amount': 'sum'
#     }).reset_index()
#     # Combine the data
#     combined_data = pd.merge(milk_summary, transaction_summary, on='farmer_ID', how='inner')
#     combined_data.rename(columns={'amount': 'transaction_amount'}, inplace=True)
#     # Assign creditworthiness based on some criteria
#     combined_data['creditworthiness'] = np.where(combined_data['quantity'] > 150, 1, 0)
#     # Prepare features and labels
#     X = combined_data[['quantity', 'sale', 'total', 'transaction_amount']]
#     y = combined_data['creditworthiness']
#     # Split the dataset
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
#     # Scale the features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     # Train the KNN classifier
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train_scaled, y_train)
#     # Get farmer data
#     farmer_data = combined_data[combined_data['farmer_ID'] == farmer_id]
#     if farmer_data.empty:
#         return "Farmer ID not found."
#     # Scale farmer data
#     farmer_data_scaled = scaler.transform(farmer_data[['quantity', 'sale', 'total', 'transaction_amount']])
#     # Predict creditworthiness
#     score = knn.predict(farmer_data_scaled)
#     creditworthiness = "High Creditworthiness" if score[0] == 1 else "Low Creditworthiness"
#     # Determine loan range based on creditworthiness
#     loan_range = (0, 0)  # Default loan range
#     if score[0] == 1:  # High creditworthiness
#         loan_range = (farmer_data['total'].values[0] * 1.5, farmer_data['total'].values[0] * 3)  # 1.5x to 3x of total sales
#     else:  # Low creditworthiness
#         loan_range = (0, farmer_data['total'].values[0] * 1.5)  # Up to 1.5x of total sales
#     return score[0], creditworthiness, loan_range
# # Example usage
# farmer_id = 103
# random_state = 42
# k = 3
# score, creditworthiness, loan_range = calculate_creditworthiness_score(farmer_id, k, random_state)
# print(f"The creditworthiness score for Farmer ID {farmer_id} is: {score}")
# print(f"Creditworthiness: {creditworthiness}")
# print(f"Loan Range: {loan_range[0]} to {loan_range[1]}")

class CreditScorer:
    def __init__(self):
        # Define scoring factors and their weights
        self.factors = {
            'monthly_income': {
                'weight': 0.4,
                'ranges': [
                    (0, 25000, 0),
                    (25001, 50000, 0.5),
                    (50001, 100000, 0.8),
                    (100001, float('inf'), 1.0)
                ]
            },
            'milk_quantity': {
                'weight': 0.3,
                'ranges': [
                    (0, 100, 0),
                    (101, 300, 0.5),
                    (301, 500, 0.8),
                    (501, float('inf'), 1.0)
                ]
            },
            'relationship_length': {
                'weight': 0.3,
                'ranges': [
                    (0, 6, 0),
                    (7, 12, 0.5),
                    (13, 24, 0.8),
                    (25, float('inf'), 1.0)
                ]
            }
        }
        
        # Define loan ranges
        self.loan_ranges = [
            (0, 30, "Not eligible for a loan at this time"),
            (31, 50, "Low: 10,000 - 50,000"),
            (51, 70, "Medium: 50,000 - 100,000"),
            (71, 100, "High: 100,000 - 500,000")
        ]

    def calculate_factor_score(self, factor, value):
        for low, high, score in self.factors[factor]['ranges']:
            if low <= value <= high:
                return score
        return 0  # Default score if no range matches

    def calculate_credit_score(self, farmer_data):
        total_score = 0
        for factor, config in self.factors.items():
            factor_score = self.calculate_factor_score(factor, farmer_data[factor])
            total_score += factor_score * config['weight']
        return total_score * 100  # Convert to a 0-100 scale

    def determine_loan_range(self, credit_score):
        for low, high, loan_range in self.loan_ranges:
            if low <= credit_score <= high:
                return loan_range
        return "Score out of expected range"

# Example usage
scorer = CreditScorer()

farmer_data = {
    'monthly_income': 10000,
    'milk_quantity': 300,
    'relationship_length': 12
}

credit_score = scorer.calculate_credit_score(farmer_data)
loan_range = scorer.determine_loan_range(credit_score)

print(f"Credit Score: {credit_score:.2f}")
print(f"Loan Range: {loan_range}")