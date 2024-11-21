# import sqlite3
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# from sqlalchemy import create_engine

# # Step 1: Create a sample SQLite database and populate it with dummy data
# def create_sample_database():
#     conn = sqlite3.connect('farmers.db')
#     cursor = conn.cursor()

#     # Create a table
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS farmers (
#             id INTEGER PRIMARY KEY,
#             fico_score INTEGER,
#             avg_milk_delivery INTEGER,
#             total_feed_spent INTEGER,
#             previous_loans INTEGER,
#             loan_eligible INTEGER
#         )
#     ''')

#     # Insert sample data
#     sample_data = [
#         (700, 150, 300, 1, 1),
#         (650, 80, 150, 0, 0),
#         (720, 200, 400, 2, 1),
#         (580, 50, 100, 0, 0),
#         (600, 60, 120, 1, 0),
#         (740, 220, 450, 3, 1)
#     ]

#     cursor.executemany('INSERT INTO farmers (fico_score, avg_milk_delivery, total_feed_spent, previous_loans, loan_eligible) VALUES (?, ?, ?, ?, ?)', sample_data)
#     conn.commit()
#     conn.close()

# # Step 2: Connect to the database and retrieve data
# def retrieve_data():
#     engine = create_engine('sqlite:///farmers.db')
#     query = "SELECT fico_score, avg_milk_delivery, total_feed_spent, previous_loans, loan_eligible FROM farmers"
#     df = pd.read_sql(query, engine)
#     return df

# # Step 3: Train the KNN model
# def train_knn_model(df):
#     X = df[['fico_score', 'avg_milk_delivery', 'total_feed_spent', 'previous_loans']]
#     y = df['loan_eligible']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     knn = KNeighborsClassifier(n_neighbors=3)
#     knn.fit(X_train, y_train)

#     y_pred = knn.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy of KNN model: {accuracy * 100:.2f}%")

#     return knn

# # Step 4: Predict loan eligibility for a new farmer
# def predict_loan_eligibility(knn, fico_score, avg_milk_delivery, total_feed_spent, previous_loans):
#     new_data = np.array([[fico_score, avg_milk_delivery, total_feed_spent, previous_loans]])
#     prediction = knn.predict(new_data)
#     return "Eligible" if prediction[0] == 1 else "Not Eligible"

# # Main function to run the demo
# def main():
#     # Create sample database and populate it
#     create_sample_database()

#     # Retrieve data from the database
#     df = retrieve_data()

#     # Train the KNN model
#     knn_model = train_knn_model(df)

#     # Example prediction for a new farmer
#     new_farmer_data = {
#         'fico_score': 675,
#         'avg_milk_delivery': 130,
#         'total_feed_spent': 200,
#         'previous_loans': 1
#     }

#     eligibility = predict_loan_eligibility(knn_model, **new_farmer_data)
#     print(f"Loan Eligibility for the new farmer: {eligibility}")

# # Run the demo
# if __name__ == "__main__":
#     main()


import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine

# Sample Data Preparation
def create_sample_database():
    conn = sqlite3.connect('farmers.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS farmers (
            id INTEGER PRIMARY KEY,
            fico_score INTEGER,
            avg_milk_delivery INTEGER,
            total_feed_spent INTEGER,
            previous_loans INTEGER,
            loan_eligible INTEGER
        )
    ''')

    sample_data = [
        (700, 150, 300, 1, 1),
        (650, 80, 150, 0, 0),
        (720, 200, 400, 2, 1),
        (580, 50, 100, 0, 0),
        (600, 60, 120, 1, 0),
        (740, 220, 450, 3, 1)
    ]

    cursor.executemany('INSERT INTO farmers (fico_score, avg_milk_delivery, total_feed_spent, previous_loans, loan_eligible) VALUES (?, ?, ?, ?, ?)', sample_data)
    conn.commit()
    conn.close()

# Retrieve data from the database
def retrieve_data():
    engine = create_engine('sqlite:///farmers.db')
    query = "SELECT fico_score, avg_milk_delivery, total_feed_spent, previous_loans, loan_eligible FROM farmers"
    df = pd.read_sql(query, engine)
    return df

# Calculate credit score for a given farmer
def calculate_credit_score(fico_score, avg_milk_delivery, total_feed_spent, previous_loans):
    # Example weights for simplicity
    weights = {
        'fico_score': 0.4,
        'avg_milk_delivery': 0.2,
        'total_feed_spent': 0.2,
        'previous_loans': 0.2
    }
    
    score = (fico_score * weights['fico_score'] +
             avg_milk_delivery * weights['avg_milk_delivery'] +
             total_feed_spent * weights['total_feed_spent'] +
             previous_loans * weights['previous_loans'])
    
    return score

# Train the KNN model
def train_knn_model(df):
    X = df[['fico_score', 'avg_milk_delivery', 'total_feed_spent', 'previous_loans']]
    y = df['loan_eligible']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of KNN model: {accuracy * 100:.2f}%")

    return knn

# Predict loan eligibility based on KNN model
def predict_loan_eligibility(knn, fico_score, avg_milk_delivery, total_feed_spent, previous_loans):
    new_data = np.array([[fico_score, avg_milk_delivery, total_feed_spent, previous_loans]])
    prediction = knn.predict(new_data)
    return "Eligible" if prediction[0] == 1 else "Not Eligible"

# Combined function to get both credit score and loan eligibility
def get_credit_score_and_eligibility(fico_score, avg_milk_delivery, total_feed_spent, previous_loans):
    # Create sample database and populate it
    create_sample_database()

    # Retrieve data from the database
    df = retrieve_data()

    # Calculate credit score
    credit_score = calculate_credit_score(fico_score, avg_milk_delivery, total_feed_spent, previous_loans)
    print(f"Calculated Credit Score: {credit_score:.2f}")

    # Train the KNN model
    knn_model = train_knn_model(df)

    # Predict loan eligibility
    eligibility = predict_loan_eligibility(knn_model, fico_score, avg_milk_delivery, total_feed_spent, previous_loans)
    print(f"Loan Eligibility: {eligibility}")

    return credit_score, eligibility

# Example usage
if __name__ == "__main__":
    # Example input data for a new farmer
    new_farmer_data = {
        'fico_score': 675,
        'avg_milk_delivery': 130,
        'total_feed_spent': 200,
        'previous_loans': 1
    }

    credit_score, eligibility = get_credit_score_and_eligibility(**new_farmer_data)
    print(f"Final Results - Credit Score: {credit_score:.2f}, Loan Eligibility: {eligibility}")

