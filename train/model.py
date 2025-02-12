import os
import joblib
# import numpy as np
import pandas as pd
from multiprocessing import Process, Queue
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def merge_csv_files(directory):
    """
    Merge all CSV files in the specified directory into a single DataFrame.
    """
    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
    df_list = [pd.read_csv(file) for file in csv_files]
    return pd.concat(df_list, ignore_index=True)


def train_model(X_train, Y_train, queue):
    """
    Train the Logistic Regression model and put the trained model in the queue.
    """
    # Increase max_iter and use a different solver if needed
    cls = LogisticRegression(max_iter=1000, solver='lbfgs')  # Increase max_iter and use lbfgs solver
    cls.fit(X_train, Y_train)
    queue.put(cls)


if __name__ == '__main__':
    # Directory containing CSV files
    data_directory = "data"  # Replace with your directory path

    # Merge all CSV files
    df = merge_csv_files(data_directory)

    # Drop unnecessary columns
    df = df.drop(["exang", "slope", "fbs", "restecg"], axis=1)

    # Separate features and target
    X = df.drop(columns='target', axis=1)
    Y = df['target']

    # Scale the data (important for convergence)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=42)

    # Use a queue to store the trained model
    model_queue = Queue()

    # Train the model using multiprocessing
    train_process = Process(target=train_model, args=(X_train, Y_train, model_queue))
    train_process.start()
    train_process.join()

    # Retrieve the trained model from the queue
    cls = model_queue.get()

    # Accuracy on training data
    X_train_prediction = cls.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy on Training data : ', training_data_accuracy)

    # Accuracy on test data
    X_test_prediction = cls.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy on Test data : ', test_data_accuracy)

    # Save the trained model using joblib
    output_directory = "output"
    os.makedirs(output_directory, exist_ok=True)

    # Save the trained model using joblib in the output directory
    filename = os.path.join(output_directory, 'trained_model.sav')
    joblib.dump(cls, filename)
    print(f"Model saved to {filename}")