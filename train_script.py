import os
import joblib
import numpy as np

# Load the trained model
def load_model(model_path):
    """
    Load the trained model from the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


# Function to get user input with default values
def get_user_input():
    """
    Get user input for prediction with default values.
    """
    print("Enter the following features to predict heart disease (press Enter to use default values):")

    def get_input(prompt, default):
        value = input(f"{prompt} [default: {default}]: ")
        return float(value) if value.strip() else default

    age = get_input("Age", 50)
    sex = get_input("Sex (1 = male; 0 = female)", 1)
    cp = get_input("Chest Pain Type (0-3)", 2)
    trestbps = get_input("Resting Blood Pressure (in mm Hg)", 130)
    chol = get_input("Serum Cholesterol (in mg/dl)", 250)
    thalach = get_input("Maximum Heart Rate Achieved", 150)
    oldpeak = get_input("ST Depression Induced by Exercise Relative to Rest", 1.5)
    ca = get_input("Number of Major Vessels (0-3) Colored by Fluoroscopy", 0)
    thal = get_input("Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)", 3)
    return np.array([age, sex, cp, trestbps, chol, thalach, oldpeak, ca, thal]).reshape(1, -1)



def main():
 
    model_path = "output/trained_model.pkl"

    # Load the model
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get user input
    try:
        user_input = get_user_input()
        print("User input:", user_input)  # Debugging: Print the input array
    except Exception as e:
        print(f"Error getting user input: {e}")
        return

    # Make prediction
    try:
        prediction = model.predict(user_input)
        print("Prediction:", prediction[0])  # Debugging: Print the prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # Display the result
    if prediction[0] == 0:
        print("✅ Prediction: The Person does not have Heart Disease.")
    else:
        print("❌ Prediction: The Person has Heart Disease.")


if __name__ == "__main__":
    main()