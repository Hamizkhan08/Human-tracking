import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from flask import Flask, render_template, request, jsonify
import joblib # To save/load scaler and encoder

app = Flask(__name__)

# --- Model Loading and Training (from your original script) ---
# It's better to load the trained model, scaler, and encoder rather than retraining on every run.
# For simplicity, we'll keep the training here, but for production, save and load them.

data = pd.read_csv("human_activity_realistic.csv")

X = data[['x', 'y', 'z']]
y = data['activity']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- Routes for the Flask App ---

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_activity = None
    if request.method == 'POST':
        try:
            x_val = float(request.form['x_val'])
            y_val = float(request.form['y_val'])
            z_val = float(request.form['z_val'])

            sample_data = np.array([[x_val, y_val, z_val]])
            sample_scaled = scaler.transform(sample_data)
            predicted_activity = encoder.inverse_transform(rf_model.predict(sample_scaled))[0]

        except ValueError:
            predicted_activity = "Invalid input. Please enter numbers for x, y, and z."
        except Exception as e:
            predicted_activity = f"An error occurred: {e}"

    return render_template('index.html', predicted_activity=predicted_activity)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    An API endpoint for external applications to make predictions.
    Expects JSON input: {"x": 0.3, "y": 0.5, "z": 9.1}
    """
    try:
        data = request.get_json(force=True)
        x_val = data['x']
        y_val = data['y']
        z_val = data['z']

        sample_data = np.array([[x_val, y_val, z_val]])
        sample_scaled = scaler.transform(sample_data)
        predicted_activity = encoder.inverse_transform(rf_model.predict(sample_scaled))[0]

        return jsonify({'prediction': predicted_activity})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # It's good practice to save your scaler and encoder
    # so you don't have to retrain every time.
    # For this example, we'll just run the app.
    # In a real scenario, you'd do:
    # joblib.dump(scaler, 'scaler.pkl')
    # joblib.dump(encoder, 'encoder.pkl')
    # joblib.dump(rf_model, 'rf_model.pkl')
    # Then load them:
    # scaler = joblib.load('scaler.pkl')
    # encoder = joblib.load('encoder.pkl')
    # rf_model = joblib.load('rf_model.pkl')

    app.run(debug=True) # debug=True allows for automatic reloading on code changes
