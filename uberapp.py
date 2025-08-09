from flask import Flask, request, render_template
import joblib
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Model Loading ---
model_path = os.path.join(os.path.dirname(__file__), 'ensemble_model.pkl')

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    model = None
    print(f"Error: Model file not found at {model_path}")

@app.route("/", methods=["GET", "POST"])
def home():
    """
    Main route for the prediction form.
    """
    prediction_result = None
    error_message = None

    if model is None:
        error_message = "Machine learning model is not available. Please check server logs."
        return render_template("index.html", error=error_message)

    if request.method == "POST":
        try:
            hour = int(request.form['hour'])
            day = int(request.form['day'])
            dayofweek = int(request.form['dayofweek'])
            month = int(request.form['month'])
            lat = float(request.form['lat'])
            lon = float(request.form['lon'])

            if not (0 <= hour <= 23):
                raise ValueError("Hour must be between 0 and 23.")
            if not (1 <= day <= 31):
                raise ValueError("Day must be between 1 and 31.")
            if not (0 <= dayofweek <= 6):
                raise ValueError("Day of week must be between 0 and 6.")
            if not (1 <= month <= 12):
                raise ValueError("Month must be between 1 and 12.")

            input_features = np.array([[hour, day, dayofweek, month, lat, lon]])
            prediction = model.predict(input_features)[0]
            prediction_result = int(round(prediction))

        except ValueError as ve:
            error_message = f"Invalid input: {ve}"
        except Exception as e:
            error_message = f"An error occurred: {e}"

    return render_template("index.html", prediction=prediction_result, error=error_message)


@app.route("/dashboard")
def dashboard():
    """
    Route for the data visualization dashboard.
    """
    return render_template("dashboard.html")


if __name__ == "__main__":
    app.run(debug=True)
