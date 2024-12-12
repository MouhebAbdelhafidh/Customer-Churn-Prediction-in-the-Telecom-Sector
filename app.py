from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model and the necessary components
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the selected feature columns (as you did during training)
selected_features = ['State', 'International plan', 'Voice mail plan', 'Total day minutes',
       'Total eve minutes', 'Total night minutes']

# Initialize the Flask app
app = Flask(__name__)

# Initialize label encoder for categorical variables
label_encoder = LabelEncoder()

# Route for rendering the HTML page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is uploaded
        if 'csv_file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['csv_file']

        # If no file is selected
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)

        # Encode categorical columns using Label Encoding
        categorical_columns = ['State', 'International plan', 'Voice mail plan']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = label_encoder.fit_transform(df[col].fillna('Unknown'))

        # Select only the columns that were used during training (feature selection)
        input_data = df[selected_features]

        # Make prediction using the model
        predictions = model.predict(input_data)

        # Convert predictions to a list of "Yes" or "No"
        predictions = ['Yes' if p == 1 else 'No' for p in predictions]

        # Return predictions as JSON
        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
