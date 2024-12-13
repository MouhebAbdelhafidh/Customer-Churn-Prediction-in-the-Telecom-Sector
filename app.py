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

# Route for rendering the HTML page and handling predictions
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # File upload prediction
            if 'csv_file' in request.files:
                file = request.files['csv_file']
                if file.filename == '':
                    return jsonify({'error': 'No selected file'}), 400

                df = pd.read_csv(file)

                # Process categorical columns with LabelEncoder
                categorical_columns = ['State', 'International plan', 'Voice mail plan']
                for col in categorical_columns:
                    if col in df.columns:
                        df[col] = label_encoder.fit_transform(df[col].fillna('Unknown'))

                input_data = df[selected_features]
                predictions = model.predict(input_data)
                predictions = ['Yes' if p == 1 else 'No' for p in predictions]

                return jsonify({'predictions': predictions})

            # Form-based prediction for a single individual
            elif request.form:
                state = request.form.get('state', 'Unknown')
                intl_plan = request.form.get('international_plan', 'No')
                voicemail_plan = request.form.get('voice_mail_plan', 'No')
                total_day_minutes = float(request.form.get('total_day_minutes', 0))
                total_eve_minutes = float(request.form.get('total_eve_minutes', 0))
                total_night_minutes = float(request.form.get('total_night_minutes', 0))

                # Create a single-row DataFrame
                input_data = pd.DataFrame([{
                    'State': label_encoder.fit_transform([state])[0],
                    'International plan': label_encoder.fit_transform([intl_plan])[0],
                    'Voice mail plan': label_encoder.fit_transform([voicemail_plan])[0],
                    'Total day minutes': total_day_minutes,
                    'Total eve minutes': total_eve_minutes,
                    'Total night minutes': total_night_minutes
                }])

                prediction = model.predict(input_data)
                prediction = 'Yes' if prediction[0] == 1 else 'No'

                return jsonify({'prediction': prediction})

            else:
                return jsonify({'error': 'Invalid request format'}), 400

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Render the main HTML page
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
