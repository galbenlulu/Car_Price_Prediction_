from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from car_data_prep import prepare_data
from datetime import datetime
import numpy as np

app = Flask(__name__)
# Load the trained model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Create a dictionary with all original columns, filled with NA in order to fit the "Prepare_data"  function. 
        all_columns = ['Year', 'model', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type', 
                       'Prev_ownership', 'Curr_ownership', 'Area', 'City', 'Pic_num', 
                       'Description', 'Color', 'Km', 'Test', 'Supply_score', 'manufactor', 
                       'Cre_date', 'Repub_date']
        input_data = {col: [np.nan] for col in all_columns}
        
        # Update the dictionary with user input for relevant features
        input_data.update({
            'manufactor': [request.form['manufactor']],
            'Year': [int(request.form['Year'])]
        })
        
        # Optional fields -For the following fields we dealt with missing values ​in  the train set
        if 'capacity_Engine' in request.form and request.form['capacity_Engine']:
            input_data['capacity_Engine'] = [int(request.form['capacity_Engine'])]
        if 'Km' in request.form and request.form['Km']:
            input_data['Km'] = [int(request.form['Km'])]
        if 'Supply_score' in request.form and request.form['Supply_score']:
            input_data['Supply_score'] = [float(request.form['Supply_score'])]
        if 'Prev_ownership' in request.form and request.form['Prev_ownership']:
            input_data['Prev_ownership'] = [request.form['Prev_ownership']]
        if 'Engine_type' in request.form and request.form['Engine_type']:
            input_data['Engine_type'] = [request.form['Engine_type']]
        
        # Create DataFrame
        df = pd.DataFrame(input_data)
        
        # Prepare data
        prepared_data = prepare_data(df)
        
        # Make prediction
        prediction = model.predict(prepared_data)[0]
        
        # car price can not be negative, therefore if the model returns a negative number the price of the car will be  0

        prediction = max(0, int(round(prediction)))
        
        # Format the prediction with commas and add the shekel symbol
        formatted_prediction = f"{prediction:,}₪"
        
        return formatted_prediction
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)