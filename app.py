from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the saved SVD model and user/item data
try:
    with open('models/svd_model.pkl', 'rb') as file:
        model = pickle.load(file)

    U = model['U']
    sigma = np.diag(model['sigma'])
    Vt = model['Vt']

    # Load item and user mappings
    item_columns = pd.read_csv('data/item_columns.csv', index_col=0).squeeze("columns")
    user_index = pd.read_csv('data/user_index.csv', index_col=0).squeeze("columns")

    print("Model and data loaded successfully.")
except Exception as e:
    print(f"Error loading model or data: {e}")
    U = sigma = Vt = item_columns = user_index = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    num_recommendations = request.args.get('num', 5)

    if user_id is None:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    try:
        user_id = int(user_id)
        num_recommendations = int(num_recommendations)
    except ValueError:
        return jsonify({'error': 'user_id and num must be integers'}), 400

    # Ensure that user_id exists in user_index
    if user_id not in user_index.values:
        return jsonify({'error': 'Invalid user_id'}), 400

    # Get the user index position
    user_index_position = user_index[user_index == user_id].index[0]
    
    # Calculate the recommendations using the SVD model
    user_ratings = np.dot(U[user_index_position, :], sigma).dot(Vt)
    
    # Sort the recommendations based on predicted ratings
    recommendations = np.argsort(user_ratings)[::-1][:num_recommendations]
    
    # Get the product names corresponding to the recommendations
    recommended_products = item_columns.iloc[recommendations].tolist()

    return jsonify({'user_id': user_id, 'recommended_products': recommended_products})


if __name__ == '__main__':
    app.run(debug=True)
