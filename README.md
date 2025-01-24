# Real-Time Recommendation System

This project is a real-time recommendation system built using Singular Value Decomposition (SVD). It recommends products to users based on their past behavior, utilizing a collaborative filtering approach. The model is trained on a dataset of Amazon Electronics product ratings.

## Project Overview

This application provides a web-based interface where users can input their `user_id` and the number of recommendations (`num`). The system then calculates and returns a list of recommended products for the given user based on the ratings and the SVD model.

## Dataset

The data used for this project is from the [Amazon Electronics Rating Dataset](https://www.kaggle.com/datasets/vibivij/amazon-electronics-rating-datasetrecommendation/data) available on Kaggle. This dataset includes product ratings by users, which we use to generate recommendations.

## Key Components

1. **SVD Model**: We use Singular Value Decomposition (SVD) to decompose the user-item rating matrix and predict the user's preference for unrated items.
   
2. **Flask Web Application**: The backend is built using Flask to serve the recommendation model via an API.

3. **Frontend Interface**: A simple HTML form allows users to input their `user_id` and the number of recommendations they want, and the system returns the recommended products.

## How It Works

1. **SVD Decomposition**: The dataset is processed, and the ratings matrix is decomposed using SVD. The SVD algorithm breaks down the matrix into three matrices: U (user features), sigma (singular values), and Vt (item features).
   
2. **Recommendation Calculation**: For a given `user_id`, the system computes the predicted ratings for all products the user has not yet rated by multiplying the U, sigma, and Vt matrices. The products are then ranked based on their predicted ratings.

3. **Frontend**: The user can submit their `user_id` and the number of recommendations (`num`) through the form. The Flask backend processes the input, calculates the recommendations, and returns the list of products.

## Installation

### Prerequisites

Ensure that you have the following installed on your machine:

- Python 3.x
- pip (Python package installer)
- Flask
- numpy
- pandas
- pickle

### Steps to Run the Project

1. **Clone the Repository**:

   Clone the repository to your local machine:

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install Dependencies**:

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Load the Dataset**:

   Download the [Amazon Electronics Rating Dataset](https://www.kaggle.com/datasets/vibivij/amazon-electronics-rating-datasetrecommendation/data) from Kaggle and place the CSV files in the `data/` directory.

   - `item_columns.csv`: Contains information about products/items.
   - `user_index.csv`: Contains the list of users in the dataset.

4. **Load and Train the SVD Model**:

   The model is pre-trained and stored in the `models/` directory as `svd_model.pkl`.

5. **Run the Flask Application**:

   Start the Flask web application by running:

   ```bash
   python app.py
   ```

   The application will run on `http://127.0.0.1:5000/`.

6. **Access the Frontend**:

   Open your browser and go to `http://127.0.0.1:5000/` to access the recommendation form. Input the `user_id` and the number of recommendations (`num`) you want.

## API Endpoints

### `/recommend`
- **Method**: GET
- **Parameters**:
  - `user_id`: The ID of the user for whom recommendations are to be generated.
  - `num`: The number of recommendations to return (default is 5).
  
- **Response**:
  - `user_id`: The user ID for which the recommendations were generated.
  - `recommended_products`: A list of recommended product IDs.

Example Request:
```
GET /recommend?user_id=2&num=5
```

Example Response:
```json
{
  "user_id": 2,
  "recommended_products": [
    "B000N99BBC",
    "B00829TIEK",
    "B008DWCRQW",
    "B00829TIA4",
    "B0088CJT4U"
  ]
}
```

## Troubleshooting

- Ensure that the `svd_model.pkl`, `item_columns.csv`, and `user_index.csv` files are correctly loaded.
- If you encounter errors related to missing dependencies, try reinstalling the requirements using `pip install -r requirements.txt`.

## Author
The model and repository is created by Sarthak Kumar Kalia
