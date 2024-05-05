import joblib
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import boto3

from sklearn.ensemble import RandomForestClassifier

"""
Deserialize fitted model
"""
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    return model

"""
input_fn
    request_body: The body of the request sent to the model.
    request_content_type: (string) specifies the format/variable type of the request
"""
def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError("This model only supports application/json input")

"""
predict_fn
    input_data: returned data from input_fn above
    model (sklearn model) returned model loaded from model_fn above
"""
def predict_fn(input_data, model):
    # Process the input data if necessary
    processed_data = process_input(input_data, model)
    # Make predictions using the model
    predictions = model.predict(processed_data)
    return predictions

def process_input(input_data, model):
    # Process input data as needed before passing to the model for prediction
    X = input_data['url']
    vectorizer = joblib.load(os.path.join("opt/ml/model", "tfidf_vectorizer.pkl"))
    X_vect = vectorizer.transform(X)
    return X_vect

"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: the content type the endpoint expects to be returned. Ex: JSON, string
"""
def output_fn(prediction, content_type):
    prediction_str = prediction[0]
    response = {"type": prediction_str}
    return json.dumps(response)
