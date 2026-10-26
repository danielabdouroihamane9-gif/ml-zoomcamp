import pickle

# Load the model pipeline
with open('pipeline.bin', 'rb') as f_in:
    model = pickle.load(f_in)

# Lead data provided in Question 3
lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "annual_income": 85000,
    "number_of_courses_viewed": 2,
    "interaction_count": 4
}

# Predict probability
pred = model.predict_proba([lead])[0, 1]
print(f"Prediction probability: {pred:.3f}")