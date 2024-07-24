# Course Prediction App

This is a Streamlit-based web application that recommends courses based on a user's field of interest after completing high school. The backend model uses a RandomForestClassifier to predict course details such as name, level, duration, and description.

## Features

- **Course Prediction:** Predicts the course name, level, duration, and description based on the selected field of interest.
- **Interactive UI:** Built with Streamlit for an interactive user experience.

## Requirements

- Python 3.7+
- `pandas`
- `scikit-learn`
- `streamlit`

  To start the Streamlit application, run:
    ```bash
    streamlit run app.py
    ```

2. Open your browser and navigate to `http://localhost:8501` to interact with the application.

## Model Details

The model used is a RandomForestClassifier, trained on the dataset to predict:

- Course Name
- Level
- Duration
- Course Description

The model and LabelEncoders are saved in `multi_output_model.pkl` and loaded for predictions.

## Example

1. **Select Field of Interest:** Choose a field from the dropdown menu.
2. **Predict:** Click the "Predict" button to get course recommendations.

## Files

- `app.py`: The main Streamlit application file.
- `multi_output_model.pkl`: The saved machine learning model.
- `courses_dataset.csv`: The dataset used for training the model.
- `requirements.txt`: List of required Python packages.
