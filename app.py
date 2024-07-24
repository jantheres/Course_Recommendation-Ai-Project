import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import streamlit as st

# Load the dataset
df = pd.read_csv('courses_dataset.csv')

# Separate features and targets
X = df[['Field of Interest']]
y = df[['Course Name', 'Level', 'Duration', 'Course Description']]

# Encode categorical features
le_X = LabelEncoder()
X_encoded = le_X.fit_transform(X['Field of Interest'])

# Encode the targets
le_y_name = LabelEncoder()
le_y_level = LabelEncoder()
le_y_duration = LabelEncoder()
le_y_description = LabelEncoder()

y_encoded_name = le_y_name.fit_transform(y['Course Name'])
y_encoded_level = le_y_level.fit_transform(y['Level'])
y_encoded_duration = le_y_duration.fit_transform(y['Duration'])
y_encoded_description = le_y_description.fit_transform(y['Course Description'])

# Prepare data for the model
X_encoded = X_encoded.reshape(-1, 1)
y_encoded = pd.DataFrame({
    'Course Name': y_encoded_name,
    'Level': y_encoded_level,
    'Duration': y_encoded_duration,
    'Course Description': y_encoded_description
})

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open('multi_output_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the model
with open('multi_output_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Define prediction function
def predict_course(field_of_interest):
    try:
        sample = le_X.transform([field_of_interest]).reshape(-1, 1)
        prediction = loaded_model.predict(sample)
        return [
            le_y_name.inverse_transform([prediction[0][0]])[0],
            le_y_level.inverse_transform([prediction[0][1]])[0],
            le_y_duration.inverse_transform([prediction[0][2]])[0],
            le_y_description.inverse_transform([prediction[0][3]])[0]
        ]
    except Exception as e:
        return [str(e)] * 4  # Return the error message in each output field

# Streamlit application
st.title('Course Prediction App')

# Dropdown menu for field of interest
field_of_interest_options = X['Field of Interest'].unique().tolist()
selected_field = st.selectbox('Select Field of Interest', field_of_interest_options)

# Button to make prediction
if st.button('Predict'):
    predictions = predict_course(selected_field)
    
    # Display predictions
    st.subheader('Predictions')
    st.write(f'**Course Name:** {predictions[0]}')
    st.write(f'**Level:** {predictions[1]}')
    st.write(f'**Duration:** {predictions[2]}')
    st.write(f'**Description:** {predictions[3]}')
