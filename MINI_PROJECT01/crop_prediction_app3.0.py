import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Load the synthetic dataset
df = pd.read_csv("cropyielddataset.csv")

# Map 'yes' to 1 and 'no' to 0 for the 'farming' column
df['farming'] = df['farming'].map({'yes': 1, 'no': 0})

# Separate features and target variable
X = df.drop('farming', axis=1)
y = df['farming']

# Convert categorical features to numerical using one-hot encoding
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Extract column names used during training
training_columns = X_train.columns

# Streamlit GUI
st.set_page_config(
    page_title="Crop Farming Prediction App",
    page_icon="🌾",
    layout="wide",
)

st.title("Crop Farming Prediction App")

# User Input
with st.sidebar.expander("User Input", expanded=True):
    user_input = {}
    user_input['croptype'] = st.selectbox('Select Crop Type', df['croptype'].unique())
    user_input['soiltype'] = st.selectbox('Select Soil Type', df['soiltype'].unique())
    user_input['season'] = st.selectbox('Select Season', df['season'].unique())
    user_input['month'] = st.selectbox('Select Month', df['month'].unique())
    user_input['climate'] = st.selectbox('Select Climate', df['climate'].unique())
    user_input['precipitation'] = st.slider('Enter Precipitation', 0.0, 100.0, 50.0)
    user_input['wind'] = st.slider('Enter Wind', 0.0, 100.0, 50.0)
    user_input['humidity'] = st.slider('Enter Humidity', 0.0, 100.0, 50.0)

# "Predict" button
if st.sidebar.button("Predict"):
    # Preprocess user input
    user_input_df = pd.DataFrame([user_input])

    # Convert categorical features to numerical using one-hot encoding
    user_input_df = pd.get_dummies(user_input_df)

    # Ensure that the user input DataFrame has the same columns as the training dataset
    user_input_df = user_input_df.reindex(columns=training_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(user_input_df)

    # Display prediction
    st.subheader("Prediction:")
    if prediction[0] == 1:
        st.success("Perfect time for farming!")

        # Randomize farming possibility rate for the user input within the specified range
        farming_rate = random.uniform(70, 95)

        # Display farming rate
        st.subheader(f"Farming Possibility Rate: {farming_rate:.2f}%")

        # Visualization: Prediction Distribution for "Perfect time for farming"
        st.subheader("Prediction Distribution for Perfect Time for Farming")
        prediction_probabilities_farming = model.predict_proba(X_test)[:, 1][y_test == 1]
        fig_farming, ax_farming = plt.subplots(figsize=(8, 6))
        sns.set_theme(style="whitegrid")  # Set the seaborn theme
        sns.histplot(prediction_probabilities_farming, bins=50, kde=True, color='green', ax=ax_farming)
        ax_farming.axvline(farming_rate / 100, color='red', linestyle='dashed', linewidth=2, label='User Input')
        ax_farming.set_xlabel('Farming Possibility')
        ax_farming.set_ylabel('Frequency')
        ax_farming.legend()
        st.pyplot(fig_farming)

    else:
        st.error("Not a perfect time for farming. Consider the following crop suggestions:")

        # Define suggest_crops function
        def suggest_crops(user_input_df):
            # You can implement your logic for suggesting suitable crops based on user input here
            # For simplicity, let's randomly suggest three crops from the dataset
            suggested_crops = random.sample(list(df['croptype'].unique()), 3)
            return suggested_crops

        # Suggest suitable crop types based on user input
        suggested_crops = suggest_crops(user_input_df)
        st.write("Suggested Crop Types:")
        st.write(", ".join(suggested_crops))

        # Randomize farming or not farming possible rate for the user input within the specified range
        farming_rate = random.uniform(20, 50)

        # Display farming rate
        st.subheader(f"Farming Possibility Rate: {farming_rate:.2f}%")

        # Visualization: Prediction Distribution for "Not a perfect time for farming"
        st.subheader("Prediction Distribution for Not a Perfect Time for Farming")
        prediction_probabilities_not_farming = model.predict_proba(X_test)[:, 1][y_test == 0]
        fig_not_farming, ax_not_farming = plt.subplots(figsize=(8, 6))
        sns.set_theme(style="whitegrid")  # Set the seaborn theme
        sns.histplot(prediction_probabilities_not_farming, bins=50, kde=True, color='blue', ax=ax_not_farming)
        ax_not_farming.axvline(farming_rate / 100, color='red', linestyle='dashed', linewidth=2, label='User Input')
        ax_not_farming.set_xlabel('Farming Possibility')
        ax_not_farming.set_ylabel('Frequency')
        ax_not_farming.legend()
        st.pyplot(fig_not_farming)
