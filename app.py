import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import gdown



# Load model and necessary files
def load_models():
    url = "https://drive.google.com/uc?id=1GFMAps0M7lXvP4vLwR_Wx1Tgr_9OMkAo"
    output = "voting_clf_model.pkl"
    gdown.download(url, output, quiet=False)
   # Modeli yükleme
    model = joblib.load(output)
    scaler = joblib.load('model_scaler.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
    with open('model_features.json', 'r') as f:
        feature_names = json.load(f)
    return model, scaler, kmeans, feature_names


# Streamlit interface
def main():
    st.title('Bank Marketing Prediction System')
    st.write('Enter customer details to predict participation in the term deposit campaign.')

    # Sidebar for user inputs
    st.sidebar.header('Customer Information')

    # Basic customer details
    age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
    job = st.sidebar.selectbox('Job',
                               ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                                'retired', 'self-employed', 'services', 'student', 'technician',
                                'unemployed', 'unknown'])

    marital = st.sidebar.selectbox('Marital Status', ['divorced', 'married', 'single'])
    education = st.sidebar.selectbox('Education Level', ['primary', 'secondary', 'tertiary', 'unknown'])

    # Financial details
    balance = st.sidebar.number_input('Account Balance', value=0)
    housing = st.sidebar.selectbox('Housing Loan', ['yes', 'no'])
    loan = st.sidebar.selectbox('Personal Loan', ['yes', 'no'])

    # Campaign details
    duration = st.sidebar.number_input('Last contact duration (seconds)', min_value=0, value=0)
    campaign = st.sidebar.number_input('Number of contacts in this campaign', min_value=0, value=0)
    pdays = st.sidebar.number_input('Days since last campaign contact', min_value=0, value=0)

    # Contact information
    contact = st.sidebar.selectbox('Contact type', ['cellular', 'telephone', 'unknown'])
    month = st.sidebar.selectbox('Last contact month',
                                 ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.sidebar.selectbox('Last contact day',
                                       ['mon', 'tue', 'wed', 'thu', 'fri'])

    # Previous campaign outcome
    poutcome = st.sidebar.selectbox('Previous campaign outcome',
                                    ['failure', 'other', 'success', 'unknown'])

    if st.sidebar.button('Predict'):
        # Load models
        model, scaler, kmeans, feature_names = load_models()

        # Prepare new data
        new_data = prepare_data(
            age, job, marital, education, balance, housing, loan,
            duration, campaign, pdays, contact, month, day_of_week,
            poutcome, feature_names
        )

        # Make predictions
        prediction = model.predict(new_data)
        prediction_proba = model.predict_proba(new_data)

        # Display results
        st.header('Prediction Results')

        if prediction[0] == 1:
            st.success('The customer is likely to participate in the term deposit campaign!')
        else:
            st.error('The customer is unlikely to participate in the term deposit campaign.')

        st.write('Participation Probability: %{:.2f}'.format(prediction_proba[0][1] * 100))

        # Visualization
        create_gauge_chart(prediction_proba[0][1])

        # Additional details
        with st.expander("Detailed Analysis"):
            st.write("Customer Profile:")
            st.write(f"- Age Group: {get_age_group(age)}")
            st.write(f"- Account Status: {get_balance_status(balance)}")
            st.write(f"- Campaign Interaction: {get_campaign_interaction(campaign)}")


def prepare_data(age, job, marital, education, balance, housing, loan,
                 duration, campaign, pdays, contact, month, day_of_week,
                 poutcome, feature_names):
    """Prepare user inputs into the appropriate format for the model"""

    # Create initial DataFrame
    data_dict = {
        'age': age,
        'balance': balance,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'debt_income_ratio': balance / (1 + campaign),
        'balance_campaign_ratio': balance / (1 + campaign)
    }

    # One-hot encoding for categorical variables
    categorical_inputs = {
        'job': job,
        'marital': marital,
        'education': education,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'poutcome': poutcome
    }

    # Apply one-hot encoding
    for feature in feature_names:
        for cat_col, value in categorical_inputs.items():
            if feature.startswith(cat_col + '_'):
                category = feature.split(cat_col + '_')[1]
                data_dict[feature] = 1 if value == category else 0

    # Create DataFrame and set column order
    df = pd.DataFrame([data_dict])
    df = df.reindex(columns=feature_names, fill_value=0)

    return df


def create_gauge_chart(probability):
    """Create a gauge chart to show prediction probability"""
    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Participation Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    st.plotly_chart(fig)


def get_age_group(age):
    if age < 30:
        return "Young (18-29)"
    elif age < 50:
        return "Middle-Aged (30-49)"
    elif age < 70:
        return "Senior (50-69)"
    else:
        return "Very Senior (70+)"


def get_balance_status(balance):
    if balance < 0:
        return "Negative Balance"
    elif balance < 1000:
        return "Low Balance"
    elif balance < 5000:
        return "Medium Balance"
    else:
        return "High Balance"


def get_campaign_interaction(campaign):
    if campaign <= 2:
        return "Low Interaction"
    elif campaign <= 5:
        return "Medium Interaction"
    else:
        return "High Interaction"


if __name__ == '__main__':
    main()
