
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load model and encoders
with open('model_CustomerChurn.pkl', 'rb') as file:
    model, SubscriptionType_encoder, PaymentMethod_encoder, PaperlessBilling_encoder, ContentType_encoder, MultiDeviceAccess_encoder, DeviceRegistered_encoder,GenrePreference_encoder, Gender_encoder, ParentalControl_encoder, SubtitlesEnabled_encoder = pickle.load(file)

# Load your DataFrame
# Replace 'your_data.csv' with the actual file name or URL
df = pd.read_csv('train.csv')
df = df.drop('CustomerID', axis=1)

# Streamlit App
st.title('Customer Churn App')

# Define a session state to remember tab selections
if 'tab_selected' not in st.session_state:
    st.session_state.tab_selected = 0

# Create tabs for prediction and visualization
tabs = ['Predict Customer Churn', 'Visualize Data', 'Predict from CSV']
selected_tab = st.radio('Select Tab:', tabs, index=st.session_state.tab_selected)

# Tab selection logic
if selected_tab != st.session_state.tab_selected:
    st.session_state.tab_selected = tabs.index(selected_tab)

# Tab 1: Predict Customer Churn
if st.session_state.tab_selected == 0:
    st.header('Predict Customer Churn')

    # User Input Form
    AccountAge = st.slider('Age', 0, 100, 30)
    MonthlyCharges = st.slider('Monthly Charges', 0, 20, 5)
    TotalCharges = st.number_input('Total Charges', 0, 5000, 1500)
    SubscriptionType = st.selectbox('Subscription Type', SubscriptionType_encoder.classes_)
    PaymentMethod = st.radio('Payment Method', PaymentMethod_encoder.classes_)
    PaperlessBilling = st.radio('Paperless Billing', PaperlessBilling_encoder.classes_)
    ContentType = st.radio('Content Type', ContentType_encoder.classes_)
    MultiDeviceAccess = st.radio('Multiple Device Access', MultiDeviceAccess_encoder.classes_)
    DeviceRegistered = st.selectbox('Device Registered', DeviceRegistered_encoder.classes_)
    ViewingHoursPerWeek = st.number_input('Viewing Hours Per Week', 0, 168, 24)
    AverageViewingDuration = st.number_input('Average Views Duration', 0, 200, 50)
    ContentDownloadsPerMonth = st.slider('Contents Downloaded Per Month', 0, 100, 50)
    GenrePreference = st.selectbox('Genre Preference', GenrePreference_encoder.classes_)
    UserRating = st.number_input('User Rating', 0.0, 5.0, 2.5)
    SupportTicketsPerMonth = st.slider('User Support Tickets Per Month', 0, 10, 0)
    Gender = st.radio('Gender', Gender_encoder.classes_)
    WatchlistSize = st.number_input('Total Watchlist', 0, 25, 5)
    ParentalControl = st.radio('Under Parental Control', ParentalControl_encoder.classes_)
    SubtitlesEnabled = st.radio('Subtitles Enabled', SubtitlesEnabled_encoder.classes_)

    # Create a DataFrame for the user input
    user_input = pd.DataFrame({
        'AccountAge': [AccountAge],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'SubscriptionType': [SubscriptionType],
        'PaymentMethod': [PaymentMethod],
        'PaperlessBilling': [PaperlessBilling],
        'ContentType': [ContentType],
        'MultiDeviceAccess': [MultiDeviceAccess],
        'DeviceRegistered': [DeviceRegistered],
        'ViewingHoursPerWeek': [ViewingHoursPerWeek],
        'AverageViewingDuration': [AverageViewingDuration],
        'ContentDownloadsPerMonth': [ContentDownloadsPerMonth],
        'GenrePreference': [GenrePreference],
        'UserRating': [UserRating],
        'SupportTicketsPerMonth': [SupportTicketsPerMonth],
        'Gender': [Gender],
        'WatchlistSize': [WatchlistSize],
        'ParentalControl': [ParentalControl],
        'SubtitlesEnabled': [SubtitlesEnabled]
    })

    # Categorical Data Encoding
    user_input['SubscriptionType'] = SubscriptionType_encoder.transform(user_input['SubscriptionType'])
    user_input['PaymentMethod'] = PaymentMethod_encoder.transform(user_input['PaymentMethod'])
    user_input['PaperlessBilling'] = PaperlessBilling_encoder.transform(user_input['PaperlessBilling'])
    user_input['ContentType'] = ContentType_encoder.transform(user_input['ContentType'])
    user_input['MultiDeviceAccess'] = MultiDeviceAccess_encoder.transform(user_input['MultiDeviceAccess'])
    user_input['DeviceRegistered'] = DeviceRegistered_encoder.transform(user_input['DeviceRegistered'])
    user_input['GenrePreference'] = GenrePreference_encoder.transform(user_input['GenrePreference'])
    user_input['Gender'] = Gender_encoder.transform(user_input['Gender'])
    user_input['ParentalControl'] = ParentalControl_encoder.transform(user_input['ParentalControl'])
    user_input['SubtitlesEnabled'] = SubtitlesEnabled_encoder.transform(user_input['SubtitlesEnabled'])


    # Predicting
    prediction = model.predict(user_input)

    # Display Result
    st.subheader('Prediction Result:')
    st.write('Customer Churn : ', prediction[0])

# Tab 2: Visualize Data
elif st.session_state.tab_selected == 1:
    st.header('Visualize Data')

    # Select condition feature
    condition_feature = st.selectbox('Select Condition Feature:', df.columns)

    # Set default condition values
    default_condition_values = ['Select All'] + df[condition_feature].unique().tolist()

    # Select condition values
    condition_values = st.multiselect('Select Condition Values:', default_condition_values)

    # Handle 'Select All' choice
    if 'Select All' in condition_values:
        condition_values = df[condition_feature].unique().tolist()

    if len(condition_values) > 0:
        # Filter DataFrame based on selected condition
        filtered_df = df[df[condition_feature].isin(condition_values)]

        # Plot the number of Customer based on Customer Churn
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.countplot(x=condition_feature, hue='Churn', data=filtered_df, palette='viridis')
        plt.title('Number of Customer based on Customer Churn')
        plt.xlabel(condition_feature)
        plt.ylabel('Number of Customer')
        st.pyplot(fig)

# Tab 3: Predict from CSV
elif st.session_state.tab_selected == 2:
    st.header('Predict from CSV')

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    # uploaded_file

    if uploaded_file is not None:
        # Read CSV file
        csv_df_org = pd.read_csv(uploaded_file)
        csv_df_org = csv_df_org.dropna()
        # csv_df_org.columns

        csv_df = csv_df_org.copy()
        csv_df = csv_df.drop('CustomerID',axis=1)



         # Categorical Data Encoding
        csv_df['SubscriptionType'] = SubscriptionType_encoder.transform(csv_df['SubscriptionType'])
        csv_df['PaymentMethod'] = PaymentMethod_encoder.transform(csv_df['PaymentMethod'])
        csv_df['PaperlessBilling'] = PaperlessBilling_encoder.transform(csv_df['PaperlessBilling'])
        csv_df['ContentType'] = ContentType_encoder.transform(csv_df['ContentType'])
        csv_df['MultiDeviceAccess'] = MultiDeviceAccess_encoder.transform(csv_df['MultiDeviceAccess'])
        csv_df['DeviceRegistered'] = DeviceRegistered_encoder.transform(csv_df['DeviceRegistered'])
        csv_df['GenrePreference'] = GenrePreference_encoder.transform(csv_df['GenrePreference'])
        csv_df['Gender'] = Gender_encoder.transform(csv_df['Gender'])
        csv_df['ParentalControl'] = ParentalControl_encoder.transform(csv_df['ParentalControl'])
        csv_df['SubtitlesEnabled'] = SubtitlesEnabled_encoder.transform(csv_df['SubtitlesEnabled'])



        # Predicting
        predictions = model.predict(csv_df)

        # Add predictions to the DataFrame
        csv_df_org['Churn'] = predictions

        # Display the DataFrame with predictions
        st.subheader('Predicted Results:')
        st.write(csv_df_org)

        # Visualize predictions based on a selected feature
        st.subheader('Visualize Predictions')

        # Select feature for visualization
        feature_for_visualization = st.selectbox('Select Feature for Visualization:', csv_df_org.columns)

        # Plot the number of employees based on KPIs for the selected feature
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.countplot(x=feature_for_visualization, hue='Churn', data=csv_df_org, palette='viridis')
        plt.title(f'Number of Customer based on Customer Churn - {feature_for_visualization}')
        plt.xlabel(feature_for_visualization)
        plt.ylabel('Number of Customer')
        st.pyplot(fig)

