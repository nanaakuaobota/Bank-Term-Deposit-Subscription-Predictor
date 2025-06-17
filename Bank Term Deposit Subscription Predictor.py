# Load model and scaler
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define categorical and numerical columns based on your preprocessing
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Load the list of all columns used in training after get_dummies (replace with actual list)
# You should save this list during training and load here; for demo, define manually:
all_columns = [
    'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous',
    # Add all dummy variable columns here, e.g.:
    'job_blue-collar', 'job_admin.', 'job_management', 'job_technician', # etc.
    'marital_married', 'marital_single',
    'education_primary', 'education_secondary', 'education_tertiary',
    'default_no', 'default_yes',
    'housing_no', 'housing_yes',
    'loan_no', 'loan_yes',
    'contact_cellular', 'contact_telephone',
    'month_apr', 'month_jul', 'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
    'poutcome_failure', 'poutcome_success', 'poutcome_unknown'
]

st.title("Bank Term Deposit Subscription Predictor")

# Collect inputs
age = st.number_input('Age', min_value=17, max_value=100, value=30)
balance = st.number_input('Balance', value=1000)
day = st.number_input('Day of Month', min_value=1, max_value=31, value=15)
duration = st.number_input('Duration (seconds)', min_value=0, value=100)
campaign = st.number_input('Number of contacts during campaign', min_value=1, value=1)
pdays = st.number_input('Days since last contact', value=999)
previous = st.number_input('Number of contacts before campaign', min_value=0, value=0)

job = st.selectbox('Job', ['admin.', 'blue-collar', 'technician', 'management', 'services', 'retired', 'self-employed', 'unemployed', 'entrepreneur', 'housemaid', 'student', 'unknown'])
marital = st.selectbox('Marital Status', ['married', 'single', 'divorced'])
education = st.selectbox('Education', ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.selectbox('Has credit in default?', ['no', 'yes'])
housing = st.selectbox('Has housing loan?', ['no', 'yes'])
loan = st.selectbox('Has personal loan?', ['no', 'yes'])
contact = st.selectbox('Contact communication type', ['cellular', 'telephone', 'unknown'])
month = st.selectbox('Last contact month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
poutcome = st.selectbox('Outcome of previous campaign', ['unknown', 'other', 'failure', 'success'])

# When user clicks predict
if st.button('Predict'):
    # Create input DataFrame
    input_dict = {
        'age': age,
        'balance': balance,
        'day': day,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'poutcome': poutcome
    }
    input_df = pd.DataFrame([input_dict])

    # One-hot encode categorical variables with drop_first=True (like training)
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Add missing columns that were in training but not in input
    missing_cols = set(all_columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0

    # Ensure columns order matches training data
    input_encoded = input_encoded[all_columns]

    # Log-transform balance and pdays as done in training
    for col in ['balance', 'pdays']:
        input_encoded[col] = np.log1p(input_encoded[col] - input_encoded[col].min() + 1)
    # Log-transform other numerical columns
    for col in ['duration', 'campaign', 'previous']:
        input_encoded[col] = np.log1p(input_encoded[col])

    # Scale numerical columns
    input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])

    # Predict
    prediction = model.predict(input_encoded)
    probability = model.predict_proba(input_encoded)[:, 1]

    # Show results
    st.success(f"Prediction: {'Subscribed' if prediction[0] == 1 else 'Not Subscribed'}")
    st.write(f"Probability of subscription: {probability[0]:.2f}")
