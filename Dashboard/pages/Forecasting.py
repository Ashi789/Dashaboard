import streamlit as st
import pandas as pd
from datetime import timedelta,datetime,time
from sqlalchemy import create_engine
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import nltk
from st_aggrid import AgGrid, GridOptionsBuilder
import json
import re
import time
# Set page config
st.set_page_config(page_title="Finit Systems Forecasting", layout="wide", initial_sidebar_state="collapsed")
 
# Custom CSS
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;} header {visibility: hidden;}
        .st-emotion-cache-1jicfl2 {padding-left: 30px; padding-right: 30px; padding-top: 10px;background: #F1EFFA !important;}
        .st-emotion-cache-4uzi61 {  border-radius: 0.5rem;background:white;}
        .st-emotion-cache-y1zvow {border-radius: 0.5rem; border: 3px solid rgb(255, 255, 255,0.4);}
        .st-emotion-cache-6qob1r { background: linear-gradient(135deg, #5D46DC, #6E52E9, #7F68F3, #947AFD, #AB8CFF); color: white; }
        .st-emotion-cache-1jicfl2 {background: #F1EFFA; backgrount-color: rgb(240, 242, 246); !important;}
        .st-emotion-cache-11wc8as {background:#F1EFFA;}
        .st-emotion-cache-1r4qj8v {background: white;} 
        .st-emotion-cache-y1zvow {background-color:rgba(255, 255, 255, 0.7);}  
        .st-emotion-cache-ocqkz7 {gap: 20px;}
        .st-emotion-cache-isgwfk{border: 3px solid rgb(255, 255, 255); background :white}
        .st-emotion-cache-169z1n8 {gap: 20px;}
        .st-emotion-cache-6qob1r { border: 1px solid; border-radius: 0px 90px 0px 90px;}}
        .st-emotion-cache-4n4ivo {background: white;}
        .st-emotion-cache-1azekzy {background: white !important;}
        .st-emotion-cache-1puwf6r ol { font-size: 16 px !important;}
        .st-emotion-cache-p0pjm p,.st-emotion-cache-1rtdyuf,.st-emotion-cache-pkbazv { color: white !important; font-size: 30px !important;}
        .st-emotion-cache-1ibsh2c{background-color: #F1EFFA;background: #F1EFFA}
        .st-emotion-cache-13fxp4v {background-color:white !important;padding: 0px; text-align:center;}
        .st-emotion-cache-4tlk2p{font-size:22px;}
        .st-emotion-cache-1rtdyuf {display: none !important;}
        .st-emotion-cache-6tkfeg{display: none !important;}
    </style>
""", unsafe_allow_html=True)

with st.container():
        st.title("DATAR HALAL GYRO & GRILL FORECASTING")
        
# Sidebar settings for comparison
with st.sidebar:
    # Set the title and header with inline HTML to change color to white
    st.markdown('<h1 style="color:white;">Finit Systems Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<h4 style="color:white;">⚙️ Settings</h4>', unsafe_allow_html=True)
    # Add a back button with a beautiful Unicode arrow
    st.page_link(page="./Dashboard.py",use_container_width=True)

def load_data():
    # Define column names for each table
    signupuser_columns = [
        "user_id", "email", "first_name", "last_name", "phone", "password",
        "street", "city", "zip_code", "reward_points", "otp",
        "otp_expiration", "is_verified"
    ]
    
    guestusers_columns = [
        "guest_id", "first_name", "last_name", "email", "phone",
        "street", "city", "zip_code"
    ]
    
    fooditems_columns = [
        "food_item_id", "name", "description", "price", "image_url", "category"
    ]
    
    orders_columns = [
        "order_id", "customer_type", "user_id", "guest_id", "food_items",
        "total_amount", "order_timestamp", "delivery_method"
    ]
    
    feedback_columns = [
        "feedback_id", "user_id", "food_item_id", "order_id", "rating",
        "feedback_date"
    ]
    
    payments_columns = [
        "payment_id", "user_id", "guest_id", "customer_type",
        "total_amount", "clover_id", "payment_timestamp", "order_id"
    ]

    # Load data with specified column names
    signupuser_data = pd.read_csv("signupuser.csv", names=signupuser_columns, header=0)
    guestusers_data = pd.read_csv("guestusers.csv", names=guestusers_columns, header=0)
    fooditems_data = pd.read_csv("fooditems.csv", names=fooditems_columns, header=0)
    orders_data = pd.read_csv("orders.csv", names=orders_columns, header=0)
    feedback_data = pd.read_csv("feedback.csv", names=feedback_columns, header=0)
    payments_data = pd.read_csv("payments.csv", names=payments_columns, header=0)

    return signupuser_data, guestusers_data, fooditems_data, orders_data, feedback_data, payments_data
   
# Load the data and convert to DataFrames
signupuser_df, guestuser_data ,fooditem_df, orders_df, feedback_df, payment_df = load_data()
print("Orders DataFrame Columns:", orders_df.columns.tolist())  # Check actual column names
print("First few rows:\n", orders_df.head())
print("Signup user columns",signupuser_df.head())

# Data Preprocessing for Orders
orders_df['order_timestamp'] = pd.to_datetime(orders_df['order_timestamp'].replace("0000-00-00", pd.NaT), errors='coerce')
base_date = pd.Timestamp('2025-01-01')  # Base date for missing timestamps
orders_df['order_timestamp'] = orders_df['order_timestamp'].fillna(base_date)
orders_df['total_amount'] = pd.to_numeric(orders_df['total_amount'], errors='coerce').fillna(0)

orders_df['food_items'] = orders_df['food_items'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

# Extract 'food_item_name' from the first food item in each order
orders_df['food_item_name'] = orders_df['food_items'].apply(lambda x: x[0]['food_item_name'] if isinstance(x, list) and x else 'Unknown')

orders_df['quantity'] = orders_df['food_items'].apply(lambda x: x[0]['quantity'] if isinstance(x, list) and x else 0)
orders_df['user_id'] = pd.to_numeric(orders_df['user_id'], errors='coerce').fillna(-1).astype(int)
orders_df['order_id'] = pd.to_numeric(orders_df['order_id'], errors='coerce').fillna(-1).astype(int)

# Data Preprocessing for Feedback
feedback_df['feedback_date'] = pd.to_datetime(feedback_df['feedback_date'].replace("0000-00-00", pd.NaT), errors='coerce')
feedback_base_date = pd.Timestamp('2025-01-01')  # Base date for missing feedback dates
feedback_df['feedback_date'] = feedback_df['feedback_date'].fillna(feedback_base_date)
rating_mapping = {
    'poor': 1,
    'average': 2,
    'neutral': 3,
    'good': 4,
    'excellent': 5} # Map categorical ratings to numeric values (poor=1, average=2, etc.)
feedback_df['rating'] = feedback_df['rating'].replace(rating_mapping)
feedback_df['rating'] = feedback_df['rating'].fillna(3)  # Default to neutral (3) if NaN or invalid ratings
feedback_df['food_item_id'] = pd.to_numeric(feedback_df['food_item_id'], errors='coerce').fillna(-1).astype(int)
feedback_df = feedback_df[feedback_df['food_item_id'] != -1]

# Data Preprocessing for Payments
payment_df['payment_timestamp'] = pd.to_datetime(payment_df['payment_timestamp'].replace("0000-00-00", pd.NaT), errors='coerce')
payment_df['payment_time'] = payment_df['payment_timestamp'].dt.strftime('%H:%M:%S')
payment_df['user_id'] = pd.to_numeric(payment_df['user_id'], errors='coerce').fillna(-1).astype(int)
payment_df['total_amount'] = pd.to_numeric(payment_df['total_amount'], errors='coerce').fillna(0)
payment_df['payment_id'] = pd.to_numeric(payment_df['payment_id'], errors='coerce').fillna(-1).astype(int)
payment_df['order_id'] = pd.to_numeric(payment_df['order_id'], errors='coerce').fillna(-1).astype(int)


# Data Preprocessing for Signup Users
signupuser_df['first_name'] = signupuser_df['first_name'].fillna('Unknown')
signupuser_df['last_name'] = signupuser_df['last_name'].fillna('Unknown')
signupuser_df['user_id'] = pd.to_numeric(signupuser_df['user_id'], errors='coerce').fillna(-1).astype(int)
signupuser_df.loc[signupuser_df['first_name'] == '', 'first_name'] = 'Unknown'
signupuser_df.loc[signupuser_df['last_name'] == '', 'last_name'] = 'Unknown'
signupuser_df['city'] = signupuser_df['city'].fillna('Unknown')
signupuser_df.loc[signupuser_df['city'] == '', 'city'] = 'Unknown'

# Data Preprocessing for FoodItems
fooditem_df['food_item_id'] = pd.to_numeric(fooditem_df['food_item_id'], errors='coerce').fillna(-1).astype(int)
fooditem_df['name'] = fooditem_df['name'].fillna('Unknown')
fooditem_df.loc[fooditem_df['name'].str.strip() == '', 'name'] = 'Unknown'
fooditem_df = fooditem_df.drop_duplicates(subset=['food_item_id'])
fooditem_df.reset_index(drop=True, inplace=True)

# Retain original DataFrames
orders_df = orders_df
feedback_df = feedback_df
payment_df = payment_df
signupuser_df = signupuser_df
fooditem_df= fooditem_df


 
# Load the events.json file
with open("events.json", "r") as file:
    event_data = json.load(file)

# Extract events from JSON and parse dates
events = []
for event, date in event_data["events"].items():
    # Use regex to remove anything after the date (e.g., "(Opening Day)")
    date_str = re.sub(r'\(.*\)', '', date).strip()  # Remove parentheses and text inside
    try:
        # Parse the cleaned date string into a datetime object
        event_date = datetime.strptime(date_str + f" {datetime.now().year}", "%B %d %Y")
        events.append({"event": event, "date": event_date})
    except ValueError as e:
        st.error(f"Error parsing date for event {event}: {e}")
# Placeholder for the warning at the top of the page
warning_placeholder = st.empty()

def prepare_forecast_data(orders_df, payment_df, group_by_columns):
    """Prepare and aggregate the data for forecasting."""

    # Ensure group_by_columns is a list (if a single string is passed, convert it)
    if isinstance(group_by_columns, str):
        group_by_columns = [group_by_columns]

    # Merge orders and payments data on order_id
    merged_df = orders_df.merge(payment_df[['order_id', 'payment_timestamp']], on='order_id', how='left')

    # Group by the specified columns + 'payment_timestamp' and sum 'quantity'
    df_grouped = merged_df.groupby(group_by_columns + ['payment_timestamp'])['quantity'].sum().reset_index()

    return df_grouped




# Print output
# print(grouped_df)
def forecast_sales(df_grouped, group_by_column):
    """Forecast sales for the next 7 days using ARIMA."""
    forecast_data = []
    
    min_data_points = 10  # Set a threshold for the number of data points
    
    print("Unique groups:", df_grouped[group_by_column].unique())
    
    for group, data in df_grouped.groupby(group_by_column):
        print(f"Processing group: {group}, Data points: {len(data)}")
        
        if len(data) < min_data_points:
            print(f"Skipping group {group} due to insufficient data")
            continue
        
        if 'quantity' not in data.columns or data['quantity'].isnull().all():
            print(f"Skipping {group} due to missing 'quantity' values")
            continue
    
        # Set 'payment_timestamp' as index
        data.set_index('payment_timestamp', inplace=True)
        
        # Resample the data by day and sum the quantities
        data = data.resample('D').sum()  # Resampling daily
        data.fillna(0, inplace=True)  # Replace any NaN values with 0
        
        # Difference the data to make it stationary (if necessary)
        data['quantity_diff'] = data['quantity'].diff().dropna()
        
        # Fit the ARIMA model
        try:
            model = ARIMA(data['quantity'], order=(5, 1, 0))  # (p,d,q) parameters
            model_fit = model.fit()

            # Forecast for the next 7 days
            forecast = model_fit.forecast(steps=7)
            forecast_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=7).date

            forecast_data.append(pd.DataFrame({
                group_by_column: [group] * 7,
                'forecast_date': forecast_dates,
                'forecasted_sales': forecast
            }))
        except Exception as e:
            print(f"Error forecasting for group {group}: {e}")
            continue  # Skip this group instead of breaking the loop
    
    # Return the concatenated forecast data
    print(forecast_data)
    return pd.concat(forecast_data, ignore_index=True) if forecast_data else pd.DataFrame()

# Example Usage
df_grouped = prepare_forecast_data(orders_df, payment_df, ['user_id', 'food_item_name'])
forecast_df = forecast_sales(df_grouped, 'user_id')


# Accuracy evaluation (using MAPE)
def evaluate_accuracy(actual, forecasted):
    """Evaluate the accuracy of predictions using MAPE."""
    return 100 * (1 - mean_absolute_percentage_error(actual, forecasted))
 
def get_top_products_forecast(df):    
    # Step 1: Prepare the grouped data
    df_grouped = prepare_forecast_data(orders_df, payment_df, ['food_item_name'])  
    # Step 2: Forecast sales using ARIMA
    forecast_data = forecast_sales(df_grouped, 'food_item_name')
    if forecast_data is None:
        print("No forecast data generated. Exiting.")
        return None, None
    # Step 3: Get the top 5 products based on forecasted sales
    top_products = forecast_data.groupby('food_item_name')['forecasted_sales'].sum().nlargest(5).reset_index()
    return forecast_data, top_products
# Get forecast data for top products
forecast_data, top_products = get_top_products_forecast(orders_df)

def get_city_forecast(orders_df, payment_df, signupuser_df):
    """Get the city-wise forecast for 7 days."""

    # Merge the data with the city information
    orders_df = orders_df.merge(signupuser_df[['user_id', 'city']], on='user_id', how='left')

    print(f"Merged data shape: {orders_df.shape}")
    print(orders_df.head())  # Verify merged data has 'city'

    # Prepare the forecast data grouped by city
    df_grouped = prepare_forecast_data(orders_df, payment_df, ['city'])  

    print("Grouped data after preparation:", df_grouped.head())  # Check grouping

    # Get the forecast data
    forecast_data = forecast_sales(df_grouped, 'city')
    if forecast_data.empty:
        print("No forecast data generated.")
        return None, None

    print("Forecasted data:", forecast_data.head())  # Check forecast data

    # Post-process the forecast data (absolute values, rounding, etc.)
    forecast_data['forecasted_sales'] = forecast_data['forecasted_sales'].abs().round().astype(int)
    print("Processed forecast data:", forecast_data.head())  # Check processed forecast data

    # Calculate city-wise distribution
    city_distribution = forecast_data.groupby('city')['forecasted_sales'].sum().reset_index()
    print("City distribution:", city_distribution.head())  # Check city distribution

    return forecast_data, city_distribution
# Get forecast data for city distribution
city_forecast_data, city_distribution = get_city_forecast(orders_df, payment_df, signupuser_df)

# Plot the charts
# Create and display any chart
def create_chart(df, x, y=None, color=None, chart_type='line', **kwargs):
    """Create a dynamic Plotly chart."""
    if chart_type == 'line':
        fig = px.line(df, x=x, y=y, color=color, **kwargs)
       
        fig.update_traces(
            mode='lines+markers+text',
            line_shape='spline',  # Smooth the line
            text=df[y],  # Show y-values as text labels
            texttemplate='%{text:.0f}',  # Format text to show absolute values (no decimals)
            textposition='top center',
            textfont=dict(color='rgb(13,13,13)', size=18),  # Position the text above the points
            line=dict(color='#5D46DC'),  # Set line color
            marker=dict(color='#5D46DC')  # Set marker color
        )
 
       
    elif chart_type == 'bar':
        fig = px.bar(df, x=x, y=y, color=color, text=y, **kwargs)
           # Update the text on bars to show absolute values without decimals
        fig.update_traces(
            texttemplate='%{text:.0f}',  # Format text to show absolute values (no decimals)
            textposition='inside',  # Position the text inside the bars
            textfont=dict(color='white', size=20),
            marker=dict(color='#5D46DC')
        )
    elif chart_type == 'pie':
        # Define gradient colors from darkest to lightest
        gradient_colors = ['#1E0F5A', '#2F1E77', '#4A3A9E', '#6556C4', '#8477E6', '#AFA3F9', '#d2ccfc']
        
        # Sort the DataFrame by the value column in descending order
        df = df.sort_values(by=y, ascending=False).reset_index(drop=True)
        
        # Assign colors dynamically based on sorted slice sizes
        num_slices = len(df)
        slice_colors = gradient_colors[:num_slices]  # Use only as many colors as there are slices
        
        # Create the pie chart
        fig = px.pie(df, names=x, values=y, **kwargs)
        
        # Update traces with dynamic colors and text formatting
        fig.update_traces(
            texttemplate='%{value}',  # Show values inside the slices
            textfont=dict(color='white', size=20),  # Set text font size to 20 and color to white
            marker=dict(colors=slice_colors)  # Apply dynamic gradient colors
        )
    
    # Update layout to customize legend font size
    fig.update_layout(
        legend_font=dict(size=20)  # Set the font size of the legend to 20
    )

 
    # Common layout updates
    fig.update_layout(
        xaxis_title=kwargs.get('xaxis_title', None),
        yaxis_title=kwargs.get('yaxis_title', None),
        xaxis_title_font=dict(color='rgb(13,13,13)', size=20),  
        yaxis_title_font=dict(color='rgb(13,13,13)', size=20),
        template="simple_white",
        height=370,
        margin=dict(t=0, b=0, l=0, r=0),
        plot_bgcolor="rgba(0,0,0,0)" ,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(tickfont=dict(color='rgb(13,13,13)', size=18))  # Update X-axis tick color
    fig.update_yaxes(tickfont=dict(color='rgb(13,13,13)', size=18)) 
    return fig
 
col1, col2 = st.columns([2, 1])
 
# Layout for the three-column view
c1, c2 ,c3 = st.columns(3)
 
# Displaying forecasted results
def display_forecasts():
    # Forecast for top 5 products
# Create the charts for the forecasts
    top_products_fig = create_chart(top_products, x='food_item_name', y='forecasted_sales', chart_type='bar')
    city_forecast_fig = create_chart(city_distribution, x='city', y='forecasted_sales', chart_type='pie', hole=0.3)
 
    # Display product forecast chart in the first column
    with c2:
        with st.container(border=True):
            st.markdown(f"<h2 style='color: {'black'};'>Top Products Sales Forecast</h2>",unsafe_allow_html=True)
            st.plotly_chart(top_products_fig, use_container_width=True)
 
    # Display city forecast chart in the second column
    # Display sales forecasting by city in the first column
    with c1:
        with st.container(border=True):
            st.markdown(f"<h2 style='color: {'black'};'>Sales Forecasting by City</h2>",unsafe_allow_html=True)
            st.plotly_chart(city_forecast_fig, use_container_width=True)  
 
# Display forecasts
display_forecasts()
# Download necessary NLTK data
nltk.download('vader_lexicon')
 
def sentiment_analysis(rating):
    """Analyze sentiment from the rating column based on the given ratings."""
    if isinstance(rating, str):  # Check if the rating is a string
        rating = rating.lower()  # Ensure case-insensitivity
    
        # Map ratings to sentiment
        if "excellent" in rating:
            return "Positive"
        elif "good" in rating:
            return "Positive"
        elif "neutral" in rating:
            return "Neutral"
        elif "average" in rating:
            return "Neutral"
        elif "poor" in rating:
            return "Negative"
        else:
            return "Neutral"  # Default to neutral if no match
    else:
        return "Neutral"  # Default to neutral for non-string or missing values

# Merge payment_df with feedback_df to get the rating
merged_df = payment_df.merge(feedback_df[['order_id', 'rating']], on='order_id', how='left')

# Now perform sentiment analysis using the 'rating' column from the merged DataFrame
merged_df['rating'] = merged_df['rating'].apply(sentiment_analysis)

# Get the sentiment distribution for each day
sentiment_dist = merged_df.groupby(['payment_timestamp', 'rating']).size().unstack(fill_value=0)

 
# Prepare training and testing data
train_data = sentiment_dist[:-7]  # Use all but the last 7 days for training
test_data = sentiment_dist[-7:]   # Use the last 7 days for testing
 
# Forecasting using ARIMA for each sentiment
predictions = {}
for sentiment in sentiment_dist.columns:
    model = ARIMA(train_data[sentiment], order=(5, 1, 0))  # Adjust ARIMA parameters if needed
    model_fit = model.fit()
   
    forecast = model_fit.forecast(steps=7)

    if np.isnan(forecast).sum() > 0:
        print(f"NaN found in forecast for {sentiment}, replacing with 0.")
        forecast = np.nan_to_num(forecast)
    predictions[sentiment] = forecast
 
# Combine predictions into a DataFrame
predicted_sentiments = pd.DataFrame(predictions, index=pd.date_range(start=train_data.index[-1] + timedelta(days=1), periods=7))
 
# Calculate accuracy using MAPE
actual_sentiment = test_data
predicted_sentiment = predicted_sentiments
sentiment_mape = mean_absolute_percentage_error(actual_sentiment, predicted_sentiment)
 
# Data Preprocessing for SARIMA
payment_df['total_amount'] = pd.to_numeric(payment_df['total_amount'].replace('[\$,]', '', regex=True), errors='coerce')
 
# Aggregate sales data by date for forecasting
sales_data = payment_df.groupby('payment_timestamp')['total_amount'].sum().reset_index()
sales_data.set_index('payment_timestamp', inplace=True)
 
# SARIMA model for forecasting
def sarima_forecast(df, steps=7):
    """Train SARIMA model and forecast the next 'steps' days."""
    # Fit SARIMA model with tuned parameters (order and seasonal_order)
    model = SARIMAX(df['total_amount'],
                    order=(1, 1, 1),        # ARIMA order (p,d,q)
                    seasonal_order=(1, 1, 1, 7),  # Seasonal order (P,D,Q,s) s=7 for weekly seasonality
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)
   
    # Forecast for the next 7 days
    forecast = results.get_forecast(steps=steps)
    forecast_index = pd.date_range(df.index[-1] + timedelta(days=1), periods=steps, freq='D')
    forecast_values = forecast.predicted_mean
   
    # Return the forecasted data
    forecast_df = pd.DataFrame({'payment_timestamp': forecast_index, 'forecasted_sales': forecast_values})
    return forecast_df, forecast_values
 
# Forecast the next 7 days for sales
forecast_df, forecast_values = sarima_forecast(sales_data)
# Extract dates and format them
forecast_dates = forecast_df['payment_timestamp'].tolist()#
formatted_dates = [date.strftime('%Y-%m-%d') for date in forecast_dates]#


 
# Calculate MAPE for accuracy (assuming you have actual sales data for comparison)
actual_sales_last_7_days = sales_data['total_amount'].tail(7).values  # Last 7 days of actual sales
 
# Ensure we have enough actual sales data to compare with forecasts
if len(actual_sales_last_7_days) == len(forecast_values):
    sales_mape = mean_absolute_percentage_error(actual_sales_last_7_days, forecast_values)
else:
    sales_mape = None
 
 
with st.container(border = True):
    # Create and display the forecast chart
    def display_forecast(forecast_df):
        """Display the forecasted sales chart."""
        sales_fig = create_chart(forecast_df, x='payment_timestamp', y='forecasted_sales', chart_type='line')
        sales_fig.update_layout(xaxis_title=None,yaxis_title="Forecasted Sales ($)", yaxis_title_font=dict(color='black',size=21))
        st.markdown(f"""<h2 style="color: {'black'};">Sales Forecasting</h2>""",unsafe_allow_html=True)
        st.plotly_chart(sales_fig, use_container_width=True)

# Aggregating sentiment predictions for 7 days
aggregated_sentiments = predicted_sentiments.sum(axis=0)
with c3:
    with st.container(border=True):
        st.markdown(f"<h2 style='color: {'black'};'>Feedbacks Sentimental Analysis</h2>", unsafe_allow_html=True)
        
        # Define gradient colors from darkest to lightest
        gradient_colors = ['#2A1A5E', '#564D9D', '#B9B4F9']
        
        # Sort the sentiment data by values in descending order
        aggregated_sentiments = aggregated_sentiments.sort_values(ascending=False)
        
        # Convert values to integers to ensure they are displayed as whole numbers
        aggregated_sentiments = aggregated_sentiments.astype(int)
        
        # Calculate total sum for percentage calculation
        total_sentiments = aggregated_sentiments.sum()
        

        # Assign colors dynamically based on sorted slice sizes
        num_slices = len(aggregated_sentiments)
        slice_colors = gradient_colors[:num_slices]  # Use only as many colors as there are slices
        
        # Create a single pie chart for the sentiment distribution over the 7 forecasted days
        fig = px.pie(
            aggregated_sentiments, 
            values=aggregated_sentiments, 
            names=aggregated_sentiments.index, 
            hole=0.3
        )
        
        # Update layout for background and size
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=370,
            margin=dict(t=0, b=0, l=0, r=0),
            legend_font=dict(size=20),  # Set legend font size to 20
        )
        
        # Manually calculate the percentage for each value
        percentages = ((aggregated_sentiments * 100) / total_sentiments).round(1)

        # Update the text and colors inside the pie chart slices
        fig.update_traces(
            texttemplate='%{value} (' + (percentages.astype(str) + '%') + ')',  # Show absolute count and percentage
            textfont=dict(color='white', size=20),  # Set text font size inside slices to 20px and color to white
            marker=dict(colors=slice_colors)  # Dynamically assign gradient colors
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)


 
# Second column for Sales forecast (Line chart)
 
with col1:
    with st.container(border = True):
     display_forecast(forecast_df)
 
    # Display sales model accuracy (MAPE)
    if sales_mape is not None:
        #st.write(f"Sales Model Accuracy (MAPE): {sales_mape * 100:.2f}%")
       
        # Calculate individual errors for each day in the last week for more insight
        individual_errors = (actual_sales_last_7_days - forecast_values) / actual_sales_last_7_days * 100
       
        # Create a DataFrame for better visualization of individual errors
        error_df = pd.DataFrame({
            'Date': forecast_df['payment_timestamp'],
            'Actual Sales': actual_sales_last_7_days,
            'Forecasted Sales': forecast_values,
            'Error (%)': individual_errors,
        })
   
    # Create a combined DataFrame to show actual and predicted values along with percentage error in one table
    combined_df = pd.DataFrame({
        'Date': forecast_df['payment_timestamp'],
        'Actual Sales': actual_sales_last_7_days,
        'Forecasted Sales': forecast_values,
    })
 
 
# Calculate total predicted sales and total predicted products sold
predicted_total_sales = forecast_values.sum()  # Total forecasted sales
# Call the function and store the results in variables
forecast_data, top_products = get_top_products_forecast(orders_df)

# Now you can access the 'forecasted_sales' column in the 'top_products' DataFrame
predicted_total_products_sold = top_products['forecasted_sales'].sum()
  # Total forecasted products sold
 
# Create a DataFrame for displaying predicted totals
predicted_totals_df = pd.DataFrame({
    'Metric': ['Predicted Total Sales ($)', 'Predicted Total Products Sold'],
    'Value': [predicted_total_sales, predicted_total_products_sold]
})
 
with col2:
    with st.container(height=503, border=True):
        # CSS styling for custom metrics
        st.markdown("""
            <style>
                .custom-metric-container {
                    text-align: center;
                    padding: 10px;
                    border: None;
                    border-radius: 8px;
                }
                .custom-metric-label {
                    font-size: 20px;
                   
                    color: rgb(13,13,13);
                }
                .custom-metric-value {
                    font-size: 32px;
                    
                    color: rgb(13,13,13);
                }
                .custom-metric-delta {
                    font-size: 20px;
                    color: green; /* Adjust for positive or negative dynamically */
                }
                .custom-metric-delta-negative {
                    color: red;
                    font-size: 20px;
                }
            </style>
        """, unsafe_allow_html=True)

        # Display total sales at the center
        c1, c2, c3 = st.columns(3)
        with c2:
            st.markdown(f"""<h2 style="color: {'black'};text-align: center;">Total Sales</h2>""",unsafe_allow_html=True)
            st.markdown(f"""
                <div class="custom-metric-container">
                    <div class="custom-metric-value">${int(predicted_total_sales)}</div>
                </div>
            """, unsafe_allow_html=True)

        # Dynamically compare deltas and assign delta_color
        c1, c2, c3 = st.columns(3)
        # Display the first metric
        with c1:
            st.markdown(f"""
                <div class="custom-metric-container">
                    <div class="custom-metric-label">{formatted_dates[0]}</div>
                    <div class="custom-metric-value">${abs(int(forecast_values[0]))}</div>
                    <div class="custom-metric-delta">${abs(int(forecast_values[0]))}</div>
                </div>
            """, unsafe_allow_html=True)

        for i in range(1, 7):
            delta = forecast_values[i] - forecast_values[i - 1]
            percentage_change = (delta / forecast_values[i - 1]) * 100 if forecast_values[i - 1] != 0 else 0

            # Determine the delta style based on whether it's positive or negative
            delta_style = "custom-metric-delta-negative" if delta < 0 else "custom-metric-delta"
            delta_value = f"{'-' if delta < 0 else ''}${abs(int(delta))} ({abs(percentage_change):.2f}%)"

            # Cycle through columns (c1, c2, c3)
            col_index = [c1, c2, c3][i % 3]
            with col_index:
                st.markdown(f"""
                    <div class="custom-metric-container">
                        <div class="custom-metric-label">{formatted_dates[i]}</div>
                        <div class="custom-metric-value">${abs(int(forecast_values[i]))}</div>
                        <div class="{delta_style}">{delta_value}</div>
                    </div>
                """, unsafe_allow_html=True)


# Check for matches and display warnings
for forecasted_date in forecast_dates:
    for event in events:
        if forecasted_date.date() == event["date"].date():
            # Display the warning message at the top of the page
            warning_placeholder.warning(
                f"Forecasted date {forecasted_date.strftime('%B %d')} coincides with event: {event['event']}!  This event might affect your future sales.",icon="⚠️", )
            # Keep the warning visible for 5 seconds
            time.sleep(5)
            # Clear the warning message
            warning_placeholder.empty()
