import streamlit as st
import pandas as pd
from datetime import timedelta
from sqlalchemy import create_engine
import plotly.express as px
import datetime
import json
 
# Set page config
st.set_page_config(page_title="Datar Halal Gyro & Grill", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;} header {visibility: hidden;}
        .st-emotion-cache-1jicfl2 { padding-left: 30px; padding-right: 30px; padding-top: 10px;}
        .st-emotion-cache-4uzi61 {  border-radius: 0.5rem;}
        .st-emotion-cache-y1zvow { border-radius: 0.5rem; border: 3px solid rgb(255, 255, 255);}
        .st-emotion-cache-6qob1r {background: linear-gradient(135deg, #5D46DC, #6E52E9, #7F68F3, #947AFD, #AB8CFF); color: white; /* Optional: Change text color in the sidebar */ }   
        .st-emotion-cache-11wc8as {background-color: #F1EFFA}
        .st-emotion-cache-ocqkz7 {gap: 20px;}
        .st-emotion-cache-isgwfk{border: 3px solid rgb(255, 255, 255); background :white}
        .st-emotion-cache-169z1n8 {gap: 20px;}
        .st-emotion-cache-1jicfl2 {background: #F1EFFA}
        .st-emotion-cache-6qob1r { border: 1px solid rgb(255, 255, 255); border-radius: 0px 90px 0px 40px;}
        .st-emotion-cache-1oflzbf {background:white;}
        .st-emotion-cache-1rtdyuf { color: white ! important; font-size: 30px ! important;}
        .st-emotion-cache-p0pjm p{color: white ! important;font-size: 25px;}
        .st-emotion-cache-pkbazv,.st-emotion-cache-1puwf6r p{font-size: 20px; color: white !important;}
        .st-emotion-cache-4tlk2p {color: white !important;font-size: 22px !important;}
        .st-emotion-cache-1rtdyuf {display: none !important;}
        .st-emotion-cache-6tkfeg{display: none !important;}
        .st-emotion-cache-fsammq p {color: rgb(255,255,255) !important;font-size: 22px !important;}
        .st-emotion-cache-1mw54nq h3 {font-size: 25px !important;}
        .st-emotion-cache-1r4qj8v{background-color: #F1EFFA;background: #F1EFFA}
        
}       
    </style>
""", unsafe_allow_html=True)

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



st.title("Finit Systems")

def format_with_commas(number):
    """Format numbers with commas for better readability (no decimals, absolute values)."""
    try:
        # Convert to absolute integer and format with commas
        return f"{abs(int(float(number))):,}"
    except (ValueError, TypeError):
        # Return the input as-is if it's not a valid number
        return str(number)

 
def smooth_series(series, window=1):
    """Smooth the series using a rolling average."""
    return series.rolling(window=window, min_periods=1).mean()
# Function to apply hover settings
def apply_hover(fig, x_label, y_label, bgcolor="rgba(144, 238, 144, 0.4)", font_color="black"):
    fig.update_traces(
        hovertemplate=f"<span style='color:{font_color}'><b>{x_label}:</b> <b>%{{x}}<br>"
                      f"<b>{y_label}:</b> <b>%{{y}}</span>",
        hoverlabel=dict(bgcolor=bgcolor, font=dict(color=font_color)))
    return fig
 
def create_chart(df, x, y=None, color=None, chart_type='line', smooth=False, smoothing_window=1, **kwargs):
    try:
        if chart_type == 'line' and smooth:
            df[y] = smooth_series(df[y], window=smoothing_window)
            fig = px.line(df, x=x, y=y, color=color, color_discrete_sequence=['#5D46DC'], **kwargs)
            fig.update_traces(mode="lines+markers", line_shape="spline")
        elif chart_type == 'bar':
            fig = px.bar(df, x=x, y=y, color=color, color_discrete_sequence=['#5D46DC'], text=y, **kwargs)
            fig.update_traces(texttemplate='%{text:.2s}', textposition='inside', textfont=dict(size=20, color='white', family='Arial'))
            fig = apply_hover(fig, x_label=kwargs.get('xaxis_title', x), y_label=kwargs.get('yaxis_title', y))
        elif chart_type == 'pie':
            fig = px.pie(df, names=x, values=y, **kwargs)

        # Update layout
        fig.update_layout(
            title=dict(text=kwargs.get('title', ""), font=dict(size=30, color="black"), x=0.5, xanchor='center'),
            xaxis_title=kwargs.get('xaxis_title', x),
            yaxis_title=kwargs.get('yaxis_title', y),
            template="simple_white",
            height=245,
            margin=dict(t=40, b=0, l=0, r=1),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        fig.update_xaxes(
            title_font=dict(size=30, color='black'),
            tickfont=dict(size=18, color='black')
        )
        fig.update_yaxes(
            title_font=dict(size=30, color='black'),
            tickfont=dict(size=18, color='black')
        )
        return fig
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None


 
 
def display_metric(col, title, chart, value=None):
    """Display a metric with a bordered container and an optional chart."""
    with col:
        with st.container(border = True,height=340):
            st.markdown(f"<p style='font-size: 30px; font-weight: bold;color:rgb(13,13,13); text-align: center;'>{title}</p>", unsafe_allow_html=True)
            if chart:  
                st.plotly_chart(chart, use_container_width=True)  # Show chart if available
            else:
                st.warning(f"No data available for the selected date range for the {title.lower()} chart.")  # Show warning inside container
 
def render_table_with_styling(df):

    st.dataframe(df) 
 
cols = st.columns(3)
# Sidebar settings for date selection
with st.sidebar:
        st.title("Finit Systems Dashboard")
        st.header("⚙ Settings")
        # Link to the second page
        st.page_link(
            page="pages/Forecasting.py",use_container_width=True)
        
        
        # Ensure orders_df is not empty before proceeding
        # Check if orders_df exists and has data
        if orders_df.empty or orders_df['order_timestamp'].dropna().empty:
            st.warning("No data available in orders for the given date range. Please select a date manually.")

            # Allow manual date input
            today = datetime.date.today()
            min_date = today - datetime.timedelta(days=365)  # Allow selection up to a year ago
            max_date = today
        else:
            # Drop NaT values to avoid errors
            orders_df = orders_df.dropna(subset=['order_timestamp'])

            # Define max and min dates based on the cleaned data
            max_date = orders_df['order_timestamp'].max().date()
            min_date = orders_df['order_timestamp'].min().date()

            # Ensure max_date is not earlier than min_date
            if min_date > max_date:
                st.error("Error: No valid date range available in the dataset.")
                st.stop()  # Stop execution if no valid data exists

        # Set default date range
        default_end_date = max_date if 'max_date' in locals() else datetime.date.today()
        default_start_date = default_end_date - datetime.timedelta(days=6)

        st.subheader("Primary Date Range")
        start_date = st.date_input("From", default_start_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("To", default_end_date, min_value=min_date, max_value=max_date)

        # Validate date range for the primary dataset
        if start_date > end_date:
            st.error("'From' date cannot be after 'To' date!")

        # Checkbox for enabling comparison
        comparison_enabled = st.checkbox("Enable Comparison")

        # Initialize comparison date variables
        comp_start_date = None
        comp_end_date = None

        if comparison_enabled:
            st.subheader("Comparison Date Range")
            comp_start_date = st.date_input("Comparison From", default_start_date, min_value=min_date, max_value=max_date)
            comp_end_date = st.date_input("Comparison To", default_end_date, min_value=min_date, max_value=max_date)
         # Validate date range for the comparison dataset
            if comp_start_date > comp_end_date:
                st.error("'Comparison From' date cannot be after 'Comparison To' date!")
            st.markdown(
                """
                <div style="display: flex; align-items: center; margin-top: 5px;">
                    <div style="width: 25px; height: 25px; background-color: #5D46DC; margin-right: 10px; font-size: 30px !important;"></div>
                    <span>Present</span>
                </div>
                <div style="display: flex; align-items: center; margin-top: 5px;">
                    <div style="width: 25px; height: 25px; background-color: #01E1D7; margin-right: 10px; font-size: 30px !important;"></div>
                    <span>Historical</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
def create_and_display_charts(df_filtered, second_df_filtered, cols):
    # Generate sales distribution chart
    sales_fig = sales_distribution_chart(df_filtered, second_df_filtered)
    feedback_fig = feedback_distribution_chart(df_filtered, second_df_filtered)
    top_products_fig = top_products_chart(df_filtered, second_df_filtered)
    time_category_fig = time_of_day_chart(df_filtered, second_df_filtered)
    people_visited_fig = people_visited_chart(df_primary, df_comparison=None)
    city_bar_chart = city_distribution_chart(df_filtered, second_df_filtered)

    display_metric(cols[0], "Sales", sales_fig)
    display_metric(cols[1], "Feedback", feedback_fig)

    with cols[2]:
        with st.container(height=340,border=True):
            st.markdown(f"<p style='font-size: 30px; color:rgb(13,13,13); font-weight: bold; text-align: center;'>Total Information</p>", unsafe_allow_html=True)
            # Check if the primary data is available
            # Initialize metrics variable
            metrics = None  

            # Check if the primary data is available
            if df_filtered is None or df_filtered.empty:
                st.warning("⚠ No data available for the selected date range.")
            elif comparison_enabled and (second_df_filtered is None or second_df_filtered.empty):
                st.warning("⚠ No data available for the selected comparison date range.")
            else:
                # Compute metrics only when data is available
                metrics = total_information(
                    df_filtered, 
                    second_df_filtered, 
                    comparison_enabled=comparison_enabled, 
                    start_date=start_date, 
                    end_date=end_date, 
                    comp_start_date=comp_start_date, 
                    comp_end_date=comp_end_date
                )

            # Only render if metrics is available
            if metrics is not None:
                render_total_information(metrics, comparison_enabled)

    display_metric(cols[0], "Top Products Sold", top_products_fig)
    display_metric(cols[1], "Time of Day Distribution", time_category_fig)

    # Display Loyal Customers Table
    with cols[2]:
        with st.container(height=340, border=True):
            loyal_customers = loyal_customers_table(df_filtered, second_df_filtered, comparison_enabled)
            display_loyal_customers_leaderboard(loyal_customers, comparison_enabled)


    display_metric(cols[0], "Visitors", people_visited_fig)
    display_metric(cols[1], "Sales by City", city_bar_chart)

#     # Display Top Feedback by Product Table
    with cols[2]:
        with st.container(height=340, border=True):
            top_feedbacks = top_feedbacks_table(df_filtered,second_df_filtered, comparison_enabled, fooditem_df, feedback_df)
            product_feedback_leaderboard_with_styles(top_feedbacks, comparison_enabled)
                  

def sales_distribution_chart(df_primary, df_secondary=None):
    """
    Generates a sales distribution chart for primary and optionally secondary datasets,
    using a "From" and "To" date range logic.
    """
    try:
        def calculate_sales_by_date(df):
            """Helper function to calculate total sales grouped by date."""
            if df is None or df.empty:
                raise ValueError("Dataframe is empty or None")
            # Ensure 'order_timestamp' column is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(df['order_timestamp']):
                df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])
            sales_by_date = df.groupby(df['order_timestamp'].dt.date)['total_amount'].sum()
            return sales_by_date

        # Calculate sales for the primary dataset
        sales_primary = calculate_sales_by_date(df_primary)
        primary_dates = pd.to_datetime(sales_primary.index)  # Ensure the index is a DatetimeIndex
        primary_labels = primary_dates.strftime('%b %d').tolist()

        # Create the primary chart
        fig = create_chart(
            sales_primary.reset_index(),
            x=sales_primary.index.name,
            y='total_amount',
            chart_type='line',
            smooth=True
        )

        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Sales ($)",
            xaxis_title_font=dict(color='black'),
            yaxis_title_font=dict(size=20, color='rgb(13,13,13)')
        )

        # Customize x-axis ticks for the primary dataset
        x_ticks = primary_dates
        x_labels = primary_labels

        # If a secondary dataset is provided, process and plot it
        if df_secondary is not None:
            sales_secondary = calculate_sales_by_date(df_secondary)
            secondary_dates = pd.to_datetime(sales_secondary.index)  # Ensure the index is a DatetimeIndex
            secondary_labels = secondary_dates.strftime('%b %d').tolist()

            # Combine primary and secondary dates for the x-axis
            x_ticks = x_ticks.union(secondary_dates)
            x_labels = x_labels + [label for label in secondary_labels if label not in x_labels]

            # Add the secondary dataset to the chart
            fig_secondary = create_chart(
                sales_secondary.reset_index(),
                x=sales_secondary.index.name,
                y='total_amount',
                chart_type='line',
                smooth=True
            )

            # Add secondary trace to the chart
            fig.add_traces(fig_secondary.data)

            # Apply custom styles to each trace
            fig.data[0].update(line=dict(color='#5D46DC', dash='solid'))
            fig.data[1].update(line=dict(color='#01E1D7', dash='dash'))
            fig.data[1].update(
                hovertemplate='<b>Date: %{x}</b><br><b>Total Sales: $%{y}</b><extra></extra>'
            )

        # Update the x-axis with combined ticks and labels
        fig.update_xaxes(
            tickvals=x_ticks,
            ticktext=x_labels,
        )

        # Customize hovertemplate for primary trace
        fig.update_traces(
            hovertemplate='<b>Date: %{x}</b><br><b>Total Sales: $%{y}</b><extra></extra>',
            line=dict(width=2)
        )

        return fig

    except ValueError as ve:
        print(f"Data error: {str(ve)}")
    except Exception as e:
        print(f"An error occurred while generating the sales distribution chart: {str(e)}")

    
def feedback_distribution_chart(df_primary, df_secondary=None):
    try:
        # Ensure 'feedback_df' has 'feedback_date' for merging
        if 'feedback_date' not in feedback_df.columns:
            raise KeyError("'feedback_date' column not found in feedback_df.")

        # Add a date column to both df_primary and feedback_df to facilitate date comparison
        df_primary['order_date'] = df_primary['order_timestamp'].dt.date
        feedback_df['feedback_date_only'] = feedback_df['feedback_date'].dt.date

        # Merge df_primary with feedback_df based on the date columns
        df_primary_with_feedback = df_primary.merge(
            feedback_df[['feedback_date_only', 'rating']], 
            left_on='order_date', 
            right_on='feedback_date_only', 
            how='left'
        )

        

        # Filter the primary dataset
        df_primary_filtered = df_primary_with_feedback[(
            df_primary_with_feedback['order_timestamp'] >= pd.Timestamp(start_date)) & 
            (df_primary_with_feedback['order_timestamp'] <= pd.Timestamp(end_date))
        ]

        # Ensure the filtered dataset is not empty
        if df_primary_filtered.empty:
            raise ValueError()#No data available for the selected primary date range.
        # Calculate feedback distribution for the primary dataset
        feedback_primary = df_primary_filtered.groupby('rating').size().reset_index(name='Count')
        c_type = 'pie' if not comparison_enabled else 'bar'

        # Helper function to assign predefined colors
        def assign_predefined_colors(data, colors):
            n = len(data)
            assigned_colors = (colors * (n // len(colors) + 1))[:n]
            return assigned_colors

        if c_type == 'pie':
            feedback_primary = feedback_primary.sort_values(by='Count', ascending=False).reset_index(drop=True)
            predefined_colors = ['#322875', '#45379e', '#5e4ec7', '#7c6ce6', '#b4a8ff']
            colors = assign_predefined_colors(feedback_primary, predefined_colors)
            feedback_fig = create_chart(
                feedback_primary, 
                x='rating', 
                y='Count', 
                chart_type=c_type
            )
            feedback_fig.update_traces(
                hole=0.4,
                text=feedback_primary['Count'].values, 
                textposition='inside', 
                textfont=dict(size=16, color='white'), 
                marker=dict(colors=colors)
            )
        elif c_type == 'bar':  
            feedback_fig = create_chart(
                feedback_primary, 
                x='Count', 
                y='rating', 
                chart_type='bar', 
                orientation='h'
            )

        feedback_fig.update_traces(text=feedback_primary['Count'].values, textposition='auto')

        # Process secondary dataset if provided
        if df_secondary is not None:
            df_secondary['order_date'] = df_secondary['order_timestamp'].dt.date
            df_secondary['order_timestamp'] = pd.to_datetime(df_secondary['order_timestamp'], errors='coerce')

            # Ensure the merge keys are both of type `datetime64[ns]` or both `object`
            df_secondary_with_feedback = df_secondary.merge(
                feedback_df[['feedback_date_only', 'rating']], 
                left_on='order_date', 
                right_on='feedback_date_only', 
                how='left'
            )

            if df_secondary_with_feedback.empty:
                raise ValueError("The merge for the secondary dataset resulted in an empty DataFrame.")

            df_secondary_filtered = df_secondary_with_feedback[(
                df_secondary_with_feedback['order_timestamp'] >= pd.Timestamp(comp_start_date)) & 
                (df_secondary_with_feedback['order_timestamp'] <= pd.Timestamp(comp_end_date))
            ]
            
            if df_secondary_filtered.empty:
                raise ValueError("No data available for the selected secondary date range.")

            feedback_secondary = df_secondary_filtered.groupby('rating').size().reset_index(name='Count')
            feedback_fig_secondary = create_chart(
                feedback_secondary, 
                x='Count', 
                y='rating', 
                chart_type='bar', 
                orientation='h'
            )
            feedback_fig_secondary.update_traces(text=feedback_secondary['Count'].values, textposition='auto')

            # Add the secondary trace to the chart
            feedback_fig.add_traces(feedback_fig_secondary.data)

            # Apply custom colors
            feedback_fig.data[0].update(marker=dict(color='#5D46DC'))
            feedback_fig.data[1].update(marker=dict(color='#01E1D7'))

        feedback_fig.update_layout(
            barmode='group',
            xaxis_title=None, 
            yaxis_title=None,
            margin=dict(t=5, b=0, l=0, r=0),
            legend=dict(
                font=dict(size=18),
                title=dict(font=dict(size=18))
            )
        )

        return feedback_fig

    except KeyError as ke:
        st.error(f"KeyError: {str(ke)}")
    except ValueError as ve:
        print(f"ValueError: {str(ve)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def total_information(df_filtered, second_df_filtered=None, comparison_enabled=False, start_date=None, end_date=None, comp_start_date=None, comp_end_date=None):
    # Helper function to calculate total sales
    def calculate_total_sales(df):
        """Calculates total sales by summing the total_amount column."""
        try:
            return df['total_amount'].sum()
        except KeyError:
            st.error("Error: 'total_amount' column not found in the dataset.")
            return 0
        except Exception as e:
            st.error(f"Error calculating total sales: {e}")
            return 0

    # Function to fetch and sum active users over the given date range
    def fetch_active_users_sum(start_date, end_date):
        try:
            # Ensure start_date and end_date are not None
            if start_date is None or end_date is None:
                # st.error("Error fetching data: start_date or end_date is None.")
                return 0

            # Convert dates to string format (YYYY-MM-DD)
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")

            # Define the date range request
            date_range = DateRange(start_date=start_date_str, end_date=end_date_str)
            request_data = RunReportRequest(
                property=f"properties/{PROPERTY_ID}",
                dimensions=[Dimension(name="date")],  # Get data per day
                metrics=[Metric(name="activeUsers")],
                date_ranges=[date_range]
            )

            # Fetch data from Google Analytics
            response = client.run_report(request_data)

            # Parse response into a DataFrame
            data = [{"date": row.dimension_values[0].value, "active_users": int(row.metric_values[0].value)}
                    for row in response.rows]
            df = pd.DataFrame(data)

            # Sum all active users correctly
            return df['active_users'].sum() if not df.empty else 0

        except KeyError as ke:
            st.error(f"KeyError: {str(ke)}")
            return 0
        except ValueError as ve:
            st.error(f"ValueError: {str(ve)}")
            return 0
        except Exception as e:
            # st.error(f"Error fetching data: {e}")
            return 0

    # Check if required columns exist in the filtered DataFrame
    if 'total_amount' not in df_filtered.columns or 'quantity' not in df_filtered.columns:
        st.error("Error: 'total_amount' or 'quantity' columns not found in the primary dataset.")
        return {}

    # Calculate metrics for the primary dataset
    metrics = {
        "Primary": {
            "Total Sales ($)": f"${format_with_commas(abs(float(calculate_total_sales(df_filtered))))}",
            "Total People Visited": format_with_commas(abs(int(fetch_active_users_sum(start_date, end_date)))),  # Correct summation
            "Total Products Sold": format_with_commas(abs(int(df_filtered['quantity'].sum()))),
        }
    }

    # If comparison is enabled, calculate comparison metrics
    if comparison_enabled and second_df_filtered is not None:
        # Ensure second_df_filtered has the required columns
        if 'total_amount' not in second_df_filtered.columns or 'quantity' not in second_df_filtered.columns:
            st.error("Error: 'total_amount' or 'quantity' columns not found in the comparison dataset.")
            return metrics  # Return primary metrics even if comparison fails

        if comp_start_date is None or comp_end_date is None:
            st.error("Comparison date range is incomplete. Please provide both start and end dates.")
            return metrics  # Return primary metrics even if comparison fails
        
        # Debugging: Print the comparison start and end dates
        print(f"Comparison Start Date: {comp_start_date}, Comparison End Date: {comp_end_date}")

        metrics["Comparison"] = {
            "Total Sales ($)": f"${format_with_commas(abs(float(calculate_total_sales(second_df_filtered))))}",
            "Total People Visited": format_with_commas(abs(int(fetch_active_users_sum(comp_start_date, comp_end_date)))),  # Correct summation
            "Total Products Sold": format_with_commas(abs(int(second_df_filtered['quantity'].sum()))),
        }

    return metrics




# Frontend: Display Total Information in a Single Container with Styled Format
def render_total_information(metrics, comparison_enabled):
    """
    Render primary and comparison metrics inside three separate cards for each metric.

    Args:
        metrics (dict): Dictionary containing metrics for primary and comparison datasets.
        comparison_enabled (bool): Whether comparison metrics should be displayed.
    """
    # Add custom CSS for card styling
    st.markdown("""
<style>
    .card {
        background: linear-gradient(135deg, #5D46DC, #6E52E9, #7F68F3, #947AFD, #AB8CFF); !important ;
        border-radius: 16px 0 16px 0;
        padding: 5px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 200px;
        width: 100%;  /* Ensures the cards are full-width */
    }
    .card h3 {
        font-size: 24px !important;
        margin-bottom: 10px !important;
        color: white !important;
        text-align: center !important;
    }
    .card p {
        font-size: 22px !important;
        margin: 5px 0;
        color: white;
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

    cols = st.columns(3)  # Three columns for the cards

    # Render Total Sales Card
    total_sales_card_content = '<div class="card"><h3> Sales<br></h3>'
    total_sales_card_content += f'<p><b><strong>Present:</strong> {metrics["Primary"]["Total Sales ($)"]}<b><p>'
    if comparison_enabled:
        total_sales_card_content += f'<p><strong>Historical: {metrics["Comparison"]["Total Sales ($)"]}</strong><p>'
    total_sales_card_content += '</div>'
    with cols[0]:
        st.markdown(total_sales_card_content, unsafe_allow_html=True)

    # Render Total People Visited Card
    total_people_visited_card_content = '<div class="card"><h3>Visitors<br></h3>'
    total_people_visited_card_content += f'<p><b> <strong>Present:</strong> {metrics["Primary"]["Total People Visited"]}<b><p>'
    if comparison_enabled:
        total_people_visited_card_content += f'<p><b><strong>Historical:</strong> {metrics["Comparison"]["Total People Visited"]}<b><p>'
    total_people_visited_card_content += '</div>'
    with cols[1]:
        st.markdown(total_people_visited_card_content, unsafe_allow_html=True)

    # Render Total Products Sold Card
    total_products_sold_card_content = '<div class="card"><h3> Products Sold<br></h3>'
    total_products_sold_card_content += f'<p><b> <strong>Present:</strong> {metrics["Primary"]["Total Products Sold"]}<b><p>'
    if comparison_enabled:
        total_products_sold_card_content += f'<p><b><strong>Historical:</strong> {metrics["Comparison"]["Total Products Sold"]}<b><p>'
    total_products_sold_card_content += '</div>'
    with cols[2]:
        st.markdown(total_products_sold_card_content, unsafe_allow_html=True)


def top_products_chart(df_filtered, df_filtered_secondary=None):
    try:
        # Ensure required columns are present in the primary dataset
        if 'order_timestamp' not in df_filtered.columns or 'food_item_name' not in df_filtered.columns or 'quantity' not in df_filtered.columns:
            raise KeyError("Primary dataset is missing required columns ('order_timestamp', 'food_item_name', 'quantity').")

        # Filter primary dataset by selected date range
        df_primary = df_filtered[
            (df_filtered['order_timestamp'] >= pd.Timestamp(start_date)) & 
            (df_filtered['order_timestamp'] <= pd.Timestamp(end_date))
        ]

        # Check if primary dataset is empty
        if df_primary.empty:
            st.warning(f"No data available for the selected date range for the top products chart.")
            return None

        # Calculate top products for the primary dataset
        top_products = df_primary.groupby('food_item_name')['quantity'].sum().nlargest(5)

        # Ensure there are valid top products data
       
        # Create the bar chart for the primary dataset with reduced bar width
        top_products_fig = px.bar(
            top_products.reset_index(),
            x='quantity',
            y='food_item_name',
            orientation='h',
            text='quantity',
            labels={'food_item_name': 'Product', 'quantity': 'Quantity'},
        )
        top_products_fig.update_traces(texttemplate='%{text}', textposition='auto', textfont=dict(size=18, color='white'))  # Set text font size and color)

        # Apply color to the primary dataset bars
        top_products_fig.update_traces(marker=dict(color='#5D46DC'))  # Primary dataset color

        # Apply the bar width reduction and chart size
        top_products_fig.update_layout(
            barmode='group',  # Bars appear side by side
            yaxis_title=None,
            xaxis_title=None,
            margin=dict(t=0, b=0, l=0, r=0),
            bargap=0.1,  # Reduce the gap between bars
            bargroupgap=0.1,  # Reduce the gap between grouped bars
            height=248,  # Set a fixed height for the chart
            width=200,   # Set a fixed width for the chart
            xaxis=dict(
                tickfont=dict(size=18, color='black')  # Set x-axis ticks font size and color
            ),
            yaxis=dict(
                tickfont=dict(size=18, color='black')  # Set y-axis ticks font size and color
            ),
        )

        # If a secondary dataset is provided, filter it and calculate top products
        if df_filtered_secondary is not None:
            # Ensure required columns are present in the secondary dataset
            if 'order_timestamp' not in df_filtered_secondary.columns or 'food_item_name' not in df_filtered_secondary.columns or 'quantity' not in df_filtered_secondary.columns:
                raise KeyError("Secondary dataset is missing required columns ('order_timestamp', 'food_item_name', 'quantity').")

            df_secondary = df_filtered_secondary[
                (df_filtered_secondary['order_timestamp'] >= pd.Timestamp(comp_start_date)) & 
                (df_filtered_secondary['order_timestamp'] <= pd.Timestamp(comp_end_date))
            ]

            # # Check if secondary dataset is empty
            

            # Calculate top products for the secondary dataset
            top_products_secondary = df_secondary.groupby('food_item_name')['quantity'].sum().nlargest(5)

           

            # Create the bar chart for the secondary dataset
            top_products_fig_secondary = px.bar(
                top_products_secondary.reset_index(),
                x='quantity',
                y='food_item_name',
                orientation='h',
                text='quantity',
                labels={'food_item_name': 'Product', 'quantity': 'Quantity'},
            )
            top_products_fig_secondary.update_traces(texttemplate='%{text}', textposition='auto', textfont=dict(size=18, color='white'))  # Set text font size and color)

            # Combine both datasets in the chart
            top_products_fig.add_traces(top_products_fig_secondary.data)

            # Style traces to differentiate between primary and secondary data
            top_products_fig.data[1].update(marker=dict(color='#01E1D7'))  # Secondary dataset color

        return top_products_fig

    except KeyError as ke:
        st.error(f"KeyError: {str(ke)}")
        return px.bar().update_layout(title="Error in data columns.")
    except ValueError as ve:
        st.error(f"ValueError: {str(ve)}")
        return px.bar().update_layout(title="Error in data processing.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return px.bar().update_layout(title="Unexpected error occurred.")


def categorize_time_of_day(hour):
    """
    Categorize the hour into time-of-day categories, including 'Noon'.
    """
    if 5 <= hour < 12:
        return "Morning"
    elif hour == 12:
        return "Noon"
    elif 13 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

def time_of_day_chart(df_filtered, df_filtered_secondary=None):
    # Ensure 'order_timestamp' is a datetime object
    df_filtered['order_timestamp'] = pd.to_datetime(df_filtered['order_timestamp'], errors='coerce')

    if df_filtered.empty or df_filtered['order_timestamp'].dropna().empty:
        st.warning("No data available for the selected date range in the primary dataset.")
        return None  # Return None to prevent errors

    # Extract the hour from 'order_timestamp' and categorize it
    df_filtered['time_category'] = df_filtered['order_timestamp'].dt.hour.map(categorize_time_of_day)

    # Filter primary dataset based on the selected date range from 'order_timestamp'
    df_primary_filtered = df_filtered[
        (df_filtered['order_timestamp'] >= pd.Timestamp(start_date)) & 
        (df_filtered['order_timestamp'] <= pd.Timestamp(end_date))
    ]
     
    # Calculate time of day distribution for the primary dataset
    time_category_data = df_primary_filtered.groupby('time_category').size().reset_index(name='Count')
    
    # Create the bar chart for the primary dataset
    time_category_fig = px.bar(
        time_category_data,
        x='time_category',
        y='Count',
        category_orders={"time_category": ["Morning", "Noon", "Afternoon", "Evening", "Night"]}
    )
    
    # Customize the color of primary bars and remove the legend
    time_category_fig.update_traces(
        marker_color="#5D46DC",
        text=time_category_data['Count'],  # Show count on bars
        textposition='inside',  # Position text above the bars
         textfont=dict(size=20, color='white')  # Set text font size and color
    )
    
    # Remove the title and update layout with axis lines
    time_category_fig.update_layout(
        xaxis_title=None,  # Remove the x-axis title
        yaxis_title='Number of People',  # Remove the y-axis title
        yaxis_title_font=dict(size=22, color='black'), 
        xaxis=dict(
            showgrid=False,  # Remove vertical gridlines
            tickfont=dict(size=18, color='black'),  # Set x-axis tick font size and color
            showline=True,  # Show x-axis line
            linecolor='black',  # Set x-axis line color
            linewidth=2  # Set x-axis line width
        ),
        yaxis=dict(
            showgrid=False,  # Remove horizontal gridlines
            tickfont=dict(size=18, color='black'),  # Set y-axis tick font size and color
            showline=True,  # Show y-axis line
            linecolor='black',  # Set y-axis line color
            linewidth=2  # Set y-axis line width
        ),
        height=250,  # Set height for the graph
        margin=dict(t=0, b=0, l=0, r=0),
        barmode='group'
    )

    # If a secondary dataset is provided, filter it and calculate its time of day distribution
    if df_filtered_secondary is not None:
        df_filtered_secondary['order_timestamp'] = pd.to_datetime(df_filtered_secondary['order_timestamp'], errors='coerce')
        # Check if secondary dataset is empty
        if df_filtered_secondary.empty or df_filtered_secondary['order_timestamp'].dropna().empty:
            st.warning("No data available for the selected date range in the comparison dataset.")
            return time_category_fig
        df_filtered_secondary['time_category'] = df_filtered_secondary['order_timestamp'].dt.hour.map(categorize_time_of_day)

        df_secondary_filtered = df_filtered_secondary[
            (df_filtered_secondary['order_timestamp'] >= pd.Timestamp(comp_start_date)) & 
            (df_filtered_secondary['order_timestamp'] <= pd.Timestamp(comp_end_date))
        ]
        
        time_category_data_secondary = df_secondary_filtered.groupby('time_category').size().reset_index(name='Count')

        time_category_fig_secondary = px.bar(
            time_category_data_secondary,
            x='time_category',
            y='Count',
            category_orders={"time_category": ["Morning", "Noon", "Afternoon", "Evening", "Night"]}
        )

        # Customize the color of secondary bars and remove the legend
        time_category_fig_secondary.update_traces(
            marker_color="#01E1D7",
            text=time_category_data_secondary['Count'],  # Show count on secondary bars
            textposition='inside'  # Position text above the secondary bars
        )

        # Remove the title and update layout for secondary dataset
        time_category_fig_secondary.update_layout(
            xaxis_title=None,  # Remove the x-axis title
            yaxis_title='Number of People',  # Remove the y-axis title
            yaxis_title_font=dict(size=22, color='black'), 
            xaxis=dict(
                tickfont=dict(size=18, color='black')  # Set x-axis tick font size and color
            ),
            yaxis=dict(
                tickfont=dict(size=18, color='black')  # Set y-axis tick font size and color
            ),
        )

        # Add traces for secondary dataset to the main figure
        for trace in time_category_fig_secondary.data:
            time_category_fig.add_trace(trace)

    return time_category_fig


def display_loyal_customers_leaderboard(loyal_customers, comparison_enabled):
    """
    Display the loyal customers leaderboard with a clean header and alternating row layout.
    """
    # CSS Styling for the leaderboard
    st.markdown("""
        <style>
            .leaderboard-title {
                font-size: 30px !important;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
                color: rgb(13, 13, 13);
            }
            .leaderboard-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                font-size: 25px;
                font-weight: bold;
                background: linear-gradient(135deg, #5D46DC, #6E52E9, #7F68F3, #947AFD, #AB8CFF);
                border-radius: 8px 8px 0 0;
            }
            .leaderboard-header .name,
            .leaderboard-header .primary,
            .leaderboard-header .comparison {
                color: white;
            }
            .leaderboard-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                font-size: 20px;
                font-weight: normal;
                color: black;
                border-bottom: 1px solid #ddd;
            }
            .row-even {
                background-color: #F1EFFA;
            }
            .row-odd {
                background-color: #ffffff;
            }
            .name, .primary, .comparison {
                flex: 1;
                text-align: center;
            }
            .primary, .comparison {
                color: black;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("<div class='leaderboard-title'>Loyal Customers</div>", unsafe_allow_html=True)

    # Header Section
    if comparison_enabled:
        st.markdown("""
            <div class="leaderboard-header">
                <div class="name">Name</div>
                <div class="primary">Present</div>
                <div class="name">Name</div>
                <div class="comparison">Historical
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="leaderboard-header">
                <div class="name">Name</div>
                <div class="primary">Present</div>
            </div>
        """, unsafe_allow_html=True)

    # Filter and sort loyal customers for top 5
    loyal_customers['Total Sales ($)_Primary'] = pd.to_numeric(loyal_customers.get('Total Sales ($)_Primary', loyal_customers.get('Total Sales ($)', 0)), errors='coerce')
    loyal_customers['Total Sales ($)_Secondary'] = pd.to_numeric(loyal_customers.get('Total Sales ($)_Secondary', 0), errors='coerce')

    # Filter top 5 for Primary Sales
    top_primary = loyal_customers[loyal_customers['Total Sales ($)_Primary'] > 0].sort_values(
        by='Total Sales ($)_Primary', ascending=False).head(5)
    if top_primary.empty:
        st.warning("No top loyal customers found for the selected date range.")
    # Filter top 5 for Comparison Sales (if comparison is enabled)
    if comparison_enabled:
        top_comparison = loyal_customers[loyal_customers['Total Sales ($)_Secondary'] > 0].sort_values(
            by='Total Sales ($)_Secondary', ascending=False).head(5)
    else:
        top_comparison = pd.DataFrame()  # Empty if comparison is not enabled

    # Display rows for Primary and Comparison side-by-side (if enabled)
    max_rows = max(len(top_primary), len(top_comparison)) if comparison_enabled else len(top_primary)

    for idx in range(max_rows):
        row_class = "row-even" if idx % 2 == 0 else "row-odd"

        # Get Primary Customer
        if idx < len(top_primary):
            primary_row = top_primary.iloc[idx]
            primary_name = f"{primary_row['First Name']} {primary_row['Last Name']}"
            primary_sales = abs(primary_row['Total Sales ($)_Primary']) if pd.notna(primary_row['Total Sales ($)_Primary']) else 0
        else:
            primary_name = ""
            primary_sales = ""

        # Get Comparison Customer (if enabled)
        if comparison_enabled and idx < len(top_comparison):
            comparison_row = top_comparison.iloc[idx]
            comparison_name = f"{comparison_row['First Name']} {comparison_row['Last Name']}"
            comparison_sales = abs(comparison_row['Total Sales ($)_Secondary']) if pd.notna(comparison_row['Total Sales ($)_Secondary']) else 0
        else:
            comparison_name = ""
            comparison_sales = ""

        # Convert sales values to float (if they are strings)
        try:
            primary_sales = float(primary_sales)
        except ValueError:
            primary_sales = 0  # Default value if conversion fails

        try:
            comparison_sales = float(comparison_sales)
        except ValueError:
            comparison_sales = 0  # Default value if conversion fails

        # Render the row
        if comparison_enabled:
            st.markdown(f"""
                <div class="leaderboard-row {row_class}">
                    <div class="name">{primary_name}</div>
                    <div class="primary">${primary_sales:,.0f}</div>
                    <div class="name">{comparison_name}</div>
                    <div class="comparison">${comparison_sales:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="leaderboard-row {row_class}">
                    <div class="name">{primary_name}</div>
                    <div class="primary">${primary_sales:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)

def loyal_customers_table(df_filtered, df_filtered_secondary, comparison_enabled):
    # Merge the primary order dataset with the payment data using order_id
    df_primary_merged = pd.merge(df_filtered, payment_df, on='order_id', how='inner')  # Merge orders with payment table
    # Rename 'user_id_x' to 'user_id' and drop 'user_id_y' from the merged DataFrame
    df_primary_merged['user_id'] = df_primary_merged['user_id_x']
    df_primary_merged.drop(columns=['user_id_x', 'user_id_y'], inplace=True)
    
    # Merge with signupuser_df using 'user_id'
    df_primary_merged = pd.merge(df_primary_merged, signupuser_df, on='user_id', how='inner')  # Merge with user table

    # Filter to match order_timestamp with payment_timestamp
    df_primary_merged = df_primary_merged[
        df_primary_merged['order_timestamp'].dt.date == df_primary_merged['payment_timestamp'].dt.date
    ]

    # Filter the primary dataset by the selected date range (start_date and end_date)
    df_primary_filtered = df_primary_merged[
        (df_primary_merged['order_timestamp'] >= pd.Timestamp(start_date)) &
        (df_primary_merged['order_timestamp'] <= pd.Timestamp(end_date))
    ]
    
    # Get the top loyal customers for the primary dataset
    loyal_customers_primary = (
        df_primary_filtered.groupby(['first_name', 'last_name'])['total_amount_x']  # Using 'total_amount_x' after merge
        .sum()
        .nlargest(5)
        .reset_index()
    )
    loyal_customers_primary.rename(
        columns={'first_name': 'First Name', 'last_name': 'Last Name', 'total_amount_x': 'Total Sales ($)'},
        inplace=True
    )
    loyal_customers_primary['Total Sales ($)'] = loyal_customers_primary['Total Sales ($)'].apply(format_with_commas)

    if comparison_enabled and df_filtered_secondary is not None:
        # Merge the secondary dataset with the user and payment data for comparison
        df_secondary_merged = pd.merge(df_filtered_secondary, payment_df, on='order_id', how='inner')  # Merge orders with payment table

        # Rename 'user_id_x' to 'user_id' and drop 'user_id_y' in the secondary DataFrame
        df_secondary_merged['user_id'] = df_secondary_merged['user_id_x']
        df_secondary_merged.drop(columns=['user_id_x', 'user_id_y'], inplace=True)

        # Filter to match order_timestamp with payment_timestamp
        df_secondary_merged = df_secondary_merged[
            df_secondary_merged['order_timestamp'].dt.date == df_secondary_merged['payment_timestamp'].dt.date
        ]

        # Merge with signupuser_df using 'user_id'
        df_secondary_merged = pd.merge(df_secondary_merged, signupuser_df, on='user_id', how='inner')  # Merge with user table

        # Filter the secondary dataset by the selected comparison date range
        df_secondary_filtered = df_secondary_merged[
            (df_secondary_merged['order_timestamp'] >= pd.Timestamp(comp_start_date)) &
            (df_secondary_merged['order_timestamp'] <= pd.Timestamp(comp_end_date))
        ]

        loyal_customers_secondary = (
            df_secondary_filtered.groupby(['first_name', 'last_name'])['total_amount_y']  # Using 'total_amount_y' after merge
            .sum()
            .nlargest(5)
            .reset_index()
        )
        loyal_customers_secondary.rename(
            columns={'first_name': 'First Name', 'last_name': 'Last Name', 'total_amount_y': 'Total Sales ($)'},
            inplace=True
        )
        loyal_customers_secondary['Total Sales ($)'] = loyal_customers_secondary['Total Sales ($)'].apply(format_with_commas)

        loyal_customers = pd.merge(
            loyal_customers_primary, loyal_customers_secondary,
            on=["First Name", "Last Name"],
            how="outer",
            suffixes=('_Primary', '_Secondary')
        )
    else:
        loyal_customers = loyal_customers_primary

    return loyal_customers






from google.oauth2 import service_account
from google.analytics.data import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest, Dimension, Metric, DateRange


# Google Analytics setup
PROPERTY_ID = "470749113"  # Replace with your Google Analytics property ID
JSON_FILE = "new2.json"  # Replace with the path to your service account JSON key

# Authentication
try:
    credentials = service_account.Credentials.from_service_account_file(JSON_FILE)
    client = BetaAnalyticsDataClient(credentials=credentials)
except Exception as e:
    st.error(f"Error during authentication: {e}")
    st.stop()

def fetch_active_users(start_date, end_date):
    """
    Fetches active user data from Google Analytics.
    """
    try:
        # Define date range and request
        date_range = DateRange(start_date=start_date, end_date=end_date)
        request_data = RunReportRequest(
            property=f"properties/{PROPERTY_ID}",
            dimensions=[Dimension(name="date")],
            metrics=[Metric(name="activeUsers")],
            date_ranges=[date_range]
        )
        # Fetch data
        response = client.run_report(request_data)
        # Parse response
        data = [
            {"date": row.dimension_values[0].value, "active_users": int(row.metric_values[0].value)}
            for row in response.rows
        ]
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        # st.error(f"Error fetching data: {e}")
        return None

# Function to create the line chart for active users using Plotly Express
def people_visited_chart(df_primary, df_comparison=None):
    """
    Creates a line chart for active users using Plotly Express.
    Ensures that markers are joined along the x-axis.
    """
    if df_primary is None or df_primary.empty:
        # st.warning("No data available for the primary period.")
        return None  # Return None to avoid errors
    # Sort primary data by date to ensure proper line connections
    df_primary = df_primary.sort_values(by="date")

    # Create the primary chart
    fig = px.line(
        df_primary,
        x='date',
        y='active_users',
        labels={'date': 'Date', 'active_users': 'Active Users'},
        line_shape='spline',
        markers=True
    )

    # Update the primary trace style
    fig.update_traces(
        line=dict(color='#5D46DC', dash='solid', width=2),
        hovertemplate='<b>Date: %{x}</b><br><b>Active Users: %{y}</b><extra></extra>',
        name='Primary Period'
    )

    # If a comparison dataset is provided, add it
    if df_comparison is not None:
        # Create a secondary dataset trace
        secondary_trace = px.line(
            df_comparison,
            x='date',
            y='active_users',
            markers=True
        )

        # Add secondary trace data to the primary figure
        for trace in secondary_trace.data:
            fig.add_trace(trace)

        # Update the secondary trace style
        fig.data[1].update(
            line=dict(color='#01E1D7', dash='dash', width=2),
            hovertemplate='<b>Date: %{x}</b><br><b>Active Users: %{y}</b><extra></extra>',
            name='Comparison Period'
        )

    # Customize the x-axis and y-axis titles and lines
    fig.update_xaxes(
        title=dict(
            font=dict(color='rgb(13,13,13)', size=20)  # Sets x-axis title font color and size
        ),
        showgrid=False,  # Removes vertical gridlines
        tickfont=dict(color='rgb(13,13,13)', size=18),  # Sets x-tick color and size
        linecolor='rgb(13,13,13)',  # Sets color of the x-axis line
        linewidth=1,  # Sets thickness of the x-axis line
        tickformat="%d %b" # Format to display only the date
        )
    fig.update_yaxes(
        title=dict(
            text="Active Users",
            font=dict(color='rgb(13,13,13)', size=20)  # Sets y-axis title font color and size
        ),
        showgrid=False,  # Removes horizontal gridlines
        tickfont=dict(color='rgb(13,13,13)', size=18),  # Sets y-tick color and size
        linecolor='rgb(13,13,13)',  # Sets color of the y-axis line
        linewidth=1  # Sets thickness of the y-axis line
    )

    # Customize layout and hover settings with graph size adjustments
    fig.update_layout(
        height=240,
        margin=dict(t=1, b=0, l=0, r=1),  # Adjust top margin for title
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


# Visualize the chart based on the date ranges from the sidebar
df_primary = fetch_active_users(str(start_date), str(end_date))
print("df_primary:", df_primary)
df_comparison = (
    fetch_active_users(str(comp_start_date), str(comp_end_date))
    if comparison_enabled and comp_start_date and comp_end_date
    else None
)

if df_primary is not None and not df_primary.empty:
    st.success("Data fetched successfully!")
else:
    print("")#No data available for the selected primary date range.

    
def assign_predefined_colors(data, colors):
    n = len(data)
    # If there are more data points than colors, repeat the color cycle
    assigned_colors = (colors * (n // len(colors) + 1))[:n]
    return assigned_colors

def city_distribution_chart(df_filtered, df_filtered_secondary=None, comparison_enabled=False):
    
    # Filter the primary dataset based on the "From" and "To" dates
    df_primary_filtered = df_filtered[ 
        (df_filtered['order_timestamp'] >= pd.Timestamp(start_date)) & 
        (df_filtered['order_timestamp'] <= pd.Timestamp(end_date))
    ]
    if df_primary_filtered.empty:
        st.warning("No orders found for the selected date range.")
        return None
    # Merge orders_df with signupuser_df to get city information
    df_primary_filtered = df_primary_filtered.merge(
        signupuser_df[['user_id', 'city']],
        on='user_id',
        how='left'
    )

    # Calculate city distribution for the primary dataset
    city_distribution_primary = df_primary_filtered.groupby('city')['quantity'].sum().reset_index()
    c_type = 'pie' if not comparison_enabled else 'bar'

    # Predefined color list for pie chart slices
    predefined_colors = ['#1E0F5A', '#2F1E77', '#4A3A9E', '#6556C4', '#8477E6', '#AFA3F9', '#d2ccfc']

    # Sort data by 'quantity' for color assignment
    city_distribution_primary = city_distribution_primary.sort_values(by='quantity', ascending=False).reset_index(drop=True)
    colors = assign_predefined_colors(city_distribution_primary, predefined_colors)

    # Create the chart for the primary city distribution
    fig = create_chart(
        city_distribution_primary,
        x='city',
        y='quantity',
        chart_type=c_type
    )

    # Update text size and colors for pie slices
    if c_type == 'pie':
        fig.update_traces(
            text=city_distribution_primary['quantity'].values,
            textposition='inside',
            textfont=dict(size=20, color='white'),  # Larger text size for slices
            marker=dict(colors=colors)  # Apply predefined colors
        )

    # If comparison is enabled and secondary data is provided, handle the comparison
    if df_filtered_secondary is not None:
        # Filter the secondary dataset based on the comparison date range
        df_secondary_filtered = df_filtered_secondary[ 
            (df_filtered_secondary['order_timestamp'] >= pd.Timestamp(comp_start_date)) & 
            (df_filtered_secondary['order_timestamp'] <= pd.Timestamp(comp_end_date))
        ]
        
        # Merge with signupuser_df
        df_secondary_filtered = df_secondary_filtered.merge(
            signupuser_df[['user_id', 'city']],
            on='user_id',
            how='left'
        )

        # Group by city for secondary dataset
        city_distribution_secondary = df_secondary_filtered.groupby('city')['quantity'].sum().reset_index()
        c_type='bar'
        # Create the bar chart for primary and secondary data
        if c_type == 'bar':
            # Create the bar chart for the primary city distribution
            fig = create_chart(
                city_distribution_primary,
                x='city',
                y='quantity',
                chart_type='bar'
            )
            
            # Create the bar chart for the secondary city distribution
            fig_secondary = create_chart(
                city_distribution_secondary,
                x='city',
                y='quantity',
                chart_type='bar'
            )
            
            # Add the secondary traces to the primary figure
            fig.add_traces(fig_secondary.data)

            # Apply custom styles to differentiate the traces
            fig.data[0].update(marker=dict(color='#5D46DC'))
            fig.data[1].update(marker=dict(color='#01E1D7'))

            # Update text for absolute positioning inside bars
            for trace in fig.data:
                trace.update(
                    text=[str(abs(int(y))) for y in trace.y if y is not None],
                    textposition='inside',  # Text inside the bars
                    insidetextanchor='middle',  # Center the text inside the bars
                    textfont=dict(
                        size=16,  # Font size of the text inside the bars
                        color='white'  # Text color
                    )
                )

            # Update layout for legend font size and margins
            fig.update_layout(
                legend=dict(
                    font=dict(size=18)  # Increase legend text size
                ),
                xaxis_title=None,
                yaxis_title="Quantity Sold",
                yaxis=dict(
                    title_font=dict(
                        size=20,  # Set the font size for the y-axis title
                        color='rgb(13,13,13)'  # Set the font color for the y-axis title
                    ),
                    tickfont=dict(
                        size=16,  # Font size of the y-axis tick labels
                        color='black'  # Color of the tick labels
                    )
                ),
                margin=dict(t=5, b=0, l=0, r=0)
            )
    return fig



def product_feedback_leaderboard_with_styles(top_feedbacks, comparison_enabled):
    # Embed the styles and the leaderboard rendering logic
    st.markdown("""
        <style>
            .leaderboard-title {
                font-size: 27px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
                color: rgb(13,13,13);
            }
            .leaderboard-header, .leaderboard-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                font-weight: bold;
                font-size: 25px;
            }
            .leaderboard-header div {
                text-align: center; /* Ensures the text is centered */
                flex: 1; /* Distribute space evenly */
                color: white !important; /* Keeps text color white */
            }
            .leaderboard-row {
                background-color: #f9f9f9;
                border-bottom: 1px solid #ddd;
                color: black; /* Black font for row text */
                font-size: 20px !important;
            }
            .row-even {
                background-color: #f3f4f6;
            }
            .row-odd {
                background-color: #ffffff;
            }
            .product, .feedback, .primary, .secondary {
                flex: 1;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)
    # Display leaderboard title
    st.markdown('<div class="leaderboard-title">Top Feedback by Product</div>', unsafe_allow_html=True)
    leaderboard_header = (
        '<div class="leaderboard-header">'
        '<div class="product">Product</div>'
        '<div class="feedback">Feedback</div>'
        '<div class="primary">Present</div>'
    )
    if comparison_enabled:
        leaderboard_header += '<div class="secondary">Historical</div>'
    leaderboard_header += '</div>'

    st.markdown(leaderboard_header, unsafe_allow_html=True)

    # Iterate over rows and display in leaderboard format
    for idx, row in top_feedbacks.iterrows():
        row_class = "row-even" if idx % 2 == 0 else "row-odd"
        leaderboard_row = (
            f'<div class="leaderboard-row {row_class}">'
            f'<div class="product">{row["Product"]}</div>'
            f'<div class="feedback">{row["Feedback"]}</div>'
            f'<div class="primary">{row["Count_Primary"]}</div>'
        )
        if comparison_enabled:
            leaderboard_row += f'<div class="secondary">{row.get("Count_Secondary", "-")}</div>'
        leaderboard_row += '</div>'
        
        st.markdown(leaderboard_row, unsafe_allow_html=True)
     # Display warning if there are no feedbacks
    if top_feedbacks.empty:
        st.warning("No feedbacks found for the selected date range.")
    # Build header dynamically

def top_feedbacks_table(df_filtered, df_filtered_secondary, comparison_enabled, fooditem_df, feedback_df):
    # Ensure necessary columns exist
    if 'name' not in fooditem_df.columns or 'rating' not in feedback_df.columns:
        raise KeyError("Required columns are missing from fooditem_df or feedback_df.")
    
    # Ensure 'feedback_date' exists in feedback_df for merging
    if 'feedback_date' not in feedback_df.columns:
        raise KeyError("'feedback_date' column not found in feedback_df.")

    # Add a date column to both df_filtered and feedback_df for comparison
    df_filtered['order_date'] = df_filtered['order_timestamp'].dt.date
    feedback_df['feedback_date_only'] = feedback_df['feedback_date'].dt.date

    # Merge orders with feedback and food items
    df_primary_with_feedback = df_filtered.merge(
        feedback_df[['feedback_date_only', 'rating', 'food_item_id']], 
        left_on='order_date', 
        right_on='feedback_date_only', 
        how='left'
    ).merge(
        fooditem_df[['food_item_id', 'name']], 
        on='food_item_id', 
        how='left'
    )

    # Group and count top feedbacks for the primary dataset
    top_feedbacks_primary = (
        df_primary_with_feedback.groupby(['name', 'rating'])
        .size()
        .reset_index(name='Count_Primary')
        .nlargest(5, 'Count_Primary')
    )

    # Rename columns for clarity
    top_feedbacks_primary.rename(columns={'name': 'Product', 'rating': 'Feedback'}, inplace=True)

    # Handle the secondary dataset if comparison is enabled
    if comparison_enabled and df_filtered_secondary is not None:
        df_filtered_secondary['order_date'] = df_filtered_secondary['order_timestamp'].dt.date

        # Merge secondary dataset with feedback and food items
        df_secondary_with_feedback = df_filtered_secondary.merge(
            feedback_df[['feedback_date_only', 'rating', 'food_item_id']], 
            left_on='order_date', 
            right_on='feedback_date_only', 
            how='left'
        ).merge(
            fooditem_df[['food_item_id', 'name']], 
            on='food_item_id', 
            how='left'
        )

        # Group and count top feedbacks for the secondary dataset
        top_feedbacks_secondary = (
            df_secondary_with_feedback.groupby(['name', 'rating'])
            .size()
            .reset_index(name='Count_Secondary')
            .nlargest(5, 'Count_Secondary')
        )

        # Rename columns for clarity
        top_feedbacks_secondary.rename(columns={'name': 'Product', 'rating': 'Feedback'}, inplace=True)
        
        


        # Merge primary and secondary feedback data
        top_feedbacks = pd.merge(
            top_feedbacks_primary, top_feedbacks_secondary,
            on=["Product", "Feedback"], how="outer"
        )

        # Fill missing values with 0 and ensure integer counts
        top_feedbacks['Count_Primary'] = top_feedbacks['Count_Primary'].fillna(0).astype(int)
        top_feedbacks['Count_Secondary'] = top_feedbacks['Count_Secondary'].fillna(0).astype(int)
    else:
        top_feedbacks = top_feedbacks_primary
        top_feedbacks['Count_Primary'] = top_feedbacks['Count_Primary'].astype(int)  # Ensure integer counts

    return top_feedbacks


# Main logic
def filter_data_by_date_range(df, start_date, end_date):
    """
    Filter the dataset based on the provided "From" and "To" date range.
    """
    # Filter primary dataset based on the provided date range
    df_filtered = df[
        (df['order_timestamp'] >= pd.Timestamp(start_date)) &
        (df['order_timestamp'] <= pd.Timestamp(end_date))
    ]
    
    return df_filtered

df_filtered = filter_data_by_date_range(orders_df, start_date, end_date)
if comparison_enabled:
    # Filter the secondary dataset by the comparison date range
    second_df_filtered = filter_data_by_date_range(orders_df, comp_start_date, comp_end_date)
else:
    # If comparison is not enabled, set the secondary dataset to None
    second_df_filtered = None
# Now, create and display the charts with the filtered datasets
create_and_display_charts(df_filtered, second_df_filtered, cols)
