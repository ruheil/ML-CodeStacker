import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import datetime
import re
import os
import tempfile
import joblib
from io import BytesIO
import base64
from PIL import Image
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="CityX Crime Watch Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        color: #4B4BFF;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FF4B4B;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
    }
    .report-card {
        background-color: #F8F9FA;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .report-field {
        font-weight: bold;
        color: #4B4BFF;
    }
    .report-value {
        margin-left: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        padding: 8px 0;
        background: #f0f2f6;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 0 24px;
        background: white;
        border-radius: 8px;
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
        font-size: 16px;
        font-weight: 500;
        color: #666 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #4B4BFF15;
        color: #4B4BFF !important;
        transform: translateY(-2px);
        box-shadow: 0 3px 8px rgba(75,75,255,0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4B4BFF, #6B72FF) !important;
        color: white !important;
        border-color: #4B4BFF;
        box-shadow: 0 4px 12px rgba(75,75,255,0.2);
    }
    .subtab [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .subtab [data-baseweb="tab"] {
        height: 40px;
        padding: 0 20px;
        background: rgba(75,75,255,0.1);
        border-radius: 6px;
        color: #4B4BFF !important;
        font-size: 14px;
        border: none;
        transition: all 0.2s ease;
    }
    .subtab [data-baseweb="tab"]:hover {
        background: rgba(75,75,255,0.2);
    }
    .subtab [aria-selected="true"] {
        background: #4B4BFF !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(75,75,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">CityX Crime Watch Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
    <div style='margin: 2rem 0; border-bottom: 2px solid #f0f2f6; padding-bottom: 1.5rem;'></div>
    """, unsafe_allow_html=True)

# Improve tab styling with custom HTML
tab_titles = [
    "📊 Crime Analytics",
    "📑 Report Processing", 
    "🔍 Prediction & Classification",
    "🗺️ Geospatial Analysis"
]

# Create navigation tabs with enhanced visibility
tabs = st.tabs(tab_titles)

# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Competition_Dataset.csv')
        
        # Convert dates
        df['Dates'] = pd.to_datetime(df['Dates'])
        
        # Extract date components
        df['Year'] = df['Dates'].dt.year
        df['Month'] = df['Dates'].dt.month
        df['Day'] = df['Dates'].dt.day
        df['Hour'] = df['Dates'].dt.hour
        df['Minute'] = df['Dates'].dt.minute
        df['DayName'] = df['Dates'].dt.day_name()
        
        # Swap longitude and latitude if they're mislabeled
        temp = df['Latitude (Y)'].copy()
        df['Latitude (Y)'] = df['Longitude (X)']
        df['Longitude (X)'] = temp
        
        # Assign severity
        df['Severity'] = df['Category'].apply(assign_severity)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Assign severity based on crime category
def assign_severity(category):
    severity_1 = ['NON-CRIMINAL', 'SUSPICIOUS OCC', 'MISSING PERSON', 'RUNAWAY', 'RECOVERED VEHICLE']
    severity_2 = ['WARRANTS', 'OTHER OFFENSES', 'VANDALISM', 'TRESPASS', 'DISORDERLY CONDUCT', 'BAD CHECKS']
    severity_3 = ['LARCENY/THEFT', 'VEHICLE THEFT', 'FORGERY/COUNTERFEITING', 'DRUG/NARCOTIC', 
                 'STOLEN PROPERTY', 'FRAUD', 'BRIBERY', 'EMBEZZLEMENT']
    severity_4 = ['ROBBERY', 'WEAPON LAWS', 'BURGLARY', 'EXTORTION']
    severity_5 = ['KIDNAPPING', 'ARSON']
    
    if category in severity_1:
        return 1
    elif category in severity_2:
        return 2
    elif category in severity_3:
        return 3
    elif category in severity_4:
        return 4
    elif category in severity_5:
        return 5
    else:
        return 0  # Unknown category

# Load classification model
@st.cache_resource
def load_model():
    try:
        if os.path.exists('crime_classifier.joblib'):
            model = joblib.load('crime_classifier.joblib')
            return model
        else:
            # If model doesn't exist, create a simple one
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            st.warning("Classification model not found. Using a default model instance.")
            return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to extract information from PDFs
def extract_from_pdf(pdf_file):
    try:
        extracted_data = {}
        
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        
        # Extract report number
        report_number_match = re.search(r'Report Number:?\s*(.+?)\n', text)
        if report_number_match:
            extracted_data['Report Number'] = report_number_match.group(1).strip()
        
        # Extract date & time
        date_match = re.search(r'Date & Time:?\s*(.+?)\n', text)
        if date_match:
            date_str = date_match.group(1).strip()
            try:
                extracted_data['Dates'] = date_str
                extracted_data['date_obj'] = pd.to_datetime(date_str)
            except:
                extracted_data['Dates'] = date_str
        
        # Extract reporting officer
        officer_match = re.search(r'Reporting Officer:?\s*(.+?)\n', text)
        if officer_match:
            extracted_data['Reporting Officer'] = officer_match.group(1).strip()
        
        # Extract incident location
        location_match = re.search(r'Incident Location:?\s*(.+?)\n', text)
        if location_match:
            extracted_data['Address'] = location_match.group(1).strip()
        
        # Extract coordinates
        coordinates_match = re.search(r'Coordinates:?\s*\((.+?),\s*(.+?)\)', text)
        if coordinates_match:
            extracted_data['Latitude (Y)'] = float(coordinates_match.group(1).strip())
            extracted_data['Longitude (X)'] = float(coordinates_match.group(2).strip())
        
        # Extract description
        description_match = re.search(r'Detailed Description:?\s*(.+?)\n(?:Police District|Resolution|Suspect Description)', text, re.DOTALL)
        if description_match:
            extracted_data['Descript'] = description_match.group(1).strip()
        
        # Extract police district
        district_match = re.search(r'Police District:?\s*(.+?)\n', text)
        if district_match:
            extracted_data['PdDistrict'] = district_match.group(1).strip()
        
        # Extract resolution
        resolution_match = re.search(r'Resolution:?\s*(.+?)\n', text)
        if resolution_match:
            extracted_data['Resolution'] = resolution_match.group(1).strip()
        
        # Extract day of week if date is available
        if 'date_obj' in extracted_data:
            extracted_data['DayOfWeek'] = extracted_data['date_obj'].day_name()
        
        return extracted_data
    
    except Exception as e:
        st.error(f"Error extracting data from PDF: {e}")
        return {}

# Predict crime category and severity
def predict_crime(description, model):
    if not model:
        return "Unknown", 0
    
    try:
        # If model is not trained, train it first
        if not hasattr(model, 'classes_'):
            st.warning("Model needs to be trained first.")
            return "Unknown", 0
        
        category = model.predict([description])[0]
        severity = assign_severity(category)
        return category, severity
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return "Error in prediction", 0

# Load data and model
df = load_data()
model = load_model()

# Define severity labels
severity_labels = {
    1: "Minor",
    2: "Low",
    3: "Medium",
    4: "High",
    5: "Critical"
}

# TAB 1: CRIME ANALYTICS
with tabs[0]:
    # Sidebar for filtering
    st.sidebar.title("Filter Options")
    
    # Time filtering
    st.sidebar.markdown("### Time Filters")
    year_options = sorted(df['Year'].unique().tolist()) if not df.empty else []
    selected_years = st.sidebar.multiselect("Select Years", year_options, default=year_options[-2:] if len(year_options) >= 2 else year_options)
    
    # Filter data by year
    filtered_df = df[df['Year'].isin(selected_years)] if not df.empty and selected_years else df
    
    # District filtering
    st.sidebar.markdown("### District Filters")
    district_options = sorted(df['PdDistrict'].unique().tolist()) if not df.empty else []
    selected_districts = st.sidebar.multiselect("Select Districts", district_options, default=district_options)
    
    # Filter data by district
    if selected_districts and not df.empty:
        filtered_df = filtered_df[filtered_df['PdDistrict'].isin(selected_districts)]
    
    # Crime type filtering
    st.sidebar.markdown("### Crime Type Filters")
    category_options = sorted(df['Category'].unique().tolist()) if not df.empty else []
    selected_categories = st.sidebar.multiselect("Select Crime Categories", category_options, default=[])
    
    # Filter data by crime category
    if selected_categories and not df.empty:
        filtered_df = filtered_df[filtered_df['Category'].isin(selected_categories)]
    
    # Severity filtering
    st.sidebar.markdown("### Severity Filters")
    severity_options = list(range(1, 6))
    selected_severities = st.sidebar.multiselect(
        "Select Severity Levels", 
        options=severity_options,
        format_func=lambda x: severity_labels.get(x, "Unknown"),
        default=severity_options
    )
    
    # Filter data by severity
    if selected_severities and not df.empty:
        filtered_df = filtered_df[filtered_df['Severity'].isin(selected_severities)]
    
    # Display metrics
    st.markdown('<h2 class="sub-header">Crime Overview:</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_incidents = len(filtered_df) if not filtered_df.empty else 0
        st.markdown(f'<div class="metric-value">{total_incidents:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Incidents</div>', unsafe_allow_html=True)
    
    with col2:
        avg_severity = filtered_df['Severity'].mean() if not filtered_df.empty else 0
        st.markdown(f'<div class="metric-value">{avg_severity:.2f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Avg. Severity</div>', unsafe_allow_html=True)
    
    with col3:
        top_district = filtered_df['PdDistrict'].value_counts().idxmax() if not filtered_df.empty else "N/A"
        st.markdown(f'<div class="metric-value">{top_district}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Highest Crime District</div>', unsafe_allow_html=True)

    # Add vertical space
    st.markdown("""
    <div style='margin: 2rem 0; border-bottom: 2px solid #f0f2f6; padding-bottom: 1.5rem;'></div>
    """, unsafe_allow_html=True)

    # Create sub-tabs for different analytics
    crime_tabs = st.tabs(["Temporal Analysis", "Categorical Analysis", "Severity Analysis"])
    
    # Temporal Analysis
    with crime_tabs[0]:
        st.markdown('<div class="subtab">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Crime Trends Over Time</h3>', unsafe_allow_html=True)
        
        # Time-based visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Crime by hour
            if not filtered_df.empty:
                hour_counts = filtered_df.groupby('Hour').size().reset_index(name='count')
                fig = px.line(hour_counts, x='Hour', y='count', 
                            title='Crime by Hour of Day',
                            labels={'count': 'Number of Incidents', 'Hour': 'Hour of Day (24h)'},
                            markers=True)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for visualization.")
        
        with col2:
            # Crime by day of week
            if not filtered_df.empty:
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_counts = filtered_df['DayOfWeek'].value_counts().reindex(day_order).reset_index()
                day_counts.columns = ['DayOfWeek', 'count']
                
                fig = px.bar(day_counts, x='DayOfWeek', y='count',
                            title='Crime by Day of Week',
                            labels={'count': 'Number of Incidents', 'DayOfWeek': 'Day of Week'},
                            category_orders={"DayOfWeek": day_order},
                            text_auto=True,
                            color_discrete_sequence=['#4B4BFF'])
                fig.update_traces(marker_line_width=1.5, opacity=0.8,
                                marker_line_color='black')
                fig.update_layout(height=400, 
                                yaxis_range=[0, day_counts['count'].max()*1.1],
                                title_font_size=20,
                                xaxis_title_font_size=16,
                                yaxis_title_font_size=16)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for visualization.")
        
        # Monthly trend
        if not filtered_df.empty:
            filtered_df['YearMonth'] = filtered_df['Dates'].dt.strftime('%Y-%m')
            monthly_counts = filtered_df.groupby('YearMonth').size().reset_index(name='count')
            
            fig = px.line(monthly_counts, x='YearMonth', y='count',
                        title='Monthly Crime Trend',
                        labels={'count': 'Number of Incidents', 'YearMonth': 'Month'},
                        markers=True)
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for visualization.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Categorical Analysis
    with crime_tabs[1]:
        st.markdown('<div class="subtab">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Crime Categories Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top crime categories
            if not filtered_df.empty:
                category_counts = filtered_df['Category'].value_counts().head(10).reset_index()
                category_counts.columns = ['Category', 'count']
                
                fig = px.bar(category_counts, y='Category', x='count', 
                            title='Top 10 Crime Categories',
                            labels={'count': 'Number of Incidents', 'Category': 'Crime Type'},
                            orientation='h')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for visualization.")
        
        with col2:
            # Crime categories by district
            if not filtered_df.empty:
                district_category = pd.crosstab(filtered_df['PdDistrict'], filtered_df['Category'])
                # Select top categories
                top_categories = filtered_df['Category'].value_counts().head(5).index
                district_category_top = district_category[top_categories]
                
                fig = px.imshow(district_category_top,
                              labels=dict(x='Crime Category', y='District', color='Number of Incidents'),
                              title='Top Crime Categories by District',
                              color_continuous_scale='Blues')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for visualization.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Severity Analysis
    with crime_tabs[2]:
        st.markdown('<div class="subtab">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Crime Severity Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity distribution
            if not filtered_df.empty:
                severity_counts = filtered_df['Severity'].value_counts().sort_index().reset_index()
                severity_counts.columns = ['Severity', 'count']
                
                # Add severity labels
                severity_counts['Severity'] = severity_counts['Severity'].apply(
                    lambda x: f"Level {x}: {severity_labels.get(x, 'Unknown')}"
                )
                
                fig = px.bar(severity_counts, x='Severity', y='count',
                            title='Crime Severity Distribution',
                            labels={'count': 'Number of Incidents', 'Severity': 'Severity Level'},
                            color='Severity', color_discrete_sequence=px.colors.sequential.RdBu_r)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for visualization.")
        
        with col2:
            # Average severity by district
            if not filtered_df.empty:
                district_severity = filtered_df.groupby('PdDistrict')['Severity'].mean().sort_values(ascending=False).reset_index()
                
                fig = px.bar(district_severity, y='PdDistrict', x='Severity',
                            title='Average Crime Severity by District',
                            labels={'Severity': 'Average Severity (1-5)', 'PdDistrict': 'District'},
                            orientation='h', color='Severity', color_continuous_scale='Reds')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for visualization.")
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: REPORT PROCESSING
with tabs[1]:
    st.markdown('<h2 class="sub-header">Police Report Extraction</h2>', unsafe_allow_html=True)
    st.markdown("""
    This module automatically extracts crime data from police reports in PDF format. 
    Upload a police report to extract information and classify the crime.
    """)
    
    # File uploader for PDF
    uploaded_file = st.file_uploader("Upload Police Report (PDF)", type="pdf")
    
    if uploaded_file:
        # Display PDF preview
        st.markdown('<h3 class="sub-header">PDF Preview</h3>', unsafe_allow_html=True)
        
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        
        # Display PDF using HTML iframe (works only in some browsers)
        base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="400" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        # Process PDF button
        if st.button("Process Report"):
            with st.spinner("Extracting information from the report..."):
                # Extract data from PDF
                extracted_data = extract_from_pdf(pdf_path)
                
                if extracted_data:
                    st.success("Report processed successfully!")
                    
                    # Display extracted information
                    st.markdown('<h3 class="sub-header">Extracted Information</h3>', unsafe_allow_html=True)
                    
                    # Create a form-like display for the extracted fields
                    with st.expander("Report Details", expanded=True):
                        report_col1, report_col2 = st.columns(2)
                        
                        with report_col1:
                            st.markdown('<div class="report-card">', unsafe_allow_html=True)
                            for field in ['Report Number', 'Dates', 'Reporting Officer', 'Address']:
                                if field in extracted_data:
                                    st.markdown(f'<span class="report-field">{field}:</span> <span class="report-value">{extracted_data[field]}</span>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with report_col2:
                            st.markdown('<div class="report-card">', unsafe_allow_html=True)
                            for field in ['PdDistrict', 'Resolution', 'DayOfWeek']:
                                if field in extracted_data:
                                    st.markdown(f'<span class="report-field">{field}:</span> <span class="report-value">{extracted_data[field]}</span>', unsafe_allow_html=True)
                            
                            # Display coordinates if available
                            if 'Latitude (Y)' in extracted_data and 'Longitude (X)' in extracted_data:
                                st.markdown(f'<span class="report-field">Coordinates:</span> <span class="report-value">({extracted_data["Latitude (Y)"]}, {extracted_data["Longitude (X)"]})</span>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display description
                    if 'Descript' in extracted_data:
                        st.markdown('<div class="report-card">', unsafe_allow_html=True)
                        st.markdown(f'<span class="report-field">Description:</span> <span class="report-value">{extracted_data["Descript"]}</span>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Predict crime category and severity
                    if 'Descript' in extracted_data and model:
                        with st.spinner("Classifying crime..."):
                            predicted_category, predicted_severity = predict_crime(extracted_data['Descript'], model)
                            
                            st.markdown('<h3 class="sub-header">Crime Classification</h3>', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f'<div class="metric-value">{predicted_category}</div>', unsafe_allow_html=True)
                                st.markdown('<div class="metric-label">Predicted Crime Category</div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f'<div class="metric-value">Level {predicted_severity}</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="metric-label">{severity_labels.get(predicted_severity, "Unknown")}</div>', unsafe_allow_html=True)
                    
                    # Convert extracted data to DataFrame row format
                    report_row = {}
                    for field in extracted_data:
                        if field not in ['date_obj']:  # Skip helper fields
                            report_row[field] = extracted_data[field]
                    
                    # Add predicted category if not present
                    if 'Category' not in report_row and 'Descript' in extracted_data:
                        report_row['Category'] = predicted_category
                    
                    # Add severity if not present
                    if 'Severity' not in report_row and 'Category' in report_row:
                        report_row['Severity'] = assign_severity(report_row['Category'])
                    
                    # Display as dataframe
                    st.markdown('<h3 class="sub-header">Structured Data for Model Input</h3>', unsafe_allow_html=True)
                    report_df = pd.DataFrame([report_row])
                    st.dataframe(report_df)
                    
                    # Show on map if coordinates are available
                    if 'Latitude (Y)' in extracted_data and 'Longitude (X)' in extracted_data:
                        st.markdown('<h3 class="sub-header">Incident Location</h3>', unsafe_allow_html=True)
                        
                        m = folium.Map(
                            location=[extracted_data['Latitude (Y)'], extracted_data['Longitude (X)']], 
                            zoom_start=15,
                            tiles='CartoDB positron'
                        )
                        
                        # Add marker with popup
                        popup_text = f"""
                        <b>Report:</b> {extracted_data.get('Report Number', 'N/A')}<br>
                        <b>Date:</b> {extracted_data.get('Dates', 'N/A')}<br>
                        <b>Category:</b> {predicted_category}<br>
                        <b>Severity:</b> Level {predicted_severity}<br>
                        <b>Description:</b> {extracted_data.get('Descript', 'N/A')}<br>
                        """
                        
                        # Color based on severity
                        color_map = {1: 'green', 2: 'blue', 3: 'orange', 4: 'red', 5: 'darkred'}
                        marker_color = color_map.get(predicted_severity, 'gray')
                        
                        folium.Marker(
                            [extracted_data['Latitude (Y)'], extracted_data['Longitude (X)']], 
                            popup=folium.Popup(popup_text, max_width=300),
                            icon=folium.Icon(color=marker_color, icon='info-sign')
                        ).add_to(m)
                        
                        # Display the map
                        folium_static(m, width=800, height=500)
                else:
                    st.error("Failed to extract information from the report. Please check the PDF format.")
        
        # Clean up temporary file
        if 'pdf_path' in locals():
            try:
                os.unlink(pdf_path)
            except:
                pass

# TAB 3: PREDICTION & CLASSIFICATION
with tabs[2]:
    st.markdown('<h2 class="sub-header">Crime Classification</h2>', unsafe_allow_html=True)
    st.markdown("""
    This module allows you to classify crime descriptions into categories and assign severity levels.
    Enter a crime description to predict its category and severity level.
    """)
    
    # Text input for crime description
    user_description = st.text_area("Enter a crime description:", height=100)
    
    # Add example descriptions
    with st.expander("Show example descriptions"):
        examples = [
            "Suspect broke into vehicle and stole personal items.",
            "Victim was robbed at gunpoint by unknown suspects.",
            "Graffiti found on building walls.",
            "Suspect was arrested for outstanding warrant.",
            "Victim reports their car was stolen from parking lot."
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{examples.index(example)}"):
                user_description = example
                st.session_state.user_description = example
    
    # Prediction button
    if st.button("Classify Crime") or user_description:
        if user_description:
            with st.spinner("Classifying crime description..."):
                # Train model if dataset is available and model isn't trained
                if not df.empty and model and not hasattr(model, 'classes_'):
                    st.info("Training classification model...")
                    X = df['Descript']
                    y = df['Category']
                    model.fit(X, y)
                
                # Make prediction
                predicted_category, predicted_severity = predict_crime(user_description, model)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f'<div class="metric-value">{predicted_category}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Predicted Crime Category</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'<div class="metric-value">Level {predicted_severity}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-label">{severity_labels.get(predicted_severity, "Unknown")}</div>', unsafe_allow_html=True)
                
                # Show similar crimes
                st.markdown('<h3 class="sub-header">Similar Crimes in Database</h3>', unsafe_allow_html=True)
                
                if not df.empty:
                    similar_crimes = df[df['Category'] == predicted_category].sample(min(5, len(df[df['Category'] == predicted_category]))) if predicted_category in df['Category'].values else pd.DataFrame()
                    
                    if not similar_crimes.empty:
                        for _, crime in similar_crimes.iterrows():
                            st.markdown('<div class="report-card">', unsafe_allow_html=True)
                            st.markdown(f"""
                            <span class="report-field">Date:</span> <span class="report-value">{crime['Dates'].strftime('%Y-%m-%d %H:%M')}</span><br>
                            <span class="report-field">Description:</span> <span class="report-value">{crime['Descript']}</span><br>
                            <span class="report-field">District:</span> <span class="report-value">{crime['PdDistrict']}</span><br>
                            <span class="report-field">Resolution:</span> <span class="report-value">{crime['Resolution']}</span>
                            """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("No similar crimes found in the database.")
                else:
                    st.info("No crime database available for comparison.")
        else:
            st.warning("Please enter a crime description.")

# TAB 4: GEOSPATIAL ANALYSIS
with tabs[3]:
    st.markdown('<h2 class="sub-header">Crime Geospatial Analysis</h2>', unsafe_allow_html=True)
    
    # Map type selection
    map_type = st.selectbox(
        "Select Map Type", 
        ["Heat Map", "Cluster Map", "Severity Map"]
    )
    
    # Function to create maps
    def create_map(data, map_type="Heat Map"):
        # Create base map centered on CityX
        m = folium.Map(
            location=[data['Latitude (Y)'].mean(), data['Longitude (X)'].mean()], 
            zoom_start=12,
            tiles='CartoDB positron'
        )
        
        # Add different map layers based on selection
        if map_type == "Heat Map":
            # Create heat map
            heat_data = [[row['Latitude (Y)'], row['Longitude (X)'], 1] 
                        for _, row in data.iterrows() if pd.notna(row['Latitude (Y)']) and pd.notna(row['Longitude (X)'])]
            HeatMap(heat_data, radius=15, max_zoom=13, blur=10).add_to(m)
        
        elif map_type == "Cluster Map":
            # Create marker cluster
            marker_cluster = MarkerCluster().add_to(m)
            
            # Add markers for crime locations
            for idx, row in data.sample(min(5000, len(data))).iterrows():
                if pd.notna(row['Latitude (Y)']) and pd.notna(row['Longitude (X)']): 
                    # Create marker popup with crime details
                    popup_text = f"""
                    <b>Category:</b> {row['Category']}<br>
                    <b>Description:</b> {row['Descript']}<br>
                    <b>Date:</b> {row['Dates'].strftime('%Y-%m-%d %H:%M')}<br>
                    <b>Day:</b> {row['DayOfWeek']}<br>
                    <b>District:</b> {row['PdDistrict']}<br>
                    <b>Resolution:</b> {row['Resolution']}<br>
                    <b>Severity:</b> Level {row['Severity']}
                    """
                    popup = folium.Popup(popup_text, max_width=300)
                    
                    # Color marker based on severity
                    color_map = {1: 'green', 2: 'blue', 3: 'orange', 4: 'red', 5: 'darkred'}
                    color = color_map.get(row['Severity'], 'gray')
                    
                    folium.Marker(
                        [row['Latitude (Y)'], row['Longitude (X)']], 
                        popup=popup,
                        icon=folium.Icon(color=color, icon='info-sign')
                    ).add_to(marker_cluster)
        
        elif map_type == "Severity Map":
            # Create marker clusters for each severity level
            clusters = {}
            for severity in range(1, 6):
                clusters[severity] = MarkerCluster(name=f"Severity {severity}: {severity_labels.get(severity, 'Unknown')}").add_to(m)
            
            # Add markers for crime locations
            for idx, row in data.sample(min(5000, len(data))).iterrows():
                if pd.notna(row['Latitude (Y)']) and pd.notna(row['Longitude (X)']): 
                    # Create marker popup with crime details
                    popup_text = f"""
                    <b>Category:</b> {row['Category']}<br>
                    <b>Description:</b> {row['Descript']}<br>
                    <b>Date:</b> {row['Dates'].strftime('%Y-%m-%d %H:%M')}<br>
                    <b>Day:</b> {row['DayOfWeek']}<br>
                    <b>District:</b> {row['PdDistrict']}<br>
                    <b>Resolution:</b> {row['Resolution']}<br>
                    <b>Severity:</b> Level {row['Severity']}
                    """
                    popup = folium.Popup(popup_text, max_width=300)
                    
                    # Color marker based on severity
                    color_map = {1: 'green', 2: 'blue', 3: 'orange', 4: 'red', 5: 'darkred'}
                    color = color_map.get(row['Severity'], 'gray')
                    
                    # Add to appropriate cluster
                    target_cluster = clusters.get(row['Severity'], clusters[1])  # Default to severity 1 if not found
                    
                    folium.Marker(
                        [row['Latitude (Y)'], row['Longitude (X)']], 
                        popup=popup,
                        icon=folium.Icon(color=color, icon='info-sign')
                    ).add_to(target_cluster)
            
            # Add layer control
            folium.LayerControl().add_to(m)
        
        return m
    
    # Create and display map if data is available
    if not filtered_df.empty and 'Latitude (Y)' in filtered_df.columns and 'Longitude (X)' in filtered_df.columns:
        crime_map = create_map(filtered_df, map_type)
        folium_static(crime_map, width=1000, height=600)
    else:
        st.warning("No data available for the selected filters or coordinates are missing.")
    
    # Allow adding a new incident to the map
    st.markdown('<h3 class="sub-header">Add New Incident to Map</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_lat = st.number_input("Latitude", value=37.7749, format="%.6f")
        new_lon = st.number_input("Longitude", value=-122.4194, format="%.6f")
    
    with col2:
        new_category = st.selectbox("Crime Category", options=sorted(df['Category'].unique()) if not df.empty else ["LARCENY/THEFT"])
        new_description = st.text_input("Description", value="Enter crime description")
    
    if st.button("Add to Map"):
        # Create a map with the existing data
        if not filtered_df.empty and 'Latitude (Y)' in filtered_df.columns and 'Longitude (X)' in filtered_df.columns:
            m = folium.Map(
                location=[new_lat, new_lon], 
                zoom_start=14,
                tiles='CartoDB positron'
            )
            
            # Add marker for the new incident
            new_severity = assign_severity(new_category)
            color_map = {1: 'green', 2: 'blue', 3: 'orange', 4: 'red', 5: 'darkred'}
            color = color_map.get(new_severity, 'gray')
            
            popup_text = f"""
            <b>NEW INCIDENT</b><br>
            <b>Category:</b> {new_category}<br>
            <b>Description:</b> {new_description}<br>
            <b>Coordinates:</b> ({new_lat}, {new_lon})<br>
            <b>Severity:</b> Level {new_severity}
            """
            
            folium.Marker(
                [new_lat, new_lon], 
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)
            
            # Add circle for emphasis
            folium.Circle(
                location=[new_lat, new_lon],
                radius=200,
                color=color,
                fill=True,
                fill_opacity=0.2
            ).add_to(m)
            
            # Display the map
            folium_static(m, width=1000, height=600)
        else:
            st.warning("No data available to create the map.")

# Footer
st.markdown("---")
st.markdown("CityX Crime Watch Dashboard - Al Ruheil Al Ruheili © 2025")