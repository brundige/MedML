import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from PIL import Image
import random

# Page configuration
st.set_page_config(
    page_title="HAM10000 Skin Cancer Dataset Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🔬 HAM10000 Skin Cancer Dataset Analysis")
st.markdown("""
**Dataset Overview:** The HAM10000 dataset contains 10,015 dermatoscopic images of common pigmented skin lesions, 
categorized into seven different classes for machine learning applications in dermatology.
""")

# Load data function
@st.cache_data
def load_data():
    """Load the HAM10000 metadata"""
    path = "C:\\Users\\chrisb\\Desktop\\EMTS\\MedML\\dataset"
    df = pd.read_csv(os.path.join(path, "HAM10000_metadata.csv"))

    # Add readable lesion type names
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    is_malignant = {
        'nv': 0, 'mel': 1, 'bkl': 0, 'bcc': 1,
        'akiec': 1, 'vasc': 0, 'df': 0
    }

    df['lesion_type'] = df['dx'].map(lesion_type_dict)
    df['malignant'] = df['dx'].map(is_malignant)
    df['malignant_label'] = df['malignant'].map({0: 'Benign', 1: 'Malignant'})

    return df, lesion_type_dict, is_malignant

# Load the data
df, lesion_type_dict, is_malignant = load_data()

# Sidebar for navigation
st.sidebar.header("📊 Navigation")
section = st.sidebar.selectbox(
    "Choose Analysis Section:",
    ["Dataset Overview", "Class Distribution", "Demographics", "Data Quality", "Sample Images"]
)

# Main content based on selection
if section == "Dataset Overview":
    st.header("📈 Dataset Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Images", f"{len(df):,}")

    with col2:
        st.metric("Number of Classes", len(df['dx'].unique()))

    with col3:
        st.metric("Unique Patients", df['lesion_id'].nunique())

    with col4:
        malignant_pct = (df['malignant'].sum() / len(df)) * 100
        st.metric("Malignant Cases", f"{malignant_pct:.1f}%")

    # Dataset structure
    st.subheader("📋 Dataset Structure")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Dataset Dimensions:**")
        st.write(f"- Rows: {df.shape[0]:,}")
        st.write(f"- Columns: {df.shape[1]}")
        st.write(f"- Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    with col2:
        st.write("**Column Information:**")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Missing Values': df.isnull().sum().values
        })
        st.dataframe(col_info, use_container_width=True)

elif section == "Class Distribution":
    st.header("📊 Class Distribution Analysis")

    # Class distribution charts
    class_counts = df['dx'].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig_bar = px.bar(
            x=class_counts.index,
            y=class_counts.values,
            labels={'x': 'Diagnosis', 'y': 'Count'},
            title='Class Distribution (Bar Chart)'
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Pie chart
        fig_pie = px.pie(
            values=class_counts.values,
            names=[lesion_type_dict[x] for x in class_counts.index],
            title='Class Distribution (Pie Chart)'
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Class imbalance analysis
    st.subheader("⚠️ Class Imbalance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Detailed class statistics
        class_stats = pd.DataFrame({
            'Class': class_counts.index,
            'Full Name': [lesion_type_dict[x] for x in class_counts.index],
            'Count': class_counts.values,
            'Percentage': (class_counts.values / len(df)) * 100,
            'Type': [is_malignant[x] for x in class_counts.index]
        })
        class_stats['Type'] = class_stats['Type'].map({0: 'Benign', 1: 'Malignant'})
        st.dataframe(class_stats, use_container_width=True)

    with col2:
        # Benign vs Malignant
        malignant_counts = df['malignant_label'].value_counts()
        fig_malignant = px.pie(
            values=malignant_counts.values,
            names=malignant_counts.index,
            title='Benign vs Malignant Distribution',
            color_discrete_map={'Benign': '#2E86AB', 'Malignant': '#F24236'}
        )
        st.plotly_chart(fig_malignant, use_container_width=True)

    # Class imbalance warning
    max_class_pct = class_counts.max() / len(df) * 100
    if max_class_pct > 50:
        st.warning(f"⚠️ **Class Imbalance Detected**: The largest class represents {max_class_pct:.1f}% of the data. Consider using techniques like SMOTE, class weights, or stratified sampling.")

elif section == "Demographics":
    st.header("👥 Demographic Analysis")

    # Age analysis
    st.subheader("📊 Age Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Age histogram
        fig_age = px.histogram(
            df.dropna(subset=['age']),
            x='age',
            nbins=30,
            title='Age Distribution',
            labels={'age': 'Age (years)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        # Age statistics
        age_stats = df['age'].describe()
        st.write("**Age Statistics:**")
        for stat, value in age_stats.items():
            if not pd.isna(value):
                st.write(f"- {stat.capitalize()}: {value:.1f} years")

        missing_age = df['age'].isnull().sum()
        if missing_age > 0:
            st.write(f"- Missing values: {missing_age} ({missing_age/len(df)*100:.1f}%)")

    # Age by diagnosis
    st.subheader("📈 Age Distribution by Diagnosis")

    fig_age_dx = px.box(
        df.dropna(subset=['age']),
        x='dx',
        y='age',
        title='Age Distribution by Diagnosis Type',
        labels={'dx': 'Diagnosis', 'age': 'Age (years)'}
    )
    fig_age_dx.update_layout(height=500)
    st.plotly_chart(fig_age_dx, use_container_width=True)

    # Gender and localization analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚻 Gender Distribution")
        if 'sex' in df.columns:
            gender_counts = df['sex'].value_counts(dropna=False)
            fig_gender = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title='Gender Distribution'
            )
            st.plotly_chart(fig_gender, use_container_width=True)

    with col2:
        st.subheader("📍 Body Location")
        if 'localization' in df.columns:
            loc_counts = df['localization'].value_counts().head(10)
            fig_loc = px.bar(
                x=loc_counts.values,
                y=loc_counts.index,
                orientation='h',
                title='Top 10 Body Locations',
                labels={'x': 'Count', 'y': 'Body Location'}
            )
            fig_loc.update_layout(height=400)
            st.plotly_chart(fig_loc, use_container_width=True)

elif section == "Data Quality":
    st.header("🔍 Data Quality Assessment")

    # Missing values analysis
    st.subheader("❌ Missing Values Analysis")

    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100

    if missing_data.sum() > 0:
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

        if len(missing_df) > 0:
            fig_missing = px.bar(
                missing_df,
                x='Column',
                y='Missing Percentage',
                title='Missing Values by Column (%)',
                labels={'Missing Percentage': 'Missing %'}
            )
            st.plotly_chart(fig_missing, use_container_width=True)

            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")
    else:
        st.success("✅ No missing values found in the dataset!")

    # Duplicates analysis
    st.subheader("🔄 Duplicate Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        duplicate_rows = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicate_rows)
        if duplicate_rows > 0:
            st.warning(f"Found {duplicate_rows} duplicate rows")

    with col2:
        duplicate_lesions = df['lesion_id'].duplicated().sum()
        st.metric("Duplicate Lesion IDs", duplicate_lesions)
        if duplicate_lesions > 0:
            st.info(f"Some lesions have multiple images")

    with col3:
        duplicate_images = df['image_id'].duplicated().sum()
        st.metric("Duplicate Image IDs", duplicate_images)
        if duplicate_images > 0:
            st.error(f"Found {duplicate_images} duplicate image IDs")

    # Data consistency checks
    st.subheader("✅ Data Consistency")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Unique Counts:**")
        st.write(f"- Unique Lesions: {df['lesion_id'].nunique():,}")
        st.write(f"- Unique Images: {df['image_id'].nunique():,}")
        st.write(f"- Total Records: {len(df):,}")

    with col2:
        st.write("**Data Range Checks:**")
        if 'age' in df.columns:
            age_outliers = df[(df['age'] < 0) | (df['age'] > 100)].shape[0]
            if age_outliers > 0:
                st.error(f"❌ {age_outliers} age outliers found")
            else:
                st.success("✅ Age values within normal range")

elif section == "Sample Images":
    st.header("🖼️ Sample Images")

    # Use images directory structure
    image_path = "C:\\Users\\chrisb\\Desktop\\EMTS\\MedML\\dataset\\images"

    if os.path.exists(image_path):
        # List available classes (subdirectories)
        available_classes = [d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))]
        st.success(f"✅ Images directory found. Classes: {', '.join(available_classes)}")

        st.subheader("📸 Sample Images by Diagnosis (from directory)")
        selected_class = st.selectbox("Select Diagnosis Type:", available_classes)

        if selected_class:
            class_dir = os.path.join(image_path, selected_class)
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            st.write(f"Found {len(image_files)} images for class '{selected_class}'.")

            # Show up to 6 random images
            sample_images = random.sample(image_files, min(6, len(image_files))) if image_files else []
            cols = st.columns(6)
            for idx, img_name in enumerate(sample_images):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = Image.open(img_path)
                    cols[idx].image(img, caption=img_name, use_column_width=True)
                except Exception as e:
                    cols[idx].error(f"Error loading image: {img_name}")
    else:
        st.error("❌ Images directory not found. Please check the path.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** HAM10000 Skin Cancer")
st.sidebar.markdown("**Source:** Kaggle")
st.sidebar.markdown("**Purpose:** ML Model Development")