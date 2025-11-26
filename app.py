# --- Imports: All libraries needed for Phases 1-7 ---
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns  # For the confusion matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import time

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# --- App Configuration ---
st.set_page_config(
    page_title="Customer Personality Analysis",
    page_icon="🤖",
    layout="wide"
)

# --- Caching: Speed up the app ---
# These @st.cache_data decorators tell Streamlit to not re-run 
# these functions if the input hasn't changed, saving a lot of time.

# --- Phase 1: Project Definition & Data Understanding ---
@st.cache_data
def load_data(file):
    """Loads and cleans the column names of an uploaded file."""
    try:
        # We know the file is tab-separated, so we tell pandas directly.
        df_raw = pd.read_csv(file, sep='\t')
    except Exception as e:
        # If this fails, then the file is truly unreadable.
        st.error(f"Error loading data: Could not parse file. {e}")
        return None
    
    # Force all columns to lowercase and strip whitespace
    df_raw.columns = df_raw.columns.str.lower().str.strip()
    return df_raw

# --- Phase 2: Automated Preprocessing & Feature Engineering ---
@st.cache_data
def preprocess_data(df):
    """
    Cleans, engineers features, and scales the customer data.
    This version is robust to missing columns.
    """
    df_processed = df.copy()
    
    # 1. Drop useless columns (if they exist)
    df_processed = df_processed.drop(columns=['id', 'z_costcontact', 'z_revenue'], errors='ignore')
    
    # --- Feature Engineering & Imputation (All Optional) ---
    current_year = date.today().year
    
    if 'income' in df_processed.columns:
        median_income = df_processed['income'].median()
        df_processed['income'] = df_processed['income'].fillna(median_income)
    
    if 'year_birth' in df_processed.columns:
        df_processed['age'] = current_year - df_processed['year_birth']

    if 'dt_customer' in df_processed.columns:
        try:
            df_processed['dt_customer'] = pd.to_datetime(df_processed['dt_customer'], format='%d-%m-%Y')
        except ValueError:
            try:
                df_processed['dt_customer'] = pd.to_datetime(df_processed['dt_customer'])
            except Exception:
                df_processed['dt_customer'] = pd.NaT # Failed to parse
        
        if pd.api.types.is_datetime64_any_dtype(df_processed['dt_customer']):
             df_processed['years_customer'] = current_year - df_processed['dt_customer'].dt.year

    if all(col in df_processed.columns for col in ['marital_status', 'kidhome', 'teenhome']):
        df_processed['adults'] = df_processed['marital_status'].apply(lambda x: 2 if x in ['married', 'together'] else 1)
        df_processed['family_size'] = df_processed['adults'] + df_processed['kidhome'] + df_processed['teenhome']
    
    mnt_cols = [col for col in df_processed.columns if 'mnt' in col]
    if mnt_cols: # Only if at least one 'mnt' column exists
        df_processed['total_spent'] = df_processed[mnt_cols].sum(axis=1)
    
    # --- Feature Selection (for clustering) ---
    # Build the list based on what was successfully created
    cluster_features = []
    possible_features = ['income', 'age', 'total_spent', 'family_size', 'years_customer']
    for feat in possible_features:
        if feat in df_processed.columns:
            # Ensure the column is numeric before adding
            if pd.api.types.is_numeric_dtype(df_processed[feat]):
                cluster_features.append(feat)
            else:
                st.warning(f"Column '{feat}' was skipped for clustering as it is not numeric.")
            
    if not cluster_features:
        st.error("Error: The uploaded file does not contain any of the required *numeric* columns (like 'income', 'age', 'total_spent', etc.) to perform an analysis.")
        return None, None, None # Stop the pipeline

    # --- Scaling ---
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_processed[cluster_features])
    df_scaled = pd.DataFrame(scaled_features, columns=cluster_features)
    
    return df_processed, df_scaled, cluster_features # Also return the list of features we used

# --- Phase 3: Unsupervised Segmentation (The Discovery) ---
@st.cache_data
def find_clusters(df_scaled, optimal_k=4):
    """Finds optimal K (Elbow) and runs K-Means."""
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(df_scaled) 
    return kmeans.labels_

# --- Phase 4: Dynamic Cluster Profiling (The Naming) ---
@st.cache_data
def profile_clusters(df_clean, cluster_labels, cluster_features):
    """Analyzes the clusters and returns a profile DataFrame."""
    df_clean['cluster'] = cluster_labels
    
    profile_features = cluster_features.copy()
    
    other_features = ['mntwines', 'mntmeatproducts', 'numdealspurchases']
    for feat in other_features:
        if feat in df_clean.columns and feat not in profile_features:
            if pd.api.types.is_numeric_dtype(df_clean[feat]):
                profile_features.append(feat)
            
    cluster_profile = df_clean.groupby('cluster')[profile_features].mean().reset_index()
    return df_clean, cluster_profile

# --- Phase 5: Predictive Model Development (The Engine) ---
@st.cache_data
def train_model(df_clean, cluster_features):
    """Trains a classifier to predict the cluster labels."""
    
    features = cluster_features.copy()
    
    # --- FIXED: Added the missing columns here ---
    other_features = ['mntwines', 'mntmeatproducts', 'numdealspurchases', 'numwebpurchases',
                      'numcatalogpurchases', 'numstorepurchases', 
                      'mntfruits', 'mntfishproducts', 'mntsweetproducts', 'mntgoldprods']
    
    for feat in other_features:
        if feat in df_clean.columns and feat not in features:
            if pd.api.types.is_numeric_dtype(df_clean[feat]):
                features.append(feat)

    if not features:
        st.error("Error: No predictive features found in the data.")
        return None, None, None, None, None

    X = df_clean[features]
    y = df_clean['cluster']

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # --- Create Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    
    return rf_model, accuracy, report, features, cm

# --- Phase 7: Streamlit Deployment (The GUI) ---

# Title
st.title("🤖 Automated Customer Personality Analysis Tool")

# Sidebar for User Journey
st.sidebar.title("Your Workflow")
st.sidebar.markdown("""
This tool builds a custom prediction model for your business.
1.  **Upload:** Add your customer CSV file.
2.  **Analyze:** The tool finds hidden groups and trains a model.
3.  **Discover:** See the profiles of your customer segments.
4.  **Act:** Use the tool to predict new customers!
""")

# --- 1. Upload ---
st.sidebar.header("1. Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload your customer CSV file", type=["csv", "tsv"])

# We use session_state to store the results of the pipeline
if "pipeline_complete" not in st.session_state:
    st.session_state.pipeline_complete = False

if uploaded_file:
    # --- 2. Analyze (Button to run the pipeline) ---
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Processing your data... This may take a moment."):
            
            st.session_state.df_raw = load_data(uploaded_file)
            if st.session_state.df_raw is None:
                st.stop() # Stop if loading failed
            
            df_clean, df_scaled, cluster_features = preprocess_data(st.session_state.df_raw)
            
            if df_clean is None:
                st.session_state.pipeline_complete = False
                st.stop() # Stop execution
            
            st.session_state.df_clean = df_clean
            st.session_state.cluster_features = cluster_features
            
            cluster_labels = find_clusters(df_scaled, optimal_k=4)
            
            df_with_labels, profile = profile_clusters(st.session_state.df_clean, cluster_labels, st.session_state.cluster_features)
            st.session_state.df_with_labels = df_with_labels
            st.session_state.profile = profile
            
            # --- Receive cm ---
            model, acc, report, model_features, cm = train_model(st.session_state.df_with_labels, st.session_state.cluster_features)
            
            # Check if model training failed
            if model is None:
                st.session_state.pipeline_complete = False
                st.stop()

            st.session_state.model = model
            st.session_state.accuracy = acc
            st.session_state.report = report
            st.session_state.model_features = model_features
            
            # --- Save cm ---
            st.session_state.cm = cm 
            
            st.session_state.pipeline_complete = True
            time.sleep(1) 
        
        st.success("Analysis complete! View the results below.")

# --- 3. Discover (Show results) ---
if st.session_state.pipeline_complete:
    st.header("Results: Your Customer Segments")
    st.markdown("Your customers have been grouped into 4 distinct segments.")
    
    st.subheader("Segment Profiles (Averages)")
    st.dataframe(st.session_state.profile.style.format("{:.2f}"))
    st.markdown(f"""
    * **Clustering was based on these features:** `{(', '.join(st.session_state.cluster_features))}`
    """)
    
    # --- Model Performance Section ---
    st.divider()
    st.header("📈 Model Performance & Metrics")
    st.markdown("Here is how well the model learned to predict your new segments.")
    
    # 1. Display Accuracy
    st.metric(label="**Model Accuracy**", value=f"{st.session_state.accuracy * 100:.2f}%")
    
    # 2. Display Classification Report
    st.subheader("Classification Report")
    st.markdown("This shows the precision, recall, and f1-score for each cluster.")
    
    report_df = pd.DataFrame(st.session_state.report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    # 3. Display Confusion Matrix
    st.subheader("Confusion Matrix")
    st.markdown("""
    This graph shows where the model got confused.
    -   The **diagonal** (top-left to bottom-right) shows **correct** predictions.
    -   Any numbers *off* the diagonal are **incorrect** predictions.
    """)
    
    # Get cluster labels (e.g., 0, 1, 2, 3)
    cluster_labels = sorted(st.session_state.profile.cluster.unique())
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(st.session_state.cm, 
                annot=True, 
                fmt='g', 
                cmap='Blues', 
                xticklabels=cluster_labels, 
                yticklabels=cluster_labels,
                ax=ax)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Model Confusion Matrix')
    st.pyplot(fig)


    # --- 3.5. Name Your Segments ---
    st.divider()
    st.subheader("Name Your Segments")
    st.markdown("Based on the profiles above, give your segments a meaningful name.")
    
    # Create text boxes to get names
    col1, col2 = st.columns(2)
    with col1:
        name_0 = st.text_input("Name for Cluster 0", value="Cluster 0")
        name_1 = st.text_input("Name for Cluster 1", value="Cluster 1")
    with col2:
        name_2 = st.text_input("Name for Cluster 2", value="Cluster 2")
        name_3 = st.text_input("Name for Cluster 3", value="Cluster 3")
    
    # --- 4. Act (Predict) ---
    st.divider()
    st.header("Act: Predict New Customers")
    st.markdown("Enter the details of a new customer to predict their segment.")
    
    expected_features = ['age', 'income', 'total_spent', 'family_size', 'years_customer',
                         'mntwines', 'mntmeatproducts', 'numdealspurchases', 'numwebpurchases',
                         'numcatalogpurchases', 'numstorepurchases', 'mntfruits', 
                         'mntfishproducts', 'mntsweetproducts', 'mntgoldprods']
    
    model_feature_set = set(st.session_state.model_features)
    expected_feature_set = set(expected_features)

    if expected_feature_set.issubset(model_feature_set):
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### Demographics")
                age = st.number_input("Age", min_value=18, max_value=120, value=40)
                income = st.number_input("Income", min_value=0, max_value=700000, value=50000)
                family_size = st.slider("Family Size", min_value=1, max_value=10, value=2)
                years_customer = st.slider("Years as Customer", min_value=0, max_value=20, value=3)
            
            with col2:
                st.markdown("##### Spending (Mnt)")
                mnt_wines = st.number_input("Spent on Wine", min_value=0, value=300)
                mnt_meat = st.number_input("Spent on Meat", min_value=0, value=150)
                mnt_fruits = st.number_input("Spent on Fruits", min_value=0, value=20)
                mnt_fish = st.number_input("Spent on Fish", min_value=0, value=20)
                mnt_sweets = st.number_input("Spent on Sweets", min_value=0, value=20)
                mnt_gold = st.number_input("Spent on Gold", min_value=0, value=20)
            
            with col3:
                st.markdown("##### Purchases (Num)")
                num_deals = st.number_input("Purchases with Deals", min_value=0, value=2)
                num_web = st.number_input("Web Purchases", min_value=0, value=4)
                num_catalog = st.number_input("Catalog Purchases", min_value=0, value=2)
                num_store = st.number_input("Store Purchases", min_value=0, value=5)
                
            submitted = st.form_submit_button("Predict Personality")

            if submitted:
                # Create the name map from the text boxes
                cluster_names_map = {
                    0: name_0,
                    1: name_1,
                    2: name_2,
                    3: name_3
                }
                
                # Correct total_spent calculation
                total_spent = mnt_wines + mnt_meat + mnt_fruits + mnt_fish + mnt_sweets + mnt_gold
                
                # Use correct column names from original file
                input_data_dict = {
                    'age': age, 'income': income, 'total_spent': total_spent, 
                    'family_size': family_size, 'years_customer': years_customer,
                    'mntwines': mnt_wines, 
                    'mntmeatproducts': mnt_meat, 
                    'numdealspurchases': num_deals, 
                    'numwebpurchases': num_web,
                    'numcatalogpurchases': num_catalog, 
                    'numstorepurchases': num_store,
                    'mntfruits': mnt_fruits,
                    'mntfishproducts': mnt_fish, 
                    'mntsweetproducts': mnt_sweets, 
                    'mntgoldprods': mnt_gold
                }
                
                # Build the input data list in the exact order the model expects
                input_data = []
                for feat in st.session_state.model_features:
                    if feat in input_data_dict:
                        input_data.append(input_data_dict[feat])
                    else:
                        input_data.append(0) 

                input_df = pd.DataFrame([input_data], columns=st.session_state.model_features)
                
                prediction = st.session_state.model.predict(input_df)
                prediction_proba = st.session_state.model.predict_proba(input_df)
                
                # Use the new map to get the name
                cluster_num = prediction[0]
                cluster_name = cluster_names_map.get(cluster_num, f"Cluster {cluster_num}")

                st.subheader(f"Prediction: This customer is a **{cluster_name}**")
                st.progress(prediction_proba.max())
                st.write(f"Confidence: {prediction_proba.max()*100:.2f}%")
                
                st.subheader("How this prediction compares to the cluster:")
                col1_res, col2_res = st.columns(2)
                with col1_res:
                    st.write("**New Customer's Profile:**")
                    st.dataframe(input_df)
                with col2_res:
                    st.write(f"**Cluster {cluster_num}'s Average Profile:**")
                    # Ensure cluster column is treated as int for comparison
                    profile_df = st.session_state.profile
                    profile_df['cluster'] = profile_df['cluster'].astype(int)
                    st.dataframe(profile_df[profile_df.cluster == cluster_num])
    
    else:
        st.warning("Prediction Form Disabled")
        st.markdown(f"""
        The prediction form is disabled because the uploaded file has a non-standard set of columns.
        
        The model was trained on the following features:
        `{st.session_state.model_features}`
        
        To use the prediction form, please upload a file with the standard columns.
        """)


# --- Initial Landing Page ---
else:
    st.info("Upload your customer CSV file in the sidebar to begin.")
    st.subheader("Project Overview")
    st.markdown("""
    This tool allows you to upload your own customer data and:
    1.  **Discover** hidden segments using K-Means clustering.
    2.  **Profile** these segments to understand them (e.g., "VIPs", "Bargain Hunters").
    3.  **Train** a custom machine learning model on your data.
    4.  **Predict** which segment a new customer will fall into.
    """)
    st.markdown("Don't have a file? You can download the test data [from Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis).")
