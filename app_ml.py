import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import urllib.parse
import difflib
import matplotlib.pyplot as plt
import seaborn as sns

sns_mar = None # Standard styling safe fallback

# ML & NLP Engineering Suite
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Model Arsenal
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor

# Metric Suite
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
)

# =====================================================================
# 1. PREMIUM PAGE SETUP & COLOR THEMING
# =====================================================================
st.set_page_config(page_title="CinemAI | Premium Movie Analytics & Insights", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        background: linear-gradient(45deg, #ff007f, #7f00ff);
        color: white; border: none; padding: 12px 30px;
        font-weight: bold; border-radius: 8px;
        box-shadow: 0 4px 15px rgba(255, 0, 127, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: scale(1.03); box-shadow: 0 6px 20px rgba(127, 0, 255, 0.6); }
    .metric-card-custom {
        background: #161b22; border: 1px solid #2d3748; border-left: 5px solid #00f0ff;
        padding: 20px; border-radius: 10px; margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0, 240, 255, 0.1);
    }
    .metric-title-custom { margin: 0; color: #8892b0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value-custom { font-size: 2.2rem; font-weight: 800; color: #00f0ff; margin: 5px 0; }
    .metric-subtitle-custom { margin: 0; color: #5cc2a5; font-size: 0.8rem; }
    h1, h2, h3, h4 { font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 CinemAI: Premium Smart Movie Analytics")
st.markdown("<p style='color: #00f0ff; font-size: 1.2rem;'>Advanced Content Performance & AI-Powered Recommendation Studio</p>", unsafe_allow_html=True)
st.write("---")

# =====================================================================
# 2. APPLICATION STATE MANAGEMENT
# =====================================================================
if "search_history" not in st.session_state:
    st.session_state["search_history"] = []
if "k_val" not in st.session_state:
    st.session_state["k_val"] = 5
if "depth_val" not in st.session_state:
    st.session_state["depth_val"] = 6
if "rf_val" not in st.session_state:
    st.session_state["rf_val"] = 40

def clean_genre_string(text):
    if pd.isna(text) or text == '': return 'Unknown'
    found_genres = re.findall(r"'name':\s*'([^']+)'", str(text))
    return ", ".join(found_genres) if found_genres else ", ".join([g.strip() for g in str(text).split(',') if g.strip()])

# =====================================================================
# 3. DATA HANDLING & CLEANING
# =====================================================================
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv('movies_metadata.csv', low_memory=False)
    except FileNotFoundError:
        st.error("🚨 Core File Missing: Please ensure 'movies_metadata.csv' is in the root folder.")
        st.stop()
        
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(df['vote_average'].mean())
    df['title'] = df['title'].fillna('Untitled Movie')
    df['overview'] = df['overview'].fillna('')
    df['clean_genres'] = df['genres'].apply(clean_genre_string)
    df['overview_word_count'] = df['overview'].apply(lambda x: len(str(x).split()))
    df = df[df['title'] != 'Untitled Movie'].sort_values(by='popularity', ascending=False).head(25000).reset_index(drop=True)
    return df

df = load_and_preprocess_data()

# =====================================================================
# 4. INTEL NLP & ML ENGINEERING ARCHITECTURE 
# =====================================================================
vectorizer = CountVectorizer(max_features=18, tokenizer=lambda x: [g.strip() for g in x.split(',') if g.strip()])
genre_features = vectorizer.fit_transform(df['clean_genres']).toarray()
genre_labels = vectorizer.get_feature_names_out()

scaler = StandardScaler()
numerical_features = df[['popularity', 'vote_count', 'overview_word_count']].values
scaled_numerical = scaler.fit_transform(numerical_features)

X = np.hstack((genre_features, scaled_numerical))
y_class = (df['vote_average'] >= 6.5).astype(int)
y_reg = df['vote_average'].values

X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

kmeans_engine = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster_signature'] = kmeans_engine.fit_predict(X)

cluster_names = {
    0: "🔥 Pop Culture Blockbusters", 1: "🎭 Intense Psychological Dramas",
    2: "🌌 High-Budget Sci-Fi & Action", 3: "🍿 Lighthearted Family Hits",
    4: "🎬 Indie Gems & Cult Classics"
}
df['taste_group'] = df['cluster_signature'].map(cluster_names).fillna("Universal Appeal")

tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))

# =====================================================================
# 5. SIDEBAR CONTROLS
# =====================================================================
st.sidebar.markdown("<h2 style='color: #ff007f;'>🎛️ Platform Tuning Studio</h2>", unsafe_allow_html=True)

user_hyperparameter_k = st.sidebar.slider("AI Neural Scan Range:", min_value=1, max_value=15, value=st.session_state["k_val"], key="k_slider")
tree_depth = st.sidebar.slider("Narrative Processing Depth:", min_value=2, max_value=15, value=st.session_state["depth_val"], key="tree_slider")
rf_estimators = st.sidebar.slider("Ensemble Computing Power:", min_value=10, max_value=100, value=st.session_state["rf_val"], key="rf_slider")

st.session_state["k_val"] = user_hyperparameter_k
st.session_state["depth_val"] = tree_depth
st.session_state["rf_val"] = rf_estimators

if st.sidebar.button("🔄 Reset Tuning Profiles"):
    st.session_state["k_val"] = 5; st.session_state["depth_val"] = 6; st.session_state["rf_val"] = 40
    st.rerun()

st.sidebar.write("---")
st.sidebar.markdown("<h3 style='color: #00f0ff;'>🍿 Your Recent Spotlight Choices</h3>", unsafe_allow_html=True)
if st.sidebar.button("🗑️ Clear Search History"):
    st.session_state["search_history"] = []; st.rerun()

if st.session_state["search_history"]:
    for past_movie in reversed(st.session_state["search_history"]): st.sidebar.text(f"🎬 {past_movie}")
else:
    st.sidebar.caption("No recent titles tracked yet.")

# =====================================================================
# 6. MOVIE EXPLORER PORTAL (WITH CLEAN BACKGROUND FUZZY MATCHING)
# =====================================================================
st.markdown("### 🔍 Discover Movie Profiles")
search_query = st.text_input("Search titles:", placeholder="Type a movie title here...")

if search_query:
    # First attempt: standard case-insensitive keyword phrase match
    query_words = [w.lower() for w in re.findall(r'\b\w+\b', search_query)]
    matched_results = df[df['title'].apply(lambda x: all(word in str(x).lower() for word in query_words))]
    
    # Fuzzy Matching Fallback: If no direct hits found, correct typos quietly in the background
    if matched_results.empty:
        all_titles = df['title'].tolist()
        closest_matches = difflib.get_close_matches(search_query, all_titles, n=1, cutoff=0.5)
        if closest_matches:
            matched_results = df[df['title'] == closest_matches[0]]
    
    if not matched_results.empty:
        matched_row = matched_results.iloc[0]
        selected_movie = matched_row['title']
        target_idx = matched_row.name  
        current_taste_group = matched_row['taste_group']
        
        if selected_movie not in st.session_state["search_history"]:
            st.session_state["search_history"].append(selected_movie)
        
        tgt_youtube_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(selected_movie + ' official trailer')}"
        tgt_netflix_url = f"https://www.netflix.com/search?q={urllib.parse.quote(selected_movie)}"
        
        st.markdown(f"""
            <div style='background: linear-gradient(90deg, #1f1f2e, #0d0d13); padding: 22px; border-radius: 12px; border: 1px solid #7f00ff; margin-bottom: 15px;'>
                <div style='display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;'>
                    <div>
                        <span style='color: #00f0ff; font-weight: bold; font-size: 1rem;'>✨ Current Spotlight:</span> 
                        <span style='font-size: 1.6rem; font-weight: bold; margin-left: 10px; color: #ffffff;'>{selected_movie}</span>
                        <br><span style='color: #aaa; font-size: 0.85rem;'>Genres: {matched_row['clean_genres']} | <strong style='color:#00f0ff;'>Category: {current_taste_group}</strong></span>
                    </div>
                    <div style='text-align: right;'>
                        <span style='color: #ff007f; font-size: 0.85rem; font-weight: bold;'>AUDIENCE SCORE</span>
                        <h3 style='color: #ff007f; margin: 0;'>{matched_row['vote_average']:.2f} <span style='font-size: 0.9rem; color: #888;'>/ 10 ★</span></h3>
                    </div>
                </div>
                <p style='color: #ddd; font-style: italic; margin: 12px 0;'>"{matched_row['overview'] if matched_row['overview'] != '' else 'No description available.'}"</p>
                <div style='display: flex; gap: 10px; max-width: 320px;'>
                    <a href='{tgt_youtube_url}' target='_blank' style='flex: 1; text-align: center; background-color: #e50914; color: white; padding: 8px 0; font-size: 0.85rem; font-weight: bold; border-radius: 6px; text-decoration: none;'>📺 Play Trailer</a>
                    <a href='{tgt_netflix_url}' target='_blank' style='flex: 1; text-align: center; background-color: #22252a; color: #00f0ff; padding: 8px 0; font-size: 0.85rem; font-weight: bold; border-radius: 6px; text-decoration: none; border: 1px solid #00f0ff;'>🎬 Netflix</a>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Match Similar Content & Run Predictive Analytics"):
            # --- MODEL COMPUTATIONS ---
            rf = RandomForestClassifier(n_estimators=rf_estimators, max_depth=tree_depth, random_state=42).fit(X_train, y_train_c)
            rf_preds = rf.predict(X_test)
            
            lr = LogisticRegression(max_iter=1000).fit(X_train, y_train_c)
            lr_preds = lr.predict(X_test)
            
            lin_reg = LinearRegression().fit(X_train_r, y_train_r)
            lin_preds = lin_reg.predict(X_test_r)
            
            knn = KNeighborsRegressor(n_neighbors=user_hyperparameter_k, metric='euclidean').fit(X_train_r, y_train_r)
            knn_preds = knn.predict(X_test_r)

            # 🛠️ Cross-Validation Logic
            cv_scores = cross_val_score(rf, X, y_class, cv=5, scoring='accuracy')
            
            # 🛠️ Bias-Variance Generalization Gap Trackers
            rf_train_preds = rf.predict(X_train)
            train_accuracy = accuracy_score(y_train_c, rf_train_preds)
            test_accuracy = accuracy_score(y_test_c, rf_preds)
            bias_variance_gap = abs(train_accuracy - test_accuracy)

            # =====================================================================
            # 7. PERFORMANCE & ADVANCED TECHNICAL VALIDATION METRICS
            # =====================================================================
            col1, col2 = st.columns([1.1, 0.9])
            with col1:
                st.markdown("#### 📊 Content Prediction Performance (Hit vs Miss)")
                metrics_data = {
                    "Evaluation Metric": ["Core Forecasting Accuracy", "Prediction Precision Rate", "Sensitivity (Recall Score)", "Overall Balance Factor (F1)"],
                    "Standard Engine": [f"{accuracy_score(y_test_c, lr_preds) * 100:.1f}%", f"{precision_score(y_test_c, lr_preds, zero_division=0) * 100:.1f}%", f"{recall_score(y_test_c, lr_preds, zero_division=0) * 100:.1f}%", f"{f1_score(y_test_c, lr_preds, zero_division=0) * 100:.1f}%"],
                    "Advanced AI Engine": [f"{accuracy_score(y_test_c, rf_preds) * 100:.1f}%", f"{precision_score(y_test_c, rf_preds, zero_division=0) * 100:.1f}%", f"{recall_score(y_test_c, rf_preds, zero_division=0) * 100:.1f}%", f"{f1_score(y_test_c, rf_preds, zero_division=0) * 100:.1f}%"]
                }
                st.table(pd.DataFrame(metrics_data))
                
                st.markdown("#### 📈 Continuous Score Prediction Engine Metrics")
                reg_comparison_data = {
                    "Algorithm Setup": ["Baseline Trend Matrix", "Advanced Proximity Matcher"],
                    "Average Error Margin (MSE)": [f"{mean_squared_error(y_test_r, lin_preds):.3f}", f"{mean_squared_error(y_test_r, knn_preds):.3f}"],
                    "Reliability Index Score (R²)": [f"{r2_score(y_test_r, lin_preds) * 100:.1f}%", f"{r2_score(y_test_r, knn_preds) * 100:.1f}%"]
                }
                st.table(pd.DataFrame(reg_comparison_data))
                
            with col2:
                st.markdown("#### 🧠 Architecture Validation Diagnostics")
                
                st.markdown(f"""
                    <div class='metric-card-custom' style='border-left: 5px solid #7f00ff;'>
                        <p class='metric-title-custom'>5-Fold Cross-Validation Accuracy</p>
                        <p class='metric-value-custom' style='color:#7f00ff;'>{cv_scores.mean() * 100:.1f}%</p>
                        <p class='metric-subtitle-custom'>🔄 Model stability score across entire dataset</p>
                    </div>
                    <div class='metric-card-custom' style='border-left: 5px solid #ff007f;'>
                        <p class='metric-title-custom'>Bias‑Variance Generalization Gap</p>
                        <p class='metric-value-custom' style='color:#ff007f;'>{bias_variance_gap * 100:.1f}%</p>
                        <p class='metric-subtitle-custom'>📊 Train vs Test divergence (Lower is better)</p>
                    </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📚 Engineering Log: Cross-Validation & Generalization Assessment"):
                    st.caption("**Robustness Diagnostics:**")
                    clean_scores_formatted = ", ".join([f"{round(float(score) * 100, 1)}%" for score in cv_scores])
                    st.write(f"The model's cross-validated scoring consistency holds within individual array blocks of: **[{clean_scores_formatted}]**.")
                    st.write(f"With a dynamic Training Accuracy of {train_accuracy*100:.1f}% vs Test Accuracy of {test_accuracy*100:.1f}%, the system's variance overhead stays tightly managed inside real-time scaling limits.")

            st.write("---")
            
            # =====================================================================
            # 8. DIAGNOSTICS & PLOTS
            # =====================================================================
            st.markdown("#### 🛡️ Platform Technical Diagnostics & Feature Weights")
            plot_col1, plot_col2 = st.columns(2)
            plt.style.use('dark_background')
            with plot_col1:
                extended_labels = list(genre_labels) + ['Scaled Popularity', 'Scaled Vote Volume', 'Plot Summary Depth']
                importance = rf.feature_importances_[:len(extended_labels)]
                feat_df = pd.DataFrame({'Engine Dimension': extended_labels, 'Influence Weight': importance}).sort_values(by='Influence Weight', ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.barplot(x='Influence Weight', y='Engine Dimension', data=feat_df, palette='spring', ax=ax)
                ax.set_title("Top 10 Computational Factors")
                st.pyplot(fig)
            with plot_col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                cm = confusion_matrix(y_test_c, rf_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Modest Performer', 'Blockbuster Hit'])
                disp.plot(cmap='magma', ax=ax, values_format='d')
                ax.set_title("Hit vs. Miss Tracking Success Matrix")
                st.pyplot(fig)

            st.markdown("#### 🌌 Automated Cohort Distribution Matrix")
            cluster_counts = df['cluster_signature'].value_counts().sort_index()
            c_labeled_names = [cluster_names[i] for i in cluster_counts.index]
            c_df = pd.DataFrame({'Audience Cohort Category': c_labeled_names, 'Active Movies Cataloged': cluster_counts.values})
            
            c_col1, c_col2 = st.columns([1.2, 0.8])
            with c_col1: st.dataframe(c_df, use_container_width=True, hide_index=True)
            with c_col2: st.write(f"**Context Evaluation:** The title matches directly into the **{current_taste_group}** catalog partition.")

            st.write("---")

            # =====================================================================
            # 9. HYBRID RECO ENGINE (UX CLEAN REWRITE)
            # =====================================================================
            st.markdown("<h3 style='color: #00f0ff;'>🍿 More Like This</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color: #8892b0; font-size: 0.9rem; margin-top:-10px; margin-bottom:20px;'>Handpicked recommendations based on your favorite titles</p>", unsafe_allow_html=True)
            
            text_sim_scores = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix).flatten()
            meta_sim_scores = cosine_similarity(X[target_idx].reshape(1, -1), X).flatten()
            hybrid_match_score = (text_sim_scores * 0.7) + (meta_sim_scores * 0.3)
            
            df['story_similarity_score'] = hybrid_match_score
            df['predicted_rating'] = knn.predict(X)
            
            final_recommendations = df[df['title'] != selected_movie].sort_values(by='story_similarity_score', ascending=False).head(5)
            
            if not final_recommendations.empty:
                for _, row in final_recommendations.iterrows():
                    raw_overview = row['overview'] if pd.notna(row['overview']) else ''
                    youtube_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(row['title'] + ' official trailer')}"
                    netflix_url = f"https://www.netflix.com/search?q={urllib.parse.quote(row['title'])}"
                    
                    st.markdown(f"""
                        <div style='background-color: #161b22; padding: 20px; border-radius: 10px; border-left: 5px solid #00f0ff; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.3);'>
                            <div style='display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;'>
                                <div style='flex: 1; min-width: 300px;'>
                                    <div style='display: flex; align-items: center; gap: 12px;'>
                                        <span style='font-size: 1.3rem; font-weight: bold; color: #ffffff;'>🎬 {row['title']}</span>
                                        <span style='background: #00f0ff; color: #0e1117; font-size: 0.75rem; padding: 3px 8px; font-weight: bold; border-radius: 4px;'>{int(row['story_similarity_score'] * 100)}% Match</span>
                                    </div>
                                    <p style='color: #ff007f; font-size: 0.85rem; margin: 4px 0; font-weight: bold;'>🎭 {row['clean_genres']} | <span style='color:#00f0ff;'>Group: {row['taste_group']}</span></p>
                                    <p style='color: #b3b3b3; font-size: 0.9rem; line-height: 1.4; margin-top: 8px;'>{raw_overview if raw_overview != '' else 'No plot dynamic captured.'}</p>
                                </div>
                                <div style='text-align: right; min-width: 160px; display: flex; flex-direction: column; justify-content: space-between;'>
                                    <div>
                                        <span style='color: #8892b0; font-size: 0.75rem;'>PREDICTED SCORE</span>
                                        <h3 style='color: #00f0ff; margin: 2px 0 12px 0;'>{row['predicted_rating']:.2f} <span style='font-size: 0.8rem; color: #888;'>/ 10 ★</span></h3>
                                    </div>
                                    <div style='display: flex; gap: 8px; justify-content: flex-end;'>
                                        <a href='{youtube_url}' target='_blank' style='padding: 6px 14px; background-color: #e50914; color: white; font-size: 0.8rem; font-weight: bold; border-radius: 5px; text-decoration: none;'>📺 Trailer</a>
                                        <a href='{netflix_url}' target='_blank' style='padding: 6px 14px; background-color: #22252a; color: #00f0ff; font-size: 0.8rem; font-weight: bold; border-radius: 5px; text-decoration: none; border: 1px solid #00f0ff;'>🎬 Netflix</a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No matching profiles parsed.")
    else:
        st.error("❌ Title not found in the archive. Try another search!")
