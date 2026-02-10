import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix

# --------------------------------------------------
# Load models & data (cached)
# --------------------------------------------------
@st.cache_resource
def load_xgb():
    model = xgb.XGBRegressor()
    model.load_model("models/xgb_model.json")
    return model

@st.cache_resource
def load_assets():
    user_item_sparse = joblib.load("models/cf_user_item_sparse.joblib")

    feature_schema = joblib.load("models/xgb_feature_schema.pkl")

    tfidf_matrix = joblib.load("models/cbf_tfidf_matrix.joblib")

    with open("models/item_ids.pkl", "rb") as f:
        item_ids = pickle.load(f)

    with open("models/hybrid_config.pkl", "rb") as f:
        hybrid_config = pickle.load(f)

    with open("models/course_metadata.pkl", "rb") as f:
        courses_df = pickle.load(f)

    with open("models/interactions.pkl", "rb") as f:
        interactions_df = pickle.load(f)

    # XGBoost assets
    with open("models/xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)

    with open("models/user_encoder.pkl", "rb") as f:
        user_encoder = pickle.load(f)

    with open("models/course_encoder.pkl", "rb") as f:
        course_encoder = pickle.load(f)

    with open("models/ohe.pkl", "rb") as f:
        ohe = pickle.load(f)

    with open("models/course_stats.pkl", "rb") as f:
        course_stats = pickle.load(f)

    with open("models/user_stats.pkl", "rb") as f:
        user_stats = pickle.load(f)

    courses_df = courses_df.drop_duplicates(subset="course_id")

    courses_df[['difficulty_level', 'instructor']] = (
    courses_df[['difficulty_level', 'instructor']]
    .fillna("unknown")
    .astype(str)
    )

    return (
        user_item_sparse, tfidf_matrix, item_ids, hybrid_config, courses_df,
        xgb_model, user_encoder, course_encoder, ohe, course_stats, user_stats,interactions_df, feature_schema
    )

(
    user_item_sparse, tfidf_matrix, item_ids, hybrid_config, courses_df,
    xgb_model, user_encoder, course_encoder, ohe, course_stats, user_stats,interactions_df, feature_schema
) = load_assets()


# --------------------------------------------------
# Recommendation Functions
# --------------------------------------------------

def get_cf_scores(course_id, top_k=20):
    idx = item_ids.index(course_id)

    # sparse row slice (NOT .values)
    item_vec = user_item_sparse[idx]

    sims = cosine_similarity(
        item_vec,
        user_item_sparse,
        dense_output=False   
    )

    sims = sims.toarray().ravel()   # small: 1 × N only

    top_idx = np.argsort(sims)[::-1][:top_k]

    return pd.Series(
        sims[top_idx],
        index=[item_ids[i] for i in top_idx]
    )

def get_cbf_scores(course_id, top_k=30):
    idx = item_ids.index(course_id)

    sims = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).ravel()

    top_idx = np.argsort(sims)[::-1][:top_k]

    return pd.Series(
        sims[top_idx],
        index=[item_ids[i] for i in top_idx]
    )


def hybrid_recommendation(course_id, n=5):
    alpha = hybrid_config["alpha"]

    cf_scores = get_cf_scores(course_id)
    cbf_scores = get_cbf_scores(course_id)

    hybrid_scores = alpha * cf_scores + (1 - alpha) * cbf_scores
    hybrid_scores = hybrid_scores.drop(course_id)

    return hybrid_scores.sort_values(ascending=False).head(n)

def get_xgb_recommendations(user_id, n=5, candidate_k=100):

    if user_id not in user_encoder.classes_:
        return pd.Series(dtype=float)

    # ---- seen courses ----
    seen_courses = interactions_df[
        interactions_df.user_id == user_id
    ]['course_id'].unique()

    # ---- LIMIT candidates (critical for speed) ----
    candidate_courses = (
        course_stats[
            ~course_stats.course_id.isin(seen_courses)
        ]
        .sort_values("rating_count", ascending=False)
        .head(candidate_k)["course_id"]
        .values
    )

    # ---- user features ----
    user_enc = user_encoder.transform([user_id])[0]
    user_row = user_stats[user_stats.user_id == user_id].iloc[0]

    feature_rows = []
    valid_course_ids = []

    for cid in candidate_courses:
        if cid not in course_encoder.classes_:
            continue

        course_row = course_stats[course_stats.course_id == cid].iloc[0]

        meta_row = courses_df.loc[
            courses_df.course_id == cid,
            ['difficulty_level', 'instructor']
        ]

        if meta_row.empty:
            continue

        meta = ohe.transform(
            meta_row.fillna("unknown").astype(str)
        )

        feature_rows.append(
            np.hstack([
                [
                    user_enc,
                    course_encoder.transform([cid])[0],
                    course_row.avg_rating,
                    course_row.rating_count,
                    user_row.user_avg_rating,
                    user_row.user_rating_count
                ],
                meta[0]
            ])
        )

        valid_course_ids.append(cid)

    if not feature_rows:
        return pd.Series(dtype=float)

    # ---- BATCH prediction (FAST) ----
    X = np.array(feature_rows, dtype="float32")
    scores = xgb_model.predict(X)

    return (
        pd.DataFrame({
            "course_id": valid_course_ids,
            "similarity_score": scores
        })
        .set_index("course_id")
        .sort_values("similarity_score", ascending=False)
        .head(n)["similarity_score"]
    )

@st.cache_data
def rmse_cf_fast(interactions_df, sample_size=1000):
    df = interactions_df.sample(
        min(sample_size, len(interactions_df)),
        random_state=42
    )

    y_true, y_pred = [], []

    for _, row in df.iterrows():
        cid = row['course_id']
        rating = row['rating']

        if cid not in item_ids:
            continue

        scores = get_cf_scores(cid, top_k=15)

        y_true.append(rating)
        y_pred.append(scores.mean())

    return np.sqrt(mean_squared_error(y_true, y_pred))


@st.cache_data
def rmse_cbf_fast(interactions_df, sample_size=1500):
    df = interactions_df.sample(
        min(sample_size, len(interactions_df)),
        random_state=42
    )

    y_true, y_pred = [], []

    for _, row in df.iterrows():
        cid = row['course_id']
        rating = row['rating']

        if cid not in item_ids:
            continue

        scores = get_cbf_scores(cid, top_k=20)
        y_true.append(rating)
        y_pred.append(scores.mean())

    return np.sqrt(mean_squared_error(y_true, y_pred))


@st.cache_data
def rmse_hybrid_fast(interactions_df, alpha, sample_size=1500):
    df = interactions_df.sample(
        min(sample_size, len(interactions_df)),
        random_state=42
    )

    y_true, y_pred = [], []

    for _, row in df.iterrows():
        cid = row['course_id']
        rating = row['rating']

        if cid not in item_ids:
            continue

        cf_pred = get_cf_scores(cid, 20).mean()
        cbf_pred = get_cbf_scores(cid, 20).mean()

        pred = alpha * cf_pred + (1 - alpha) * cbf_pred

        y_true.append(rating)
        y_pred.append(pred)

    return np.sqrt(mean_squared_error(y_true, y_pred))

@st.cache_data
def rmse_xgb_fast(interactions_df, sample_size=3000):
    """
    Fast RMSE for XGBoost recommender using fixed feature schema
    """

    # -----------------------------
    # 1. Sample interactions (speed)
    # -----------------------------
    df = interactions_df.sample(
        min(sample_size, len(interactions_df)),
        random_state=42
    )

    y_true = []
    y_pred = []

    # -----------------------------
    # 2. Iterate interactions
    # -----------------------------
    for _, row in df.iterrows():
        user_id = row["user_id"]
        course_id = row["course_id"]
        rating = row["rating"]

        # Skip unseen entities
        if (
            user_id not in user_encoder.classes_ or
            course_id not in course_encoder.classes_
        ):
            continue

        # -----------------------------
        # 3. Encode IDs
        # -----------------------------
        user_enc = user_encoder.transform([user_id])[0]
        course_enc = course_encoder.transform([course_id])[0]

        # -----------------------------
        # 4. Aggregate features
        # -----------------------------
        try:
            course_row = course_stats.loc[course_stats.course_id == course_id].iloc[0]
            user_row = user_stats.loc[user_stats.user_id == user_id].iloc[0]
        except IndexError:
            continue

        numeric_features = [
            user_enc,
            course_enc,
            course_row.avg_rating,
            course_row.rating_count,
            user_row.user_avg_rating,
            user_row.user_rating_count
        ]

        # -----------------------------
        # 5. Metadata OHE
        # -----------------------------
        meta_df = courses_df.loc[
            courses_df.course_id == course_id,
            ["difficulty_level", "instructor"]
        ].head(1)

        if meta_df.empty:
            continue

        meta_ohe_raw = ohe.transform(meta_df)

        # Handle both sparse and dense outputs
        if hasattr(meta_ohe_raw, "toarray"):
            meta_ohe = meta_ohe_raw.toarray()[0]
        else:
            meta_ohe = meta_ohe_raw[0]

        # -----------------------------
        # 6. Final feature vector (ORDER MATTERS)
        # -----------------------------
        features = np.hstack([numeric_features, meta_ohe]).reshape(1, -1)

        # -----------------------------
        # 7. Predict
        # -----------------------------
        pred = xgb_model.predict(features)[0]

        y_true.append(rating)
        y_pred.append(pred)

    # -----------------------------
    # 8. RMSE
    # -----------------------------
    if len(y_true) == 0:
        return np.nan

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse


@st.cache_data
def compute_all_rmse(interactions_df, alpha=0.5):
    return {
        "Collaborative Filtering": rmse_cf_fast(interactions_df),
        "Content-Based Filtering": rmse_cbf_fast(interactions_df),
        "Hybrid": rmse_hybrid_fast(interactions_df, alpha),
        "XGBoost": rmse_xgb_fast(interactions_df)
    }

rmse_scores = compute_all_rmse(interactions_df)


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("📚 Course Recommendation System")
st.caption(
    "Hybrid Recommendation Engine using Collaborative Filtering, "
    "Content-Based Filtering & XGBoost"
)

st.divider()


with st.spinner("Evaluating models..."):
    rmse_scores = compute_all_rmse(
        interactions_df,
        hybrid_config["alpha"]
    )

# st.subheader("📊 Model Evaluation (RMSE)")

# rmse_df = (
#     pd.DataFrame.from_dict(rmse_scores, orient="index", columns=["RMSE"])
#     .sort_values("RMSE")
# )

# st.dataframe(rmse_df, use_container_width=True)

# best_model = rmse_df.index[0]
# st.success(f"🏆 Best Model (Lowest RMSE): {best_model}")

# st.write(interactions_df[['user_id', 'course_id', 'rating']].head())

with st.sidebar:
    st.header("⚙️ Recommendation Settings")

    model_type = st.selectbox(
        "Recommendation Model",
        [
            "Collaborative Filtering",
            "Content-Based Filtering",
            "Hybrid",
            "XGBoost (Personalized)"
        ]
    )

    if model_type == "XGBoost (Personalized)":
        xgb_model = load_xgb()
        
    course_id_input = st.text_input(
        "Course ID",
        placeholder="e.g. 101",
        help="Required for CF, CBF & Hybrid models"
    )

    user_id_input = st.text_input(
        "User ID",
        placeholder="e.g. 501",
        help="Required only for XGBoost model"
    )

    top_n = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=20,
        value=5
    )

    run_btn = st.button("🚀 Get Recommendations", use_container_width=True)

# --------------------------------------------------
# Run Recommendation
# --------------------------------------------------

if run_btn:
    with st.spinner(f"Generating recommendations using {model_type}..."):

        st.markdown(f"### 🎯 Recommendations ({model_type})")
        st.caption("Personalized suggestions based on your preferences and behavior for Online Courses")

    
    if not course_id_input.isdigit():
        st.error("❌ Please enter a valid numeric Course ID")
        st.stop()

    course_id = int(course_id_input)

    if course_id not in item_ids:
        st.warning("⚠️ Course ID not found in dataset")
        st.stop()

    with st.spinner(f"Generating recommendations using {model_type}..."):

        if model_type == "Collaborative Filtering":
            results = get_cf_scores(course_id).drop(course_id).head(top_n)

        elif model_type == "Content-Based Filtering":
            results = get_cbf_scores(course_id).drop(course_id).head(top_n)

        elif model_type == "Hybrid":
            results = hybrid_recommendation(course_id, top_n)

        else:
            if not user_id_input.isdigit():
                st.error("❌ User ID required for XGBoost")
                st.stop()

            results = get_xgb_recommendations(int(user_id_input), top_n)

            if results.empty:
                st.info("ℹ️ No recommendations available for this user")
                st.stop()

   # st.subheader("🎯 Recommended Courses")

    results_df = (
        results.reset_index()
        .rename(columns={"index": "course_id"})
        .merge(courses_df, on="course_id", how="left")
    )

    st.dataframe(
        results_df[
            ["course_id", "course_name", "rating"]
        ].sort_values("rating", ascending=False),
        use_container_width=True
    )

st.divider()
st.subheader("📊 Model Performance (RMSE)")

rmse_df = (
    pd.DataFrame.from_dict(rmse_scores, orient="index", columns=["RMSE"])
    .sort_values("RMSE")
)

st.dataframe(rmse_df, use_container_width=True)

best_model = rmse_df.index[0]
st.success(f"🏆 Best Model: **{best_model}**")

#st.download_button(
#    "⬇️ Download Recommendations",
#    results_df.to_csv(index=False),
#    file_name="recommendations.csv"
#)


