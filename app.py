# ============================================================
# Tourism Experience Analytics - Streamlit App
# Run AFTER train_tourism_model.py
# Command: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# App configuration
# ============================================================
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
)

# ============================================================
# Load data and models (cached so it loads only once)
# ============================================================
@st.cache_data
def load_data():
    data = pd.read_csv("tourism_master_data.csv")
    data = data.dropna(subset=["VisitMode", "Continent", "Country", "AttractionType"])
    return data

@st.cache_resource
def load_models():
    clf_model   = joblib.load("visit_mode_model.pkl")
    reg_model   = joblib.load("rating_model.pkl")
    mode_enc    = joblib.load("mode_encoder.pkl")
    feat_encs   = joblib.load("feature_encoders.pkl")
    scaler      = joblib.load("feature_scaler.pkl")
    collab      = joblib.load("collab_filtering_data.pkl")
    content     = joblib.load("content_filtering_data.pkl")
    defaults    = joblib.load("global_defaults.pkl")
    try:
        comparison = joblib.load("model_comparison.pkl")
    except:
        comparison = None
    return clf_model, reg_model, mode_enc, feat_encs, scaler, collab, content, defaults, comparison

df = load_data()
clf_model, reg_model, mode_enc, feat_encs, scaler, collab, content, DEFAULTS, cmp_data = load_models()

# Unpack useful variables from loaded files
CLF_FEATS   = DEFAULTS["clf_features"]
REG_FEATS   = DEFAULTS["reg_features"]
SEASON_MAP  = DEFAULTS["season_map"]
G_ATTR_AVG  = DEFAULTS["global_attr_avg"]
G_ATTR_CNT  = DEFAULTS["global_attr_cnt"]
G_REG_AVG   = DEFAULTS["global_region_avg"]
G_CTRY_AVG  = DEFAULTS["global_country_avg"]
G_TYPE_AVG  = DEFAULTS["global_type_avg"]
G_USER_STD  = DEFAULTS["global_user_std"]
G_ATTR_STD  = DEFAULTS["global_attr_std"]

user_factors = collab["user_factors"]
item_factors = collab["item_factors"]
collab_mat   = collab["user_item_matrix"]
collab_aids  = collab["attraction_id_order"]

attr_df  = content["attractions_df"]
sim_mat  = content["sim_matrix"]

# ============================================================
# Sidebar navigation
# ============================================================
st.sidebar.title("🌍 Tourism Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to page:", [
    "📊 Market Insights",
    "🧳 Predict Visit Mode",
    "⭐ Predict Rating",
    "🗺️ Recommender",
    "📈 Model Performance",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Info**")
st.sidebar.write("Total visits:", df.shape[0])
st.sidebar.write("Unique users:", df["UserId"].nunique())
st.sidebar.write("Attractions:", df["Attraction"].nunique())
st.sidebar.write("Countries:", df["Country"].nunique())
st.sidebar.write("Avg rating:", round(df["Rating"].mean(), 2), "/ 5")
if cmp_data:
    st.sidebar.markdown("---")
    st.sidebar.write("Best Classifier:", cmp_data.get("best_clf", "-"))
    st.sidebar.write("Best Regressor:", cmp_data.get("best_reg", "-"))


# ============================================================
# PAGE 1 - Market Insights (EDA dashboard)
# ============================================================
if page == "📊 Market Insights":
    st.title("📊 Tourism Market Insights")
    st.write("Explore patterns and trends from the full tourism dataset.")

    # Summary numbers in 5 columns
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Visits",   df.shape[0])
    col2.metric("Unique Users",   df["UserId"].nunique())
    col3.metric("Attractions",    df["Attraction"].nunique())
    col4.metric("Countries",      df["Country"].nunique())
    col5.metric("Avg Rating",     round(df["Rating"].mean(), 2))

    st.markdown("---")

    # Row 1 charts
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Visit Mode Distribution")
        st.write("Couples (41%) are the most common. Business (1.2%) is the rarest.")
        st.bar_chart(df["VisitMode"].value_counts())

    with col_b:
        st.subheader("Rating Distribution")
        st.write("All ratings from 1 to 5 are valid. Ratings 1 and 2 are NOT outliers.")
        st.bar_chart(df["Rating"].value_counts().sort_index())

    # Row 2 charts
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Top 10 Countries")
        st.bar_chart(df["Country"].value_counts().head(10))

    with col_d:
        st.subheader("Continent Share")
        continent_table = df["Continent"].value_counts().reset_index()
        continent_table.columns = ["Continent", "Visits"]
        continent_table["Share %"] = (continent_table["Visits"] / continent_table["Visits"].sum() * 100).round(1)
        st.dataframe(continent_table, hide_index=True, use_container_width=True)

    # Row 3 charts
    col_e, col_f = st.columns(2)
    with col_e:
        st.subheader("Most Popular Attraction Types")
        st.bar_chart(df["AttractionType"].value_counts().head(10))

    with col_f:
        st.subheader("Average Rating by Attraction Type")
        avg_by_type = df.groupby("AttractionType")["Rating"].mean().sort_values(ascending=False).round(2).reset_index()
        avg_by_type.columns = ["Attraction Type", "Avg Rating"]
        st.dataframe(avg_by_type, hide_index=True, use_container_width=True)

    # Row 4 charts
    col_g, col_h = st.columns(2)
    with col_g:
        st.subheader("Monthly Visit Trend")
        monthly = df.groupby("VisitMonth")["UserId"].count().rename("Visits")
        st.line_chart(monthly)

    with col_h:
        st.subheader("Seasonal Distribution")
        if "Season" in df.columns:
            st.bar_chart(df["Season"].value_counts())
        else:
            st.bar_chart(df.groupby("VisitYear")["UserId"].count().rename("Visits"))

    # Visit Mode x Attraction Type
    st.subheader("Visit Mode by Attraction Type (top 6 types)")
    top6_types = df["AttractionType"].value_counts().head(6).index
    pivot = df[df["AttractionType"].isin(top6_types)].groupby(["AttractionType", "VisitMode"]).size().unstack(fill_value=0)
    st.bar_chart(pivot)

    # EDA Image
    st.subheader("EDA Charts (from training)")
    try:
        st.image("eda_charts.png", use_column_width=True)
    except:
        st.info("Run train_tourism_model.py to generate eda_charts.png")

    # Business insights table
    st.subheader("Key Business Insights")
    insights = pd.DataFrame({
        "Insight": ["Top segment", "Rarest segment", "Peak months", "Best rated type", "New users"],
        "Finding": ["Couples (41%)", "Business (1.2%)", "July to September", "Parks and Nature", "No visit history"],
        "Action":  [
            "Promote couples packages",
            "Add business travel features",
            "Run off-season promotions",
            "Feature nature attractions",
            "Use content-based filtering",
        ],
    })
    st.dataframe(insights, hide_index=True, use_container_width=True)


# ============================================================
# PAGE 2 - Predict Visit Mode (Classification)
# ============================================================
elif page == "🧳 Predict Visit Mode":
    st.title("🧳 Predict Visit Mode")
    st.write("Predict if a visit will be: Business, Couples, Family, Friends, or Solo")

    st.info("This model uses 3 types of input: location, attraction type, and your past visit patterns.")

    st.markdown("---")
    st.subheader("Step 1: Location Details")
    col1, col2 = st.columns(2)

    with col1:
        continent = st.selectbox("Continent", sorted(df["Continent"].dropna().unique()), key="vm_cont")
        country   = st.selectbox("Country", sorted(df[df["Continent"] == continent]["Country"].dropna().unique()), key="vm_ctry")
        region_options = sorted(df[df["Country"] == country]["Region"].dropna().unique())
        if len(region_options) == 0:
            region_options = sorted(df["Region"].dropna().unique())
        region = st.selectbox("Region", region_options, key="vm_reg")

    with col2:
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        attr_type = st.selectbox("Attraction Type", sorted(df["AttractionType"].dropna().unique()), key="vm_atype")
        rating    = st.slider("Rating Given (1=Poor to 5=Excellent)", 1, 5, 4, key="vm_rat")
        month     = st.selectbox("Visit Month", list(range(1, 13)), key="vm_mon",
                                 format_func=lambda m: month_labels[m - 1])
        year      = st.number_input("Visit Year", 2000, 2030, 2024, key="vm_yr")

    st.markdown("---")
    st.subheader("Step 2: Your Past Visit Patterns (improves accuracy)")
    col3, col4 = st.columns(2)
    with col3:
        user_avg_rating   = st.slider("Your Typical Rating",     1.0, 5.0, 3.5, 0.1, key="vm_h1")
        user_visit_count  = st.slider("Total Places Visited",    1, 100, 10, 1, key="vm_h2")
        pct_couples       = st.slider("% Past Trips as Couples", 0.0, 1.0, 0.0, 0.05, key="vm_h3")
        pct_family        = st.slider("% Past Trips as Family",  0.0, 1.0, 0.0, 0.05, key="vm_h4")
    with col4:
        pct_friends       = st.slider("% Past Trips as Friends", 0.0, 1.0, 0.0, 0.05, key="vm_h5")
        pct_solo          = st.slider("% Past Trips as Solo",    0.0, 1.0, 0.0, 0.05, key="vm_h6")
        pct_business      = st.slider("% Past Trips as Business",0.0, 1.0, 0.0, 0.05, key="vm_h7")

    if st.button("🔮 Predict Visit Mode", use_container_width=True):
        try:
            season = SEASON_MAP.get(int(month), "Summer")

            # Encode text values to numbers
            cont_enc  = feat_encs["Continent"].transform([continent])[0]
            ctry_enc  = feat_encs["Country"].transform([country])[0]
            reg_enc   = feat_encs["Region"].transform([region])[0]
            atype_enc = feat_encs["AttractionType"].transform([attr_type])[0]
            seas_enc  = feat_encs["Season"].transform([season])[0]

            # Scale numeric features the same way as training
            raw_numbers = np.array([[user_avg_rating, user_visit_count, G_USER_STD,
                                     G_ATTR_AVG, G_ATTR_CNT, G_ATTR_STD,
                                     G_REG_AVG, G_CTRY_AVG, G_TYPE_AVG]])
            try:
                scaled = scaler.transform(raw_numbers)[0]
            except:
                scaled = [user_avg_rating / 5, user_visit_count / 100, 0.5,
                          G_ATTR_AVG, G_ATTR_CNT, 0.5, G_REG_AVG, G_CTRY_AVG, G_TYPE_AVG]

            # Build the feature row
            user_pcts = {
                "UserPct_Business": pct_business,
                "UserPct_Couples":  pct_couples,
                "UserPct_Family":   pct_family,
                "UserPct_Friends":  pct_friends,
                "UserPct_Solo":     pct_solo,
            }
            attr_pcts = {}
            for col_name in CLF_FEATS:
                if col_name.startswith("AttrPct_"):
                    attr_pcts[col_name] = float(df[col_name].mean()) if col_name in df.columns else 0.0

            feature_row = {
                "Rating":              rating,
                "VisitYear":           year,
                "VisitMonth":          month,
                "Season_enc":          seas_enc,
                "Continent_enc":       cont_enc,
                "Country_enc":         ctry_enc,
                "Region_enc":          reg_enc,
                "AttractionType_enc":  atype_enc,
                "UserAvgRating":       scaled[0],
                "UserVisitCount":      scaled[1],
                "UserRatingStd":       scaled[2],
                "AttractionAvgRating": scaled[3],
                "AttractionVisitCount": scaled[4],
                "AttractionRatingStd": scaled[5],
                "RegionAvgRating":     scaled[6],
                "CountryAvgRating":    scaled[7],
                "TypeAvgRating":       scaled[8],
            }
            feature_row.update(user_pcts)
            feature_row.update(attr_pcts)

            X_input = pd.DataFrame([feature_row]).reindex(columns=CLF_FEATS, fill_value=0)
            predicted_enc   = clf_model.predict(X_input)[0]
            predicted_label = mode_enc.inverse_transform([predicted_enc])[0]

            st.success("Predicted Visit Mode: " + predicted_label)

            # Show confidence probabilities if available
            if hasattr(clf_model, "predict_proba"):
                probs = clf_model.predict_proba(X_input)[0]
                all_modes = mode_enc.classes_
                prob_df = pd.DataFrame({
                    "Visit Mode":  all_modes,
                    "Probability": [str(round(p * 100, 1)) + "%" for p in probs],
                    "Confidence":  ["High" if p > 0.5 else "Medium" if p > 0.2 else "Low" for p in probs],
                }).sort_values("Probability", ascending=False)
                st.write("**Confidence breakdown:**")
                st.dataframe(prob_df, hide_index=True, use_container_width=True)
                top_prob = max(probs)
                if top_prob > 0.8:
                    st.write("Model is very confident in this prediction:", str(round(top_prob * 100, 0)) + "%")
                elif top_prob > 0.5:
                    st.write("Model has good confidence:", str(round(top_prob * 100, 0)) + "%")
                else:
                    st.write("Tip: Enter your past visit patterns above to improve accuracy.")

        except Exception as error:
            st.error("Prediction failed: " + str(error))
            st.info("Make sure train_tourism_model.py ran without errors.")


# ============================================================
# PAGE 3 - Predict Rating (Regression)
# ============================================================
elif page == "⭐ Predict Rating":
    st.title("⭐ Predict Attraction Rating")
    st.write("Predict what star rating (1 to 5) a visitor would give an attraction.")

    st.info("The model considers visitor history, attraction popularity, and location to make predictions.")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visitor Info")
        r_continent = st.selectbox("Continent", sorted(df["Continent"].dropna().unique()), key="rat_cont")
        r_country   = st.selectbox("Country", sorted(df[df["Continent"] == r_continent]["Country"].dropna().unique()), key="rat_ctry")
        r_regs = sorted(df[df["Country"] == r_country]["Region"].dropna().unique())
        if len(r_regs) == 0:
            r_regs = sorted(df["Region"].dropna().unique())
        r_region    = st.selectbox("Region", r_regs, key="rat_reg")
        r_user_avg  = st.slider("Visitor Typical Rating", 1.0, 5.0, 3.5, 0.1, key="rat_uavg")
        r_user_cnt  = st.slider("Places Visited Before",  1, 100, 10, 1, key="rat_ucnt")

    with col2:
        st.subheader("Attraction & Visit Info")
        r_attr_type = st.selectbox("Attraction Type", sorted(df["AttractionType"].dropna().unique()), key="rat_atype")
        r_mode      = st.selectbox("Visit Mode", sorted(df["VisitMode"].dropna().unique()), key="rat_mode")
        r_month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        r_month     = st.selectbox("Visit Month", list(range(1, 13)), key="rat_mon",
                                   format_func=lambda m: r_month_labels[m - 1])
        r_year      = st.number_input("Visit Year", 2000, 2030, 2024, key="rat_yr")

    if st.button("⭐ Predict Rating", use_container_width=True):
        try:
            r_season = SEASON_MAP.get(int(r_month), "Summer")

            cont_enc  = feat_encs["Continent"].transform([r_continent])[0]
            ctry_enc  = feat_encs["Country"].transform([r_country])[0]
            reg_enc   = feat_encs["Region"].transform([r_region])[0]
            atype_enc = feat_encs["AttractionType"].transform([r_attr_type])[0]
            mode_enc_val = mode_enc.transform([r_mode])[0]
            seas_enc  = feat_encs["Season"].transform([r_season])[0]

            raw_numbers = np.array([[r_user_avg, r_user_cnt, G_USER_STD,
                                     G_ATTR_AVG, G_ATTR_CNT, G_ATTR_STD,
                                     G_REG_AVG, G_CTRY_AVG, G_TYPE_AVG]])
            try:
                scaled = scaler.transform(raw_numbers)[0]
            except:
                scaled = [r_user_avg / 5, r_user_cnt / 100, 0.5,
                          G_ATTR_AVG, G_ATTR_CNT, 0.5, G_REG_AVG, G_CTRY_AVG, G_TYPE_AVG]

            feature_row = {
                "VisitYear":           r_year,
                "VisitMonth":          r_month,
                "Season_enc":          seas_enc,
                "Continent_enc":       cont_enc,
                "Country_enc":         ctry_enc,
                "Region_enc":          reg_enc,
                "AttractionType_enc":  atype_enc,
                "VisitMode_enc":       mode_enc_val,
                "UserAvgRating":       scaled[0],
                "UserVisitCount":      scaled[1],
                "UserRatingStd":       scaled[2],
                "AttractionAvgRating": scaled[3],
                "AttractionVisitCount": scaled[4],
                "AttractionRatingStd": scaled[5],
                "RegionAvgRating":     scaled[6],
                "CountryAvgRating":    scaled[7],
                "TypeAvgRating":       scaled[8],
            }

            X_input = pd.DataFrame([feature_row]).reindex(columns=REG_FEATS, fill_value=0)
            raw_pred = reg_model.predict(X_input)[0]
            predicted_rating = float(np.clip(round(raw_pred, 1), 1.0, 5.0))
            stars = "⭐" * int(round(predicted_rating))
            rating_labels = {1: "Poor", 2: "Below Average", 3: "Average", 4: "Good", 5: "Excellent"}
            label = rating_labels.get(int(round(predicted_rating)), "Good")

            st.success("Predicted Rating: " + str(predicted_rating) + " / 5.0   " + stars)
            st.write("This is considered:", label)
            st.write("Average model error is about 0.04 stars.")

            st.subheader("What affects the rating most?")
            drivers = pd.DataFrame({
                "Factor":  ["Visitor's typical rating", "Attraction average rating", "Attraction type average", "Region/Country average", "Visit mode"],
                "Impact":  ["Highest", "High", "Medium", "Medium", "Lower"],
                "Value":   [str(r_user_avg) + "/5", "Dataset average (LOO)", r_attr_type, r_region, r_mode],
            })
            st.dataframe(drivers, hide_index=True, use_container_width=True)

        except Exception as error:
            st.error("Prediction failed: " + str(error))


# ============================================================
# PAGE 4 - Attraction Recommender
# ============================================================
elif page == "🗺️ Recommender":
    st.title("🗺️ Attraction Recommender")
    st.write("Get personalised attraction recommendations using 3 different methods.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 Collaborative",
        "🏛️ Content-Based",
        "🔀 Hybrid",
        "🏆 Top Rated",
    ])

    # ---- Tab 1: Collaborative Filtering ----
    with tab1:
        st.subheader("👥 Collaborative Filtering")
        st.write("Finds users who liked what you liked and recommends what they visited next.")

        all_attractions = sorted(attr_df["Attraction"].dropna().tolist())
        selected_collab = st.selectbox("Attraction You Visited:", all_attractions, key="col_sel")
        rating_collab   = st.slider("Your Rating for It:", 1, 5, 4, key="col_rat")
        num_recs_collab = st.slider("Number of Recommendations:", 3, 10, 5, key="col_n")

        if st.button("Get Collaborative Recommendations", use_container_width=True):
            try:
                row = attr_df[attr_df["Attraction"] == selected_collab]
                if len(row) == 0:
                    st.error("Attraction not found.")
                else:
                    attraction_id = row.iloc[0]["AttractionId"]

                    # Create a rating vector for the selected attraction
                    user_vec = np.zeros(len(collab_aids))
                    if attraction_id in collab_aids:
                        user_vec[collab_aids.index(attraction_id)] = rating_collab

                    # Find similar users using SVD taste vectors
                    taste_vec = user_vec @ item_factors
                    similarities = cosine_similarity(taste_vec.reshape(1, -1), user_factors)[0]
                    top_user_indices = np.argsort(similarities)[-6:-1]

                    # Get top attractions those similar users visited
                    scores = {}
                    for user_idx in top_user_indices:
                        for attr_id, rating_val in collab_mat.iloc[user_idx].items():
                            if rating_val > 0 and attr_id != attraction_id:
                                if attr_id not in scores:
                                    scores[attr_id] = 0
                                scores[attr_id] += similarities[user_idx] * rating_val

                    top_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_recs_collab]

                    st.write("**Top", num_recs_collab, "Recommendations:**")
                    for rank, (attr_id, score) in enumerate(top_recs, 1):
                        attr_row = attr_df[attr_df["AttractionId"] == attr_id]
                        if len(attr_row) > 0:
                            a = attr_row.iloc[0]
                            st.write(str(rank) + ". **" + a["Attraction"] + "**")
                            st.write("   Type:", a["AttractionType"], " | Region:", a["Region"], ", ", a["Continent"])
                            st.write("   Score:", round(score, 2))
                            st.markdown("---")
            except Exception as e:
                st.error("Error: " + str(e))

    # ---- Tab 2: Content-Based Filtering ----
    with tab2:
        st.subheader("🏛️ Content-Based Filtering")
        st.write("Finds attractions similar to one you already liked based on type, region, and continent.")
        st.write("Works for brand new users who have no visit history.")

        col_cb1, col_cb2 = st.columns([3, 1])
        with col_cb1:
            selected_cb = st.selectbox("Attraction You Liked:", sorted(attr_df["Attraction"].dropna().tolist()), key="cb_sel")
        with col_cb2:
            num_recs_cb = st.slider("How many:", 3, 10, 5, key="cb_n")

        filter_type = st.selectbox("Filter by Type (optional):",
            ["Any"] + sorted(df["AttractionType"].dropna().unique().tolist()), key="cb_type")

        if st.button("Find Similar Attractions", use_container_width=True):
            try:
                selected_index = attr_df[attr_df["Attraction"] == selected_cb].index[0]
                all_scores = list(enumerate(sim_mat[selected_index]))

                if filter_type != "Any":
                    valid_indices = set(attr_df[attr_df["AttractionType"] == filter_type].index)
                    all_scores = [(i, s) for i, s in all_scores if i in valid_indices]

                top_similar = sorted(
                    [(i, s) for i, s in all_scores if i != selected_index],
                    key=lambda x: x[1], reverse=True
                )[:num_recs_cb]

                sel_row = attr_df.iloc[selected_index]
                st.write("**Selected attraction:**", sel_row["Attraction"])
                st.write("Type:", sel_row["AttractionType"], " | Region:", sel_row["Region"], " | ", sel_row["Continent"])
                st.markdown("---")
                st.write("**Similar Attractions:**")

                for rank, (idx, score) in enumerate(top_similar, 1):
                    a = attr_df.iloc[idx]
                    similarity_pct = int(score * 100)
                    if similarity_pct >= 80:
                        label = "Very Similar"
                    elif similarity_pct >= 50:
                        label = "Similar"
                    else:
                        label = "Related"
                    st.write(str(rank) + ". **" + a["Attraction"] + "**  (" + label + " - " + str(similarity_pct) + "%)")
                    st.write("   Type:", a["AttractionType"], " | Region:", a["Region"], ", ", a["Continent"])
                    st.markdown("---")
            except Exception as e:
                st.error("Error: " + str(e))

    # ---- Tab 3: Hybrid Recommender ----
    with tab3:
        st.subheader("🔀 Hybrid Recommender")
        st.write("Combines Collaborative and Content-Based recommendations.")
        st.write("You can control how much weight to give each method.")

        selected_hybrid = st.selectbox("Attraction You Visited:", sorted(attr_df["Attraction"].dropna().tolist()), key="hyb_sel")
        rating_hybrid   = st.slider("Your Rating for It:", 1, 5, 4, key="hyb_rat")
        num_recs_hybrid = st.slider("Number of Recommendations:", 3, 10, 5, key="hyb_n")
        collab_weight   = st.slider(
            "Collaborative Weight (0 = content-only, 1 = collaborative-only, 0.5 = balanced):",
            0.0, 1.0, 0.5, 0.1, key="hyb_w"
        )

        if st.button("Get Hybrid Recommendations", use_container_width=True):
            try:
                rh = attr_df[attr_df["Attraction"] == selected_hybrid]
                if len(rh) == 0:
                    st.error("Attraction not found.")
                else:
                    aid_h   = rh.iloc[0]["AttractionId"]
                    si_h    = rh.index[0]

                    # Collaborative scores
                    uv_h = np.zeros(len(collab_aids))
                    if aid_h in collab_aids:
                        uv_h[collab_aids.index(aid_h)] = rating_hybrid
                    taste_h  = uv_h @ item_factors
                    sims_h   = cosine_similarity(taste_h.reshape(1, -1), user_factors)[0]
                    top_u_h  = np.argsort(sims_h)[-6:-1]
                    c_scores = {}
                    for ui in top_u_h:
                        for ac, rv in collab_mat.iloc[ui].items():
                            if rv > 0 and ac != aid_h:
                                if ac not in c_scores:
                                    c_scores[ac] = 0
                                c_scores[ac] += sims_h[ui] * rv
                    max_c = max(c_scores.values()) if c_scores else 1
                    c_norm = {k: v / max_c for k, v in c_scores.items()}

                    # Content scores
                    cb_norm = {attr_df.iloc[i]["AttractionId"]: s
                               for i, s in enumerate(sim_mat[si_h]) if i != si_h}

                    # Blend the two
                    all_attrs = set(c_norm) | set(cb_norm)
                    hybrid_scores = {
                        a: collab_weight * c_norm.get(a, 0) + (1 - collab_weight) * cb_norm.get(a, 0)
                        for a in all_attrs
                    }
                    top_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:num_recs_hybrid]

                    st.write("**Top", num_recs_hybrid, "Hybrid Recommendations**")
                    st.write("Collaborative weight:", str(round(collab_weight * 100)) + "% | Content weight:", str(round((1 - collab_weight) * 100)) + "%")
                    st.markdown("---")

                    for rank, (a_id, h_score) in enumerate(top_hybrid, 1):
                        ri = attr_df[attr_df["AttractionId"] == a_id]
                        if len(ri) == 0:
                            continue
                        ri = ri.iloc[0]
                        st.write(str(rank) + ". **" + ri["Attraction"] + "**")
                        st.write("   Type:", ri["AttractionType"], " | Region:", ri["Region"], ", ", ri["Continent"])
                        st.write("   Hybrid Score:", round(h_score, 3),
                                 " | Collaborative:", round(c_norm.get(a_id, 0), 2),
                                 " | Content:", round(cb_norm.get(a_id, 0), 2))
                        st.markdown("---")
            except Exception as e:
                st.error("Error: " + str(e))

    # ---- Tab 4: Top Rated ----
    with tab4:
        st.subheader("🏆 Top Rated Attractions")
        st.write("Shows the highest-rated attractions with at least 10 visits.")

        col_ft1, col_ft2 = st.columns(2)
        with col_ft1:
            filter_type2 = st.selectbox("Filter by Type:", ["All"] + sorted(df["AttractionType"].dropna().unique().tolist()), key="top_type")
        with col_ft2:
            filter_cont2 = st.selectbox("Filter by Continent:", ["All"] + sorted(df["Continent"].dropna().unique().tolist()), key="top_cont")

        top_df = df.copy()
        if filter_type2 != "All":
            top_df = top_df[top_df["AttractionType"] == filter_type2]
        if filter_cont2 != "All":
            top_df = top_df[top_df["Continent"] == filter_cont2]

        top_rated = (
            top_df.groupby(["Attraction", "AttractionType", "Region", "Continent"])
            .agg(AvgRating=("Rating", "mean"), Visits=("UserId", "count"))
            .reset_index()
            .query("Visits >= 10")
            .sort_values("AvgRating", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )
        top_rated.insert(0, "Rank", range(1, len(top_rated) + 1))
        top_rated["AvgRating"] = top_rated["AvgRating"].round(2)

        st.dataframe(top_rated, hide_index=True, use_container_width=True)
        if len(top_rated) > 0:
            st.success("Top Attraction: " + top_rated.iloc[0]["Attraction"] +
                       " (" + top_rated.iloc[0]["AttractionType"] + ") - Avg Rating: " + str(top_rated.iloc[0]["AvgRating"]))


# ============================================================
# PAGE 5 - Model Performance
# ============================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance and Comparison")
    st.write("See how all trained models performed. The best model is auto-selected and saved.")

    tab_clf, tab_reg = st.tabs(["🧳 Classification Results", "⭐ Regression Results"])

    # ---- Classification tab ----
    with tab_clf:
        st.subheader("Visit Mode Prediction (Classification)")

        if cmp_data and "clf_results" in cmp_data:
            best = cmp_data.get("best_clf", "")
            rows = []
            for model_name, metrics in cmp_data["clf_results"].items():
                label = model_name + " (Best)" if model_name == best else model_name
                rows.append({
                    "Model":     label,
                    "Accuracy":  str(round(metrics["accuracy"] * 100, 2)) + "%",
                    "Precision": round(metrics["precision"], 4),
                    "Recall":    round(metrics["recall"], 4),
                    "F1 Score":  round(metrics["f1"], 4),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.success("Best model: " + best + " (saved as visit_mode_model.pkl)")
        else:
            st.info("Run train_tourism_model.py to see comparison results.")

        # Show charts if they exist
        try:
            st.image("clf_feature_importance.png", caption="Top 20 Most Important Features", use_column_width=True)
        except:
            pass
        try:
            st.image("clf_model_comparison.png", caption="Model Accuracy and F1 Comparison", use_column_width=True)
        except:
            pass

        st.markdown("---")
        st.subheader("What each metric means")
        metric_info = pd.DataFrame({
            "Metric":    ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Meaning":   [
                "% of visits where mode was predicted correctly",
                "Of all predicted Couples visits, how many were really Couples?",
                "Of all actual Couples visits, how many did we find?",
                "Balance between Precision and Recall",
            ],
            "Good Value": ["> 85%", "> 0.80", "> 0.80", "> 0.85"],
        })
        st.dataframe(metric_info, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("Data Quality Fixes Applied")
        fixes = pd.DataFrame({
            "Problem": [
                "UserAvgRating data leakage",
                "UserPct columns data leakage",
                "AttractionAvgRating leakage",
                "Region/Country/Type avg leakage",
                "13 redundant columns",
                "Rating 1 and 2 removal",
                "CityName overfitting",
                "AttractionAddress noise",
            ],
            "What was wrong": [
                "It included the current row's rating in the average",
                "Current row's visit mode was counted in the proportion",
                "Same leakage problem as UserAvgRating",
                "Current row's rating was included in group average",
                "Duplicate ID columns after joining 9 tables",
                "IQR method wrongly flags them as outliers",
                "5545 unique city names cause overfitting",
                "Partial addresses like Kuta are not useful for ML",
            ],
            "Fix applied": [
                "Leave-One-Out: exclude current row before averaging",
                "Leave-One-Out: exclude current row from proportion",
                "Leave-One-Out: exclude current row before averaging",
                "Leave-One-Out for all group averages",
                "Dropped all redundant ID columns after merge",
                "Kept - they are valid discrete values on 1-5 scale",
                "Removed from features",
                "Removed from features",
            ],
        })
        st.dataframe(fixes, hide_index=True, use_container_width=True)

    # ---- Regression tab ----
    with tab_reg:
        st.subheader("Rating Prediction (Regression)")

        if cmp_data and "reg_results" in cmp_data:
            best = cmp_data.get("best_reg", "")
            rows = []
            for model_name, metrics in cmp_data["reg_results"].items():
                label = model_name + " (Best)" if model_name == best else model_name
                rows.append({
                    "Model":        label,
                    "R2 Score":     round(metrics["r2"], 4),
                    "RMSE (stars)": round(metrics["rmse"], 4),
                    "MAE (stars)":  round(metrics["mae"], 4),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.success("Best model: " + best + " (saved as rating_model.pkl)")
        else:
            st.info("Run train_tourism_model.py to see comparison results.")

        try:
            st.image("reg_model_comparison.png", caption="Model R2 and RMSE Comparison", use_column_width=True)
        except:
            pass

        st.markdown("---")
        st.subheader("What each metric means")
        reg_metric_info = pd.DataFrame({
            "Metric":    ["R2 Score", "RMSE", "MAE"],
            "Meaning":   [
                "% of rating variation explained by the model (1.0 is perfect)",
                "Average prediction error in star units (lower is better)",
                "Average absolute error in star units (lower is better)",
            ],
            "Good Value": ["> 0.95", "< 0.10 stars", "< 0.05 stars"],
        })
        st.dataframe(reg_metric_info, hide_index=True, use_container_width=True)


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.write("Tourism Experience Analytics  |  Classification · Regression · Collaborative · Content-Based · Hybrid")
