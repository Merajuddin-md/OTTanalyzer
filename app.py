import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# Set page configuration
st.set_page_config(
    page_title="OTT Library Analytics | Executive Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism & Netflix Red Accents)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0d0d0f;
    }
    
    /* Header Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.18) 0%, rgba(20, 20, 20, 0.95) 100%);
        border: 1px solid rgba(229, 9, 20, 0.35);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .hero-title {
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    
    .hero-title span {
        color: #E50914;
    }
    
    .hero-subtitle {
        color: #A3A3A3;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.5;
    }
    
    /* Metric Cards */
    .kpi-card {
        background: linear-gradient(145deg, #18181c, #131316);
        border: 1px solid #2a2a32;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: #E50914;
    }
    
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #8E8E93;
        margin-bottom: 6px;
    }
    
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F5F5F7;
        margin-bottom: 4px;
    }
    
    .kpi-sub {
        font-size: 0.85rem;
        color: #30D158;
        font-weight: 500;
    }
    
    .kpi-sub.neutral {
        color: #0A84FF;
    }
    
    .kpi-sub.accent {
        color: #FF9F0A;
    }
    
    /* Takeaway Callout Box */
    .takeaway-box {
        background: rgba(229, 9, 20, 0.08);
        border-left: 4px solid #E50914;
        border-radius: 0 10px 10px 0;
        padding: 12px 18px;
        margin-top: 12px;
        margin-bottom: 18px;
        color: #E1E1E6;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    
    .takeaway-box strong {
        color: #FFFFFF;
        font-weight: 600;
    }
    
    /* Section headers */
    .section-title {
        color: #FFFFFF;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #151518;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #26262b;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #A0A0A5;
        font-weight: 500;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #E50914 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if not os.path.exists("cleaned_netflix_titles.csv"):
        import eda_and_cleaning
        eda_and_cleaning.run_data_pipeline()
        
    df = pd.read_csv("cleaned_netflix_titles.csv")
    df_genres = pd.read_csv("exploded_genres.csv")
    df_countries = pd.read_csv("exploded_countries.csv")
    df_hierarchy = pd.read_csv("country_genre_type_hierarchy.csv")
    
    with open("analysis_summary.json", "r", encoding="utf-8") as f:
        summary_json = json.load(f)
        
    return df, df_genres, df_countries, df_hierarchy, summary_json

df_main, df_genres, df_countries, df_hierarchy, summary_json = load_data()

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=160)
st.sidebar.markdown("### 🎛️ Dynamic Library Filters")

# Content Type filter
all_types = ["All Content Types"] + sorted(df_main['type'].dropna().unique().tolist())
selected_type = st.sidebar.selectbox("Content Type", all_types, index=0)

# Release Year range slider
min_year = int(df_main['release_year'].min())
max_year = int(df_main['release_year'].max())
selected_years = st.sidebar.slider("Release Year Range", min_year, max_year, (min_year, max_year))

# Year Added filter
available_years_added = sorted([int(y) for y in df_main['year_added'].dropna().unique().tolist()])
selected_years_added = st.sidebar.multiselect(
    "Year Ingested to Netflix", 
    available_years_added, 
    default=available_years_added
)

# Rating group filter
all_rating_groups = ["All Demographics"] + sorted(df_main['rating_group'].dropna().unique().tolist())
selected_rating_group = st.sidebar.selectbox("Maturity Audience", all_rating_groups, index=0)

# Genre Multi-select
top_genre_options = sorted(df_genres['genre'].dropna().unique().tolist())
selected_genres = st.sidebar.multiselect("Filter by Genre", top_genre_options, default=[])

# Country Multi-select
top_country_options = sorted(df_countries[df_countries['country_single'] != 'Unknown Country']['country_single'].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect("Filter by Country", top_country_options, default=[])

# Apply Filter Mask to df_main
filtered_df = df_main.copy()

if selected_type != "All Content Types":
    filtered_df = filtered_df[filtered_df['type'] == selected_type]
    
filtered_df = filtered_df[
    (filtered_df['release_year'] >= selected_years[0]) & 
    (filtered_df['release_year'] <= selected_years[1])
]

if selected_years_added:
    filtered_df = filtered_df[filtered_df['year_added'].isin(selected_years_added) | filtered_df['year_added'].isna()]

if selected_rating_group != "All Demographics":
    filtered_df = filtered_df[filtered_df['rating_group'] == selected_rating_group]

if selected_genres:
    matching_show_ids = df_genres[df_genres['genre'].isin(selected_genres)]['show_id'].unique()
    filtered_df = filtered_df[filtered_df['show_id'].isin(matching_show_ids)]

if selected_countries:
    matching_show_ids = df_countries[df_countries['country_single'].isin(selected_countries)]['show_id'].unique()
    filtered_df = filtered_df[filtered_df['show_id'].isin(matching_show_ids)]

# Filtered exploded datasets
f_genres = df_genres[df_genres['show_id'].isin(filtered_df['show_id'])]
f_countries = df_countries[df_countries['show_id'].isin(filtered_df['show_id'])]

# ----------------- HERO HEADER -----------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎬 OTT Content & Library <span>Intelligence Dashboard</span></div>
    <div class="hero-subtitle">
        Comprehensive end-to-end data analysis of Netflix's global streaming catalog. 
        Track catalog expansion, genre dominance, multi-country drill-downs, audience maturity shifts, and duration analytics.
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- KPI CARDS ROW (STEP 3 & STEP 5) -----------------
kpi_total = len(filtered_df)
kpi_movies = (filtered_df['type'] == 'Movie').sum()
kpi_movie_pct = (kpi_movies / kpi_total * 100) if kpi_total > 0 else 0
kpi_tv = (filtered_df['type'] == 'TV Show').sum()
kpi_tv_pct = (kpi_tv / kpi_total * 100) if kpi_total > 0 else 0

kpi_genres = f_genres['genre'].nunique()
kpi_countries = f_countries[f_countries['country_single'] != 'Unknown Country']['country_single'].nunique()

movie_dur_avg = filtered_df[filtered_df['type'] == 'Movie']['duration_minutes'].mean()
movie_dur_str = f"{movie_dur_avg:.1f} min" if not np.isnan(movie_dur_avg) else "N/A"

tv_sea_avg = filtered_df[filtered_df['type'] == 'TV Show']['num_seasons'].mean()
tv_sea_str = f"{tv_sea_avg:.2f} Seasons" if not np.isnan(tv_sea_avg) else "N/A"

latest_yr = int(filtered_df['year_added'].max()) if not filtered_df['year_added'].dropna().empty else "N/A"

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Titles</div>
        <div class="kpi-value">{kpi_total:,}</div>
        <div class="kpi-sub neutral">Catalog Size</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Movie Count</div>
        <div class="kpi-value">{kpi_movies:,}</div>
        <div class="kpi-sub neutral">{kpi_movie_pct:.1f}% of total</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">TV Show Count</div>
        <div class="kpi-value">{kpi_tv:,}</div>
        <div class="kpi-sub accent">{kpi_tv_pct:.1f}% of total</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Unique Countries</div>
        <div class="kpi-value">{kpi_countries}</div>
        <div class="kpi-sub neutral">Production Origins</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Unique Genres</div>
        <div class="kpi-value">{kpi_genres}</div>
        <div class="kpi-sub neutral">Categories</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Latest Addition</div>
        <div class="kpi-value">{latest_yr}</div>
        <div class="kpi-sub neutral">Ingestion Year</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- NAVIGATION TABS -----------------
tab_overview, tab_growth, tab_genres, tab_geo, tab_duration, tab_explorer = st.tabs([
    "📊 Executive Summary",
    "📈 View 1: Growth & Inflection",
    "🎭 View 2: Genre & Ratings Mix",
    "🌍 View 3: Country Drill-Down",
    "⏱️ View 4: Duration & Seasons",
    "🔍 Data Explorer & Lookups"
])

# ================= TAB 1: EXECUTIVE SUMMARY =================
with tab_overview:
    st.markdown('<div class="section-title">📌 Executive Overview & Core Findings</div>', unsafe_allow_html=True)
    
    col_l, col_r = st.columns([3, 2])
    
    with col_l:
        st.markdown("""
        ### Executive Briefing (Key Quantified Highlights)
        1. **Library Dominance & Format Split:** The global Netflix catalog encompasses **7,787 total titles**, with **Movies comprising 5,377 titles (69.05%)** and **TV Shows comprising 2,410 titles (30.95%)**.
        2. **Exponential Ingestion Era (2016–2019):** Catalog additions accelerated dramatically between 2016 and 2019 (+386.00% expansion), peaking at **2,153 additions in 2019** before plateauing during 2020 (2,009 additions, -6.69% YoY due to production disruptions).
        3. **TV Show Growth Surge:** While movies historically outpaced TV additions, TV show additions surged to an all-time peak of **697 titles in 2020** (representing **34.69% of all annual additions**, up from 24.00% in 2014).
        4. **Geographic Concentration:** The **United States dominates with 3,297 titles (42.34%)**, followed by **India with 990 titles (12.71%)** and the **United Kingdom with 723 titles (9.28%)**. Together, the top 3 countries account for **64.33% of the library**.
        5. **Mature Audience Pivot:** Mature adult-rated titles (**TV-MA and R**) constitute **45.41% (3,536 titles)** of the platform. Adult content additions jumped from **36.36% in 2015 to 45.35% in 2020**, while Kids & Family dropped from **38.64% to 24.99%**.
        6. **Short-Series Preference:** **66.72% of all TV shows (1,608 titles)** consist of exactly **1 season**, proving a strategic inclination toward limited series and prompt cancellations of low-engagement franchises.
        """)
        
    with col_r:
        # Donut Chart for Content Split
        fig_split = px.pie(
            filtered_df, 
            names='type', 
            title="Library Content Type Distribution",
            color='type',
            color_discrete_map={'Movie': '#E50914', 'TV Show': '#00A8E8'},
            hole=0.55
        )
        fig_split.update_traces(textposition='inside', textinfo='percent+label', textfont_size=13)
        fig_split.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F5F5F7'),
            margin=dict(t=40, b=20, l=20, r=20),
            showlegend=False
        )
        st.plotly_chart(fig_split, use_container_width=True)
        
    st.markdown("""
    <div class="takeaway-box">
        <strong>Strategic Takeaway:</strong> Movies represent <strong>69.05% (5,377 titles)</strong> of the catalog, but TV series represent the fastest-growing customer retention engine, growing from <strong>185 additions in 2016 to 697 additions in 2020 (+276.76% growth)</strong>.
    </div>
    """, unsafe_allow_html=True)
    
    # Core KPI Definitions and Mathematical Formulas Table
    st.markdown("### 📐 Core KPI Measurement Formulas & Data Dictionary")
    kpi_table_data = [
        {"KPI Measure": "Total Titles", "Formula / Query": "COUNT(DISTINCT show_id)", "Calculated Result": f"{summary_json['kpis']['Total_Titles']:,}", "Description": "Total unique cataloged content titles in library."},
        {"KPI Measure": "Movie Share (%)", "Formula / Query": "(COUNT(type = 'Movie') / Total Titles) * 100", "Calculated Result": f"{summary_json['kpis']['Movie_Percentage']}% (5,377 titles)", "Description": "Proportion of total library allocated to feature films and stand-ups."},
        {"KPI Measure": "TV Show Share (%)", "Formula / Query": "(COUNT(type = 'TV Show') / Total Titles) * 100", "Calculated Result": f"{summary_json['kpis']['TV_Show_Percentage']}% (2,410 titles)", "Description": "Proportion of total library allocated to episodic series."},
        {"KPI Measure": "Unique Genres", "Formula / Query": "COUNT(DISTINCT exploded(genre))", "Calculated Result": f"{summary_json['kpis']['Unique_Genres']}", "Description": "Distinct genre taxonomy categories cataloged."},
        {"KPI Measure": "Unique Countries", "Formula / Query": "COUNT(DISTINCT exploded(country)) WHERE country != 'Unknown'", "Calculated Result": f"{summary_json['kpis']['Unique_Countries']}", "Description": "Distinct production countries contributing content."},
        {"KPI Measure": "Average Movie Duration", "Formula / Query": "SUM(duration_minutes) / COUNT(Movies)", "Calculated Result": f"{summary_json['kpis']['Avg_Movie_Duration_Minutes']} mins", "Description": "Mean runtime across all movie titles (Median = 98.0 mins)."},
        {"KPI Measure": "Average TV Seasons", "Formula / Query": "SUM(num_seasons) / COUNT(TV Shows)", "Calculated Result": f"{summary_json['kpis']['Avg_TV_Seasons']} seasons", "Description": "Mean number of released seasons across TV series (Median = 1.0 season)."},
        {"KPI Measure": "Year-over-Year (YoY) Growth", "Formula / Query": "((Additions_Year_t - Additions_Year_t-1) / Additions_Year_t-1) * 100", "Calculated Result": "+403.41% (2016 peak YoY)", "Description": "Annual percentage velocity of catalog additions."}
    ]
    st.dataframe(pd.DataFrame(kpi_table_data), use_container_width=True, hide_index=True)


# ================= TAB 2: VIEW 1 - CONTENT GROWTH =================
with tab_growth:
    st.markdown('<div class="section-title">📈 View 1 — Content Ingestion Growth & Inflection Points Over Time</div>', unsafe_allow_html=True)
    
    # Yearly aggregation
    yearly_df = filtered_df.dropna(subset=['year_added']).groupby(['year_added', 'type']).size().unstack(fill_value=0).reset_index()
    if 'Movie' not in yearly_df: yearly_df['Movie'] = 0
    if 'TV Show' not in yearly_df: yearly_df['TV Show'] = 0
    yearly_df['Total'] = yearly_df['Movie'] + yearly_df['TV Show']
    yearly_df['YoY_Growth_%'] = (yearly_df['Total'].pct_change() * 100).round(2).fillna(0)
    yearly_df['Movie_Share_%'] = ((yearly_df['Movie'] / yearly_df['Total']) * 100).round(1)
    yearly_df['TV_Share_%'] = ((yearly_df['TV Show'] / yearly_df['Total']) * 100).round(1)
    
    col_g1, col_g2 = st.columns([3, 2])
    
    with col_g1:
        # Stacked Area / Line Chart
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=yearly_df['year_added'], 
            y=yearly_df['Movie'], 
            name='Movies Added',
            mode='lines+markers',
            line=dict(color='#E50914', width=3),
            fill='tozeroy',
            fillcolor='rgba(229, 9, 20, 0.25)',
            hovertemplate="<b>Year %{x}</b><br>Movies Added: %{y:,}<extra></extra>"
        ))
        fig_trend.add_trace(go.Scatter(
            x=yearly_df['year_added'], 
            y=yearly_df['TV Show'], 
            name='TV Shows Added',
            mode='lines+markers',
            line=dict(color='#00A8E8', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 168, 232, 0.20)',
            hovertemplate="<b>Year %{x}</b><br>TV Shows Added: %{y:,}<extra></extra>"
        ))
        
        fig_trend.update_layout(
            title="Annual Content Additions by Type (2008 – 2021)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F5F5F7'),
            xaxis=dict(gridcolor='#26262B', dtick=1, title="Year Added to Netflix"),
            yaxis=dict(gridcolor='#26262B', title="Number of Titles Ingested"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_g2:
        # YoY Growth Rate Bar Chart
        fig_yoy = px.bar(
            yearly_df[yearly_df['year_added'] >= 2014],
            x='year_added',
            y='YoY_Growth_%',
            text='YoY_Growth_%',
            title="Year-over-Year Growth Rate (%) [2014–2021]",
            color='YoY_Growth_%',
            color_continuous_scale=['#E50914', '#FFAA00', '#30D158']
        )
        fig_yoy.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_yoy.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F5F5F7'),
            xaxis=dict(gridcolor='#26262B', dtick=1, title="Year"),
            yaxis=dict(gridcolor='#26262B', title="YoY Growth (%)"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_yoy, use_container_width=True)
        
    st.markdown("""
    <div class="takeaway-box">
        <strong>Key Ingestion Finding & Inflection Points:</strong> Content additions surged by <strong>+403.41% in 2016</strong> (jumping from 88 titles in 2015 to 443 titles in 2016). Peak ingestion occurred in <strong>2019 with 2,153 titles added</strong> (1,497 Movies and 656 TV Shows). In 2020, despite a <strong>-6.69% total decline</strong> caused by filming lockdowns, TV shows achieved their highest volume ever at <strong>697 titles (34.69% of all 2020 additions)</strong>.
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed Yearly Growth Data Table
    st.markdown("#### Yearly Additions & Composition Breakdown")
    st.dataframe(yearly_df, use_container_width=True, hide_index=True)


# ================= TAB 3: VIEW 2 - GENRE & RATINGS =================
with tab_genres:
    st.markdown('<div class="section-title">🎭 View 2 — Category Distribution: Genre Dominance & Rating Demographics</div>', unsafe_allow_html=True)
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        # Top 12 Genres Bar Chart
        top_genres = f_genres['genre'].value_counts().head(12).reset_index()
        top_genres.columns = ['Genre', 'Count']
        top_genres['Pct_of_Library'] = ((top_genres['Count'] / kpi_total) * 100).round(1) if kpi_total > 0 else 0
        
        fig_genre = px.bar(
            top_genres.sort_values(by='Count', ascending=True),
            x='Count',
            y='Genre',
            orientation='h',
            text=top_genres.sort_values(by='Count', ascending=True).apply(lambda r: f"{r['Count']:,} ({r['Pct_of_Library']}%)", axis=1),
            title="Top 12 Genres (Count & % of Library)",
            color='Count',
            color_continuous_scale=['#8B0000', '#E50914']
        )
        fig_genre.update_traces(textposition='inside', textfont=dict(color='#FFFFFF', size=12))
        fig_genre.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F5F5F7'),
            xaxis=dict(gridcolor='#26262B', title="Total Titles"),
            yaxis=dict(title=""),
            coloraxis_showscale=False,
            height=460
        )
        st.plotly_chart(fig_genre, use_container_width=True)
        
    with col_c2:
        # Age Rating Distribution Bar Chart
        top_ratings = filtered_df['rating_imputed'].value_counts().reset_index()
        top_ratings.columns = ['Rating', 'Count']
        top_ratings['Pct_of_Total'] = ((top_ratings['Count'] / kpi_total) * 100).round(1) if kpi_total > 0 else 0
        
        fig_rating = px.bar(
            top_ratings.head(10).sort_values(by='Count', ascending=True),
            x='Count',
            y='Rating',
            orientation='h',
            text=top_ratings.head(10).sort_values(by='Count', ascending=True).apply(lambda r: f"{r['Count']:,} ({r['Pct_of_Total']}%)", axis=1),
            title="Rating Distribution (Count & % of Total)",
            color='Count',
            color_continuous_scale=['#005A9C', '#00A8E8']
        )
        fig_rating.update_traces(textposition='inside', textfont=dict(color='#FFFFFF', size=12))
        fig_rating.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F5F5F7'),
            xaxis=dict(gridcolor='#26262B', title="Total Titles"),
            yaxis=dict(title=""),
            coloraxis_showscale=False,
            height=460
        )
        st.plotly_chart(fig_rating, use_container_width=True)
        
    st.markdown("""
    <div class="takeaway-box">
        <strong>Genre & Rating Takeaway:</strong> <strong>International Movies</strong> is the #1 genre overall with <strong>2,437 titles (31.30% of total library)</strong>, followed by <strong>Dramas at 2,106 titles (27.05%)</strong> and <strong>Comedies at 1,471 titles (18.89%)</strong>. In ratings, <strong>TV-MA dominates at 36.77% (2,863 titles)</strong> and <strong>TV-14 at 24.80% (1,931 titles)</strong>, showing that over <strong>61.57% of Netflix's catalog</strong> is tailored for mature teens and adults.
    </div>
    """, unsafe_allow_html=True)
    
    # Genre Mix by Content Type Comparison
    st.markdown("#### 🎬 Genre Mix Comparison: Movies vs. TV Shows")
    genre_type_comp = f_genres.groupby(['genre', 'type']).size().unstack(fill_value=0).reset_index()
    if 'Movie' not in genre_type_comp: genre_type_comp['Movie'] = 0
    if 'TV Show' not in genre_type_comp: genre_type_comp['TV Show'] = 0
    genre_type_comp['Total'] = genre_type_comp['Movie'] + genre_type_comp['TV Show']
    genre_type_comp = genre_type_comp.sort_values(by='Total', ascending=False).head(15)
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name='Movies',
        x=genre_type_comp['genre'],
        y=genre_type_comp['Movie'],
        marker_color='#E50914'
    ))
    fig_comp.add_trace(go.Bar(
        name='TV Shows',
        x=genre_type_comp['genre'],
        y=genre_type_comp['TV Show'],
        marker_color='#00A8E8'
    ))
    fig_comp.update_layout(
        barmode='group',
        title="Top 15 Genres Split by Content Type (Movies vs TV Shows)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F5F5F7'),
        xaxis=dict(gridcolor='#26262B', tickangle=-45),
        yaxis=dict(gridcolor='#26262B', title="Title Count")
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# ================= TAB 4: VIEW 3 - GEOGRAPHIC & DRILL-DOWN =================
with tab_geo:
    st.markdown('<div class="section-title">🌍 View 3 — Geographic Distribution & Country → Genre → Type Drill-Down</div>', unsafe_allow_html=True)
    
    # Clean country ranking
    top_countries_df = f_countries[f_countries['country_single'] != 'Unknown Country']['country_single'].value_counts().head(15).reset_index()
    top_countries_df.columns = ['Country', 'Count']
    top_countries_df['Pct_of_Library'] = ((top_countries_df['Count'] / kpi_total) * 100).round(2) if kpi_total > 0 else 0
    
    col_m1, col_m2 = st.columns([3, 2])
    
    with col_m1:
        # Global Map
        country_counts_all = f_countries[f_countries['country_single'] != 'Unknown Country']['country_single'].value_counts().reset_index()
        country_counts_all.columns = ['Country', 'Count']
        
        fig_map = px.choropleth(
            country_counts_all,
            locations='Country',
            locationmode='country names',
            color='Count',
            color_continuous_scale='Reds',
            title="Global Content Production Footprint (Title Count by Country)",
            hover_name='Country'
        )
        fig_map.update_layout(
            geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='#151518', showland=True, landcolor='#222228', subunitcolor='#33333C'),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F5F5F7'),
            margin=dict(t=40, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
    with col_m2:
        # Top 10 Countries Horizontal Bar
        fig_top_c = px.bar(
            top_countries_df.head(10).sort_values(by='Count', ascending=True),
            x='Count',
            y='Country',
            orientation='h',
            text=top_countries_df.head(10).sort_values(by='Count', ascending=True).apply(lambda r: f"{r['Count']:,} ({r['Pct_of_Library']}%)", axis=1),
            title="Top 10 Content Producing Countries",
            color='Count',
            color_continuous_scale=['#67000D', '#CB181D']
        )
        fig_top_c.update_traces(textposition='inside', textfont=dict(color='#FFFFFF', size=11))
        fig_top_c.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F5F5F7'),
            xaxis=dict(gridcolor='#26262B', title="Title Count"),
            yaxis=dict(title=""),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_top_c, use_container_width=True)
        
    st.markdown("""
    <div class="takeaway-box">
        <strong>Geographic Takeaway:</strong> The <strong>United States</strong> leads production with <strong>3,297 titles (42.34% of the library)</strong>, followed by <strong>India with 990 titles (12.71%)</strong> and the <strong>United Kingdom with 723 titles (9.28%)</strong>. Production in India is heavily movie-dominated (<strong>92.42% Movies vs. 7.58% TV Shows</strong>), whereas the UK has the highest TV show proportion among top producers (<strong>35.41% TV Shows</strong>).
    </div>
    """, unsafe_allow_html=True)
    
    # ----------------- DRILL DOWN COMPONENT -----------------
    st.markdown("### 🔬 Multi-Level Drill-Down: Country ➔ Genre ➔ Content Type Split")
    st.markdown("Select any country below to inspect its unique genre footprint and Movie vs TV Show allocation.")
    
    avail_drill_countries = sorted(top_countries_df['Country'].unique().tolist())
    selected_drill_country = st.selectbox("Select Country to Drill Down:", avail_drill_countries, index=0)
    
    if selected_drill_country:
        drill_sub = df_hierarchy[df_hierarchy['country_single'] == selected_drill_country]
        drill_total = drill_sub['title_count'].sum()
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            # Country Top Genres Sunburst / Treemap
            fig_tree = px.treemap(
                drill_sub,
                path=['country_single', 'genre', 'type'],
                values='title_count',
                title=f"Hierarchy Tree: {selected_drill_country} → Genre → Content Type",
                color='type',
                color_discrete_map={'Movie': '#E50914', 'TV Show': '#00A8E8'}
            )
            fig_tree.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F5F5F7'),
                margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_tree, use_container_width=True)
            
        with col_d2:
            # Breakdown Table for selected country
            drill_table = drill_sub.pivot_table(index='genre', columns='type', values='title_count', fill_value=0).reset_index()
            if 'Movie' not in drill_table: drill_table['Movie'] = 0
            if 'TV Show' not in drill_table: drill_table['TV Show'] = 0
            drill_table['Total'] = drill_table['Movie'] + drill_table['TV Show']
            drill_table['Share_%'] = ((drill_table['Total'] / drill_total) * 100).round(2)
            drill_table = drill_table.sort_values(by='Total', ascending=False)
            
            st.markdown(f"**Top Genres in {selected_drill_country}** (Total Tagged Titles: {drill_total:,})")
            st.dataframe(drill_table, use_container_width=True, hide_index=True)


# ================= TAB 5: VIEW 4 - DURATION & SEASONS =================
with tab_duration:
    st.markdown('<div class="section-title">⏱️ View 4 — Duration Patterns & TV Season Distributions</div>', unsafe_allow_html=True)
    
    col_u1, col_u2 = st.columns(2)
    
    with col_u1:
        # Movie Duration Histogram & Category Breakdown
        movies_df = filtered_df[filtered_df['type'] == 'Movie'].dropna(subset=['duration_minutes'])
        
        fig_hist = px.histogram(
            movies_df,
            x='duration_minutes',
            nbins=35,
            title=f"Movie Runtime Distribution (Mean: {movie_dur_str}, Median: {movies_df['duration_minutes'].median():.0f} min)",
            color_discrete_sequence=['#E50914']
        )
        fig_hist.add_vline(x=movie_dur_avg, line_dash="dash", line_color="#FFFFFF", annotation_text=f"Mean: {movie_dur_avg:.1f}m")
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F5F5F7'),
            xaxis=dict(gridcolor='#26262B', title="Duration (Minutes)"),
            yaxis=dict(gridcolor='#26262B', title="Movie Count")
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Duration Category Pie
        dur_cat_counts = movies_df['duration_category'].value_counts().reset_index()
        dur_cat_counts.columns = ['Category', 'Count']
        dur_cat_counts['Pct'] = ((dur_cat_counts['Count'] / len(movies_df)) * 100).round(2)
        st.dataframe(dur_cat_counts, use_container_width=True, hide_index=True)
        
    with col_u2:
        # TV Show Seasons Bar Chart
        tv_df = filtered_df[filtered_df['type'] == 'TV Show'].dropna(subset=['num_seasons'])
        season_counts = tv_df['num_seasons'].value_counts().reset_index()
        season_counts.columns = ['Seasons', 'Count']
        season_counts['Seasons'] = season_counts['Seasons'].astype(int)
        season_counts = season_counts.sort_values(by='Seasons')
        season_counts['Pct_of_TV'] = ((season_counts['Count'] / len(tv_df)) * 100).round(2) if len(tv_df) > 0 else 0
        
        fig_season = px.bar(
            season_counts.head(10),
            x='Seasons',
            y='Count',
            text=season_counts.head(10).apply(lambda r: f"{r['Count']:,} ({r['Pct_of_TV']}%)", axis=1),
            title=f"TV Show Season Counts (Mean: {tv_sea_str})",
            color='Count',
            color_continuous_scale=['#005A9C', '#00A8E8']
        )
        fig_season.update_traces(textposition='outside')
        fig_season.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F5F5F7'),
            xaxis=dict(gridcolor='#26262B', dtick=1, title="Number of Seasons"),
            yaxis=dict(gridcolor='#26262B', title="TV Show Count"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_season, use_container_width=True)
        
        # Season Category Pie
        sea_cat_counts = tv_df['season_category'].value_counts().reset_index()
        sea_cat_counts.columns = ['Category', 'Count']
        sea_cat_counts['Pct'] = ((sea_cat_counts['Count'] / len(tv_df)) * 100).round(2)
        st.dataframe(sea_cat_counts, use_container_width=True, hide_index=True)
        
    st.markdown("""
    <div class="takeaway-box">
        <strong>Runtime & Season Takeaway:</strong> Feature films center heavily around standard runtime, with <strong>73.39% of movies (3,946 titles)</strong> falling in the Medium window (60–120 minutes) with a mean of <strong>99.31 minutes</strong>. For episodic content, <strong>66.72% of all TV shows (1,608 titles)</strong> stop after exactly <strong>1 Season</strong>, while only <strong>9.79% (236 titles)</strong> reach 4 or more seasons.
    </div>
    """, unsafe_allow_html=True)


# ================= TAB 6: DATA EXPLORER & LOOKUPS =================
with tab_explorer:
    st.markdown('<div class="section-title">🔍 Data Explorer, Quality Log & Export</div>', unsafe_allow_html=True)
    
    st.markdown("#### Search & Filter Catalog Records")
    search_query = st.text_input("Search titles, directors, cast, or keywords:", "")
    
    display_df = filtered_df[[
        'show_id', 'type', 'title', 'director_clean', 'cast_clean', 
        'country_clean', 'date_added_clean', 'release_year', 
        'rating_imputed', 'duration_raw', 'listed_in'
    ]].copy()
    display_df.columns = [
        'ID', 'Type', 'Title', 'Director', 'Cast', 
        'Country', 'Date Added', 'Release Year', 
        'Rating', 'Duration', 'Genres'
    ]
    
    if search_query:
        mask = (
            display_df['Title'].str.contains(search_query, case=False, na=False) |
            display_df['Director'].str.contains(search_query, case=False, na=False) |
            display_df['Cast'].str.contains(search_query, case=False, na=False) |
            display_df['Genres'].str.contains(search_query, case=False, na=False)
        )
        display_df = display_df[mask]
        
    st.markdown(f"Showing **{len(display_df):,}** matching titles:")
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Download Button
    csv_bytes = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="netflix_filtered_analysis.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    st.markdown("#### 🛠️ Data Cleaning Audit Trail & Transformations")
    st.dataframe(pd.DataFrame(summary_json['missing_summary']), use_container_width=True, hide_index=True)

