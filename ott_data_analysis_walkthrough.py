"""
========================================================================================
OTT CONTENT & CONTENT LIBRARY ANALYSIS — COMPLETE STEP-BY-STEP DATA ANALYST WALKTHROUGH
========================================================================================
Author: Data Analysis Team
Dataset: Netflix Streaming Titles Catalog (7,787 titles across 12 metadata attributes)

This learning script walks through every data cleaning, transformation, feature engineering,
KPI computation, and analytical breakdown with clear explanations for each step.
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("STEP 1: DATA INGESTION & DATA QUALITY AUDIT")
print("=" * 80)
print("""
[Why this step matters]:
Before performing any analysis or modeling, a data analyst must inspect the structure,
data types, and missingness of the raw dataset. Never silently drop missing rows!
""")

# Load the raw dataset
raw_df = pd.read_csv("netflix_titles.csv")
total_rows = len(raw_df)
print(f"Total Records: {total_rows:,}")
print(f"Total Columns: {raw_df.shape[1]}")
print(f"Columns: {list(raw_df.columns)}")

# Missing Values Audit
missing_table = []
for col in raw_df.columns:
    nulls = raw_df[col].isnull().sum()
    pct = (nulls / total_rows) * 100
    missing_table.append({
        "Column Name": col,
        "Missing Count": int(nulls),
        "Missing %": f"{pct:.2f}%",
        "Data Type": str(raw_df[col].dtype),
        "Cleaning Strategy": (
            "Impute 'Unknown Director' (30.68% missing; dropping would bias sample)" if col == 'director' else
            "Impute 'Unknown Cast' (9.22% missing)" if col == 'cast' else
            "Impute 'Unknown Country' (6.51% missing)" if col == 'country' else
            "Keep null or flag (0.13% missing; uncataloged date added)" if col == 'date_added' else
            "Impute 'Unknown' rating (0.09% missing)" if col == 'rating' else
            "No missing values (Clean)"
        )
    })
print("\nMissing Values & Quality Audit:")
print(pd.DataFrame(missing_table).to_string(index=False))

# Check for duplicate rows
dups = raw_df.duplicated().sum()
print(f"\nExact duplicate rows: {dups} (No row deduplication needed).")


print("\n" + "=" * 80)
print("STEP 2: DATA STANDARDIZATION & FEATURE ENGINEERING")
print("=" * 80)
print("""
[Why this step matters]:
Raw data contains mixed formats, compound fields (e.g. multi-value genres),
and unseparated units. We create clean standardized fields.
""")

df = raw_df.copy()

# 1. Standardize string casing and trim leading/trailing whitespace
for col in ['show_id', 'type', 'title', 'director', 'cast', 'country', 'rating', 'listed_in', 'description']:
    df[col] = df[col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})

# 2. Standardize 'type' to exactly 'Movie' or 'TV Show'
df['type'] = df['type'].str.title().replace({'Tv Show': 'TV Show', 'Tv': 'TV Show'})
print("Content Types:", df['type'].value_counts().to_dict())

# 3. Parse date_added into datetime and extract year and month
# Note the difference:
# - release_year: Year the title originally premiered (in theaters or TV network)
# - date_added: Date Netflix acquired and published the title to its streaming library
df['date_added_clean'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
df['year_added'] = df['date_added_clean'].dt.year.astype('Int64')
df['month_added'] = df['date_added_clean'].dt.month.astype('Int64')
df['month_name_added'] = df['date_added_clean'].dt.month_name()

# 4. Split Duration into Movies (duration_minutes) vs TV Shows (num_seasons)
# Formula: Extract digits into separate numeric columns
df['duration_minutes'] = np.where(
    df['type'] == 'Movie',
    df['duration'].str.extract(r'(\d+)')[0].astype(float),
    np.nan
)
df['num_seasons'] = np.where(
    df['type'] == 'TV Show',
    df['duration'].str.extract(r'(\d+)')[0].astype(float),
    np.nan
)

# 5. Feature Engineering: Bucketing and Multi-Value Metrics
df['release_decade'] = (df['release_year'] // 10) * 10
df['release_decade_label'] = df['release_decade'].astype(str) + "s"

def get_movie_duration_cat(m):
    if pd.isna(m): return np.nan
    if m < 60: return 'Short (<60 min)'
    elif m <= 120: return 'Medium (60–120 min)'
    else: return 'Long (>120 min)'

df['duration_category'] = df['duration_minutes'].apply(get_movie_duration_cat)

def get_tv_season_cat(s):
    if pd.isna(s): return np.nan
    if s == 1: return '1 Season'
    elif s <= 3: return '2–3 Seasons'
    else: return '4+ Seasons'

df['season_category'] = df['num_seasons'].apply(get_tv_season_cat)

# Multi-value genres & countries
df['genre_list'] = df['listed_in'].fillna('').apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])
df['num_genres'] = df['genre_list'].apply(len)
df['has_multiple_genres'] = df['num_genres'] > 1

df['country_list'] = df['country'].fillna('Unknown Country').apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
df['num_countries'] = df['country_list'].apply(len)
df['has_multiple_countries'] = df['num_countries'] > 1

print(f"Titles with multiple genres: {df['has_multiple_genres'].sum():,} ({df['has_multiple_genres'].sum()/total_rows*100:.1f}%)")
print(f"Titles with multiple production countries: {df['has_multiple_countries'].sum():,} ({df['has_multiple_countries'].sum()/total_rows*100:.1f}%)")


print("\n" + "=" * 80)
print("STEP 3: CORE KPI MEASURES & MATHEMATICAL FORMULAS")
print("=" * 80)

total_titles = len(df)
movie_count = int((df['type'] == 'Movie').sum())
movie_pct = (movie_count / total_titles) * 100

tv_count = int((df['type'] == 'TV Show').sum())
tv_pct = (tv_count / total_titles) * 100

# Exploded Genres and Countries
df_genres = df.explode('genre_list').rename(columns={'genre_list': 'genre'})
df_genres['genre'] = df_genres['genre'].str.strip()

df_countries = df.explode('country_list').rename(columns={'country_list': 'country_single'})
df_countries['country_single'] = df_countries['country_single'].str.strip()

unique_genres = df_genres['genre'].nunique()
unique_countries = df_countries[df_countries['country_single'] != 'Unknown Country']['country_single'].nunique()

avg_mov_dur = df[df['type'] == 'Movie']['duration_minutes'].mean()
avg_tv_sea = df[df['type'] == 'TV Show']['num_seasons'].mean()

latest_year = int(df['year_added'].max())

print(f"1. Total Catalog Titles: {total_titles:,}")
print(f"2. Movie Count: {movie_count:,} ({movie_pct:.2f}% of catalog)")
print(f"3. TV Show Count: {tv_count:,} ({tv_pct:.2f}% of catalog)")
print(f"4. Unique Genres Tagged: {unique_genres}")
print(f"5. Unique Content Countries: {unique_countries}")
print(f"6. Average Movie Duration: {avg_mov_dur:.2f} minutes")
print(f"7. Average TV Show Seasons: {avg_tv_sea:.2f} seasons")
print(f"8. Latest Year Added: {latest_year}")


print("\n" + "=" * 80)
print("STEP 4: DETAILED ANALYTICAL FINDINGS & RESEARCH QUESTIONS")
print("=" * 80)

# Question 1: Content Growth Over Time & Inflection Points
print("\n--- Question 1: Ingestion Velocity & Growth Over Time ---")
yearly_df = df.dropna(subset=['year_added']).groupby(['year_added', 'type']).size().unstack(fill_value=0)
yearly_df['Total'] = yearly_df['Movie'] + yearly_df['TV Show']
yearly_df['YoY_Growth_%'] = (yearly_df['Total'].pct_change() * 100).round(2)
yearly_df['Movie_Share_%'] = (yearly_df['Movie'] / yearly_df['Total'] * 100).round(2)
yearly_df['TV_Share_%'] = (yearly_df['TV Show'] / yearly_df['Total'] * 100).round(2)

print(yearly_df.to_string())
print("\n[Inflection Point Insight]:")
print("From 2008 to 2015, Netflix added fewer than 100 titles annually.")
print("In 2016, additions exploded by +403.41% YoY (443 titles vs 88 in 2015).")
print("Catalog growth peaked in 2019 at 2,153 additions (1,497 Movies, 656 TV Shows).")
print("In 2020, while total additions dipped -6.69% to 2,009, TV shows reached an all-time high of 697 additions (34.69% of all additions).")

# Question 2: Genre Distribution
print("\n--- Question 2: Genre Distribution & Mix Differences ---")
top_10_genres = df_genres['genre'].value_counts().head(10)
for rank, (genre_name, count) in enumerate(top_10_genres.items(), 1):
    print(f"{rank:2d}. {genre_name}: {count:,} titles ({count/total_titles*100:.2f}% of library)")

# Question 3: Country Comparisons & Drill-down
print("\n--- Question 3: Geographic Production & Drill-Down ---")
top_countries = df_countries[df_countries['country_single'] != 'Unknown Country']['country_single'].value_counts().head(5)
for c, count in top_countries.items():
    c_sub = df[df['country'].fillna('').str.contains(c)]
    m_sub = (c_sub['type'] == 'Movie').sum()
    t_sub = (c_sub['type'] == 'TV Show').sum()
    print(f"  - {c}: {count:,} titles ({count/total_titles*100:.2f}% of library) -> Movies: {m_sub} ({m_sub/len(c_sub)*100:.1f}%), TV: {t_sub} ({t_sub/len(c_sub)*100:.1f}%)")

# Question 4: Ratings & Maturity Shifts
print("\n--- Question 4: Rating Distribution & Audience Shifts ---")
print(df['rating'].fillna('Unknown').value_counts().to_string())

# Question 5: Duration Patterns
print("\n--- Question 5: Duration & Season Distributions ---")
print("Movie Categories:")
print(df[df['type'] == 'Movie']['duration_category'].value_counts(normalize=True)*100)
print("\nTV Show Season Categories:")
print(df[df['type'] == 'TV Show']['season_category'].value_counts(normalize=True)*100)
print(f"Single-season TV shows: {(df[df['type'] == 'TV Show']['num_seasons'] == 1).sum():,} out of {tv_count:,} ({(df[df['type'] == 'TV Show']['num_seasons'] == 1).sum()/tv_count*100:.2f}%)")

print("\n" + "=" * 80)
print("PROJECT EXECUTION COMPLETE!")
print("=" * 80)
