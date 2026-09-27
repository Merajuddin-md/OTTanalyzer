import pandas as pd
import numpy as np
import json
import os

def run_data_pipeline():
    print("=" * 60)
    print("STEP 1: LOADING AND AUDITING RAW DATA")
    print("=" * 60)
    
    raw_path = "netflix_titles.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")
        
    df = pd.read_csv(raw_path)
    total_raw_rows = len(df)
    print(f"Total raw rows: {total_raw_rows:,}")
    print(f"Columns: {list(df.columns)}")
    
    # 1. Missing Values Audit
    missing_summary = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / total_raw_rows) * 100
        missing_summary.append({
            "Column": col,
            "Missing_Count": int(null_count),
            "Missing_Percentage": round(null_pct, 2),
            "Action_Taken": (
                "Filled with 'Unknown Director'" if col == 'director' else
                "Filled with 'Unknown Cast'" if col == 'cast' else
                "Filled with 'Unknown Country'" if col == 'country' else
                "Kept null in date fields; flagged as unrecorded addition date" if col == 'date_added' else
                "Filled with 'Unknown' rating" if col == 'rating' else
                "No missing values"
            )
        })
    missing_df = pd.DataFrame(missing_summary)
    print("\nData Quality & Missing Values Audit:")
    print(missing_df.to_string(index=False))
    
    # 2. Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    print(f"\nExact duplicate rows detected: {duplicate_count}")
    
    print("\n" + "=" * 60)
    print("STEP 2: CLEANING & STANDARDIZATION")
    print("=" * 60)
    
    # Clean text columns (strip whitespace, normalize)
    str_cols = ['show_id', 'type', 'title', 'director', 'cast', 'country', 'rating', 'listed_in', 'description']
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        
    # Standardize 'type'
    df['type'] = df['type'].str.title().replace({'Tv Show': 'TV Show', 'Tv': 'TV Show'})
    
    # Standardize 'rating'
    rating_fixes = {
        'UR': 'UR',
        'NR': 'NR'
    }
    df['rating'] = df['rating'].replace(rating_fixes)
    df['rating_imputed'] = df['rating'].fillna('Unknown')
    
    # Parse date_added
    # The format in Netflix dataset is typically "September 25, 2021" or "March 2, 2019"
    df['date_added_clean'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df['year_added'] = df['date_added_clean'].dt.year.astype('Int64')
    df['month_added'] = df['date_added_clean'].dt.month.astype('Int64')
    df['month_name_added'] = df['date_added_clean'].dt.month_name()
    
    # Duration parsing
    # Movies have 'XX min', TV Shows have 'X Season' / 'X Seasons'
    df['duration_raw'] = df['duration']
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
    
    # Impute missing strings for analytics safety
    df['director_clean'] = df['director'].fillna('Unknown Director')
    df['cast_clean'] = df['cast'].fillna('Unknown Cast')
    df['country_clean'] = df['country'].fillna('Unknown Country')
    
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 60)
    
    # 1. Decade and Release Year Buckets
    df['release_decade'] = (df['release_year'] // 10) * 10
    df['release_decade_label'] = df['release_decade'].astype(str) + "s"
    
    def bucket_release_year(yr):
        if yr < 2000:
            return "Classic (< 2000)"
        elif yr <= 2010:
            return "2000–2010"
        elif yr <= 2015:
            return "2011–2015"
        elif yr <= 2018:
            return "2016–2018"
        else:
            return "2019–Present"
            
    df['release_year_bucket'] = df['release_year'].apply(bucket_release_year)
    
    # 2. Duration Categories for Movies
    def categorize_movie_duration(mins):
        if pd.isna(mins):
            return np.nan
        if mins < 60:
            return "Short (<60 min)"
        elif mins <= 120:
            return "Medium (60–120 min)"
        else:
            return "Long (>120 min)"
            
    df['duration_category'] = df['duration_minutes'].apply(categorize_movie_duration)
    
    # 3. Season Categories for TV Shows
    def categorize_tv_seasons(seasons):
        if pd.isna(seasons):
            return np.nan
        if seasons == 1:
            return "1 Season"
        elif seasons <= 3:
            return "2–3 Seasons"
        else:
            return "4+ Seasons"
            
    df['season_category'] = df['num_seasons'].apply(categorize_tv_seasons)
    
    # 4. Multi-value Genre and Country features
    df['genre_list'] = df['listed_in'].fillna('').apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])
    df['num_genres'] = df['genre_list'].apply(len)
    df['has_multiple_genres'] = df['num_genres'] > 1
    
    df['country_list'] = df['country'].fillna('Unknown Country').apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
    df['num_countries'] = df['country_list'].apply(len)
    df['has_multiple_countries'] = df['num_countries'] > 1
    
    # 5. Rating Demographics Grouping
    def group_maturity_rating(rating):
        if pd.isna(rating) or rating == 'Unknown':
            return "Unknown / Unrated"
        r = rating.upper()
        if r in ['TV-MA', 'R', 'NC-17', 'UR']:
            return "Adult (18+)"
        elif r in ['TV-14', 'PG-13']:
            return "Teens (13-17)"
        elif r in ['TV-PG', 'PG', 'TV-Y', 'TV-Y7', 'TV-Y7-FV', 'TV-G', 'G']:
            return "Kids & Family"
        elif r in ['NR']:
            return "Not Rated (NR)"
        else:
            return "Other / Special"
            
    df['rating_group'] = df['rating_imputed'].apply(group_maturity_rating)
    
    # 6. Library Ingestion Lag (Years to Platform)
    df['years_to_platform'] = df['year_added'] - df['release_year']
    
    print("\n" + "=" * 60)
    print("STEP 4: GENERATING EXPLODED LOOKUP DATASETS")
    print("=" * 60)
    
    # Exploded Genres Dataframe
    df_genres = df.explode('genre_list').rename(columns={'genre_list': 'genre'})
    df_genres['genre'] = df_genres['genre'].str.strip()
    
    # Exploded Countries Dataframe
    df_countries = df.explode('country_list').rename(columns={'country_list': 'country_single'})
    df_countries['country_single'] = df_countries['country_single'].str.strip()
    
    # Multi-level drill-down: Country -> Genre -> Content Type
    df_country_genre = df.explode('country_list').explode('genre_list')
    df_country_genre['country_single'] = df_country_genre['country_list'].str.strip()
    df_country_genre['genre'] = df_country_genre['genre_list'].str.strip()
    
    drilldown_df = df_country_genre.groupby(['country_single', 'genre', 'type']).agg(
        title_count=('show_id', 'count')
    ).reset_index()
    
    print(f"Exploded Genres rows: {len(df_genres):,}")
    print(f"Exploded Countries rows: {len(df_countries):,}")
    print(f"Drill-down combinations: {len(drilldown_df):,}")
    
    print("\n" + "=" * 60)
    print("STEP 5: CALCULATING CORE KPIS & METRICS")
    print("=" * 60)
    
    total_titles = len(df)
    movie_count = int((df['type'] == 'Movie').sum())
    movie_pct = round((movie_count / total_titles) * 100, 2)
    
    tv_count = int((df['type'] == 'TV Show').sum())
    tv_pct = round((tv_count / total_titles) * 100, 2)
    
    unique_genres_count = int(df_genres['genre'].nunique())
    unique_countries_count = int(df_countries[df_countries['country_single'] != 'Unknown Country']['country_single'].nunique())
    
    avg_movie_duration = round(float(df[df['type'] == 'Movie']['duration_minutes'].mean()), 2)
    median_movie_duration = round(float(df[df['type'] == 'Movie']['duration_minutes'].median()), 2)
    
    avg_tv_seasons = round(float(df[df['type'] == 'TV Show']['num_seasons'].mean()), 2)
    median_tv_seasons = round(float(df[df['type'] == 'TV Show']['num_seasons'].median()), 2)
    
    max_year_added = int(df['year_added'].max())
    min_year_added = int(df['year_added'].min())
    
    # Yearly Growth Analysis
    yearly_growth = df.dropna(subset=['year_added']).groupby(['year_added', 'type']).size().unstack(fill_value=0)
    yearly_growth['Total'] = yearly_growth['Movie'] + yearly_growth['TV Show']
    yearly_growth['YoY_Total_Change_Count'] = yearly_growth['Total'].diff().fillna(0).astype(int)
    yearly_growth['YoY_Growth_Rate_Pct'] = (yearly_growth['Total'].pct_change() * 100).round(2).fillna(0.0)
    yearly_growth['Movie_Share_Pct'] = ((yearly_growth['Movie'] / yearly_growth['Total']) * 100).round(2)
    yearly_growth['TV_Share_Pct'] = ((yearly_growth['TV Show'] / yearly_growth['Total']) * 100).round(2)
    
    # Top Genres Overall, Movies, and TV Shows
    genre_overall = df_genres['genre'].value_counts().reset_index()
    genre_overall.columns = ['Genre', 'Count']
    genre_overall['Pct_of_Total_Library'] = ((genre_overall['Count'] / total_titles) * 100).round(2)
    
    genre_by_type = df_genres.groupby(['genre', 'type']).size().unstack(fill_value=0).reset_index()
    genre_by_type['Total'] = genre_by_type['Movie'] + genre_by_type['TV Show']
    genre_by_type = genre_by_type.sort_values(by='Total', ascending=False)
    
    # Top Countries
    country_summary = df_countries[df_countries['country_single'] != 'Unknown Country']['country_single'].value_counts().reset_index()
    country_summary.columns = ['Country', 'Count']
    country_summary['Pct_of_Total_Library'] = ((country_summary['Count'] / total_titles) * 100).round(2)
    
    # Ratings Distribution
    rating_summary = df['rating_imputed'].value_counts().reset_index()
    rating_summary.columns = ['Rating', 'Count']
    rating_summary['Pct_of_Total'] = ((rating_summary['Count'] / total_titles) * 100).round(2)
    
    rating_group_summary = df['rating_group'].value_counts().reset_index()
    rating_group_summary.columns = ['Rating_Group', 'Count']
    rating_group_summary['Pct_of_Total'] = ((rating_group_summary['Count'] / total_titles) * 100).round(2)
    
    # Movie Duration Category Distribution
    movie_dur_dist = df[df['type'] == 'Movie']['duration_category'].value_counts().reset_index()
    movie_dur_dist.columns = ['Duration_Category', 'Count']
    movie_dur_dist['Pct_of_Movies'] = ((movie_dur_dist['Count'] / movie_count) * 100).round(2)
    
    # TV Season Category Distribution
    tv_season_dist = df[df['type'] == 'TV Show']['season_category'].value_counts().reset_index()
    tv_season_dist.columns = ['Season_Category', 'Count']
    tv_season_dist['Pct_of_TV_Shows'] = ((tv_season_dist['Count'] / tv_count) * 100).round(2)
    
    # Ratings Shift Over Time (Early vs Recent)
    rating_shift = df.dropna(subset=['year_added']).groupby(['year_added', 'rating_group']).size().unstack(fill_value=0)
    rating_shift_pct = rating_shift.div(rating_shift.sum(axis=1), axis=0) * 100
    
    kpis = {
        "Total_Titles": total_titles,
        "Movie_Count": movie_count,
        "Movie_Percentage": movie_pct,
        "TV_Show_Count": tv_count,
        "TV_Show_Percentage": tv_pct,
        "Unique_Genres": unique_genres_count,
        "Unique_Countries": unique_countries_count,
        "Avg_Movie_Duration_Minutes": avg_movie_duration,
        "Median_Movie_Duration_Minutes": median_movie_duration,
        "Avg_TV_Seasons": avg_tv_seasons,
        "Median_TV_Seasons": median_tv_seasons,
        "Latest_Year_Added": max_year_added,
        "Earliest_Year_Added": min_year_added
    }
    
    print("\n--- KPI METRICS SUMMARY ---")
    for k, v in kpis.items():
        print(f"  {k}: {v}")
        
    print("\n--- SAVING PROCESSED ARTIFACTS ---")
    
    # Save cleaned primary dataset
    clean_csv_path = "cleaned_netflix_titles.csv"
    df_clean_export = df.drop(columns=['genre_list', 'country_list'])
    df_clean_export.to_csv(clean_csv_path, index=False)
    print(f"Saved: {clean_csv_path} ({len(df_clean_export):,} rows)")
    
    # Save exploded genres
    genres_export_path = "exploded_genres.csv"
    df_genres[['show_id', 'title', 'type', 'genre', 'release_year', 'year_added', 'rating_imputed', 'rating_group']].to_csv(genres_export_path, index=False)
    print(f"Saved: {genres_export_path} ({len(df_genres):,} rows)")
    
    # Save exploded countries
    countries_export_path = "exploded_countries.csv"
    df_countries[['show_id', 'title', 'type', 'country_single', 'release_year', 'year_added', 'rating_imputed', 'rating_group']].to_csv(countries_export_path, index=False)
    print(f"Saved: {countries_export_path} ({len(df_countries):,} rows)")
    
    # Save drilldown hierarchy
    drilldown_path = "country_genre_type_hierarchy.csv"
    drilldown_df.to_csv(drilldown_path, index=False)
    print(f"Saved: {drilldown_path} ({len(drilldown_df):,} rows)")
    
    # Save JSON summary
    summary_data = {
        "kpis": kpis,
        "missing_summary": missing_summary,
        "yearly_growth": yearly_growth.reset_index().to_dict(orient='records'),
        "top_genres_overall": genre_overall.head(15).to_dict(orient='records'),
        "genre_by_type": genre_by_type.head(15).to_dict(orient='records'),
        "top_countries": country_summary.head(15).to_dict(orient='records'),
        "rating_distribution": rating_summary.to_dict(orient='records'),
        "rating_group_distribution": rating_group_summary.to_dict(orient='records'),
        "movie_duration_distribution": movie_dur_dist.to_dict(orient='records'),
        "tv_season_distribution": tv_season_dist.to_dict(orient='records')
    }
    
    with open("analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print("Saved: analysis_summary.json")
    print("Data Pipeline Completed Successfully!")

if __name__ == "__main__":
    run_data_pipeline()
