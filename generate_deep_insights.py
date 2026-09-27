import pandas as pd
import numpy as np
import json

def run_deep_analysis():
    df = pd.read_csv("cleaned_netflix_titles.csv")
    df_genres = pd.read_csv("exploded_genres.csv")
    df_countries = pd.read_csv("exploded_countries.csv")
    
    total_titles = len(df)
    total_movies = (df['type'] == 'Movie').sum()
    total_tv = (df['type'] == 'TV Show').sum()
    
    print("=" * 70)
    print("1. LIBRARY COMPOSITION & CORE KPIS")
    print("=" * 70)
    print(f"Total Titles Cataloged: {total_titles:,}")
    print(f"Movies: {total_movies:,} ({total_movies/total_titles*100:.2f}% of library)")
    print(f"TV Shows: {total_tv:,} ({total_tv/total_titles*100:.2f}% of library)")
    print(f"Unique Genres: {df_genres['genre'].nunique()}")
    print(f"Unique Countries: {df_countries[df_countries['country_single'] != 'Unknown Country']['country_single'].nunique()}")
    print(f"Average Movie Duration: {df[df['type'] == 'Movie']['duration_minutes'].mean():.2f} minutes")
    print(f"Median Movie Duration: {df[df['type'] == 'Movie']['duration_minutes'].median():.1f} minutes")
    print(f"Average TV Show Seasons: {df[df['type'] == 'TV Show']['num_seasons'].mean():.2f} seasons")
    print(f"Median TV Show Seasons: {df[df['type'] == 'TV Show']['num_seasons'].median():.1f} seasons")
    
    print("\n" + "=" * 70)
    print("2. CONTENT ADDITION GROWTH & INFLECTION POINTS")
    print("=" * 70)
    yearly = df.dropna(subset=['year_added']).groupby(['year_added', 'type']).size().unstack(fill_value=0)
    yearly['Total'] = yearly['Movie'] + yearly['TV Show']
    yearly['YoY_Growth_%'] = (yearly['Total'].pct_change() * 100).round(2)
    yearly['Movie_Share_%'] = (yearly['Movie'] / yearly['Total'] * 100).round(2)
    yearly['TV_Share_%'] = (yearly['TV Show'] / yearly['Total'] * 100).round(2)
    print(yearly.to_string())
    
    # 2016 to 2019 expansion
    added_2016 = yearly.loc[2016, 'Total']
    added_2019 = yearly.loc[2019, 'Total']
    print(f"\nExpansion Check (2016 vs 2019):")
    print(f"Titles added in 2016: {added_2016:,}")
    print(f"Titles added in 2019: {added_2019:,} (an increase of {((added_2019 - added_2016)/added_2016*100):.2f}%)")
    
    print("\n" + "=" * 70)
    print("3. GENRE DISTRIBUTION (OVERALL, MOVIES, TV SHOWS)")
    print("=" * 70)
    top_genres_overall = df_genres['genre'].value_counts().head(10)
    print("Top 10 Genres Across Entire Library:")
    for g, count in top_genres_overall.items():
        print(f"  - {g}: {count:,} titles ({count/total_titles*100:.2f}% of library)")
        
    print("\nTop 5 Movie Genres:")
    top_movie_genres = df_genres[df_genres['type'] == 'Movie']['genre'].value_counts().head(5)
    for g, count in top_movie_genres.items():
        print(f"  - {g}: {count:,} movies ({count/total_movies*100:.2f}% of all movies)")
        
    print("\nTop 5 TV Show Genres:")
    top_tv_genres = df_genres[df_genres['type'] == 'TV Show']['genre'].value_counts().head(5)
    for g, count in top_tv_genres.items():
        print(f"  - {g}: {count:,} TV shows ({count/total_tv*100:.2f}% of all TV shows)")

    print("\n" + "=" * 70)
    print("4. COUNTRY COMPARISON & DRILL-DOWN")
    print("=" * 70)
    valid_countries = df_countries[df_countries['country_single'] != 'Unknown Country']
    top_10_countries = valid_countries['country_single'].value_counts().head(10)
    print("Top 10 Content-Producing Countries:")
    for c, count in top_10_countries.items():
        print(f"  - {c}: {count:,} titles ({count/total_titles*100:.2f}% of library)")

    print("\nCountry Drill-Down Profiles for Top 3 Countries:")
    for c in ['United States', 'India', 'United Kingdom']:
        c_sub = df[df['country_clean'].str.contains(c, na=False)]
        c_movies = (c_sub['type'] == 'Movie').sum()
        c_tv = (c_sub['type'] == 'TV Show').sum()
        c_tot = len(c_sub)
        print(f"\n[{c}] Total Titles: {c_tot:,} (Movies: {c_movies} [{c_movies/c_tot*100:.1f}%], TV Shows: {c_tv} [{c_tv/c_tot*100:.1f}%])")
        c_genres = df_genres[df_genres['show_id'].isin(c_sub['show_id'])]['genre'].value_counts().head(3)
        print("  Top 3 Genres:")
        for g, gc in c_genres.items():
            print(f"    * {g}: {gc:,} titles ({gc/c_tot*100:.1f}%)")

    print("\n" + "=" * 70)
    print("5. AGE RATINGS & MATURITY SHIFT")
    print("=" * 70)
    ratings = df['rating_imputed'].value_counts()
    for r, count in ratings.items():
        print(f"  - {r}: {count:,} titles ({count/total_titles*100:.2f}%)")
        
    print("\nRating Groups:")
    r_groups = df['rating_group'].value_counts()
    for rg, count in r_groups.items():
        print(f"  - {rg}: {count:,} titles ({count/total_titles*100:.2f}%)")
        
    print("\nMaturity Shift Over Time (2015 vs 2020):")
    early_year = df[df['year_added'] == 2015]['rating_group'].value_counts(normalize=True) * 100
    recent_year = df[df['year_added'] == 2020]['rating_group'].value_counts(normalize=True) * 100
    print("2015 Rating Group Distribution (%):")
    print(early_year.round(2).to_string())
    print("2020 Rating Group Distribution (%):")
    print(recent_year.round(2).to_string())

    print("\n" + "=" * 70)
    print("6. DURATION & SEASON DISTRIBUTIONS")
    print("=" * 70)
    print("Movie Duration Buckets:")
    mov_dur = df[df['type'] == 'Movie']['duration_category'].value_counts()
    for cat, count in mov_dur.items():
        print(f"  - {cat}: {count:,} movies ({count/total_movies*100:.2f}%)")
        
    print("\nTV Show Season Buckets:")
    tv_sea = df[df['type'] == 'TV Show']['season_category'].value_counts()
    for cat, count in tv_sea.items():
        print(f"  - {cat}: {count:,} TV shows ({count/total_tv*100:.2f}%)")
        
    exact_1_season = (df[df['type'] == 'TV Show']['num_seasons'] == 1).sum()
    print(f"\nSingle-season TV shows: {exact_1_season:,} out of {total_tv:,} TV shows ({exact_1_season/total_tv*100:.2f}%)")

if __name__ == "__main__":
    run_deep_analysis()
