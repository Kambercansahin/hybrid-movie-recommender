from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from app.db import get_custom_movies_df
import os
import joblib
import numpy as np
import pandas as pd
import requests
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

BASE = "saved_models/"


# TMDB

TMDB_API_KEY = "API_KEY"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

# -------------------------
# CSV
# -------------------------
movies = pd.read_csv("movies.csv")    # movieId,title,genres
ratings = pd.read_csv("ratings.csv")  # userId,movieId,rating,timestamp
links = pd.read_csv("links.csv")      # movieId,imdbId,tmdbId

# global stats
movie_mean = ratings.groupby("movieId")["rating"].mean()
movie_count = ratings.groupby("movieId")["rating"].size()

# -------------------------
# PKL MODELS
# -------------------------
svd_model = joblib.load(os.path.join(BASE, "svd_cf_model.pkl"))
xgb_model = joblib.load(os.path.join(BASE, "xgb_hybrid_model.pkl"))
movie_to_index = joblib.load(os.path.join(BASE, "movie_to_index.pkl"))

# TF-IDF matrix from training
# Must match the training vector space
tfidf_matrix = joblib.load(os.path.join(BASE, "tfidf_matrix.pkl"))

# Optional: Pre-computed user profiles
try:
    user_profiles_scb = joblib.load(os.path.join(BASE, "user_profiles_scb.pkl"))
except Exception:
    user_profiles_scb = None



# Posters

@lru_cache(maxsize=200)
def fetch_poster_by_title(title: str):
    """Finds poster via TMDB Search API by title.."""
    if not TMDB_API_KEY: return None

    import urllib.parse
    safe_title = urllib.parse.quote(title)

    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={safe_title}"
    try:
        r = requests.get(search_url, timeout=5)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:

                path = results[0].get("poster_path")
                if path:
                    return f"https://image.tmdb.org/t/p/w500{path}"
    except Exception as e:
        print(f"Poster fetch error: {e}")
    return None


@lru_cache(maxsize=4096)
def get_poster(movie_id: int):
    mid = int(movie_id)

    # Step 1: Check links.csv for old movies
    row = links[links.movieId == mid]
    if not row.empty:
        tmdb_id = row.tmdbId.values[0]
        if not pd.isna(tmdb_id):
            # Fetch directly if TMDB ID exists
            url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
            try:
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    path = r.json().get("poster_path")
                    if path: return f"https://image.tmdb.org/t/p/w500{path}"
            except:
                pass

    # Step 2: Search by title for new movies (300k+) or missing IDs
    movie_row = movies[movies.movieId == mid]
    if not movie_row.empty:
        title = movie_row.iloc[0]['title']

        clean_title = title.split('(')[0].strip()
        return fetch_poster_by_title(clean_title)

    return None


def enrich(df: pd.DataFrame, user_rating_map: dict[int, int] | None):
    df = df.copy()
    df["poster"] = df["movieId"].apply(get_poster)
    df["avg_rating"] = df["movieId"].apply(lambda x: round(float(movie_mean.get(int(x), 0.0)), 2))
    df["rating_count"] = df["movieId"].apply(lambda x: int(movie_count.get(int(x), 0)))

    if user_rating_map is None:
        df["your_rating"] = None
    else:
        df["your_rating"] = df["movieId"].apply(lambda x: user_rating_map.get(int(x)))
    return df


def get_movie(movie_id: int, user_rating_map: dict[int, int] | None):
    mid = int(movie_id)
    row = movies[movies.movieId == mid]
    if row.empty:
        return None

    # Basic data (like Poster) comes from enrich
    movie_data = enrich(row, user_rating_map).iloc[0].to_dict()

    # Overview
    overview = "No description available."

    #Rule 1: Check DB for admin-provided overview
    db_overview = row.iloc[0].get('overview')
    if pd.notna(db_overview) and len(str(db_overview).strip()) > 10:
        overview = db_overview
    else:
        # Rule 2: Fetch from TMDB if new movie (ID >= 300k) or no admin text
        if mid >= 300000:
            title = movie_data.get('title')
            search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
            try:
                r = requests.get(search_url, timeout=5)
                if r.status_code == 200:
                    results = r.json().get("results", [])
                    if results:
                        overview = results[0].get("overview", overview)
            except:
                pass
        else:
            #Rule 3: Fetch via TMDB ID for old movies
            link_row = links[links.movieId == mid]
            if not link_row.empty:
                tmdb = link_row.tmdbId.values[0]
                if not pd.isna(tmdb):
                    try:
                        url = f"https://api.themoviedb.org/3/movie/{int(tmdb)}?api_key={TMDB_API_KEY}&language=en-US"
                        r = requests.get(url, timeout=2)
                        if r.status_code == 200:
                            overview = r.json().get("overview", overview)
                    except:
                        pass

    movie_data["overview"] = overview
    return movie_data


def get_popular(n=16, user_rating_map=None):
    top = movie_count.sort_values(ascending=False).head(n).index
    df = movies[movies.movieId.isin(top)]
    return enrich(df, user_rating_map).to_dict("records")


def search_movies(q: str, n=24, user_rating_map=None):
    q = (q or "").strip()
    if not q:
        return []
    df = movies[movies.title.str.contains(q, case=False, na=False)].head(n)
    return enrich(df, user_rating_map).to_dict("records")


def recommend_similar_movies(movie_id: int, n=16, user_rating_map=None):

    movie_id = int(movie_id)

    if movie_id not in movie_to_index:
        return []

    idx = int(movie_to_index[movie_id])

    cosine_sim_vector = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get all similarity scores (index, score)
    sims = list(enumerate(cosine_mm[idx]))

    # Sort descending
    sims.sort(key=lambda x: x[1], reverse=True)

    # Index to MovieID mapping
    index_to_movie = {v: k for k, v in movie_to_index.items()}

    ids = []

    for i in sim_indices:
        if i == idx: continue

        rec_id = index_to_movie.get(i)
        if rec_id is None: continue
        rec_id = int(rec_id)


        if movie_count.get(rec_id, 0) < 20: continue

        ids.append(rec_id)
        if len(ids) >= n: break

    df = movies[movies.movieId.isin(ids)]
    if not df.empty:
        df = df.set_index('movieId').loc[ids].reset_index()

    return enrich(df, user_rating_map).to_dict("records")


# SCB (Content-based using TF-IDF user profile cosine)
def build_user_profile_from_ratings(user_ratings_map: dict[int, int]) -> np.ndarray:

    feat_dim = tfidf_matrix.shape[1]
    user_vec = np.zeros(feat_dim, dtype=np.float32)

    if not user_ratings_map:
        return user_vec

    #3.5 is neutral
    u_mean = 3.5

    for mid, r in user_ratings_map.items():
        mid = int(mid)
        if mid not in movie_to_index:
            continue
        idx = int(movie_to_index[mid])

        # Weight: +1.5 for 5 stars, -2.5 for 1 star.
        diff = float(r) - u_mean
        if diff == 0: diff = 0.1 # Nötr durumu hafif pozitif yap

        # normalize movie vector
        mv_vec = tfidf_matrix[idx].toarray().ravel().astype(np.float32)
        mv_norm = float(np.linalg.norm(mv_vec))
        if mv_norm > 0:
            mv_vec = mv_vec / mv_norm

        # Weighted addition to profile vector
        user_vec += (mv_vec * diff)

    # Normalize final user vector
    norm = float(np.linalg.norm(user_vec))
    if norm > 0:
        user_vec = user_vec / norm

    return user_vec


def recommend_scb(user_id: int, user_ratings_map: dict[int, int], n=16):
    """
    [Cold Start] Diversity + Weighted logic
    """
    if not user_ratings_map: return []

    # Reverse mapping (Index -> MovieID)
    index_to_movie = {v: k for k, v in movie_to_index.items()}

    # 1. Separate Liked vs Disliked
    liked_items = sorted(user_ratings_map.items(), key=lambda x: x[1], reverse=True)
    liked_movies = [int(mid) for mid, r in liked_items if r >= 4]
    disliked_movies = [int(mid) for mid, r in user_ratings_map.items() if r <= 2]

    # If no high ratings (4+), take the most recently rated ones as fallback
    if not liked_movies:
        liked_movies = [int(mid) for mid, _ in liked_items[:5]]

    # 2. Blacklist (
    # OPTIMIZATION: Using linear_kernel instead of pre-computed cosine_mm
    black_list = set()
    for d_mid in disliked_movies:
        if d_mid in movie_to_index:
            idx = int(movie_to_index[d_mid])

            # ON-THE-FLY CALCULATION
            # Calculates similarity only for this specific movie against all others
            sim_vec = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

            # Find movies with similarity > 0.80
            # We check only the top 20 most similar to save time
            top_indices = sim_vec.argsort()[-20:][::-1]

            for t_idx in top_indices:
                if sim_vec[t_idx] > 0.80:
                    target_id = index_to_movie.get(t_idx)
                    if target_id: black_list.add(target_id)

    # 3. Candidate Pool & Scoring
    candidate_scores = {}

    # Performance: Look at only the last 20 liked movies (Checking all would be too slow)
    recent_liked_movies = liked_movies[:20]

    for mid in recent_liked_movies:
        if mid not in movie_to_index: continue
        idx = int(movie_to_index[mid])

        user_rating = user_ratings_map.get(mid, 4)
        weight = (user_rating - 3.5) * 2.0

        # OPTIMIZATION: Calculate similarity vector for this movie only
        sim_vec = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

        # Get top 40 similar movies
        top_indices = sim_vec.argsort()[-41:][::-1]

        for target_idx in top_indices:
            if target_idx == idx: continue  # Skip self

            score = sim_vec[target_idx]
            target_mid = index_to_movie.get(target_idx)

            # Filters: Blacklist, Already rated, Low global vote count
            if not target_mid or target_mid in user_ratings_map or target_mid in black_list: continue
            if movie_count.get(target_mid, 0) < 20: continue

            # Accumulate weighted score
            candidate_scores[target_mid] = candidate_scores.get(target_mid, 0) + (score * weight)

    # 4. Diversity Logic
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

    genres_seen = {}
    final_ids = []
    genre_limit = 4  # Max 4 movies per primary genre

    for mid, score in sorted_candidates:
        # Safely fetch genre info
        movie_row = movies[movies.movieId == mid]
        if movie_row.empty: continue

        m_genres = movie_row['genres'].values[0].split('|')
        primary_genre = m_genres[0]

        if genres_seen.get(primary_genre, 0) < genre_limit:
            final_ids.append(mid)
            genres_seen[primary_genre] = genres_seen.get(primary_genre, 0) + 1

        if len(final_ids) >= n: break

    # If diversity filtering reduced the list too much, fill the rest from top candidates
    if len(final_ids) < n:
        for mid, _ in sorted_candidates:
            if mid not in final_ids:
                final_ids.append(mid)
            if len(final_ids) >= n: break

    # 5. Return Results
    df_recs = movies[movies.movieId.isin(final_ids)]
    results = enrich(df_recs, user_ratings_map).to_dict("records")

    # Sort by score (enriching might break the order)
    results.sort(key=lambda x: candidate_scores.get(x['movieId'], 0), reverse=True)

    return results



# CF Score for Hybrid

def cf_score_svd(user_id: int, movie_id: int) -> float:
    try:
        return float(svd_model.predict(int(user_id), int(movie_id)).est)
    except Exception:
        return 0.0


def cf_score_user_user_online(user_id: int, movie_id: int) -> float:
    """
    Predict CF score for (user, movie) using online user-user cosine
    Based on WEB DB ratings only.
    """
    from app.db import get_all_web_ratings_df

    df = get_all_web_ratings_df()
    if df.empty:
        return 0.0

    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df["rating"] = df["rating"].astype(float)

    if int(user_id) not in df["user_id"].values:
        return 0.0

    pivot = df.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0.0)
    if int(user_id) not in pivot.index:
        return 0.0

    # If movie not in pivot columns, can't predict from neighbors
    if int(movie_id) not in pivot.columns:
        return 0.0

    target = pivot.loc[int(user_id)].values.reshape(1, -1)
    sims = cosine_similarity(target, pivot.values)[0]
    users = pivot.index.tolist()

    # neighbors excluding itself
    neigh = [(u, float(s)) for u, s in zip(users, sims) if u != int(user_id)]
    neigh.sort(key=lambda x: x[1], reverse=True)
    neigh = [(u, s) for u, s in neigh if s > 0][:10]
    if not neigh:
        return 0.0

    col_idx = list(pivot.columns).index(int(movie_id))
    num = 0.0
    den = 0.0
    for u, s in neigh:
        r = float(pivot.loc[int(u)].values[col_idx])
        if r <= 0:
            continue
        num += s * r
        den += s

    if den == 0:
        return 0.0
    return float(num / den)


def cf_score(user_id: int, movie_id: int, user_ratings_map: dict[int, int]) -> float:
    """
    [Weighted Hybrid CF Score]
    Blends Online User-User (DB) and Global SVD (Dataset).
    """
    global_mean = 3.5

    # 1. SVD prediction (Base)
    svd_val = cf_score_svd(user_id, movie_id)
    if svd_val <= 0: svd_val = global_mean

    # 2. Online User-User score
    web_val = cf_score_user_user_online(user_id, movie_id)

    # If online similarity exists
    # Weighted Avg: 40% Online, 60% Global.
    if web_val > 0:

        return (web_val * 0.4) + (svd_val * 0.6)

    #Fallback to SVD
    return svd_val


# HYBRID (XGBoost)
def user_avg_rating(user_ratings_map: dict[int, int]) -> float:
    if not user_ratings_map:
        return 0.0
    return float(sum(user_ratings_map.values()) / len(user_ratings_map))


def scb_score_for_candidate(user_id: int, user_ratings_map: dict[int, int], candidate_movie_id: int) -> float:
    """Calculates SCB score for XGBoost feature."""
    if not user_ratings_map: return 0.0
    cand_mid = int(candidate_movie_id)
    if cand_mid not in movie_to_index: return 0.0

    # 1. User Vector
    user_vec = build_user_profile_from_ratings(user_ratings_map)
    if float(np.linalg.norm(user_vec)) == 0.0: return 0.0

    # 2. Movie Vector
    j = int(movie_to_index[cand_mid])
    movie_vec = tfidf_matrix[j].toarray().ravel().astype(np.float32)

    # Normalize
    mnorm = float(np.linalg.norm(movie_vec))
    if mnorm > 0:
        movie_vec = movie_vec / mnorm
    else:
        return 0.0

    return float(np.dot(user_vec, movie_vec))


def recommend_hybrid(user_id: int, user_ratings_map: dict[int, int], n=16):

    cnt = len(user_ratings_map)

    # 1. No ratings
    if cnt == 0:
        return []

    # 2. <10 ratings, use SCB
    if cnt < 10:
        return recommend_scb(user_id, user_ratings_map, n=n)

    # 3. 10+ ratings, use Hybrid XGBoost
    rated = set(int(m) for m in user_ratings_map.keys())
    uavg = user_avg_rating(user_ratings_map)

    # Filter: Min 25 ratings, unrated by user
    potential_candidates = [mid for mid, count in movie_count.items() if count >= 25 and mid not in rated]

    # dynamic result
    np.random.shuffle(potential_candidates)

    rows = []
    checked_count = 0
    for mid in potential_candidates:
        # Calculate Features
        scb = scb_score_for_candidate(user_id, user_ratings_map, mid)
        mavg = float(movie_mean.get(mid, 0.0))

        # Quality Threshold: SCB > 0.15 or High Avg Rating
        if scb < 0.15 and mavg < 3.0:
            continue

        cfv = cf_score(user_id, mid, user_ratings_map)
        rows.append([scb, cfv, uavg, mavg, mid])

        checked_count += 1
        # Check max 2000 candidates
        if checked_count > 2000:
            break

    if not rows:
        return []

    # Predict
    df_feat = pd.DataFrame(rows, columns=["SCB_score", "CF_score", "user_avg_rating", "movie_avg_rating", "movieId"])
    X = df_feat[["SCB_score", "CF_score", "user_avg_rating", "movie_avg_rating"]]
    df_feat["hybrid_score"] = xgb_model.predict(X)


    all_candidates = df_feat.sort_values("hybrid_score", ascending=False)

    genres_seen = {}
    final_selected_rows = []
    genre_limit = 4
    for _, row in all_candidates.iterrows():
        mid = int(row['movieId'])

        m_genres = movies[movies.movieId == mid]['genres'].values[0].split('|')
        primary_genre = m_genres[0]


        if genres_seen.get(primary_genre, 0) < genre_limit:
            final_selected_rows.append(row)
            genres_seen[primary_genre] = genres_seen.get(primary_genre, 0) + 1

        if len(final_selected_rows) >= n:
            break


    if len(final_selected_rows) < n:
        already_ids = [r['movieId'] for r in final_selected_rows]
        remaining = all_candidates[~all_candidates['movieId'].isin(already_ids)].head(n - len(final_selected_rows))
        for _, r in remaining.iterrows():
            final_selected_rows.append(r)

    # final table
    top = pd.DataFrame(final_selected_rows)

    # Reasoning labels for UI
    def get_reason(row):
        if row['SCB_score'] > 0.45 and row['CF_score'] > 4.2: return "🎯 Super Match"
        if row['SCB_score'] > 0.35: return "🎬 Content Match"
        if row['CF_score'] > 4.0: return "👥 Popular Choice"
        return "⭐ Highly Recommended"

    top = top.copy()
    top['reason_text'] = top.apply(get_reason, axis=1)

    reason_map = top.set_index('movieId')['reason_text'].to_dict()
    score_map = top.set_index('movieId')['hybrid_score'].to_dict()
    ids = top["movieId"].tolist()

    dfm = movies[movies.movieId.isin(ids)]
    results = enrich(dfm, user_ratings_map).to_dict("records")

    for item in results:
        item['reason'] = reason_map.get(item['movieId'], "")
        item['h_score'] = score_map.get(item['movieId'], 0)

    results.sort(key=lambda x: x['h_score'], reverse=True)
    return results


# CF Page Logic
def recommend_cf_user_user(user_id: int, user_ratings_map: dict[int, int], n=24):
    from app.db import get_all_web_ratings_df

    df = get_all_web_ratings_df()
    if df.empty:
        return []

    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df["rating"] = df["rating"].astype(float)

    if int(user_id) not in df["user_id"].values:
        return []

    pivot = df.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0.0)
    if int(user_id) not in pivot.index:
        return []

    target_vec = pivot.loc[int(user_id)].values.reshape(1, -1)
    sims = cosine_similarity(target_vec, pivot.values)[0]
    sim_users = list(zip(pivot.index.tolist(), sims.tolist()))
    sim_users = [(u, s) for u, s in sim_users if u != int(user_id)]
    sim_users.sort(key=lambda x: x[1], reverse=True)
    sim_users = [(u, s) for u, s in sim_users if s > 0][:10]
    if not sim_users:
        return []

    top_users = [int(u) for u, _ in sim_users[:5]]
    df_sim = df[df["user_id"].isin(top_users)]

    rated = set(int(mid) for mid in user_ratings_map.keys())
    top = (
        df_sim.groupby("movie_id")["rating"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    top = [int(mid) for mid in top if int(mid) not in rated][:n]
    if not top:
        return []

    dfm = movies[movies.movieId.isin(top)]
    return enrich(dfm, user_ratings_map).to_dict("records")


def recommend_cf_svd(user_id: int, user_ratings_map: dict[int, int], n=24):
    rated = set(int(mid) for mid in user_ratings_map.keys())
    preds = []
    for mid in movies.movieId.tolist():
        mid = int(mid)
        if mid in rated:
            continue
        est = cf_score_svd(user_id, mid)
        preds.append((mid, est))
    preds.sort(key=lambda x: x[1], reverse=True)
    ids = [mid for mid, _ in preds[:n]]
    dfm = movies[movies.movieId.isin(ids)]
    return enrich(dfm, user_ratings_map).to_dict("records")


def recommend_cf(user_id: int, user_ratings_map: dict[int, int], n=24):
    if not user_ratings_map or len(user_ratings_map) < 10:
        recs = recommend_cf_user_user(user_id, user_ratings_map, n=n)
        if recs:
            return recs, "CF (User-User Cosine) - Online"
        return get_popular(n=n, user_rating_map=user_ratings_map), "CF (Cold Start Popular)"
    return recommend_cf_svd(user_id, user_ratings_map, n=n), "CF (SVD)"


def get_combined_profile(user_id: int, user_ratings_map_from_db: dict[int, int]) -> dict[int, int]:
    """
    Merges Web (DB) and Dataset (CSV) ratings.
    Skip CSV for new users (ID > 200k).
    """
    final_ratings = user_ratings_map_from_db.copy()

    if int(user_id) >= 200000:
        return final_ratings

    user_csv_data = ratings[ratings.userId == int(user_id)]
    if not user_csv_data.empty:
        for _, row in user_csv_data.iterrows():
            mid = int(row['movieId'])
            if mid not in final_ratings:
                final_ratings[mid] = int(row['rating'])
    return final_ratings


#Reload Logic for Custom Movies

def reload_system_with_new_movie():
    global movies, tfidf_matrix, movie_to_index, movie_mean, movie_count

    df_csv = pd.read_csv("movies.csv")
    df_custom = get_custom_movies_df()
    if df_custom.empty: return

    movies_combined = pd.concat([df_csv, df_custom], ignore_index=True)

    # 1. Update TF-IDF and Indices
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

    soup = (movies_combined['title'] + " " +
            movies_combined['genres'].str.replace('|', ' ') + " " +
            movies_combined.get('overview', '').fillna(''))

    tfidf_matrix = tfidf.fit_transform(soup.str.lower())
    movie_to_index = {mid: i for i, mid in enumerate(movies_combined.movieId)}
    movies = movies_combined

    # 2. Recompute similarity matrix immediately.
    from sklearn.metrics.pairwise import cosine_similarity
    _cosine_movie_movie = cosine_similarity(tfidf_matrix)

