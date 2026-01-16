from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.db import add_custom_movie_to_db
from app.recommender import reload_system_with_new_movie, movies, enrich
from app.db import init_db, create_user, auth_user, set_rating, get_user_ratings,get_username
from app.db import get_username
from app.recommender import (
    get_popular,
    search_movies,
    get_movie,
    recommend_similar_movies,
    recommend_hybrid,
    recommend_cf,
    get_combined_profile
)

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize DB and system on startup
init_db()
from app.recommender import reload_system_with_new_movie
reload_system_with_new_movie()

@app.get("/", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "msg": ""})

@app.post("/register")
def register(request: Request, username: str = Form(...), password: str = Form(...)):
    ok, msg = create_user(username, password)
    text = "Registered! Now login." if ok else msg
    return templates.TemplateResponse("login.html", {"request": request, "msg": text})

@app.post("/login")
def do_login(request: Request, username: str = Form(...), password: str = Form(...)):
    uid = auth_user(username, password)
    if uid is None:
        return templates.TemplateResponse("login.html", {"request": request, "msg": "Login failed."})
    return RedirectResponse(url=f"/home/{uid}", status_code=302)


@app.get("/home/{user_id}", response_class=HTMLResponse)
def home(request: Request, user_id: int):
    # 1. Fetch ratings from DB
    ur_db = get_user_ratings(user_id)
    # 2. Combine DB ratings with dataset ratings (for Demo Users)
    ur_combined = get_combined_profile(user_id, ur_db)
    # Generate popular recommendations
    popular = get_popular(n=16, user_rating_map=ur_combined)

    # Generate Hybrid recommendations
    hybrid = recommend_hybrid(user_id, ur_combined, n=16)


    latest_df = movies[movies.movieId >= 300000].sort_values("movieId", ascending=False).head(10)


    latest_recs = enrich(latest_df, ur_combined).to_dict("records") if not latest_df.empty else []

    # Display name logic for dataset users
    u_name = get_username(user_id)
    if u_name == "User" and int(user_id) < 138400: u_name = f"Dataset User {user_id}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_id": user_id,
        "username": u_name,
        "popular": popular,
        "hybrid": hybrid,
        "latest": latest_recs,
        "q": "",
        "is_search": False,
    })

@app.get("/search/{user_id}", response_class=HTMLResponse)
def search_page(request: Request, user_id: int, q: str = ""):
    ur_db = get_user_ratings(user_id)
    ur_combined = get_combined_profile(user_id, ur_db)

    results = search_movies(q, n=24, user_rating_map=ur_combined) if q else []

    u_name = get_username(user_id)
    if u_name == "User" and int(user_id) < 138400: u_name = f"Dataset User {user_id}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_id": user_id,
        "username": u_name,
        "popular": results,
        "hybrid": [],
        "q": q,
        "is_search": True,
    })


@app.get("/movie/{user_id}/{movie_id}", response_class=HTMLResponse)
def movie_page(request: Request, user_id: int, movie_id: int):
    ur_db = get_user_ratings(user_id)
    ur_combined = get_combined_profile(user_id, ur_db)

    movie = get_movie(movie_id, user_rating_map=ur_combined)
    recs = recommend_similar_movies(movie_id, n=16, user_rating_map=ur_combined)

    u_name = get_username(user_id)
    if u_name == "User" and int(user_id) < 138400: u_name = f"Dataset User {user_id}"

    return templates.TemplateResponse("movie.html", {
        "request": request,
        "user_id": user_id,
        "username": u_name,
        "movie": movie,
        "recs": recs,
    })


@app.post("/rate/{user_id}/{movie_id}")
def rate_movie(user_id: int, movie_id: int, rating: int = Form(...)):
    rating = max(1, min(5, int(rating)))
    set_rating(user_id, movie_id, rating)

    # If rating a new custom movie (ID 300k+), reload system to update vectors immediately
    if movie_id >= 300000:
        reload_system_with_new_movie()

    return RedirectResponse(url=f"/movie/{user_id}/{movie_id}", status_code=302)

@app.get("/recs/{user_id}/{mode}", response_class=HTMLResponse)
def recs_page(request: Request, user_id: int, mode: str):
    ur_db = get_user_ratings(user_id)
    ur_combined = get_combined_profile(user_id, ur_db)  # <-- Eklendi

    mode = (mode or "").lower().strip()

    u_name = get_username(user_id)
    if u_name == "User" and int(user_id) < 138400: u_name = f"Dataset User {user_id}"

    if mode == "hybrid":
        if not ur_combined:
            recs = get_popular(n=24, user_rating_map=ur_combined)
            title = "Cold Start (Popular Movies)"
        else:
            recs = recommend_hybrid(user_id, ur_combined, n=24)
            title = "Recommended For You (Hybrid)"
    elif mode == "cf":
        recs, title = recommend_cf(user_id, ur_combined, n=24)
    else:
        return RedirectResponse(url=f"/home/{user_id}", status_code=302)

    return templates.TemplateResponse("recs.html", {
        "request": request,
        "user_id": user_id,
        "username": u_name,
        "title": title,
        "recs": recs,
    })



@app.get("/logout")
def logout(request: Request):

    return RedirectResponse(url="/", status_code=302)


@app.get("/profile/{user_id}", response_class=HTMLResponse)
def profile_page(request: Request, user_id: int):

    ur_db = get_user_ratings(user_id)
    ur_combined = get_combined_profile(user_id, ur_db)

    u_name = get_username(user_id)
    if u_name == "User" and int(user_id) < 138400: u_name = f"Dataset User {user_id}"

    rated_movie_ids = list(ur_combined.keys())


    from app.recommender import movies, enrich

    if rated_movie_ids:
        df_rated = movies[movies.movieId.isin(rated_movie_ids)].copy()
        results = enrich(df_rated, ur_combined).to_dict("records")

        results.sort(key=lambda x: x['your_rating'], reverse=True)
    else:
        results = []

    return templates.TemplateResponse("recs.html", {
        "request": request,
        "user_id": user_id,
        "username": u_name,
        "title": f"My Watched & Rated Movies ({len(results)})",
        "recs": results,
    })




@app.get("/admin/{user_id}", response_class=HTMLResponse)
def admin_page(request: Request, user_id: int):
    #Current Username
    current_username = get_username(user_id)

   #adminList
    ALLOWED_ADMINS = ["admin"]


    # 2. 2. Access Control: Block if not in list
    if current_username not in ALLOWED_ADMINS:
        return HTMLResponse(
            f"""
            <div style='display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; background:#111827; color:white; font-family:sans-serif;'>
                <h1 style='color:#ef4444; font-size:50px;'>⛔ ACCESS DENIED</h1>
                <p>Sorry <b>{current_username}</b>, only admin  can use this area.</p>
                <p></p>
                <a href='/home/{user_id}' style='padding:10px 20px; background:#3b82f6; color:white; text-decoration:none; border-radius:5px;'>HomePage</a>
            </div>
            """,
            status_code=403
        )


    return templates.TemplateResponse("admin.html", {"request": request, "user_id": user_id})



@app.on_event("startup")
def startup_event():
    init_db()
    reload_system_with_new_movie()


@app.post("/admin/add_movie")
def add_movie_logic(
        request: Request,
        title: str = Form(...),
        genres: str = Form(...),
        overview: str = Form(...)
):
    # 1. Add movie to DB
    new_id = add_custom_movie_to_db(title, genres, overview)

    # 2. Reload system
    reload_system_with_new_movie()

    success_msg = f"Movie Added Successfully! ID: {new_id}. '{title}' is now live."

    referer = request.headers.get("Referer")
    if referer:
        return RedirectResponse(url=referer + "?msg=success", status_code=303)

    return RedirectResponse(url="/", status_code=303)



