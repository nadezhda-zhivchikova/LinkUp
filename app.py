import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# ============ Storage Setup ============
# На Streamlit Cloud репозиторий read-only. Пишем в временную директорию.
DATA_DIR = Path(tempfile.gettempdir()) / "recsys_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PACKAGE_DATA_DIR = Path(__file__).parent / "data"  # ваши стартовые CSV из репозитория

# ============ Helpers ============

def do_rerun():
    """Совместимый rerun для новых/старых версий Streamlit."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()  # fallback для старых версий

def ensure_data_files():
    """Если в tmp нет CSV — скопировать их из репозитория (или создать пустые)."""
    expected = [
        "items_films.csv", "items_music.csv", "items_books.csv",
        "users.csv", "ratings.csv"
    ]
    for name in expected:
        dst = DATA_DIR / name
        if not dst.exists():
            src = PACKAGE_DATA_DIR / name
            if src.exists():
                shutil.copy(src, dst)
            else:
                # создать минимальную заготовку
                if name.startswith("items_films"):
                    pd.DataFrame(columns=["id","title","year","genres","language","age_rating","type"]).to_csv(dst, index=False)
                elif name.startswith("items_music"):
                    pd.DataFrame(columns=["id","title","artist","year","genres","language","age_rating","type"]).to_csv(dst, index=False)
                elif name.startswith("items_books"):
                    pd.DataFrame(columns=["id","title","author","year","genres","language","age_rating","type"]).to_csv(dst, index=False)
                elif name == "users.csv":
                    pd.DataFrame(columns=[
                        "user_id","name","grade",
                        "favorite_genres_films","favorite_genres_music","favorite_genres_books"
                    ]).to_csv(dst, index=False)
                elif name == "ratings.csv":
                    pd.DataFrame(columns=["user_id","item_id","type","value"]).to_csv(dst, index=False)

@st.cache_data
def load_data():
    ensure_data_files()
    films = pd.read_csv(DATA_DIR / "items_films.csv")
    music = pd.read_csv(DATA_DIR / "items_music.csv")
    books = pd.read_csv(DATA_DIR / "items_books.csv")
    users = pd.read_csv(DATA_DIR / "users.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    return films, music, books, users, ratings

def save_users(df):
    df.to_csv(DATA_DIR / "users.csv", index=False)
    load_data.clear()  # сброс кэша

def save_ratings(df):
    df.to_csv(DATA_DIR / "ratings.csv", index=False)
    load_data.clear()  # сброс кэша

def ensure_user(users_df, user_id, name="", grade=""):
    if (users_df["user_id"] == user_id).any():
        return users_df
    new = pd.DataFrame([{
        "user_id": user_id,
        "name": name or user_id.split("@")[0],
        "grade": grade,
        "favorite_genres_films": "",
        "favorite_genres_music": "",
        "favorite_genres_books": ""
    }])
    users_df = pd.concat([users_df, new], ignore_index=True)
    save_users(users_df)
    return users_df

def parse_genres(s):
    if pd.isna(s) or not str(s).strip():
        return set()
    return set(g.strip() for g in str(s).split(",") if g.strip())

def jaccard(a:set, b:set):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def user_likes_set(ratings_df, user_id, kind):
    return set(ratings_df[(ratings_df.user_id==user_id) & (ratings_df.type==kind) & (ratings_df.value==1)].item_id.values)

def top_neighbors(target_id, ratings_df, users_df, kind, K=10):
    target_likes = user_likes_set(ratings_df, target_id, kind)
    sims = []
    for uid in users_df.user_id.unique():
        if uid == target_id:
            continue
        likes = user_likes_set(ratings_df, uid, kind)
        sims.append((uid, jaccard(target_likes, likes)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return [pair for pair in sims if pair[1] > 0][:K]

def recommend_items(target_id, ratings_df, users_df, items_df, kind, K=10, N=10):
    neighbors = top_neighbors(target_id, ratings_df, users_df, kind, K)
    target_seen = user_likes_set(ratings_df, target_id, kind)
    scores = {}
    for (uid, sim) in neighbors:
        likes = user_likes_set(ratings_df, uid, kind)
        for it in likes:
            if it in target_seen:
                continue
            scores[it] = scores.get(it, 0) + sim
    if scores:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:N]
        ids = [it for it,_ in ranked]
        return items_df[items_df["id"].isin(ids)]

    # Fallback: жанры из профиля
    user_row = users_df[users_df.user_id==target_id]
    if user_row.empty:
        return items_df.sample(min(N, len(items_df)))
    user_row = user_row.iloc[0]
    fav_col = {"film":"favorite_genres_films","music":"favorite_genres_music","book":"favorite_genres_books"}[kind]
    fav_genres = parse_genres(user_row.get(fav_col, ""))

    if "genres" in items_df.columns and len(items_df):
        items_df = items_df.copy()
        items_df["genre_overlap"] = items_df["genres"].fillna("").apply(
            lambda s: len(fav_genres & set(g.strip() for g in str(s).split(",")))
        )
        fallback = items_df[~items_df["id"].isin(list(target_seen))].sort_values("genre_overlap", ascending=False).head(N)
        return fallback
    return items_df.sample(min(N, len(items_df)))

def like_button(ratings_df, user_id, item_id, kind, key):
    liked = ((ratings_df.user_id==user_id) & (ratings_df.item_id==item_id) & (ratings_df.type==kind) & (ratings_df.value==1)).any()
    label = "👍 Liked" if liked else "👍 Like"
    if st.button(label, key=key):
        if liked:
            ratings_df = ratings_df[~((ratings_df.user_id==user_id) & (ratings_df.item_id==item_id) & (ratings_df.type==kind))]
        else:
            new = pd.DataFrame([{"user_id":user_id,"item_id":item_id,"type":kind,"value":1}])
            ratings_df = pd.concat([ratings_df, new], ignore_index=True)
        save_ratings(ratings_df)
        do_rerun()
    return ratings_df

# ============ UI ============

def main():
    st.set_page_config(page_title="School Recommender (Films • Music • Books)", layout="wide")
    st.title("🎒 School Recommender — Films • Music • Books")
    st.caption("MVP: user-based collaborative filtering + genre fallback")

    films, music, books, users, ratings = load_data()

    with st.sidebar:
        st.header("Login")
        email = st.text_input("School email", value=st.session_state.get("user_id", "demo@student.school"))
        name = st.text_input("Name", value=st.session_state.get("name", "Demo Student"))
        grade = st.text_input("Class/Grade", value=st.session_state.get("grade", "9A"))
        if st.button("Sign in / Register"):
            st.session_state["user_id"] = email.strip().lower()
            st.session_state["name"] = name.strip()
            st.session_state["grade"] = grade.strip()
            # создаём пользователя при входе
            users2 = ensure_user(users, st.session_state["user_id"], st.session_state["name"], st.session_state["grade"])
            do_rerun()

        st.markdown("---")
        st.subheader("Data location")
        st.code(str(DATA_DIR), language="bash")

    user_id = st.session_state.get("user_id", None)
    if not user_id:
        st.info("⬅️ Войдите в систему через боковую панель.")
        st.stop()

    # Обновим данные (на случай, если user только что добавлен)
    films, music, books, users, ratings = load_data()

    tabs = st.tabs(["👤 Profile", "🎞️ Films", "🎵 Music", "📚 Books", "✨ Recommendations"])

    # PROFILE
    with tabs[0]:
        st.subheader("Favorite Genres")
        urow = users[users.user_id==user_id]
        if urow.empty:
            users = ensure_user(users, user_id, st.session_state.get("name",""), st.session_state.get("grade",""))
            films, music, books, users, ratings = load_data()
            urow = users[users.user_id==user_id]
        urow = urow.iloc[0].copy()

        film_genres = sorted(list({g for s in films.genres.fillna('') for g in str(s).split(",") if g}))
        music_genres = sorted(list({g for s in music.genres.fillna('') for g in str(s).split(",") if g}))
        book_genres  = sorted(list({g for s in books.genres.fillna('') for g in str(s).split(",") if g}))

        sel_f = st.multiselect("Film genres", film_genres, default=list(parse_genres(urow.get("favorite_genres_films",""))))
        sel_m = st.multiselect("Music genres", music_genres, default=list(parse_genres(urow.get("favorite_genres_music",""))))
        sel_b = st.multiselect("Book genres",  book_genres,  default=list(parse_genres(urow.get("favorite_genres_books",""))))

        if st.button("Save Profile"):
            users.loc[users.user_id==user_id, "favorite_genres_films"] = ",".join(sel_f)
            users.loc[users.user_id==user_id, "favorite_genres_music"] = ",".join(sel_m)
            users.loc[users.user_id==user_id, "favorite_genres_books"] = ",".join(sel_b)
            save_users(users)
            st.success("Saved!")

        st.markdown("> Совет: отметьте минимум 5 любимых тайтлов в каждой категории для лучших рекомендаций.")

    # FILMS
    with tabs[1]:
        st.subheader("Catalog — Films")
        if len(films) == 0:
            st.warning("Каталог фильмов пуст. Добавьте записи в items_films.csv (id,title,year,genres,language,age_rating,type).")
        for _, row in films.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['title']}** ({int(row['year'])}) · *{row['genres']}*")
                ratings = like_button(ratings, user_id, row['id'], "film", key=f"film_{row['id']}")

    # MUSIC
    with tabs[2]:
        st.subheader("Catalog — Music")
        if len(music) == 0:
            st.warning("Каталог музыки пуст. Добавьте записи в items_music.csv.")
        for _, row in music.iterrows():
            with st.container(border=True):
                more = f"{row.get('artist','')} · {int(row['year'])} · *{row['genres']}*"
                st.markdown(f"**{row['title']}** — {more}")
                ratings = like_button(ratings, user_id, row['id'], "music", key=f"music_{row['id']}")

    # BOOKS
    with tabs[3]:
        st.subheader("Catalog — Books")
        if len(books) == 0:
            st.warning("Каталог книг пуст. Добавьте записи в items_books.csv.")
        for _, row in books.iterrows():
            with st.container(border=True):
                who = row.get('author','')
                st.markdown(f"**{row['title']}** — {who} ({int(row['year'])}) · *{row['genres']}*")
                ratings = like_button(ratings, user_id, row['id'], "book", key=f"book_{row['id']}")

    # RECOMMEND
    with tabs[4]:
        st.subheader("Recommendations")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🎞️ Films")
            rec = recommend_items(user_id, ratings, users, films.copy(), "film", K=10, N=10)
            if isinstance(rec, pd.DataFrame) and len(rec):
                for _, row in rec.iterrows():
                    st.write(f"• {row['title']} ({int(row['year'])})")
            else:
                st.caption("Пока нечего рекомендовать — отметьте фильмы лайками.")

        with col2:
            st.markdown("### 🎵 Music")
            rec = recommend_items(user_id, ratings, users, music.copy(), "music", K=10, N=10)
            if isinstance(rec, pd.DataFrame) and len(rec):
                for _, row in rec.iterrows():
                    artist = row.get('artist','')
                    st.write(f"• {row['title']} — {artist}")
            else:
                st.caption("Пока нечего рекомендовать — отметьте треки лайками.")

        with col3:
            st.markdown("### 📚 Books")
            rec = recommend_items(user_id, ratings, users, books.copy(), "book", K=10, N=10)
            if isinstance(rec, pd.DataFrame) and len(rec):
                for _, row in rec.iterrows():
                    who = row.get('author','')
                    st.write(f"• {row['title']} — {who}")
            else:
                st.caption("Пока нечего рекомендовать — отметьте книги лайками.")

    st.caption("Privacy tip: данные пишутся в временную директорию контейнера. Для продакшена используйте БД и авторизацию (например, Firebase).")

if __name__ == "__main__":
    main()
