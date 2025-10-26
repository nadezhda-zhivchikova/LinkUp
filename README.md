
# School Recommender (Streamlit MVP)

## Как запустить
1) Установите зависимости:
```bash
pip install streamlit pandas numpy
```
2) Запустите локально:
```bash
streamlit run app.py
```
3) Файлы данных лежат в `data/`:
- `items_films.csv`, `items_music.csv`, `items_books.csv`
- `users.csv`
- `ratings.csv`

## Как пользоваться
- Введите школьный email слева и нажмите **Sign in / Register**.
- Во вкладке **Profile** отметьте любимые жанры.
- В разделах **Films / Music / Books** ставьте **Like** на тайтлы.
- Во вкладке **Recommendations** появятся рекомендации (user-based CF, fallback на жанры).

## Заметки
- Это учебный MVP. Для школы в продакшене используйте БД (например, Firestore/Postgres), аутентификацию и модерацию каталога.
