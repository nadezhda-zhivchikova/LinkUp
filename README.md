
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

## Заметки
- Это учебный MVP. Для школы в продакшене используйте БД (например, Firestore/Postgres), аутентификацию и модерацию каталога.
