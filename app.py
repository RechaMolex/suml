import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import re
from datetime import datetime
import pickle
import os


class BookRecommender:
    def __init__(self):
        self.df = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.svd = None

    def preprocess_data(self, df):
        """Preprocessuj dane - parsuj daty, usuń duplikaty, standaryzuj język"""
        st.info("Przetwarzanie danych...")

        # Usuń duplikaty
        df = df.drop_duplicates(subset=['title', 'author'], keep='first')

        # Parsuj daty
        df['year'] = df['year'].astype(str)
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)

        # Filtruj realistyczne lata
        df = df[(df['year'] >= 1000) & (df['year'] <= datetime.now().year)]

        # Standaryzuj opisy do angielskiego
        df['description'] = df['description'].fillna('')
        df['description'] = df['description'].astype(str)

        # Usuń nieanglojęzyczne znaki z opisów
        df['description'] = df['description'].apply(self.clean_text)

        # Usuń rekordy z pustymi opisami
        df = df[df['description'].str.len() > 20]

        # Standaryzuj kolumny
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce')

        return df.reset_index(drop=True)

    def clean_text(self, text):
        """Oczyść tekst, pozostaw tylko angielskie znaki"""
        text = re.sub(r'[^\x00-\x7F]+', ' ', str(text))
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def prepare_features(self, df):
        """Przygotuj macierz cech dla rekomendacji"""
        # Połącz wszystkie tekstowe cechy
        df['combined_features'] = (
                df['title'].fillna('') + ' ' +
                df['author'].fillna('') + ' ' +
                df['description'].fillna('') + ' ' +
                df['genres'].fillna('')
        )

        # TF-IDF wektoryzacja
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(df['combined_features'])

        # Redukcja wymiarowości
        self.svd = TruncatedSVD(n_components=100, random_state=42)
        self.tfidf_matrix = self.svd.fit_transform(self.tfidf_matrix)

        return df

    def get_recommendations(self, liked_books, genre_pref=None, year_range=None,
                            rating_min=None, num_recs=10):
        """Generuj rekomendacje na podstawie preferencji"""
        if self.df is None or self.tfidf_matrix is None:
            return []

        # Znajdź indeksy polubionych książek
        liked_indices = []
        for book in liked_books:
            matches = self.df[self.df['title'].str.contains(book, case=False, na=False)]
            if not matches.empty:
                liked_indices.append(matches.index[0])

        if not liked_indices:
            return self.get_popular_books(num_recs)

        # Oblicz średni wektor preferencji
        user_profile = np.mean(self.tfidf_matrix[liked_indices], axis=0)

        # Oblicz podobieństwo cosinusowe
        similarities = cosine_similarity([user_profile], self.tfidf_matrix)[0]

        # Filtruj według preferencji
        filtered_df = self.df.copy()

        if genre_pref:
            filtered_df = filtered_df[
                filtered_df['genres'].str.contains(genre_pref, case=False, na=False)
            ]

        if year_range:
            filtered_df = filtered_df[
                (filtered_df['year'] >= year_range[0]) &
                (filtered_df['year'] <= year_range[1])
                ]

        if rating_min:
            filtered_df = filtered_df[filtered_df['rating'] >= rating_min]

        # Usuń już polubione książki
        filtered_df = filtered_df[~filtered_df.index.isin(liked_indices)]

        if filtered_df.empty:
            return self.get_popular_books(num_recs)

        # Sortuj według podobieństwa
        filtered_indices = filtered_df.index.tolist()
        filtered_similarities = [(i, similarities[i]) for i in filtered_indices]
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)

        # Zwróć najlepsze rekomendacje
        recommendations = []
        for idx, sim in filtered_similarities[:num_recs]:
            book = self.df.iloc[idx]
            recommendations.append({
                'title': book['title'],
                'author': book['author'],
                'year': book['year'],
                'rating': book['rating'],
                'genres': book['genres'],
                'similarity': sim
            })

        return recommendations

    def get_popular_books(self, num_recs=10):
        """Zwróć popularne książki jako fallback"""
        popular = self.df.nlargest(num_recs, 'rating')
        return [
            {
                'title': row['title'],
                'author': row['author'],
                'year': row['year'],
                'rating': row['rating'],
                'genres': row['genres'],
                'similarity': 0.0
            }
            for _, row in popular.iterrows()
        ]


def load_data():
    """Załaduj dane z pliku"""
    try:
        # Sprawdź czy istnieje przetworzony plik
        if os.path.exists('processed_books.pkl'):
            with open('processed_books.pkl', 'rb') as f:
                return pickle.load(f)

        # Załaduj surowe dane
        if os.path.exists('books.csv'):
            df = pd.read_csv('books.csv')
        else:
            # Przykładowe dane jeśli nie ma pliku
            df = pd.DataFrame({
                'title': ['1984', 'To Kill a Mockingbird', 'Pride and Prejudice'],
                'author': ['George Orwell', 'Harper Lee', 'Jane Austen'],
                'year': [1949, 1960, 1813],
                'rating': [4.2, 4.3, 4.1],
                'num_pages': [328, 376, 432],
                'genres': ['Dystopian Fiction', 'Classic Literature', 'Romance'],
                'description': ['A dystopian novel about totalitarian control',
                                'A story about racial injustice in the American South',
                                'A romantic novel about manners and marriage']
            })

        return df
    except Exception as e:
        st.error(f"Błąd wczytywania danych: {e}")
        return None


def main():
    st.set_page_config(
        page_title="Rekomendacje Książek",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 System Rekomendacji Książek")
    st.markdown("Znajdź nowe książki na podstawie swoich preferencji!")

    # Inicjalizacja sesji
    if 'recommender' not in st.session_state:
        st.session_state.recommender = BookRecommender()
        st.session_state.data_loaded = False

    # Wczytaj dane
    if not st.session_state.data_loaded:
        with st.spinner("Wczytywanie i przetwarzanie danych..."):
            df = load_data()
            if df is not None:
                df_processed = st.session_state.recommender.preprocess_data(df)
                st.session_state.recommender.df = st.session_state.recommender.prepare_features(df_processed)
                st.session_state.data_loaded = True
                st.success(f"Załadowano {len(df_processed)} książek!")

    if not st.session_state.data_loaded:
        st.error("Nie udało się wczytać danych")
        return

    # Sidebar z preferencjami
    st.sidebar.header("Twoje preferencje")

    # Polubione książki
    liked_books = st.sidebar.text_area(
        "Książki które Ci się podobały (po jednej w linii):",
        placeholder="Wpisz tytuły książek..."
    ).split('\n')
    liked_books = [book.strip() for book in liked_books if book.strip()]

    # Preferencje gatunku
    genres = st.session_state.recommender.df['genres'].str.split(',').explode().str.strip().unique()
    genres = [g for g in genres if pd.notna(g)]
    genre_pref = st.sidebar.selectbox("Preferowany gatunek:", ['Wszystkie'] + sorted(genres))
    if genre_pref == 'Wszystkie':
        genre_pref = None

    # Zakres lat
    min_year = int(st.session_state.recommender.df['year'].min())
    max_year = int(st.session_state.recommender.df['year'].max())
    year_range = st.sidebar.slider(
        "Zakres lat publikacji:",
        min_year, max_year,
        (min_year, max_year)
    )

    # Minimalna ocena
    rating_min = st.sidebar.slider("Minimalna ocena:", 0.0, 5.0, 0.0, 0.1)

    # Liczba rekomendacji
    num_recs = st.sidebar.slider("Liczba rekomendacji:", 5, 20, 10)

    # Generuj rekomendacje
    if st.sidebar.button("Generuj rekomendacje", type="primary"):
        with st.spinner("Generowanie rekomendacji..."):
            recommendations = st.session_state.recommender.get_recommendations(
                liked_books=liked_books,
                genre_pref=genre_pref,
                year_range=year_range,
                rating_min=rating_min,
                num_recs=num_recs
            )

            st.header("📖 Twoje rekomendacje:")

            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {rec['title']} - {rec['author']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Rok:** {rec['year']}")
                            st.write(f"**Ocena:** {rec['rating']:.1f}/5")
                        with col2:
                            st.write(f"**Gatunek:** {rec['genres']}")
                            if rec['similarity'] > 0:
                                st.write(f"**Podobieństwo:** {rec['similarity']:.2f}")
            else:
                st.warning("Nie znaleziono rekomendacji dla podanych kryteriów.")

    # Statystyki
    with st.expander("📊 Statystyki datasetu"):
        df = st.session_state.recommender.df
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Liczba książek", len(df))
        with col2:
            st.metric("Średnia ocena", f"{df['rating'].mean():.2f}")
        with col3:
            st.metric("Zakres lat", f"{df['year'].min()}-{df['year'].max()}")


if __name__ == "__main__":
    main()