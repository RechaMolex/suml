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

        # Wyświetl dostępne kolumny dla debugowania
        st.info(f"Dostępne kolumny: {list(df.columns)}")

        # Mapowanie kolumn dla Best Books Ever Dataset
        column_mapping = {
            'year': ['firstPublishDate', 'publishDate'],
            'title': ['title'],
            'author': ['author'],
            'rating': ['rating'],
            'description': ['description'],
            'genres': ['genres'],
            'num_pages': ['pages'],
            'language': ['language'],
            'series': ['series'],
            'characters': ['characters'],
            'publisher': ['publisher'],
            'awards': ['awards'],
            'numRatings': ['numRatings'],
            'likedPercent': ['likedPercent'],
            'bbeScore': ['bbeScore'],
            'setting': ['setting']
        }

        # Znajdź i zmapuj kolumny
        for standard_name, possible_names in column_mapping.items():
            found_column = None
            for possible_name in possible_names:
                if possible_name in df.columns:
                    found_column = possible_name
                    break

            if found_column and found_column != standard_name:
                df[standard_name] = df[found_column]

        # Ustaw domyślne wartości dla brakujących kolumn
        required_columns = ['title', 'author', 'year', 'rating', 'description', 'genres']
        for col in required_columns:
            if col not in df.columns:
                if col == 'year':
                    df[col] = 2000
                elif col == 'rating':
                    df[col] = 3.5
                else:
                    df[col] = 'Unknown'

        # Parsuj daty publikacji
        if 'year' in df.columns:
            # Spróbuj najpierw firstPublishDate, potem publishDate
            if 'firstPublishDate' in df.columns:
                df['year'] = pd.to_datetime(df['firstPublishDate'], errors='coerce').dt.year
            elif 'publishDate' in df.columns:
                df['year'] = pd.to_datetime(df['publishDate'], errors='coerce').dt.year

            # Wypełnij NaN i filtruj
            df['year'] = df['year'].fillna(2000).astype(int)
            df = df[(df['year'] >= 1000) & (df['year'] <= datetime.now().year)]

        # Filtruj tylko angielskie książki (jeśli kolumna language istnieje)
        if 'language' in df.columns:
            df = df[df['language'].str.contains('en|english', case=False, na=False) | df['language'].isna()]

        # Usuń duplikaty
        subset_cols = [col for col in ['title', 'author'] if col in df.columns]
        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols, keep='first')

        # Standaryzuj opisy (usuń tylko bardzo krótkie)
        if 'description' in df.columns:
            df['description'] = df['description'].fillna('').astype(str)
            df['description'] = df['description'].apply(self.clean_text)
            # Zachowaj książki nawet z krótkimi opisami
            df = df[df['description'].str.len() > 5]

        # Standaryzuj inne kolumny numeryczne
        numeric_columns = ['rating', 'pages', 'numRatings', 'likedPercent', 'bbeScore']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col == 'rating':
                    df[col] = df[col].fillna(3.5)
                elif col in ['pages']:
                    df[col] = df[col].fillna(300)
                else:
                    df[col] = df[col].fillna(0)

        # Standaryzuj kolumny tekstowe
        text_columns = ['title', 'author', 'genres', 'series', 'characters', 'publisher', 'setting']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str)

        st.success(f"Przetworzono {len(df)} książek")
        return df.reset_index(drop=True)

    def clean_text(self, text):
        """Oczyść tekst, pozostaw tylko angielskie znaki"""
        text = re.sub(r'[^\x00-\x7F]+', ' ', str(text))
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def prepare_features(self, df):
        """Przygotuj macierz cech dla rekomendacji"""
        # Użyj wszystkich dostępnych kolumn tekstowych z datasetu
        feature_columns = ['title', 'author', 'description', 'genres', 'series', 'characters', 'setting']
        feature_parts = []

        for col in feature_columns:
            if col in df.columns:
                feature_parts.append(df[col].fillna('').astype(str))

        if feature_parts:
            df['combined_features'] = pd.concat(feature_parts, axis=1).apply(lambda x: ' '.join(x), axis=1)
        else:
            df['combined_features'] = df['title'].fillna('').astype(str)

        # TF-IDF wektoryzacja z większą liczbą cech
        self.vectorizer = TfidfVectorizer(
            max_features=8000,  # Zwiększone dla bogatszego datasetu
            stop_words='english',
            ngram_range=(1, 3),  # Trigramy dla lepszego rozpoznawania
            min_df=2,
            max_df=0.95
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(df['combined_features'])

        # Redukcja wymiarowości
        self.svd = TruncatedSVD(n_components=150, random_state=42)  # Więcej komponentów
        self.tfidf_matrix = self.svd.fit_transform(self.tfidf_matrix)

        return df

    def get_recommendations(self, liked_books, genre_pref=None, year_range=None,
                            rating_min=None, num_recs=10):
        """Generuj rekomendacje na podstawie preferencji"""
        if self.df is None or self.tfidf_matrix is None:
            return []

        # Znajdź indeksy polubionych książek
        liked_indices = []
        title_col = 'title' if 'title' in self.df.columns else self.df.columns[0]

        for book in liked_books:
            matches = self.df[self.df[title_col].str.contains(book, case=False, na=False)]
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

        if genre_pref and 'genres' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['genres'].str.contains(genre_pref, case=False, na=False)
            ]

        if year_range and 'year' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['year'] >= year_range[0]) &
                (filtered_df['year'] <= year_range[1])
                ]

        if rating_min and 'rating' in filtered_df.columns:
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
                'title': book.get('title', 'Unknown Title'),
                'author': book.get('author', 'Unknown Author'),
                'year': book.get('year', 'Unknown'),
                'rating': book.get('rating', 0),
                'genres': book.get('genres', 'Unknown'),
                'series': book.get('series', ''),
                'pages': book.get('pages', 0),
                'publisher': book.get('publisher', ''),
                'numRatings': book.get('numRatings', 0),
                'bbeScore': book.get('bbeScore', 0),
                'similarity': sim
            })

        return recommendations

    def get_popular_books(self, num_recs=10):
        """Zwróć popularne książki jako fallback"""
        if 'rating' in self.df.columns:
            popular = self.df.nlargest(num_recs, 'rating')
        else:
            popular = self.df.head(num_recs)

        return [
            {
                'title': row.get('title', 'Unknown Title'),
                'author': row.get('author', 'Unknown Author'),
                'year': row.get('year', 'Unknown'),
                'rating': row.get('rating', 0),
                'genres': row.get('genres', 'Unknown'),
                'series': row.get('series', ''),
                'pages': row.get('pages', 0),
                'publisher': row.get('publisher', ''),
                'numRatings': row.get('numRatings', 0),
                'bbeScore': row.get('bbeScore', 0),
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
    st.sidebar.header("🎯 Twoje preferencje")

    # Polubione książki
    liked_books = st.sidebar.text_area(
        "📚 Książki które Ci się podobały (po jednej w linii):",
        placeholder="np. Harry Potter\n1984\nLord of the Rings",
        height=100
    ).split('\n')
    liked_books = [book.strip() for book in liked_books if book.strip()]

    # Dodatkowe filtry w expanderze
    with st.sidebar.expander("🔍 Zaawansowane filtry"):
        # Preferencje gatunku
        available_genres = []
        if 'genres' in st.session_state.recommender.df.columns:
            genres = st.session_state.recommender.df['genres'].str.split(',').explode().str.strip().unique()
            available_genres = [g for g in genres if pd.notna(g) and g != 'Unknown' and len(g) > 1]

        if available_genres:
            genre_pref = st.selectbox("🎭 Preferowany gatunek:", ['Wszystkie'] + sorted(available_genres[:50]))
            if genre_pref == 'Wszystkie':
                genre_pref = None
        else:
            genre_pref = None
            st.info("Brak informacji o gatunkach")

        # Zakres lat
        if 'year' in st.session_state.recommender.df.columns:
            years = st.session_state.recommender.df['year']
            min_year = int(years.min())
            max_year = int(years.max())
            year_range = st.slider(
                "📅 Zakres lat publikacji:",
                min_year, max_year,
                (max(min_year, 1900), max_year)
            )
        else:
            year_range = None
            st.info("Brak informacji o latach")

        # Minimalna ocena
        if 'rating' in st.session_state.recommender.df.columns:
            ratings = st.session_state.recommender.df['rating']
            min_rating = float(ratings.min())
            max_rating = float(ratings.max())
            rating_min = st.slider("⭐ Minimalna ocena:", min_rating, max_rating, min_rating, 0.1)
        else:
            rating_min = None
            st.info("Brak informacji o ocenach")

        # Minimalna liczba ocen (popularność)
        if 'numRatings' in st.session_state.recommender.df.columns:
            num_ratings = st.session_state.recommender.df['numRatings']
            min_num_ratings = int(num_ratings.quantile(0.1))  # 10% najmniej popularnych
            max_num_ratings = int(num_ratings.max())
            min_popularity = st.slider(
                "👥 Minimalna liczba ocen:",
                0, min(max_num_ratings, 10000),
                min_num_ratings
            )
            # Dodaj to do filtrowania
        else:
            min_popularity = None

    # Liczba rekomendacji
    num_recs = st.sidebar.slider("📊 Liczba rekomendacji:", 5, 25, 10)

    # Generuj rekomendacje
    if st.sidebar.button("🚀 Generuj rekomendacje", type="primary"):
        with st.spinner("🔍 Analizuję Twoje preferencje i generuję rekomendacje..."):
            recommendations = st.session_state.recommender.get_recommendations(
                liked_books=liked_books,
                genre_pref=genre_pref,
                year_range=year_range,
                rating_min=rating_min,
                num_recs=num_recs
            )

            st.header("📖 Twoje spersonalizowane rekomendacje:")

            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"**{i}. {rec['title']}** - *{rec['author']}*", expanded=i <= 3):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.write(f"**📅 Rok:** {rec['year']}")
                            st.write(f"**⭐ Ocena:** {rec['rating']:.1f}/5")
                            if rec.get('numRatings', 0) > 0:
                                st.write(f"**👥 Liczba ocen:** {rec['numRatings']:,}")

                        with col2:
                            st.write(f"**🎭 Gatunek:** {rec['genres']}")
                            if rec.get('pages', 0) > 0:
                                st.write(f"**📄 Strony:** {rec['pages']}")
                            if rec.get('series') and rec['series'] != 'Unknown':
                                st.write(f"**📚 Seria:** {rec['series']}")

                        with col3:
                            if rec.get('publisher') and rec['publisher'] != 'Unknown':
                                st.write(f"**🏢 Wydawca:** {rec['publisher']}")
                            if rec.get('bbeScore', 0) > 0:
                                st.write(f"**🏆 BBE Score:** {rec['bbeScore']:.1f}")
                            if rec['similarity'] > 0:
                                st.write(f"**🎯 Podobieństwo:** {rec['similarity']:.2f}")
            else:
                st.warning("😕 Nie znaleziono rekomendacji dla podanych kryteriów.")
                st.info("💡 Spróbuj rozszerzyć kryteria wyszukiwania lub dodać więcej polubionych książek.")

    # Statystyki
    with st.expander("📊 Statystyki datasetu Best Books Ever"):
        df = st.session_state.recommender.df
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📚 Liczba książek", f"{len(df):,}")
            if 'language' in df.columns:
                eng_books = len(df[df['language'].str.contains('en|english', case=False, na=True)])
                st.metric("🇬🇧 Angielskie", f"{eng_books:,}")

        with col2:
            if 'rating' in df.columns:
                st.metric("⭐ Średnia ocena", f"{df['rating'].mean():.2f}")
                st.metric("🏆 Najwyższa ocena", f"{df['rating'].max():.2f}")

        with col3:
            if 'year' in df.columns:
                st.metric("📅 Najstarsza", f"{df['year'].min()}")
                st.metric("📅 Najnowsza", f"{df['year'].max()}")

        with col4:
            if 'numRatings' in df.columns:
                total_ratings = df['numRatings'].sum()
                st.metric("👥 Łączne oceny", f"{total_ratings:,}")
                most_rated = df['numRatings'].max()
                st.metric("👑 Najwięcej ocen", f"{most_rated:,}")

        # Dodatkowe statystyki
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'genres' in df.columns:
                top_genres = df['genres'].str.split(',').explode().str.strip().value_counts().head(5)
                st.markdown("**🎭 Top 5 gatunków:**")
                for genre, count in top_genres.items():
                    if genre and genre != 'Unknown':
                        st.write(f"• {genre}: {count}")

        with col2:
            if 'bbeScore' in df.columns:
                st.metric("🏆 Śr. BBE Score", f"{df['bbeScore'].mean():.1f}")
                top_bbe = df.nlargest(1, 'bbeScore')['title'].iloc[0] if len(df) > 0 else "Brak danych"
                st.write(f"**🥇 Najwyższy BBE:** {top_bbe}")

        with col3:
            if 'pages' in df.columns:
                avg_pages = df['pages'].mean()
                st.metric("📄 Średnie strony", f"{avg_pages:.0f}")
                longest = df.nlargest(1, 'pages')['title'].iloc[0] if len(df) > 0 else "Brak danych"
                st.write(f"**📖 Najdłuższa:** {longest}")


if __name__ == "__main__":
    main()