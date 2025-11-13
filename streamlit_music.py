"""Streamlit-ready music recommendation app.

Usage: run `streamlit run music_recommandation.py` from the directory
that contains `high_popularity_spotify_data.csv`.
"""

from typing import Tuple
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]


@st.cache_data
def load_data(path: str = 'high_popularity_spotify_data.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize artist column name if necessary
    if 'artist_name' not in df.columns and 'track_artist' in df.columns:
        df = df.rename(columns={'track_artist': 'artist_name'})
    if 'artist_name' not in df.columns:
        df['artist_name'] = ''

    # Ensure track_name exists
    if 'track_name' not in df.columns and 'name' in df.columns:
        df = df.rename(columns={'name': 'track_name'})

    df = df.drop_duplicates(subset='track_name')
    df = df.dropna(subset=FEATURES)
    return df.reset_index(drop=True)


@st.cache_data
def build_similarity(df: pd.DataFrame, features: list = FEATURES) -> Tuple[pd.DataFrame, any]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    sim = cosine_similarity(scaled)
    return df, sim


def recommend(song_name: str, df: pd.DataFrame, similarity_matrix, n: int = 5) -> pd.DataFrame:
    if song_name not in df['track_name'].values:
        return pd.DataFrame({'error': [f"'{song_name}' not found in dataset."]})

    idx = int(df[df['track_name'] == song_name].index[0])
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    rec_idx = [i[0] for i in sim_scores]
    recommendations = df.iloc[rec_idx][['track_name', 'artist_name']].copy()
    recommendations['score'] = [round(similarity_matrix[idx][i], 4) for i in rec_idx]
    return recommendations.reset_index(drop=True)


def main():
    st.set_page_config(page_title='Music Recommendation', layout='wide')
    st.title('🎵 Music Recommendation')

    st.markdown('Upload or use the bundled `high_popularity_spotify_data.csv`.')

    uploaded = st.file_uploader('Upload CSV (optional)', type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        # apply same normalization as load_data
        if 'track_artist' in df.columns and 'artist_name' not in df.columns:
            df = df.rename(columns={'track_artist': 'artist_name'})
        if 'name' in df.columns and 'track_name' not in df.columns:
            df = df.rename(columns={'name': 'track_name'})
        df = df.drop_duplicates(subset='track_name')
        df = df.dropna(subset=FEATURES)
    else:
        try:
            df = load_data()
        except FileNotFoundError:
            st.error('CSV file not found. Upload a CSV or place `high_popularity_spotify_data.csv` in this folder.')
            return

    df, similarity_matrix = build_similarity(df)

    st.sidebar.header('Options')
    n = st.sidebar.slider('Number of recommendations', 1, 20, 5)
    search_mode = st.sidebar.selectbox('Choose input', ['Select song', 'Type song name'])

    if search_mode == 'Select song':
        song = st.selectbox('Pick a song', df['track_name'].sort_values().unique())
    else:
        song = st.text_input('Song name', '')

    if st.button('Recommend'):
        if not song:
            st.warning('Please select or type a song name.')
        else:
            recs = recommend(song, df, similarity_matrix, n=n)
            if 'error' in recs.columns:
                st.error(recs['error'].iloc[0])
            else:
                st.subheader(f'Recommendations for: {song}')
                st.dataframe(recs)

    with st.expander('Sample songs'):
        st.write(df['track_name'].dropna().sample(min(20, len(df))).values)


if __name__ == '__main__':
    main()

