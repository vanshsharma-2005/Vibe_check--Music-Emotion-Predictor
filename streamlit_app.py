import streamlit as st
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURE_NAMES = ['danceability', 'loudness', 'speechiness', 'acousticness',
                 'instrumentalness', 'liveness', 'tempo']

st.set_page_config(page_title="VibeCheck 🎵", page_icon="🎵", layout="centered")

st.markdown("""
<style>
h1 { color: #1DB954; text-align: center; font-weight: 800; margin-bottom: 0px; }
.subtitle { text-align: center; color: #a0a0a0; font-size: 1.1rem; margin-bottom: 30px; }
.mood-box { padding: 2rem; border-radius: 15px; text-align: center; margin-top: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2); transition: transform 0.3s ease; }
.mood-box:hover { transform: translateY(-5px); }
.mood-title { font-size: 1.2rem; margin-bottom: 10px; text-transform: uppercase;
              letter-spacing: 1px; font-weight: 600; }
.mood-result { font-size: 2.5rem; font-weight: 800; }
.happy  { background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); color: #333; }
.sad    { background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%); color: #333; }
.calm   { background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); color: #333; }
.tense  { background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%); color: #fff; }
.default-mood { background: linear-gradient(135deg, #8e2de2 0%, #4a00e0 100%); color: #fff; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="🎵 Training model on first load — takes ~30 seconds...")
def train_and_load_model():
    data_path = str(Path(__file__).parent / 'Music Info.csv')
    df = pd.read_csv(data_path)
    df = df.dropna(subset=FEATURE_NAMES + ['valence', 'energy'])

    v_med = df['valence'].median()
    e_med = df['energy'].median()

    def get_mood(row):
        v, e = row['valence'], row['energy']
        if v >= v_med and e >= e_med:   return 'Happy/Joyful'
        elif v >= v_med and e < e_med:  return 'Calm/Relaxed'
        elif v < v_med and e >= e_med:  return 'Tense/Angry'
        else:                           return 'Sad/Depressed'

    df['mood'] = df.apply(get_mood, axis=1)
    X = df[FEATURE_NAMES]
    y = df['mood']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    return model, scaler


@st.cache_data
def load_data():
    data_path = str(Path(__file__).parent / 'Music Info.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        cols_to_keep = ['spotify_id', 'name', 'artist'] + FEATURE_NAMES + ['valence', 'energy']
        existing_cols = [c for c in cols_to_keep if c in df.columns]
        return df[existing_cols]
    return None


def get_mood_html(prediction):
    pred_lower = str(prediction).lower()
    if 'happy' in pred_lower or 'joy' in pred_lower:
        css_class, emoji = 'happy', '☀️'
    elif 'sad' in pred_lower or 'depress' in pred_lower:
        css_class, emoji = 'sad', '🌧️'
    elif 'calm' in pred_lower or 'relax' in pred_lower:
        css_class, emoji = 'calm', '🍃'
    elif 'tense' in pred_lower or 'angry' in pred_lower:
        css_class, emoji = 'tense', '🔥'
    else:
        css_class, emoji = 'default-mood', '🎶'
    return f'''<div class="mood-box {css_class}">
        <div class="mood-title">Predicted Mood</div>
        <div class="mood-result">{emoji} {prediction}</div>
    </div>'''


def display_playlist_generator(prediction, music_df, data_path):
    st.divider()
    st.markdown("### 🎧 Automatic Playlist Generator")
    st.write(f"Based on the predicted mood (**{prediction}**), here are some matching tracks:")
    with st.spinner("Curating your playlist..."):
        if music_df is not None and not music_df.empty:
            try:
                full_df = pd.read_csv(data_path)
                pred_lower = str(prediction).lower()
                if 'happy' in pred_lower or 'joy' in pred_lower:
                    subset = full_df[(full_df['valence'] > 0.6) & (full_df['energy'] > 0.6)]
                elif 'sad' in pred_lower or 'depress' in pred_lower:
                    subset = full_df[(full_df['valence'] < 0.4) & (full_df['energy'] < 0.6) & (full_df['acousticness'] > 0.4)]
                elif 'calm' in pred_lower or 'relax' in pred_lower:
                    subset = full_df[(full_df['energy'] < 0.5) & (full_df['acousticness'] > 0.6)]
                elif 'tense' in pred_lower or 'angry' in pred_lower:
                    subset = full_df[(full_df['energy'] > 0.8) & (full_df['acousticness'] < 0.2)]
                else:
                    subset = full_df
                if not subset.empty:
                    playlist = subset.sample(n=min(5, len(subset)))
                    for _, row in playlist.iterrows():
                        st.markdown(f'''<div style="padding:10px;border-radius:8px;
                            background:rgba(255,255,255,0.05);margin-bottom:10px;">
                            <div style="font-weight:bold;font-size:1.1rem;">{row['name']}</div>
                            <div style="color:#a0a0a0;">by {row['artist']}</div>
                            <a href="https://open.spotify.com/track/{row['spotify_id']}"
                               target="_blank" style="color:#1DB954;font-weight:bold;">
                               ▶ Listen on Spotify</a></div>''', unsafe_allow_html=True)
                else:
                    st.warning("No matching tracks found for this mood.")
            except Exception as e:
                st.error(f"Playlist error: {e}")
        else:
            st.warning("Dataset not available.")


# ── Main App ──────────────────────────────────────────────────────────────────
st.title("🎵 VibeCheck")
st.markdown("<div class='subtitle'>Predict the emotional mood of any song using Spotify audio features.</div>",
            unsafe_allow_html=True)

model, scaler = train_and_load_model()
music_df = load_data()

tab1, tab2 = st.tabs(["🎧 Predict from Spotify", "🎛️ Manual Audio Features"])

with tab1:
    st.markdown("### Enter a Spotify Track ID")
    st.write("E.g., `0keNu0t0tqsWtExGM3nT1D` (Mr. Brightside)")
    track_input = st.text_input("Track ID or Spotify Link", "")

    if "open.spotify.com/track/" in track_input:
        track_id = track_input.split("open.spotify.com/track/")[1].split("?")[0]
    else:
        track_id = track_input.strip()

    if st.button("Predict 🔮", key="spotify_predict"):
        if not track_id:
            st.warning("Please enter a valid Track ID.")
        else:
            client_id = os.environ.get("SPOTIPY_CLIENT_ID")
            client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
            if not client_id or not client_secret:
                try:
                    client_id = client_id or st.secrets.get("SPOTIPY_CLIENT_ID")
                    client_secret = client_secret or st.secrets.get("SPOTIPY_CLIENT_SECRET")
                except Exception:
                    pass

            if not client_id or not client_secret:
                st.error("Spotify credentials not configured. Add SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in Streamlit Secrets.")
            else:
                try:
                    with st.spinner("Fetching from Spotify..."):
                        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
                        sp = spotipy.Spotify(auth_manager=auth_manager)
                        track_info = sp.track(track_id)
                        name = track_info['name']
                        artist_name = track_info['artists'][0]['name']
                        album_art_url = track_info['album']['images'][0]['url'] if track_info['album']['images'] else None

                        song_features = None
                        try:
                            features_list = sp.audio_features(track_id)
                            if features_list and features_list[0]:
                                song_features = features_list[0]
                        except Exception:
                            pass

                        # Fallback to local dataset
                        if song_features is None and music_df is not None:
                            search = music_df[music_df['spotify_id'] == track_id]
                            if not search.empty:
                                row = search.iloc[0]
                                song_features = {k: row[k] for k in FEATURE_NAMES}

                        # Simulated fallback
                        if song_features is None:
                            import hashlib
                            st.info("Using simulated features (Spotify API restricted).")
                            hash_val = int(hashlib.md5(track_id.encode()).hexdigest(), 16)
                            def get_val(mn, mx, i):
                                return mn + (mx - mn) * ((hash_val >> (i * 4)) % 1000) / 1000.0
                            song_features = {
                                'danceability': get_val(0.3, 0.9, 0),
                                'loudness': get_val(-15.0, -3.0, 1),
                                'speechiness': get_val(0.02, 0.25, 2),
                                'acousticness': get_val(0.01, 0.8, 3),
                                'instrumentalness': get_val(0.0, 0.5, 4),
                                'liveness': get_val(0.05, 0.4, 5),
                                'tempo': get_val(80.0, 160.0, 6)
                            }

                        feature_values = [[float(song_features[f]) for f in FEATURE_NAMES]]
                        X_scaled = scaler.transform(feature_values)
                        prediction = model.predict(X_scaled)[0]

                        st.divider()
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if album_art_url:
                                st.image(album_art_url, use_container_width=True)
                        with col2:
                            st.subheader(name)
                            st.write(f"by **{artist_name}**")
                            st.markdown(get_mood_html(prediction), unsafe_allow_html=True)

                        with st.expander("View Audio Features"):
                            st.json({k: song_features[k] for k in FEATURE_NAMES})

                        display_playlist_generator(prediction, music_df,
                                                   str(Path(__file__).parent / 'Music Info.csv'))
                except Exception as e:
                    if "404" in str(e):
                        st.error("Track not found. Please check the ID.")
                    else:
                        st.error(f"Error: {e}")

with tab2:
    st.markdown("### Tweak Audio Features manually")
    col1, col2 = st.columns(2)
    with col1:
        danceability     = st.slider("Danceability", 0.0, 1.0, 0.5)
        speechiness      = st.slider("Speechiness", 0.0, 1.0, 0.05)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
        tempo            = st.slider("Tempo (BPM)", 50.0, 200.0, 120.0)
    with col2:
        loudness     = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2)
        liveness     = st.slider("Liveness", 0.0, 1.0, 0.1)

    if st.button("Predict Emotion 🔮", key="manual_predict"):
        feature_values = [[danceability, loudness, speechiness,
                           acousticness, instrumentalness, liveness, tempo]]
        X_scaled = scaler.transform(feature_values)
        prediction = model.predict(X_scaled)[0]
        st.markdown(get_mood_html(prediction), unsafe_allow_html=True)
        display_playlist_generator(prediction, music_df,
                                   str(Path(__file__).parent / 'Music Info.csv'))
