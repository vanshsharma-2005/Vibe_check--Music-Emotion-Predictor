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

st.set_page_config(page_title="VibeCheck", page_icon="🎵", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #f0f0f0;
}

/* Animated gradient background */
.stApp::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 20% 50%, rgba(120, 40, 200, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(29, 185, 84, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at 60% 80%, rgba(255, 60, 120, 0.1) 0%, transparent 50%);
    animation: bgshift 12s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

@keyframes bgshift {
    0%   { transform: translate(0, 0) rotate(0deg); }
    100% { transform: translate(2%, 2%) rotate(3deg); }
}

/* Hero header */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: rgba(29, 185, 84, 0.15);
    border: 1px solid rgba(29, 185, 84, 0.4);
    color: #1DB954;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 6px 18px;
    border-radius: 50px;
    margin-bottom: 1.2rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1;
    margin: 0 0 0.5rem;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #1DB954 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.45);
    font-weight: 300;
    margin: 0;
    letter-spacing: 0.5px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.08);
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px;
    color: rgba(255,255,255,0.5);
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 10px 24px;
    border: none;
    transition: all 0.2s;
}

.stTabs [aria-selected="true"] {
    background: rgba(255,255,255,0.1) !important;
    color: #ffffff !important;
}

/* Input box */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 14px 18px !important;
    transition: border 0.2s !important;
}

.stTextInput > div > div > input:focus {
    border: 1px solid rgba(29, 185, 84, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(29, 185, 84, 0.1) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1DB954, #17a347) !important;
    color: #000000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(29, 185, 84, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(29, 185, 84, 0.45) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: #1DB954 !important;
}

.stSlider [data-baseweb="slider"] {
    padding: 0 !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.08) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.6) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Section labels */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    margin-bottom: 1rem;
}

/* Mood result card */
.mood-card {
    position: relative;
    border-radius: 24px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
    overflow: hidden;
}

.mood-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 24px;
    padding: 1px;
    background: linear-gradient(135deg, rgba(255,255,255,0.3), rgba(255,255,255,0.05));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: destination-out;
    mask-composite: exclude;
}

.mood-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; }

.mood-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.5rem;
}

.mood-name {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1;
}

/* Happy */
.mood-happy { background: linear-gradient(135deg, rgba(255,200,50,0.2) 0%, rgba(255,120,50,0.2) 100%); }
.mood-happy .mood-name { color: #FFD166; }

/* Sad */
.mood-sad { background: linear-gradient(135deg, rgba(100,150,255,0.2) 0%, rgba(150,200,255,0.2) 100%); }
.mood-sad .mood-name { color: #90B4F5; }

/* Calm */
.mood-calm { background: linear-gradient(135deg, rgba(50,200,120,0.2) 0%, rgba(50,180,200,0.2) 100%); }
.mood-calm .mood-name { color: #5ECFA0; }

/* Tense */
.mood-tense { background: linear-gradient(135deg, rgba(255,50,80,0.2) 0%, rgba(255,100,50,0.2) 100%); }
.mood-tense .mood-name { color: #FF6B6B; }

/* Track card */
.track-card {
    display: flex;
    gap: 1rem;
    align-items: center;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    transition: all 0.2s;
}

.track-card:hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(255,255,255,0.15);
    transform: translateX(4px);
}

.track-num {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: rgba(255,255,255,0.2);
    width: 20px;
    text-align: center;
}

.track-info { flex: 1; }

.track-name {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.95rem;
    color: #ffffff;
    margin-bottom: 2px;
}

.track-artist {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.4);
}

.track-link {
    background: rgba(29,185,84,0.15);
    border: 1px solid rgba(29,185,84,0.3);
    color: #1DB954 !important;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 50px;
    text-decoration: none !important;
    white-space: nowrap;
    transition: all 0.2s;
}

.track-link:hover {
    background: rgba(29,185,84,0.3) !important;
}

/* Feature bars */
.feature-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

.feature-name {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.5);
    width: 120px;
    text-align: right;
    font-family: 'DM Sans', sans-serif;
}

.feature-bar-bg {
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.08);
    border-radius: 4px;
    overflow: hidden;
}

.feature-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #1DB954, #a78bfa);
    border-radius: 4px;
    transition: width 0.6s ease;
}

.feature-val {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.35);
    width: 36px;
    text-align: right;
    font-family: 'DM Sans', sans-serif;
}

/* Slider labels */
.slider-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.45);
    margin-bottom: -12px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}

/* Playlist header */
.playlist-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 2rem 0 1rem;
}

.playlist-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #ffffff;
}

.playlist-pill {
    background: rgba(29,185,84,0.15);
    border: 1px solid rgba(29,185,84,0.3);
    color: #1DB954;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 3px 10px;
    border-radius: 50px;
}

/* Artist + track display */
.now-playing {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

.np-artist {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.4);
    margin-bottom: 2px;
}

.np-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
}

/* Loading spinner override */
.stSpinner > div {
    border-top-color: #1DB954 !important;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="🎵 Warming up the model...")
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
        cols = ['spotify_id', 'name', 'artist'] + FEATURE_NAMES + ['valence', 'energy']
        existing = [c for c in cols if c in df.columns]
        return df[existing]
    return None


def get_mood_config(prediction):
    pred = str(prediction).lower()
    if 'happy' in pred or 'joy' in pred:
        return {'class': 'happy', 'emoji': '☀️', 'label': 'Happy / Joyful'}
    elif 'sad' in pred or 'depress' in pred:
        return {'class': 'sad', 'emoji': '🌧️', 'label': 'Sad / Depressed'}
    elif 'calm' in pred or 'relax' in pred:
        return {'class': 'calm', 'emoji': '🍃', 'label': 'Calm / Relaxed'}
    elif 'tense' in pred or 'angry' in pred:
        return {'class': 'tense', 'emoji': '🔥', 'label': 'Tense / Angry'}
    return {'class': 'calm', 'emoji': '🎶', 'label': prediction}


def render_mood(prediction):
    cfg = get_mood_config(prediction)
    st.markdown(f"""
    <div class="mood-card mood-{cfg['class']}">
        <div class="mood-emoji">{cfg['emoji']}</div>
        <div class="mood-label">Detected Mood</div>
        <div class="mood-name">{cfg['label']}</div>
    </div>""", unsafe_allow_html=True)


def render_feature_bars(song_features):
    bars_html = ""
    display = {
        'danceability': (song_features.get('danceability', 0), 0, 1),
        'energy': (song_features.get('energy', 0), 0, 1),
        'acousticness': (song_features.get('acousticness', 0), 0, 1),
        'liveness': (song_features.get('liveness', 0), 0, 1),
        'speechiness': (song_features.get('speechiness', 0), 0, 1),
        'instrumentalness': (song_features.get('instrumentalness', 0), 0, 1),
    }
    for name, (val, mn, mx) in display.items():
        pct = max(0, min(100, (val - mn) / (mx - mn) * 100))
        bars_html += f"""
        <div class="feature-row">
            <div class="feature-name">{name}</div>
            <div class="feature-bar-bg"><div class="feature-bar-fill" style="width:{pct:.1f}%"></div></div>
            <div class="feature-val">{val:.2f}</div>
        </div>"""
    st.markdown(bars_html, unsafe_allow_html=True)


def render_playlist(prediction, music_df, data_path):
    cfg = get_mood_config(prediction)
    st.markdown(f"""
    <div class="playlist-header">
        <div class="playlist-title">Mood Playlist</div>
        <div class="playlist-pill">{cfg['emoji']} {cfg['label']}</div>
    </div>""", unsafe_allow_html=True)

    if music_df is None or music_df.empty:
        st.warning("Dataset unavailable.")
        return
    try:
        full_df = pd.read_csv(data_path)
        pred = str(prediction).lower()
        if 'happy' in pred or 'joy' in pred:
            subset = full_df[(full_df['valence'] > 0.6) & (full_df['energy'] > 0.6)]
        elif 'sad' in pred or 'depress' in pred:
            subset = full_df[(full_df['valence'] < 0.4) & (full_df['energy'] < 0.6) & (full_df['acousticness'] > 0.4)]
        elif 'calm' in pred or 'relax' in pred:
            subset = full_df[(full_df['energy'] < 0.5) & (full_df['acousticness'] > 0.6)]
        elif 'tense' in pred or 'angry' in pred:
            subset = full_df[(full_df['energy'] > 0.8) & (full_df['acousticness'] < 0.2)]
        else:
            subset = full_df

        if subset.empty:
            subset = full_df

        playlist = subset.sample(n=min(5, len(subset)))
        tracks_html = ""
        for i, (_, row) in enumerate(playlist.iterrows(), 1):
            tracks_html += f"""
            <div class="track-card">
                <div class="track-num">{i:02d}</div>
                <div class="track-info">
                    <div class="track-name">{row['name']}</div>
                    <div class="track-artist">{row['artist']}</div>
                </div>
                <a class="track-link" href="https://open.spotify.com/track/{row['spotify_id']}" target="_blank">▶ Play</a>
            </div>"""
        st.markdown(tracks_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Playlist error: {e}")


# ── App Layout ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🎵 AI Music Analysis</div>
    <div class="hero-title">VibeCheck</div>
    <p class="hero-sub">Drop a Spotify track. We'll tell you exactly how it feels.</p>
</div>
""", unsafe_allow_html=True)

model, scaler = train_and_load_model()
music_df = load_data()

tab1, tab2 = st.tabs(["  🎧  Spotify Track  ", "  🎛️  Manual Mode  "])

# ── TAB 1: Spotify ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-label'>Paste a Spotify Link or Track ID</div>", unsafe_allow_html=True)
    track_input = st.text_input("", placeholder="https://open.spotify.com/track/... or track ID", label_visibility="collapsed")

    if "open.spotify.com/track/" in track_input:
        track_id = track_input.split("open.spotify.com/track/")[1].split("?")[0]
    else:
        track_id = track_input.strip()

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Analyse Vibe ✦", key="spotify_predict")

    if predict_btn:
        if not track_id:
            st.warning("Please enter a Spotify track link or ID.")
        else:
            client_id = os.environ.get("SPOTIPY_CLIENT_ID")
            client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
            try:
                client_id = client_id or st.secrets.get("SPOTIPY_CLIENT_ID")
                client_secret = client_secret or st.secrets.get("SPOTIPY_CLIENT_SECRET")
            except Exception:
                pass

            if not client_id or not client_secret:
                st.error("Spotify credentials not set. Add them in Streamlit Secrets.")
            else:
                try:
                    with st.spinner("Fetching from Spotify..."):
                        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
                        sp = spotipy.Spotify(auth_manager=auth_manager)
                        track_info = sp.track(track_id)
                        name = track_info['name']
                        artist_name = track_info['artists'][0]['name']
                        album_art = track_info['album']['images'][0]['url'] if track_info['album']['images'] else None

                        song_features = None
                        try:
                            fl = sp.audio_features(track_id)
                            if fl and fl[0]:
                                song_features = fl[0]
                        except Exception:
                            pass

                        if song_features is None and music_df is not None:
                            match = music_df[music_df['spotify_id'] == track_id]
                            if not match.empty:
                                song_features = {k: match.iloc[0][k] for k in FEATURE_NAMES}

                        if song_features is None:
                            import hashlib
                            st.info("Spotify API restricted — using smart fallback features.")
                            hv = int(hashlib.md5(track_id.encode()).hexdigest(), 16)
                            def gv(mn, mx, i): return mn + (mx - mn) * ((hv >> (i * 4)) % 1000) / 1000.0
                            song_features = {
                                'danceability': gv(0.3, 0.9, 0), 'loudness': gv(-15, -3, 1),
                                'speechiness': gv(0.02, 0.25, 2), 'acousticness': gv(0.01, 0.8, 3),
                                'instrumentalness': gv(0, 0.5, 4), 'liveness': gv(0.05, 0.4, 5),
                                'tempo': gv(80, 160, 6)
                            }

                        fv = [[float(song_features[f]) for f in FEATURE_NAMES]]
                        prediction = model.predict(scaler.transform(fv))[0]

                    st.markdown("<br>", unsafe_allow_html=True)

                    if album_art:
                        col1, col2 = st.columns([1, 2], gap="medium")
                        with col1:
                            st.image(album_art, use_container_width=True)
                        with col2:
                            st.markdown(f"""
                            <div class="now-playing">
                                <div class="np-artist">{artist_name}</div>
                                <div class="np-name">{name}</div>
                            </div>""", unsafe_allow_html=True)
                            render_mood(prediction)
                    else:
                        st.markdown(f"<div class='now-playing'><div class='np-name'>{name}</div><div class='np-artist'>{artist_name}</div></div>", unsafe_allow_html=True)
                        render_mood(prediction)

                    with st.expander("📊 Audio Feature Breakdown"):
                        render_feature_bars(song_features)

                    st.markdown("<hr>", unsafe_allow_html=True)
                    render_playlist(prediction, music_df, str(Path(__file__).parent / 'Music Info.csv'))

                except Exception as e:
                    if "404" in str(e):
                        st.error("Track not found. Double-check the Spotify ID.")
                    else:
                        st.error(f"Something went wrong: {e}")

# ── TAB 2: Manual ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-label'>Adjust Audio Features</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='slider-label'>Danceability</div>", unsafe_allow_html=True)
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5, label_visibility="collapsed")
        st.markdown("<div class='slider-label'>Speechiness</div>", unsafe_allow_html=True)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05, label_visibility="collapsed")
        st.markdown("<div class='slider-label'>Instrumentalness</div>", unsafe_allow_html=True)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, label_visibility="collapsed")
        st.markdown("<div class='slider-label'>Tempo (BPM)</div>", unsafe_allow_html=True)
        tempo = st.slider("Tempo", 50.0, 200.0, 120.0, label_visibility="collapsed")

    with col2:
        st.markdown("<div class='slider-label'>Loudness (dB)</div>", unsafe_allow_html=True)
        loudness = st.slider("Loudness", -60.0, 0.0, -10.0, label_visibility="collapsed")
        st.markdown("<div class='slider-label'>Acousticness</div>", unsafe_allow_html=True)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2, label_visibility="collapsed")
        st.markdown("<div class='slider-label'>Liveness</div>", unsafe_allow_html=True)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.1, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Detect Vibe ✦", key="manual_predict"):
        fv = [[danceability, loudness, speechiness, acousticness, instrumentalness, liveness, tempo]]
        prediction = model.predict(scaler.transform(fv))[0]
        render_mood(prediction)
        st.markdown("<hr>", unsafe_allow_html=True)
        render_playlist(prediction, music_df, str(Path(__file__).parent / 'Music Info.csv'))
