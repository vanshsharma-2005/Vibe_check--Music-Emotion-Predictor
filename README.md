<div align="center">

<br>

# 🎵 VibeCheck
### *Know the mood before you press play.*

<br>

![Python](https://img.shields.io/badge/Python-3.10+-1DB954?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Spotify](https://img.shields.io/badge/Spotify_API-1DB954?style=for-the-badge&logo=spotify&logoColor=white)

<br>

> **VibeCheck** is a machine learning web app that predicts the emotional mood of any song using Spotify audio features. Paste a track link — get the vibe instantly.

<br>

[🚀 Live Demo](#) &nbsp;·&nbsp; [📖 How It Works](#how-it-works) &nbsp;·&nbsp; [⚙️ Run Locally](#run-locally)

<br>

---

</div>

<br>

## ✨ Features

- 🎧 &nbsp;**Spotify Integration** — paste any Spotify track link and get an instant mood prediction
- 🎛️ &nbsp;**Manual Mode** — adjust audio feature sliders to explore how sound affects emotion
- 📊 &nbsp;**Feature Breakdown** — visual bars showing danceability, energy, acousticness and more
- 🎶 &nbsp;**Mood Playlist** — auto-generates 5 songs from the dataset matching your predicted mood
- 🤖 &nbsp;**Smart Fallback** — works even when Spotify API restricts audio features, using local dataset lookup

<br>

## 🎭 Mood Categories

| Mood | Vibe | Audio Profile |
|------|------|---------------|
| ☀️ Happy / Joyful | Upbeat, positive, energetic | High valence + High energy |
| 🌧️ Sad / Depressed | Melancholic, slow, emotional | Low valence + Low energy |
| 🍃 Calm / Relaxed | Peaceful, acoustic, gentle | High valence + Low energy |
| 🔥 Tense / Angry | Aggressive, intense, driven | Low valence + High energy |

> Moods are mapped using **Russell's Circumplex Model of Emotion** — a psychology-backed framework that maps feelings across valence and energy axes.

<br>

## 🧠 How It Works

```
Spotify Track ID
       │
       ▼
Fetch Audio Features (danceability, tempo, loudness, acousticness...)
       │
       ▼
StandardScaler (normalize features)
       │
       ▼
Random Forest Classifier (100 estimators, trained on Music Info dataset)
       │
       ▼
Predicted Mood + Playlist Recommendations
```

The model is trained on-the-fly when the app starts using `Music Info.csv`. It uses 7 audio features to classify each song into one of the four emotional quadrants.

<br>

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Custom CSS |
| ML Model | Scikit-learn (Random Forest Classifier) |
| Data Processing | Pandas, NumPy |
| Spotify Integration | Spotipy (Spotify Web API) |
| Deployment | Streamlit Community Cloud |
| Language | Python 3.10+ |

<br>

## ⚙️ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/your-username/vibecheck-music-emotion.git
cd vibecheck-music-emotion
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up Spotify credentials**

Get your free API keys from [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) and set them as environment variables:

```bash
# Mac / Linux
export SPOTIPY_CLIENT_ID="your_client_id"
export SPOTIPY_CLIENT_SECRET="your_client_secret"

# Windows
$env:SPOTIPY_CLIENT_ID="your_client_id"
$env:SPOTIPY_CLIENT_SECRET="your_client_secret"
```

**4. Run the app**
```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

<br>

## 🚀 Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select this repo → set main file to `streamlit_app.py`
4. Under **Advanced settings → Secrets**, add:
```toml
SPOTIPY_CLIENT_ID = "your_client_id"
SPOTIPY_CLIENT_SECRET = "your_client_secret"
```
5. Click **Deploy!** 🎉

<br>

## 📁 Project Structure

```
vibecheck/
├── streamlit_app.py       # Main Streamlit web app
├── src/
│   └── train_model.py     # Model training script
├── api/
│   └── main.py            # FastAPI backend (optional)
├── Music Info.csv         # Song dataset with audio features
├── requirements.txt       # Python dependencies
└── README.md
```

<br>

## 👥 Authors

Built with 🎵 by **[Your Name]** and **[Sarthak](https://github.com/SARTHAKG8-debug)**

<br>

---

<div align="center">
<sub>If you found this project useful, drop a ⭐ on the repo!</sub>
</div>
