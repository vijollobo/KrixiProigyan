"""
Fetches a representative photo for a crop from Wikipedia's public REST API
at runtime (https://en.wikipedia.org/api/rest_v1/page/summary/{title}).

This is intentionally done live, from the app itself, rather than
hardcoding image URLs: Wikipedia's summary endpoint is a stable, CORS-
friendly, purpose-built endpoint for exactly this kind of thumbnail
hotlinking, and staying dynamic means we don't depend on any one
hardcoded link going stale or being wrong for a given crop.

Requires internet access on the machine running Streamlit. If a request
fails (no internet, page has no image, etc.) we fall back to an emoji so
the app never breaks.
"""

import requests
import streamlit as st

WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

# Emoji fallback shown if the live photo can't be fetched
CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫘", "mothbeans": "🫘", "mungbean": "🫘", "blackgram": "🫘",
    "lentil": "🫘", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🥭", "coconut": "🥥", "cotton": "☁️",
    "jute": "🌿", "coffee": "☕",
}


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_crop_photo_url(wiki_title: str):
    """Returns a direct image URL for the given Wikipedia page title,
    or None if unavailable (caller should fall back to an emoji)."""
    try:
        resp = requests.get(
            WIKI_SUMMARY_URL.format(wiki_title.replace(" ", "_")),
            headers={"User-Agent": "KrixiProigyan-Demo/1.0"},
            timeout=4,
        )
        if resp.status_code == 200:
            data = resp.json()
            thumb = data.get("thumbnail", {}).get("source")
            return thumb
    except requests.RequestException:
        pass
    return None


def show_crop_photo(st_container, crop_key: str, wiki_title: str, caption: str):
    """Renders a crop photo if fetchable, otherwise a large emoji fallback."""
    url = get_crop_photo_url(wiki_title)
    if url:
        st_container.image(url, caption=caption, use_container_width=True)
    else:
        st_container.markdown(
            f"<div style='font-size:90px; text-align:center'>{CROP_EMOJI.get(crop_key.lower(), '🌱')}</div>"
            f"<div style='text-align:center'>{caption}</div>",
            unsafe_allow_html=True,
        )
