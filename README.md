# 🌾 KrixiProigyan — Smart Farming Assistant

**KrixiProigyan** (কৃষিপ্ৰজ্ঞান, "agricultural wisdom" in Axomiya (Assamese)) is an open-source, machine-learning-powered farming advisory dashboard built for the Indian agricultural context. It recommends the most suitable crop for a given plot of land based on soil and climate readings, and layers on irrigation, fertilizer, and early-warning advisories — all presented in 13 languages, including 12 major Indian languages.

This project has been built from the ground up as a standalone, open-source demonstration: every feature originally sketched across slide decks (soil health monitoring, crop recommendation, smart irrigation, fertilizer advisory, multilingual farmer alerts) has been implemented as real, runnable code in a single unified application, with the IoT hardware layer simulated so the ML/software side can be evaluated on its own merits.

This project can be accessed without downloading using this [link].(https://krixiproigyan.streamlit.app/)

> 📊 A slide deck (`কৃষিপ্ৰজ্ঞান Krixiproigyan.pptx`) describing the original concept and motivation is included in this repository for background context. The application itself supersedes the deck — this README documents what is actually implemented.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
- [The Machine Learning Model](#the-machine-learning-model)
  - [Problem Framing](#problem-framing)
  - [Dataset](#dataset)
  - [Algorithm: Random Forest Classifier](#algorithm-random-forest-classifier)
  - [Why Random Forest](#why-random-forest)
  - [Training Pipeline](#training-pipeline)
  - [Model Performance](#model-performance)
  - [Inference / Prediction](#inference--prediction)
- [The Rule-Based Agronomy Engine](#the-rule-based-agronomy-engine)
  - [Soil Health Index](#soil-health-index)
  - [Irrigation Advisory](#irrigation-advisory)
  - [Fertilizer Advisory](#fertilizer-advisory)
  - [Disease & pH Alerts](#disease--ph-alerts)
  - [Yield Outlook](#yield-outlook)
- [Multilingual System](#multilingual-system)
- [Live Crop Photos](#live-crop-photos)
- [Tech Stack](#tech-stack)
- [Installation & Running](#installation--running)
- [Usage Guide](#usage-guide)
- [Dataset License & Attribution](#dataset-license--attribution)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## What This Project Does

Given seven soil and climate readings — nitrogen, phosphorus, potassium, temperature, humidity, soil pH, and rainfall — KrixiProigyan:

1. **Recommends the best-suited crop** out of 22 common Indian crops, with a confidence score and two runner-up alternatives, using a trained Random Forest classifier.
2. **Scores the soil's overall health** on a 0–100 index derived from the same readings.
3. **Advises on irrigation timing** (irrigate now / soon / not needed) based on current moisture and rainfall.
4. **Advises on fertilizer (NPK) correction** by comparing the current readings against the statistical nutrient profile the recommended crop actually needs, learned directly from data.
5. **Raises early-warning alerts** for elevated pest/disease risk and extreme soil pH.
6. **Gives an indicative yield outlook** (favourable / moderate / sub-optimal).
7. **Presents everything in the farmer's own language** — 13 languages are supported, with crop names, UI text, and all numeric values (rounded to 2 decimal places) rendered in the correct native numeral script.
8. **Shows a live photo of the recommended crop**, fetched at runtime rather than hardcoded.

All of this runs in a single Streamlit web application, backed by a real dataset and a real trained model — there is no mock data anywhere in the ML pipeline.

---

## Project Structure

```
krixiproigyan/
├── app.py                       # Streamlit application — UI, tabs, layout, orchestration
├── model.py                     # ML pipeline: data loading, training, prediction
├── agri_logic.py                # Rule-based agronomy engine (soil health, irrigation, fertilizer, alerts, yield)
├── translations.py              # 13-language UI strings, crop names, numeral localization
├── crop_media.py                # Live crop-photo fetching (Wikipedia API) with emoji fallback
├── data/
│   └── crop_recommendation.csv  # Training dataset (2,200 rows, 22 crops)
├── requirements.txt             # Python dependencies
├── কৃষিপ্ৰজ্ঞান Krixiproigyan.pptx  # Original concept slide deck (background/context only)
└── README.md                    # You are here
```

---

## System Architecture

```
                     ┌─────────────────────────────┐
                     │   Soil & Climate Input       │
                     │  (manual sliders OR           │
                     │   simulated sensor draw)      │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │  Random Forest Classifier     │
                     │  (model.py)                   │
                     │  → Top-3 crops + confidence   │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │  Rule-Based Agronomy Engine   │
                     │  (agri_logic.py)              │
                     │  → Soil Health Index          │
                     │  → Irrigation advisory        │
                     │  → Fertilizer advisory        │
                     │  → Disease / pH alerts        │
                     │  → Yield outlook              │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │  Presentation Layer            │
                     │  (app.py + translations.py    │
                     │   + crop_media.py)             │
                     │  → 13-language dashboard       │
                     │  → SMS/voice-style advisory     │
                     │  → Live crop photo              │
                     └─────────────────────────────┘
```

The ML model and the rule engine are deliberately decoupled: the classifier's only job is "which crop fits this soil/climate profile," and every other piece of advice (irrigation, fertilizer, alerts, yield) is a separate, inspectable, hand-written rule that consumes the same raw sensor values (and, for fertilizer, the model's top prediction). This makes the system's behaviour auditable — you can trace exactly why it said what it said, which matters for anything giving agricultural advice.

---

## The Machine Learning Model

### Problem Framing

This is a **multi-class classification problem**: given a 7-dimensional feature vector (N, P, K, temperature, humidity, pH, rainfall), predict which one of 22 crop classes is best suited to those conditions. It is *not* a regression or yield-prediction problem — the model does not predict how much you'll harvest, only what to plant. (Yield is estimated separately by a simple, clearly-labelled heuristic — see [Yield Outlook](#yield-outlook).)

### Dataset

- **Source**: The "Crop Recommendation Dataset" originally published on Kaggle by **Atharva Ingle**, built by augmenting rainfall, climate, and fertilizer datasets available for India. It's one of the most widely used datasets for this exact task in the Indian agri-tech space. This repository fetches it via a GitHub mirror (`gireesh777/Crop_Recommendation_System_using_ML`); the CSV is committed directly into `data/` so the project runs without any external download step.
- **Size**: 2,200 rows.
- **Classes**: 22 crops, **perfectly balanced at 100 samples each** — rice, maize, chickpea, kidney beans, pigeon peas, moth beans, mung bean, black gram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, and coffee.
- **Features** (all numeric, no missing values):

  | Feature | Meaning | Typical Range |
  |---|---|---|
  | N | Nitrogen content in soil (kg/ha) | 0–140 |
  | P | Phosphorus content in soil (kg/ha) | 5–145 |
  | K | Potassium content in soil (kg/ha) | 5–205 |
  | temperature | Ambient temperature (°C) | 8–44 |
  | humidity | Relative humidity (%) | 14–100 |
  | ph | Soil pH | 3.5–10 |
  | rainfall | Rainfall (mm) | 20–300 |

- **No feature scaling is applied.** Tree-based models like Random Forest split on raw threshold values per feature, so they're invariant to monotonic scaling — normalizing would add a preprocessing step for no benefit here.

### Algorithm: Random Forest Classifier

The model is `sklearn.ensemble.RandomForestClassifier` — an **ensemble of decision trees**. Here's how it actually works, end to end:

1. **Bootstrap sampling (bagging)**: for each of the 200 trees in the forest, the training algorithm draws a random sample of the training rows *with replacement* — so each tree sees a slightly different subset of the 1,760 training examples (some rows repeated, some left out).
2. **Random feature subsetting**: at every split point inside every tree, the algorithm doesn't consider all 7 features — it considers a random subset of them (scikit-learn's default for classification is √7 ≈ 2–3 features per split) and picks the best split among just that subset.
3. **Growing each tree**: each tree recursively splits the data on the feature/threshold that best separates the crop classes at that node (measured by Gini impurity, scikit-learn's default criterion), continuing until nodes are pure or a stopping condition is hit. Individually, a single deep decision tree like this tends to overfit badly.
4. **Aggregating predictions**: at prediction time, all 200 trees each "vote" on a crop class; the forest's final output is the **average of each tree's class-probability estimate** (this is what `predict_proba` returns, and it's what the app uses to rank the top-3 crops by confidence).

The two sources of randomness — bootstrap sampling and random feature subsetting — decorrelate the trees from each other. Individually noisy, overfitting-prone trees average out into a stable, well-generalizing forest. This is the core idea behind why Random Forests reliably outperform single decision trees.

### Why Random Forest

For this specific problem, Random Forest was chosen over alternatives for concrete reasons:

- **Non-linear, non-monotonic relationships**: crop suitability isn't a linear function of rainfall or pH — rice wants high rainfall and slight acidity, chickpea wants the opposite — and tree-based splits capture these "sweet spot" ranges naturally, where a linear model (e.g. logistic regression) would struggle without manual feature engineering.
- **No scaling/normalization required**: with 7 features on very different scales (pH ranges 3–10, rainfall ranges 20–300), Random Forest handles this natively.
- **Built-in feature importance**: the trained model directly exposes which of the 7 inputs matter most (see [Model Insights tab](#usage-guide) and the numbers below) — useful for explaining the system's reasoning, which a black-box model like a neural network would obscure.
- **Robust with a small, clean dataset**: 2,200 rows is small by deep-learning standards but is exactly the regime where Random Forest tends to shine — it doesn't need thousands of examples per class to generalize, unlike a neural network which would likely overfit or underperform here.
- **Well-calibrated probability output**: `predict_proba` gives usable confidence scores for the "top-3 recommended crops" feature, without needing a separate calibration step.

A neural network or gradient-boosted model (XGBoost/LightGBM) could likely match or marginally beat this accuracy, but would add complexity, dependencies, and reduced interpretability for a dataset this size and clean — not a good trade for a demonstration project that specifically wants to show *why* it's recommending what it recommends.

### Training Pipeline

Implemented in `model.py::train_model()`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
```

- **80/20 train/test split** (1,760 training rows, 440 test rows).
- **Stratified split** (`stratify=y`): guarantees each of the 22 crop classes is represented proportionally in both the training and test sets — important given the classes are small (100 samples each) and perfectly balanced to begin with.
- **200 trees** (`n_estimators=200`): a reasonably large forest for stability; increasing further gives diminishing returns on a dataset this size.
- **Fixed random seed** (`random_state=42`) for both the split and the forest, so results are reproducible run to run.
- Training happens once per Streamlit session and is cached via `@st.cache_resource`, so the model isn't retrained on every UI interaction.

### Model Performance

On the held-out 20% test set (440 rows never seen during training):

- **Test accuracy: 99.55%** (438 out of 440 correctly classified).
- **Feature importance ranking** (how much each feature contributes to the forest's splitting decisions, normalized to sum to 1):

  | Rank | Feature | Importance |
  |---|---|---|
  | 1 | Rainfall | 0.2196 |
  | 2 | Humidity | 0.2171 |
  | 3 | Potassium (K) | 0.1808 |
  | 4 | Phosphorus (P) | 0.1513 |
  | 5 | Nitrogen (N) | 0.1034 |
  | 6 | Temperature | 0.0755 |
  | 7 | Soil pH | 0.0523 |

  This matches agronomic intuition — rainfall and humidity are the two strongest differentiators between, say, rice (very high rainfall) and cotton (moderate rainfall, distinct climate band) — and is visible live in the app's **Model Insights** tab, along with a full 22×22 confusion matrix.

- The confusion matrix shows near-perfect diagonal classification, with only a couple of the visually and agronomically similar crop pairs occasionally confused (e.g. two pulse crops with overlapping NPK profiles).

**A note on the 99.5% figure**: this is a genuinely strong number, and it's strong *because* the dataset's classes are cleanly separated in feature space (this is a well-known, somewhat "easy" benchmark dataset) — not because of any special tuning. It should be read as "this dataset is highly learnable by tree-based models," not as a claim that real-world crop recommendation is this deterministic. Real soil/climate data is noisier than this dataset.

### Inference / Prediction

`model.py::predict_top_n()` takes the 7 current sensor values, runs them through the trained forest's `predict_proba`, and returns the top-3 crops ranked by predicted probability. The app displays the #1 result as the headline recommendation and all three as a horizontal confidence bar chart.

---

## The Rule-Based Agronomy Engine

Everything besides "which crop" is handled by transparent, hand-written rules in `agri_logic.py` — deliberately *not* by the ML model, so that this logic stays inspectable and easy to justify or adjust.

### Soil Health Index

A composite 0–100 score, not a scientific/industry-standard index, computed as a weighted average of how close each soil parameter is to an ideal midpoint:

- Nitrogen: ideal ≈ 80 kg/ha (weight 0.20)
- Phosphorus: ideal ≈ 50 kg/ha (weight 0.20)
- Potassium: ideal ≈ 50 kg/ha (weight 0.20)
- pH: ideal ≈ 6.5 (weight 0.25 — pH gets the highest weight since extreme pH affects nutrient uptake regardless of NPK levels)
- Humidity (used as a moisture proxy): ideal ≈ 65% (weight 0.15)

Each parameter's deviation from its ideal is converted into a 0–1 "goodness" score (`1 − distance/scale`, clipped at 0), then combined via the weights above and scaled to 0–100. Scores ≥70 are labelled "good," 45–69 "moderate," below 45 "poor."

### Irrigation Advisory

A simple three-way rule based on current rainfall, humidity, and temperature:

- **Not needed**: rainfall > 150mm OR humidity > 80% (the soil/air is already wet enough).
- **Now**: rainfall < 60mm AND humidity < 50% AND temperature > 28°C (hot, dry conditions with no recent rain).
- **Soon** (default): anything in between — monitor and irrigate within 2 days.

### Fertilizer Advisory

This is the one piece of advice that's genuinely **data-driven rather than hand-picked**: for the crop the model just recommended, the app computes that crop's actual mean and standard deviation for N, P, and K *from the training dataset itself* (`model.py::crop_nutrient_stats()`, grouping all 100 samples of that crop). The farmer's current NPK readings are then compared against a band of `mean ± 0.75×std`:

- Below the band → **"Low — apply more."**
- Within the band → **"Adequate."**
- Above the band → **"High — reduce application."**

This means the fertilizer advice for, say, rice, is grounded in what the 100 real rice samples in the dataset actually looked like — not a static lookup table that could go stale or be wrong for a specific crop variety.

### Disease & pH Alerts

Two independent heuristic checks:

- **Pest/disease risk**: triggered when temperature > 30°C *and* humidity > 75% simultaneously — the classic hot-and-humid combination that favors fungal and insect pressure in most Indian field crops.
- **Extreme pH**: triggered below pH 5.5 (acidic — suggests liming) or above pH 8.0 (alkaline — suggests organic matter/gypsum).

### Yield Outlook

An explicitly **indicative-only** proxy (not a trained model, and clearly labelled as such in the UI and here): "favourable" if the Soil Health Index is ≥70 and irrigation isn't urgently needed, "moderate" if SHI ≥45, otherwise "sub-optimal." This exists to complete the advisory picture but should not be read as a real yield forecast — a genuine yield model would need a dataset with actual historical yield outcomes (area, season, weather history, harvest quantity), which this dataset does not contain.

---

## Multilingual System

`translations.py` holds a single `TR` dictionary keyed by language code, with **13 languages fully populated**: English, Bangla, Assamese, Hindi, Marathi, Konkani, Kannada, Gujarati, Punjabi, Odia, Tamil, Telugu, and Malayalam. Every language has the exact same set of translation keys (verified programmatically — nothing silently falls back to English).

Three extra layers of localization beyond UI text:

- **`CROP_INFO`**: everyday local names for all 22 crops in every language (e.g. mung bean → মুগ ডাল / मूग / ಹೆಸರುಕಾಳು / பாசிப்பயறு), used everywhere a crop name is displayed — recommendation, chart labels, fertilizer tab, confusion matrix axes, and the generated SMS/voice text.
- **`NUMERAL_GROUPS` / `local_number()`**: converts Western digits into the correct native numeral script — Bengali-Assamese, Devanagari (Hindi/Marathi/Konkani), Gurmukhi (Punjabi), Gujarati, Odia, Tamil, Telugu, Kannada, and Malayalam digit sets.
- **`format_number()`**: rounds any numeric value to 2 decimal places and renders it through `local_number()` — applied consistently across every number shown in the app (confidence %, Soil Health Index, NPK readings, model accuracy, chart value labels).

One honest limitation: Streamlit's native slider widgets are browser components and always render their handle value in Western digits — that's outside what the Python layer can override. A caption underneath the manual-input sliders mirrors the same values in the selected language's script as a workaround.

**Translation quality note**: Bangla, Assamese, Hindi, and Marathi strings have had more scrutiny during development. Konkani, Kannada, Gujarati, Punjabi, Odia, Tamil, Telugu, and Malayalam use standard agricultural vocabulary but have not been checked by a native speaker of each — worth a review pass before using this with real farmers or in a formal context.

## Live Crop Photos

Rather than hardcoding 22 image URLs (which would go stale or be wrong), `crop_media.py` fetches a representative photo of the recommended crop **live, at runtime**, from Wikipedia's public REST summary API (`https://en.wikipedia.org/api/rest_v1/page/summary/{title}`) — a stable, purpose-built, CORS-friendly endpoint designed for exactly this kind of thumbnail hotlinking. The result is cached for 24 hours (`@st.cache_data(ttl=86400)`) to avoid repeat network calls. If the request fails for any reason (no internet, page has no image), the app falls back to a large crop-appropriate emoji so the interface never breaks.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI framework | [Streamlit](https://streamlit.io/) |
| ML | scikit-learn (`RandomForestClassifier`) |
| Data handling | pandas, NumPy |
| Charts | Plotly (`plotly.express`, `plotly.graph_objects`) |
| Live image fetch | `requests` → Wikipedia REST API |
| Language | Python 3.9+ |

---

## Installation & Running

### Prerequisites

- Python 3.9 or newer
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/krixiproigyan.git
cd krixiproigyan

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Streamlit will start a local server and print a URL (typically `http://localhost:8501`) — open it in your browser. No API keys, external accounts, or hardware are required. An internet connection is only needed for the live crop-photo feature; everything else (model training, recommendations, advisory logic, translations) runs fully offline.

### Troubleshooting

- **`ModuleNotFoundError`**: make sure you activated the virtual environment and ran `pip install -r requirements.txt` inside it.
- **Port already in use**: run `streamlit run app.py --server.port 8502` (or any free port).
- **No crop photo showing**: this means the live Wikipedia fetch failed (usually no internet access) — the app will show an emoji instead and continue working normally.

---

## Usage Guide

1. **Sidebar** — pick a language, a state/region, and how to provide soil/climate data: either move the sliders yourself, or click "Simulate IoT sensor reading" to draw a random realistic sample (standing in for a real sensor, since no physical hardware is wired up in this demo).
2. **🌱 Crop Recommendation tab** — the top recommended crop, its confidence score, a live photo, an indicative yield outlook, and a bar chart of the top 3 candidate crops.
3. **💧 Soil Health & Irrigation tab** — a gauge showing the Soil Health Index, and an irrigation recommendation (now / soon / not needed).
4. **🧪 Fertilizer Advisory tab** — current N/P/K readings compared against the recommended crop's data-driven ideal range, with a low/adequate/high verdict for each.
5. **📋 Farmer Advisory tab** — any active early-warning alerts, plus auto-generated SMS-style and voice-assistant-style summary text in the selected language.
6. **📊 Model Insights tab** — the model's test accuracy, a feature importance chart, and the full confusion matrix, all with localized crop names and numerals.

---

## Dataset License & Attribution

The training dataset (`data/crop_recommendation.csv`) originates from the **"Crop Recommendation Dataset" by Atharva Ingle**, published on Kaggle, built by augmenting Indian rainfall, climate, and fertilizer data. It is widely redistributed for this exact use case. If you plan to redistribute this repository, please retain this attribution and check the current licensing terms on the dataset's Kaggle page before commercial use.

---

## Known Limitations

Being upfront about these matters for anyone evaluating or building on this project:

- **IoT layer is simulated.** There is no physical sensor integration. "Simulate IoT sensor reading" draws a random real row from the training dataset — it demonstrates the software pipeline end-to-end but is not connected to any hardware.
- **Yield outlook is indicative only**, built from a simple heuristic (Soil Health Index + irrigation status), not a trained model — no yield-labeled dataset was used or is available here.
- **Fertilizer/irrigation/disease logic is rule-based, not learned.** This is a deliberate design choice for transparency, not a limitation of available techniques — but it means these rules reflect simplified agronomic heuristics, not field-validated recommendations, and should not be used as-is for real farming decisions without expert review.
- **Non-native-reviewed translations** for 8 of the 12 Indian languages (see [Multilingual System](#multilingual-system)).
- **Voice output is text only** — the "Voice Assistant Script" tab shows the text a TTS engine would speak; no actual text-to-speech is wired up.
- **Streamlit sliders can't be localized to native digits** — a browser-widget constraint, worked around with a caption below the sliders.
- **The 99.5% model accuracy reflects a clean, well-separated benchmark dataset** — real-world field data would likely be noisier and yield lower accuracy.

## Roadmap

- [ ] Real sensor integration (ESP8266/ESP32 + soil NPK sensor + DHT22, reporting over MQTT or serial)
- [ ] A genuine yield-prediction model, once a yield-labeled dataset (area, season, historical output) is available
- [ ] Native-speaker review of Konkani, Kannada, Gujarati, Punjabi, Odia, Tamil, Telugu, and Malayalam strings
- [ ] Real text-to-speech for the voice advisory script, using a regional-language TTS engine
- [ ] More languages (e.g. Urdu, Nepali, Sindhi, Sanskrit — the numeral/translation system is built to extend easily)
- [ ] District/state-level soil baseline calibration instead of the current single global ideal range

## Contributing

This is an open-source demonstration project — contributions, corrections (especially translation reviews), and issues are welcome. If you're extending this:

- Follow the existing file boundaries: ML logic in `model.py`, rule-based advisory in `agri_logic.py`, translations in `translations.py`, media fetching in `crop_media.py`, and UI/orchestration in `app.py`.
- To add a new language, add one key block to `TR` in `translations.py` with the exact same keys as the `"en"` block, add local crop names to `CROP_INFO`, and — if the language uses non-Latin digits — add its numeral script to `NUMERAL_GROUPS`.
- Please open an issue before submitting large structural changes.

## License

Choose and add a license file appropriate for your use (e.g. MIT) before publishing — none is currently specified in this repository. The bundled dataset has its own attribution requirements; see [Dataset License & Attribution](#dataset-license--attribution) above.
