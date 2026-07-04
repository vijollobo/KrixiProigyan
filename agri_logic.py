"""
Rule-based agronomy logic layer.
This merges the "Smart Irrigation", "Fertilizer Advisory", "Soil Health Index"
and "Early Warning" features described across the original KrixiProigyan deck
into one consistent module, operating on the same sensor inputs used by the
ML crop-recommendation model.
"""

import numpy as np


def soil_health_index(n, p, k, ph, moisture_proxy_humidity):
    """
    A simple composite Soil Health Index (0-100), inspired by the deck's
    'Soil Health Index (SHI): derived from pH, NPK, and moisture' feature.
    Not a scientific standard — a transparent, explainable demo heuristic.
    """
    # Normalize each factor to a 0-1 "goodness" score against ideal Indian topsoil ranges
    n_score = 1 - min(abs(n - 80) / 80, 1)         # ideal ~ 60-100 kg/ha
    p_score = 1 - min(abs(p - 50) / 60, 1)         # ideal ~ 30-70 kg/ha
    k_score = 1 - min(abs(k - 50) / 60, 1)         # ideal ~ 30-70 kg/ha
    ph_score = 1 - min(abs(ph - 6.5) / 3.0, 1)     # ideal ~ 6.0-7.0
    moisture_score = 1 - min(abs(moisture_proxy_humidity - 65) / 65, 1)

    weights = {"n": 0.2, "p": 0.2, "k": 0.2, "ph": 0.25, "moisture": 0.15}
    score = (
        n_score * weights["n"]
        + p_score * weights["p"]
        + k_score * weights["k"]
        + ph_score * weights["ph"]
        + moisture_score * weights["moisture"]
    )
    return round(score * 100, 1)


def soil_health_label(score):
    if score >= 70:
        return "good"
    elif score >= 45:
        return "moderate"
    return "poor"


def irrigation_advice(humidity, rainfall, temperature):
    """
    Rule-based smart-irrigation recommendation, standing in for the deck's
    'LoRaWAN-enabled smart valves' + 'Rainfall Forecast Integration' logic.
    """
    if rainfall > 150 or humidity > 80:
        return "not_needed"
    if rainfall < 60 and humidity < 50 and temperature > 28:
        return "now"
    return "soon"


def fertilizer_advice(n, p, k, crop_stats):
    """
    Compares current NPK values against the data-driven ideal range
    (mean ± std) for the recommended crop, learned from the training data
    itself — this replaces the original static rule tables with values
    the model actually learned.
    crop_stats: dict with 'N', 'P', 'K' each -> (mean, std)
    """
    advice = {}
    for nutrient, value in [("N", n), ("P", p), ("K", k)]:
        mean, std = crop_stats[nutrient]
        low = mean - 0.75 * std
        high = mean + 0.75 * std
        if value < low:
            advice[nutrient] = "low"
        elif value > high:
            advice[nutrient] = "high"
        else:
            advice[nutrient] = "ok"
    return advice


def disease_risk_alerts(temperature, humidity, ph):
    alerts = []
    if temperature > 30 and humidity > 75:
        alerts.append("disease")
    if ph < 5.5:
        alerts.append("ph_low")
    if ph > 8.0:
        alerts.append("ph_high")
    return alerts


def yield_outlook(soil_score, irrigation_status):
    """Lightweight indicative yield-outlook proxy (no real yield dataset used)."""
    if soil_score >= 70 and irrigation_status != "now":
        return "high"
    if soil_score >= 45:
        return "medium"
    return "low"
