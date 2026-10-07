"""
Serveur de prédiction : lit /log dans Firebase, prédit la température et
l'humidité des prochaines minutes, écrit le résultat dans /predictions.
Utilise la clé de compte de service (serviceAccountKey.json) : ne la partagez jamais.
"""
import time
import numpy as np
import firebase_admin
from firebase_admin import credentials, db

KEY_FILE = "serviceAccountKey.json"
DATABASE_URL = "https://projetiot-e1bdb-default-rtdb.europe-west1.firebasedatabase.app"
STEPS = 5          # nombre de mesures futures prédites
MIN_POINTS = 5     # minimum de mesures dans /log pour prédire
EVERY = 30         # secondes entre deux prédictions

firebase_admin.initialize_app(credentials.Certificate(KEY_FILE), {"databaseURL": DATABASE_URL})


def read_log():
    """Retourne les mesures de /log triées par temps (liste ou dict selon Firebase)."""
    raw = db.reference("/log").get()
    if not raw:
        return []
    items = raw.values() if isinstance(raw, dict) else raw
    rows = [r for r in items if r and "ts" in r and "temperature" in r and "humidity" in r]
    return sorted(rows, key=lambda r: r["ts"])


def predict(ts, values, future_ts, lo=None, hi=None):
    """Régression linéaire sur les mesures (temps en minutes) puis extrapolation."""
    x = (np.array(ts) - ts[0]) / 60000.0                 # minutes depuis la 1re mesure
    slope, intercept = np.polyfit(x, values, 1)
    fx = (np.array(future_ts) - ts[0]) / 60000.0
    pred = slope * fx + intercept
    if lo is not None:
        pred = np.clip(pred, lo, hi)
    return [round(float(p), 1) for p in pred], float(slope)


def trend(slope_per_min):
    if slope_per_min > 0.05:
        return "hausse"
    if slope_per_min < -0.05:
        return "baisse"
    return "stable"


def run_once():
    rows = read_log()
    if len(rows) < MIN_POINTS:
        print(f"[Prédiction] {len(rows)} mesures seulement, il en faut {MIN_POINTS}.")
        return
    ts = [r["ts"] for r in rows]
    step_ms = int(np.median(np.diff(ts)))                # intervalle réel entre mesures
    future = [ts[-1] + step_ms * (i + 1) for i in range(STEPS)]

    t_pred, t_slope = predict(ts, [r["temperature"] for r in rows], future)
    h_pred, h_slope = predict(ts, [r["humidity"] for r in rows], future, 0, 100)

    result = {
        "updated": int(time.time() * 1000),
        "based_on": len(rows),
        "temperature": {"trend": trend(t_slope), "per_min": round(t_slope, 2)},
        "humidity": {"trend": trend(h_slope), "per_min": round(h_slope, 2)},
        "points": [{"ts": f, "temperature": t, "humidity": h}
                   for f, t, h in zip(future, t_pred, h_pred)],
    }
    db.reference("/predictions").set(result)
    print(f"[Prédiction] T → {t_pred[-1]} °C ({result['temperature']['trend']}) | "
          f"H → {h_pred[-1]} % ({result['humidity']['trend']})")


if __name__ == "__main__":
    print("Serveur de prédiction démarré (Ctrl+C pour arrêter)")
    while True:
        try:
            run_once()
        except Exception as e:
            print("[Erreur]", e)
        time.sleep(EVERY)
