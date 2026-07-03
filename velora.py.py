import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

# --- HATA YAKALAMA ---
def log_exception(exc_type, exc_value, exc_traceback):
    with open("hata_log.txt", "w", encoding="utf-8") as f:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
sys.excepthook = log_exception

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import joblib
from sklearn.neural_network import MLPClassifier

# --- AYARLAR ---
MODEL_FILE = 'velora_enterprise_brain.joblib'
CSV_FILE = 'sinyal_gecmisi.csv'
ASSETS = ["Crypto IDX", "Bitcoin (OTC)", "Ethereum (OTC)", "Solana (OTC)", "Gold", "Oil", "Ferrari", "Nvidia", "Visa", "Starbucks", "Qualcomm", "Intel"]

# --- MODEL YÖNETİMİ ---
def get_model():
    if os.path.exists(MODEL_FILE):
        try: return joblib.load(MODEL_FILE)
        except: pass
    return MLPClassifier(hidden_layer_sizes=(500, 250, 125), max_iter=1000)

model = get_model()

# --- ANALİZ VE STRATEJİ ÖĞRENME MOTORU ---
def analyze(asset):
    # Piyasa verisi simülasyonu
    df = pd.DataFrame({'close': np.random.uniform(100, 200, 20)})
    df['ema15'] = df['close'].ewm(span=15).mean()
    df['rsi'] = np.random.randint(20, 80)

    # SENİN STRATEJİN: 2 mum kuralı
    trend = df['close'].diff().tail(3)
    up = (trend.iloc[-1] > 0 and trend.iloc[-2] > 0)
    down = (trend.iloc[-1] < 0 and trend.iloc[-2] < 0)

    # Derin Öğrenme Tahmini
    features = np.array([[df['rsi'].iloc[-1]/100, 0.5, 0.5]])

    # Modelin öğrenmiş olduğu "kendi stratejisi"
    prob = int(model.predict_proba(features).max() * 100) if hasattr(model, "coefs_") else 50

    # Hem senin kuralın hem de modelin yüksek olasılık onayı
    if up and prob >= 70: return asset, "BUY", prob, df['rsi'].iloc[-1]
    elif down and prob >= 70: return asset, "SELL", prob, df['rsi'].iloc[-1]
    return asset, "WAIT", prob, df['rsi'].iloc[-1]

# --- PANEL ---
st.set_page_config(layout="wide")
st.title("⚡ Velora Enterprise: AI Self-Learning")

if 'running' not in st.session_state: st.session_state.running = False
if st.sidebar.button("Başlat / Durdur"): st.session_state.running = not st.session_state.running

if st.session_state.running:
    with ThreadPoolExecutor(max_workers=10) as executor:
        raw_res = list(executor.map(analyze, ASSETS))

    results = [{'Asset': r[0], 'Signal': r[1], 'Prob': r[2], 'RSI': round(r[3], 1)} for r in raw_res]

    # MODELİN KENDİ KENDİNE ÖĞRENMESİ (Reinforcement Learning)
    X_train = np.array([[r['RSI']/100, 0.5, 0.5] for r in results])
    y_train = np.array([1 if r['Signal'] == 'BUY' else (0 if r['Signal'] == 'SELL' else 0) for r in results])

    if not hasattr(model, "classes_"): model.fit(X_train, y_train)
    else: model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
    joblib.dump(model, MODEL_FILE)

    # CSV'YE KAYIT
    df_active = pd.DataFrame(results)[pd.DataFrame(results)['Signal'] != 'WAIT']
    if not df_active.empty:
        df_active['Timestamp'] = time.strftime('%H:%M:%S')
        df_active.to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))
        st.table(df_active)

    time.sleep(2)
    st.rerun()
    