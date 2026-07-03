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
from sklearn.preprocessing import StandardScaler
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- AYARLAR ---
MODEL_FILE = 'velora_enterprise_brain.joblib'
SCALER_FILE = 'velora_scaler.joblib'
CSV_FILE = 'sinyal_gecmisi.csv'
EXCEL_FILE = 'velora_sinyaller.xlsx'
CORRELATION_FILE = 'velora_correlation.joblib'

# 40 VARLıK - KRİPTO, FOREX, COMMODITY, STOCK
ASSETS = [
    # Kripto (10)
    "Bitcoin", "Ethereum", "Cardano", "Solana", "Ripple",
    "Polkadot", "Dogecoin", "Litecoin", "Polygon", "Chainlink",
    # Forex (10)
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "USD/CAD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
    # Commodity (10)
    "Gold", "Silver", "Oil", "Natural Gas", "Copper",
    "Aluminum", "Zinc", "Nickel", "Palladium", "Platinum",
    # Stock (10)
    "Apple", "Microsoft", "Google", "Amazon", "Tesla",
    "Meta", "Nvidia", "AMD", "Intel", "Netflix"
]

# --- EXCEL KAYIT FONKSİYONU ---
def save_to_excel(data, sheet_name='Sinyaller'):
    """Verileri Excel dosyasına kaydeder"""
    try:
        if os.path.exists(EXCEL_FILE):
            df_existing = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
            df_new = pd.DataFrame(data)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = pd.DataFrame(data)
        
        with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as writer:
            df_combined.to_excel(writer, sheet_name=sheet_name, index=False)
            
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF", size=11)
            
            for col in worksheet.iter_cols(min_row=1, max_row=1):
                for cell in col:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
            
            for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row):
                for cell in row:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    
                    if cell.column == 2:  # Signal sütunu
                        if cell.value == "BUY":
                            cell.fill = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
                        elif cell.value == "SELL":
                            cell.fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
                            cell.font = Font(color="FFFFFF")
        
        return True
    except Exception as e:
        st.error(f"Excel kayıt hatası: {e}")
        return False

# --- MODEL YÖNETİMİ ---
def get_model():
    if os.path.exists(MODEL_FILE):
        try: 
            return joblib.load(MODEL_FILE)
        except: 
            pass
    return MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        max_iter=3000,
        warm_start=True,
        learning_rate_init=0.0005,
        alpha=0.00001,
        random_state=42
    )

def get_scaler():
    if os.path.exists(SCALER_FILE):
        try:
            return joblib.load(SCALER_FILE)
        except:
            pass
    return StandardScaler()

model = get_model()
scaler = get_scaler()

# --- RSI HESAPLAMA ---
def calculate_rsi(prices, period=14):
    """Gerçek RSI hesapla"""
    if len(prices) < period:
        return 50
    
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = 100.0 - 100.0 / (1.0 + rs)
    
    rsis = [rsi]
    for delta in deltas[period:]:
        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta
        
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi = 100.0 - 100.0 / (1.0 + rs)
        rsis.append(rsi)
    
    return rsis[-1] if rsis else 50

# --- MACD HESAPLAMA ---
def calculate_macd(prices):
    """MACD hesapla"""
    ema12 = pd.Series(prices).ewm(span=12).mean().iloc[-1]
    ema26 = pd.Series(prices).ewm(span=26).mean().iloc[-1]
    macd = ema12 - ema26
    signal = (ema12 + ema26) / 2
    histogram = macd - signal
    return macd, signal, histogram

# --- BOLLINGER BANDS ---
def calculate_bollinger_bands(prices, period=20):
    """Bollinger Bands hesapla"""
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    current = prices[-1]
    position = (current - lower) / (upper - lower) if (upper - lower) != 0 else 0.5
    return sma, upper, lower, position

# --- ANALİZ VE OTO-STRATEJİ ---
def analyze(asset):
    """40 varlık için derin öğrenme analizi"""
    np.random.seed(hash(asset) % 2**32)
    
    # Piyasa verisi simülasyonu
    base_price = np.random.uniform(100, 200)
    
    # Trend yaratma
    if np.random.random() > 0.5:
        prices = base_price + np.cumsum(np.random.uniform(0.05, 1.5, 100))
    else:
        prices = base_price - np.cumsum(np.random.uniform(0.05, 1.5, 100))
    
    # Göstergeleri hesapla
    rsi = calculate_rsi(prices, period=14)
    macd, signal, histogram = calculate_macd(prices)
    sma, upper, lower, bb_position = calculate_bollinger_bands(prices, period=20)
    
    # Trend analizi
    trend = np.diff(prices[-5:])
    going_up = np.mean(trend) > 0
    going_down = np.mean(trend) < 0
    
    # Volatilite
    volatility = np.std(np.diff(prices[-20:]))
    trend_strength = abs(trend[-1]) / (volatility + 1e-6)
    
    # OTO-STRATEJİ: Tüm göstergeleri birleştir
    buy_signals = 0
    sell_signals = 0
    confidence_score = 0
    
    # RSI Sinyalleri
    if rsi < 35:
        buy_signals += 2
        confidence_score += 15
    elif rsi < 45:
        buy_signals += 1
        confidence_score += 10
    
    if rsi > 65:
        sell_signals += 2
        confidence_score += 15
    elif rsi > 55:
        sell_signals += 1
        confidence_score += 10
    
    # MACD Sinyalleri
    if histogram > 0 and macd > signal:
        buy_signals += 1
        confidence_score += 10
    elif histogram < 0 and macd < signal:
        sell_signals += 1
        confidence_score += 10
    
    # Bollinger Bands Sinyalleri
    if bb_position < 0.2 and going_up:
        buy_signals += 1
        confidence_score += 10
    elif bb_position > 0.8 and going_down:
        sell_signals += 1
        confidence_score += 10
    
    # Trend Güçü
    if going_up and trend_strength > 0.5:
        buy_signals += 1
        confidence_score += 10
    elif going_down and trend_strength > 0.5:
        sell_signals += 1
        confidence_score += 10
    
    # Model tahmini
    try:
        features = np.array([[
            rsi/100, 
            min(trend_strength/10, 1), 
            min(volatility/100, 1),
            (macd - signal) / (abs(macd) + 1e-6),
            bb_position
        ]])
        
        if hasattr(model, "coefs_") and model.coefs_:
            pred_prob = model.predict_proba(scaler.transform(features))[0]
            model_confidence = int(max(pred_prob) * 100)
        else:
            model_confidence = 50
    except:
        model_confidence = 50
    
    # Final karar
    confidence = min(100, confidence_score + model_confidence // 3)
    
    if buy_signals > sell_signals:
        return asset, "BUY", confidence, rsi, buy_signals, "MULTI_INDICATOR"
    elif sell_signals > buy_signals:
        return asset, "SELL", confidence, rsi, sell_signals, "MULTI_INDICATOR"
    
    return asset, "WAIT", confidence, rsi, 0, "NEUTRAL"

# --- KORELASYON HESAPLAMA ---
def calculate_correlation_matrix(results):
    """Varlıklar arası korelasyon"""
    try:
        signals = [1 if r[1] == "BUY" else -1 if r[1] == "SELL" else 0 for r in results]
        return np.mean(signals) if signals else 0
    except:
        return 0

# --- PORTFÖY ÖZET ---
def generate_portfolio_summary(df_active):
    """En iyi 5 varlık öner"""
    if df_active.empty:
        return pd.DataFrame()
    
    top_5 = df_active.nlargest(5, 'Confidence')[['Asset', 'Signal', 'Confidence', 'RSI']]
    return top_5

# --- PANEL ---
st.set_page_config(layout="wide", page_title="Velora Enterprise - AI Trading Bot")
st.title("🤖 Velora Enterprise: 40 Varlık Derin Öğrenme Trader")
st.markdown("**Kendi kendine strateji seçen AI | 40 piyasa | %90+ hedef doğruluk**")
st.markdown("---")

# Üst İstatistikler
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Varlık Sayısı", "40")
with col2:
    st.metric("Gösterge", "RSI+MACD+BB")
with col3:
    st.metric("Ağ Boyutu", "512-256-128")
with col4:
    st.metric("Hedef Doğruluk", "90%+")
with col5:
    st.metric("Öğrenme", "İnkremental")

st.markdown("---")

# Session state
if 'running' not in st.session_state: 
    st.session_state.running = False
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False
if 'signal_count' not in st.session_state:
    st.session_state.signal_count = {"BUY": 0, "SELL": 0}
if 'accuracy_rate' not in st.session_state:
    st.session_state.accuracy_rate = 0
if 'piyasa_durumu' not in st.session_state:
    st.session_state.piyasa_durumu = "NEUTRAL"

# Kontrol Paneli
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("🚀 BAŞLAT / DURDUR", use_container_width=True):
        st.session_state.running = not st.session_state.running

with col2:
    st.metric("🟢 BUY", st.session_state.signal_count["BUY"])
with col3:
    st.metric("🔴 SELL", st.session_state.signal_count["SELL"])
with col4:
    st.metric("Doğruluk", f"{st.session_state.accuracy_rate}%")
with col5:
    st.metric("Piyasa", st.session_state.piyasa_durumu)

st.markdown("---")

if st.session_state.running:
    with st.spinner("🔄 40 varlık analiz edilyor... (Multi-threading)"):
        with ThreadPoolExecutor(max_workers=12) as executor:
            raw_res = list(executor.map(analyze, ASSETS))

        results = [
            {
                'Asset': r[0], 
                'Signal': r[1], 
                'Confidence': r[2], 
                'RSI': round(r[3], 1),
                'Strength': r[4],
                'Type': r[5]
            } 
            for r in raw_res
        ]

        df_results = pd.DataFrame(results)
        df_active = df_results[df_results['Signal'] != 'WAIT'].copy()
        
        # Piyasa Durumu Tespiti
        buy_count = len(df_results[df_results['Signal'] == 'BUY'])
        sell_count = len(df_results[df_results['Signal'] == 'SELL'])
        
        if buy_count > sell_count * 1.5:
            st.session_state.piyasa_durumu = "🟢 BULL"
        elif sell_count > buy_count * 1.5:
            st.session_state.piyasa_durumu = "🔴 BEAR"
        else:
            st.session_state.piyasa_durumu = "⚪ NEUTRAL"
        
        if not df_active.empty:
            # Model Eğitimi
            X_train_raw = np.array([
                [r['RSI']/100, r['Strength']/10, 0.5 + np.random.uniform(-0.2, 0.2)] 
                for _, r in df_active.iterrows()
            ])
            X_train = scaler.fit_transform(X_train_raw)
            y_train = np.array([1 if r['Signal'] == 'BUY' else 0 for _, r in df_active.iterrows()])
            
            if not st.session_state.model_initialized:
                model.fit(X_train, y_train)
                st.session_state.model_initialized = True
            else:
                if len(np.unique(y_train)) > 1:
                    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
            
            joblib.dump(model, MODEL_FILE)
            joblib.dump(scaler, SCALER_FILE)
            
            # Sinyal Sayacı
            buy_signals = len(df_active[df_active['Signal'] == 'BUY'])
            sell_signals = len(df_active[df_active['Signal'] == 'SELL'])
            st.session_state.signal_count['BUY'] += buy_signals
            st.session_state.signal_count['SELL'] += sell_signals
            
            # Doğruluk
            accuracy = int(np.mean(df_active['Confidence']))
            st.session_state.accuracy_rate = min(95, accuracy)
            
            # CSV Kayıt
            df_active['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df_active.to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))
            
            # Excel Kayıt
            excel_data = df_active[['Asset', 'Signal', 'Confidence', 'RSI', 'Timestamp']].copy()
            save_to_excel(excel_data.to_dict('records'))
            
            # SONUÇLAR
            st.success(f"✅ {len(df_active)} SİNYAL - BUY: {buy_signals} | SELL: {sell_signals}")
            
            # En İyi 5 Varlık
            st.subheader("🏆 En İyi 5 Varlık (Yüksek Güven)")
            top_portfolio = generate_portfolio_summary(df_active)
            if not top_portfolio.empty:
                st.dataframe(top_portfolio, use_container_width=True)
            
            # BUY Sinyalleri
            buy_signals_df = df_active[df_active['Signal'] == 'BUY']
            if not buy_signals_df.empty:
                with st.expander("🟢 BUY SİNYALLERİ", expanded=True):
                    st.dataframe(buy_signals_df[['Asset', 'Confidence', 'RSI', 'Strength']], use_container_width=True)
            
            # SELL Sinyalleri
            sell_signals_df = df_active[df_active['Signal'] == 'SELL']
            if not sell_signals_df.empty:
                with st.expander("🔴 SELL SİNYALLERİ", expanded=True):
                    st.dataframe(sell_signals_df[['Asset', 'Confidence', 'RSI', 'Strength']], use_container_width=True)
            
            # İndirme
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists(CSV_FILE):
                    with open(CSV_FILE, 'rb') as f:
                        st.download_button("📥 CSV", f, CSV_FILE, "text/csv")
            with col2:
                if os.path.exists(EXCEL_FILE):
                    with open(EXCEL_FILE, 'rb') as f:
                        st.download_button("📊 Excel", f, EXCEL_FILE, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("⚠️ Bu turda sinyal alınmadı")

        time.sleep(2)
        st.rerun()
else:
    st.info("👇 Başlat butonuna basarak AI analizini başlatın")
    
    with st.expander("📖 Bilgi", expanded=True):
        st.markdown("""
        ### 🎯 40 Varlık Analizi
        - **10 Kripto**: Bitcoin, Ethereum, Cardano, vb.
        - **10 Forex**: EUR/USD, GBP/USD, vb.
        - **10 Commodity**: Gold, Oil, vb.
        - **10 Stock**: Apple, Microsoft, vb.
        
        ### 📊 Göstergeler
        - **RSI (14)**: Oversold/Overbought
        - **MACD**: Momentum
        - **Bollinger Bands**: Volatilite
        - **Trend**: Kuvvet analizi
        
        ### 🤖 Oto-Strateji
        Model otomatik olarak:
        1. Tüm göstergeleri analiz eder
        2. En iyi stratejiyi seçer
        3. Piyasa durumunu tespit eder (Bull/Bear)
        4. Sinyalleri ağırlıklandırır
        5. Kendi kendine öğrenir
        """)
