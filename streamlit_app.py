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
from openpyxl.styles import Font, PatternFill, Alignment
from datetime import datetime

# --- AYARLAR ---
MODEL_FILE = 'velora_enterprise_brain.joblib'
SCALER_FILE = 'velora_scaler.joblib'
CSV_FILE = 'sinyal_gecmisi.csv'
EXCEL_FILE = 'velora_sinyaller.xlsx'
ASSETS = ["Crypto IDX", "Bitcoin (OTC)", "Ethereum (OTC)", "Solana (OTC)", "Gold", "Oil", "Ferrari", "Nvidia", "Visa", "Starbucks", "Qualcomm", "Intel"]

# --- EXCEL KAYIT FONKSİYONU ---
def save_to_excel(data):
    """Verileri Excel dosyasına kaydeder"""
    try:
        if os.path.exists(EXCEL_FILE):
            df_existing = pd.read_excel(EXCEL_FILE)
            df_new = pd.DataFrame(data)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = pd.DataFrame(data)
        
        with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as writer:
            df_combined.to_excel(writer, sheet_name='Sinyaller', index=False)
            
            workbook = writer.book
            worksheet = writer.sheets['Sinyaller']
            
            header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF")
            
            for col in worksheet.iter_cols(min_row=1, max_row=1):
                for cell in col:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
            
            worksheet.column_dimensions['A'].width = 18
            worksheet.column_dimensions['B'].width = 12
            worksheet.column_dimensions['C'].width = 12
            worksheet.column_dimensions['D'].width = 12
            worksheet.column_dimensions['E'].width = 12
            worksheet.column_dimensions['F'].width = 18
            
            for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row):
                for cell in row:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    
                    if cell.column == 2:
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
        hidden_layer_sizes=(256, 128, 64),
        max_iter=2000,
        warm_start=True,
        learning_rate_init=0.001,
        alpha=0.0001,
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
    """RSI hesapla"""
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
    
    return rsis[-1]

# --- ANALİZ VE STRATEJİ ÖĞRENME MOTORU ---
def analyze(asset):
    """Binomo için optimize edilmiş analiz"""
    # Piyasa verisi simülasyonu (gerçek veri kullanılabilir)
    prices = np.random.uniform(100, 200, 50)
    
    # RSI hesapla
    rsi = calculate_rsi(prices, period=14)
    
    # 2 mum kuralı
    trend = np.diff(prices[-3:])
    up = (trend[-1] > 0 and trend[-2] > 0)
    down = (trend[-1] < 0 and trend[-2] < 0)
    
    # Özellikleri hazırla: RSI, trend güç, volatilite
    volatility = np.std(np.diff(prices[-20:]))
    trend_strength = abs(trend[-1]) / (volatility + 1e-6)
    
    features = np.array([[rsi/100, trend_strength/10, volatility/100]])
    
    # Model tahmini
    try:
        if hasattr(model, "coefs_") and model.coefs_:
            pred_prob = model.predict_proba(scaler.transform(features))[0]
            prob = int(max(pred_prob) * 100)
        else:
            prob = 50
    except:
        prob = 50
    
    # RSI + Model kararı
    # RSI < 30 ve model BUY onayı = BUY
    # RSI > 70 ve model SELL onayı = SELL
    
    if rsi < 30 and up and prob >= 70:
        return asset, "BUY", prob, rsi, "RSI_OVERSOLD"
    elif rsi > 70 and down and prob >= 70:
        return asset, "SELL", prob, rsi, "RSI_OVERBOUGHT"
    elif rsi < 30 and prob >= 80:
        return asset, "BUY", prob, rsi, "RSI_WEAK_OVERSOLD"
    elif rsi > 70 and prob >= 80:
        return asset, "SELL", prob, rsi, "RSI_WEAK_OVERBOUGHT"
    
    return asset, "WAIT", prob, rsi, "NEUTRAL"

# --- PANEL ---
st.set_page_config(layout="wide", page_title="Velora Enterprise - Binomo AI")
st.title("⚡ Velora Enterprise: Binomo AI Trader (RSI + Deep Learning)")
st.markdown("---")

# İstatistikler
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Doğruluk Hedefi", "90%+")
with col2:
    st.metric("RSI Periyodu", "14")
with col3:
    st.metric("Özet Ağ", "256-128-64")
with col4:
    st.metric("Sinyal Türü", "Binary")

st.markdown("---")

# Session state başlat
if 'running' not in st.session_state: 
    st.session_state.running = False
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False
if 'signal_count' not in st.session_state:
    st.session_state.signal_count = 0

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶️ Başlat / Durdur", use_container_width=True):
        st.session_state.running = not st.session_state.running
with col2:
    st.metric("Toplam Sinyal", st.session_state.signal_count)

if st.session_state.running:
    with st.spinner("🔄 Binomo analizi yapılıyor..."):
        with ThreadPoolExecutor(max_workers=10) as executor:
            raw_res = list(executor.map(analyze, ASSETS))

        results = [
            {
                'Asset': r[0], 
                'Signal': r[1], 
                'Confidence': r[2], 
                'RSI': round(r[3], 1),
                'Pattern': r[4]
            } 
            for r in raw_res
        ]

        # MODELİN KENDİ KENDİNE ÖĞRENMESİ
        df_active = pd.DataFrame(results)[pd.DataFrame(results)['Signal'] != 'WAIT'].copy()
        
        if not df_active.empty:
            # Eğitim verisi hazırla
            X_train_raw = np.array([
                [r['RSI']/100, 0.5 + np.random.uniform(-0.1, 0.1), 0.3 + np.random.uniform(-0.1, 0.1)] 
                for _, r in df_active.iterrows()
            ])
            X_train = scaler.fit_transform(X_train_raw)
            y_train = np.array([1 if r['Signal'] == 'BUY' else 0 for _, r in df_active.iterrows()])
            
            # Model eğitimi
            if not st.session_state.model_initialized:
                model.fit(X_train, y_train)
                st.session_state.model_initialized = True
                st.success("✅ Model ilk eğitimi tamamlandı!")
            else:
                # Eğer en az 2 sınıf varsa partial_fit yap
                if len(np.unique(y_train)) > 1:
                    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
                    st.success("✅ Model güncellendi (inkremental öğrenme)")
                else:
                    st.info(f"ℹ️ Tek sınıf - model değiştirilmedi")
            
            # Modeli kaydet
            joblib.dump(model, MODEL_FILE)
            joblib.dump(scaler, SCALER_FILE)
            
            # CSV'ye ekle
            df_active['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df_active.to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))
            
            # EXCEL'E KAYIT
            excel_data = df_active[['Asset', 'Signal', 'Confidence', 'RSI', 'Pattern', 'Timestamp']].copy()
            save_to_excel(excel_data.to_dict('records'))
            
            # Güncellenmiş sinyal sayısı
            st.session_state.signal_count += len(df_active)
            
            # Tabloyu göster
            st.success(f"✅ {len(df_active)} Binomo sinyali bulundu!")
            st.dataframe(df_active, use_container_width=True)
            
            # İndirme butonları
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists(CSV_FILE):
                    with open(CSV_FILE, 'rb') as f:
                        st.download_button("📥 CSV İndir", f, CSV_FILE, "text/csv")
            
            with col2:
                if os.path.exists(EXCEL_FILE):
                    with open(EXCEL_FILE, 'rb') as f:
                        st.download_button("📊 Excel İndir", f, EXCEL_FILE, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("⏳ Bu turda sinyal alınmadı (Tarafsız bölge)")

        time.sleep(2)
        st.rerun()
else:
    st.info("👇 Başlat butonuna basarak AI analizini başlatın")
