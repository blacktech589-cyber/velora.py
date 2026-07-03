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
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from datetime import datetime

# --- AYARLAR ---
MODEL_FILE = 'velora_enterprise_brain.joblib'
CSV_FILE = 'sinyal_gecmisi.csv'
EXCEL_FILE = 'velora_sinyaller.xlsx'
ASSETS = ["Crypto IDX", "Bitcoin (OTC)", "Ethereum (OTC)", "Solana (OTC)", "Gold", "Oil", "Ferrari", "Nvidia", "Visa", "Starbucks", "Qualcomm", "Intel"]

# --- EXCEL KAYIT FONKSİYONU ---
def save_to_excel(data):
    """Verileri Excel dosyasına kaydeder"""
    try:
        if os.path.exists(EXCEL_FILE):
            # Mevcut dosyayı aç
            df_existing = pd.read_excel(EXCEL_FILE)
            df_new = pd.DataFrame(data)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = pd.DataFrame(data)
        
        # Excel dosyasını oluştur/güncelle
        with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as writer:
            df_combined.to_excel(writer, sheet_name='Sinyaller', index=False)
            
            # Stil uygula
            workbook = writer.book
            worksheet = writer.sheets['Sinyaller']
            
            # Header formatı
            header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF")
            
            for col in worksheet.iter_cols(min_row=1, max_row=1):
                for cell in col:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
            
            # Sütun genişlikleri
            worksheet.column_dimensions['A'].width = 20
            worksheet.column_dimensions['B'].width = 12
            worksheet.column_dimensions['C'].width = 12
            worksheet.column_dimensions['D'].width = 12
            worksheet.column_dimensions['E'].width = 18
            
            # Veri satırlarını ortala
            for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row):
                for cell in row:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    
                    # Sinyal rengini ayarla
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
st.set_page_config(layout="wide", page_title="Velora Enterprise")
st.title("⚡ Velora Enterprise: AI Self-Learning")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶️ Başlat / Durdur", use_container_width=True):
        st.session_state.running = not st.session_state.running if 'running' in st.session_state else True

if 'running' not in st.session_state: 
    st.session_state.running = False

if st.session_state.running:
    with st.spinner("Analiz yapılıyor..."):
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
        df_active = pd.DataFrame(results)[pd.DataFrame(results)['Signal'] != 'WAIT'].copy()
        if not df_active.empty:
            df_active['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # CSV'ye ekle
            df_active.to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))
            
            # EXCEL'E KAYIT
            excel_data = df_active[['Asset', 'Signal', 'Prob', 'RSI', 'Timestamp']].copy()
            save_to_excel(excel_data.to_dict('records'))
            
            # Tabloyu göster
            st.success(f"✅ {len(df_active)} sinyal bulundu ve Excel'e kaydedildi!")
            st.dataframe(df_active, use_container_width=True)
            
            # İndirme butonları
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists(CSV_FILE):
                    with open(CSV_FILE, 'rb') as f:
                        st.download_button("📥 CSV İndir", f, EXCEL_FILE, "text/csv")
            
            with col2:
                if os.path.exists(EXCEL_FILE):
                    with open(EXCEL_FILE, 'rb') as f:
                        st.download_button("📊 Excel İndir", f, EXCEL_FILE, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        time.sleep(2)
        st.rerun()
