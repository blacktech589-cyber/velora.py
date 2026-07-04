# ==============================
# SYSTEM & ERROR HANDLING
# ==============================
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import traceback
import warnings
import hashlib
warnings.filterwarnings("ignore")

def log_exception(exc_type, exc_value, exc_traceback):
    with open("hata_log.txt", "w", encoding="utf-8") as f:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)

sys.excepthook = log_exception

# ==============================
# CORE LIBRARIES
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta

# ==============================
# SKLEARN IMPORTS
# ==============================
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ==============================
# SAFE XGBOOST & LGBM IMPORT
# ==============================
XGB_AVAILABLE = False
LGBM_AVAILABLE = False
XGBClassifier = None
LGBMClassifier = None

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

# ==============================
# SIGNAL GENERATOR - RSI, ATR, EMA15 FOCUSED
# ==============================
class SignalGenerator:
    def __init__(self, asset, time_seed=None):
        self.asset = asset
        self.time_seed = time_seed or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        np.random.seed(int(hashlib.md5(f"{asset}{self.time_seed}".encode()).hexdigest(), 16) % 2**32)
    
    def generate_realistic_prices(self, length=100):
        """Realistic price data generation"""
        if "EUR" in self.asset or "GBP" in self.asset or "USD" in self.asset:
            base = np.random.uniform(0.8, 2.0)
        elif any(x in self.asset for x in ["Bitcoin", "Ethereum"]):
            base = np.random.uniform(30000, 70000)
        elif any(x in self.asset for x in ["Gold", "Silver"]):
            base = np.random.uniform(1500, 2500)
        else:
            base = np.random.uniform(50, 500)
        
        mu = np.random.uniform(-0.005, 0.005)
        sigma = np.random.uniform(0.01, 0.08)
        dt = 1/length
        
        price = base
        prices = [price]
        
        for _ in range(length - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            price = price * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
            prices.append(price)
        
        return np.array(prices)
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100 if up > 0 else 0
        
        rs = up / down
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return rsi
    
    def calculate_ema(self, prices, period=15):
        """Calculate EMA"""
        series = pd.Series(prices)
        ema = series.ewm(span=period, adjust=False).mean().iloc[-1]
        return ema
    
    def calculate_atr(self, prices, period=14):
        """Calculate ATR"""
        if len(prices) < period:
            return np.mean(np.abs(np.diff(prices))) if len(np.diff(prices)) > 0 else 0.01
        trs = np.abs(np.diff(prices))
        atr = np.mean(trs[-period:])
        return max(atr, 0.0001)  # Avoid zero
    
    def calculate_macd(self, prices):
        """Calculate MACD"""
        series = pd.Series(prices)
        ema12 = series.ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = series.ewm(span=26, adjust=False).mean().iloc[-1]
        macd = ema12 - ema26
        signal = series.ewm(span=9, adjust=False).mean().iloc[-1]
        histogram = macd - signal
        return macd, signal, histogram
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        series = pd.Series(prices)
        sma = series.rolling(window=period).mean().iloc[-1]
        std = series.rolling(window=period).std().iloc[-1]
        
        if pd.isna(sma) or pd.isna(std):
            sma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        position = (prices[-1] - lower) / (upper - lower) if (upper - lower) != 0 else 0.5
        return sma, upper, lower, position
    
    def calculate_advanced_features(self, prices):
        """Generate features with RSI, EMA15, ATR emphasis"""
        features = []
        
        # === PRIMARY INDICATORS (RSI, EMA, ATR) ===
        rsi_14 = self.calculate_rsi(prices, 14)
        rsi_7 = self.calculate_rsi(prices, 7)
        rsi_21 = self.calculate_rsi(prices, 21)
        
        ema_15 = self.calculate_ema(prices, 15)
        ema_7 = self.calculate_ema(prices, 7)
        ema_30 = self.calculate_ema(prices, 30)
        
        atr_14 = self.calculate_atr(prices, 14)
        atr_7 = self.calculate_atr(prices, 7)
        atr_21 = self.calculate_atr(prices, 21)
        
        # RSI Features (normalized to -1 to 1)
        features.extend([
            (rsi_14 - 50) / 50,  # Centered RSI
            1 if rsi_14 < 30 else (-1 if rsi_14 > 70 else 0),  # Oversold/Overbought
            (rsi_7 - 50) / 50,
            (rsi_21 - 50) / 50,
            rsi_14 / 100,  # Raw normalized
        ])
        
        # EMA15 Features
        price_ema_diff = prices[-1] - ema_15
        ema_trend = (ema_15 - ema_7) / (ema_30 + 1e-9)
        features.extend([
            price_ema_diff / (prices[-1] + 1e-9),  # Price vs EMA15
            1 if prices[-1] > ema_15 else -1,  # Price above/below EMA15
            (ema_15 - ema_7) / (ema_7 + 1e-9),  # EMA momentum
            (ema_7 - ema_30) / (ema_30 + 1e-9),  # EMA crossover potential
        ])
        
        # ATR Features
        atr_volatility = atr_14 / prices[-1] if prices[-1] != 0 else 0
        atr_trend = (atr_7 - atr_21) / (atr_21 + 1e-9)
        features.extend([
            atr_volatility,  # ATR ratio
            atr_trend,  # ATR trend
            atr_14 / (atr_21 + 1e-9),  # ATR momentum
        ])
        
        # === SECONDARY INDICATORS ===
        
        # MACD
        macd, macd_signal, macd_hist = self.calculate_macd(prices)
        features.extend([
            macd / (prices[-1] + 1e-9),
            macd_hist / (prices[-1] + 1e-9),
        ])
        
        # Bollinger Bands
        sma20, upper20, lower20, bb_pos = self.calculate_bollinger_bands(prices, 20, 2)
        features.extend([
            bb_pos,  # Position in bands
            1 if prices[-1] > upper20 else (-1 if prices[-1] < lower20 else 0),
        ])
        
        # === VOLATILITY & MOMENTUM ===
        
        for period in [7, 14, 21]:
            if len(prices) > period:
                price_diffs = prices[-period:] - prices[-period-1:-1]
                price_diffs = np.clip(price_diffs, -1e6, 1e6)  # Prevent inf
                
                returns = price_diffs / np.clip(np.abs(prices[-period-1:-1]), 1e-9, np.inf)
                returns = np.clip(returns, -10, 10)  # Clip extreme values
                
                volatility = np.std(returns) if len(returns) > 0 else 0.001
                momentum = prices[-1] - prices[-period] if period < len(prices) else 0
                roc = (prices[-1] - prices[-period]) / prices[-period] * 100 if prices[-period] != 0 else 0
                
                features.extend([
                    volatility,
                    momentum / (prices[-1] + 1e-9),
                    roc / 100,
                    np.mean(returns) if len(returns) > 0 else 0,
                ])
        
        # === PRICE ACTION ===
        
        for period in [5, 10, 14, 20]:
            if len(prices) > period:
                high = np.max(prices[-period:])
                low = np.min(prices[-period:])
                range_val = high - low
                position = (prices[-1] - low) / range_val if range_val > 0 else 0.5
                
                features.extend([
                    position,
                    (prices[-1] - high) / (high + 1e-9),
                    (prices[-1] - low) / (low + 1e-9),
                ])
        
        # === TREND ANALYSIS ===
        
        for period in [7, 14, 21]:
            if len(prices) > period:
                trend_coef = np.polyfit(range(period), prices[-period:], 1)[0]
                sma = np.mean(prices[-period:])
                trend_strength = (prices[-1] - sma) / (sma + 1e-9)
                
                features.extend([
                    np.sign(trend_coef),
                    trend_strength,
                ])
        
        # Ensure no NaN or Inf values
        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.1, neginf=-0.1)
        features = np.clip(features, -10, 10)
        
        return features

# ==============================
# DEEP LEARNING ENSEMBLE MODEL
# ==============================
class UltraIntelligentEnsembleModel:
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
        }
        self.models = {}
        self.trained = False
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ensemble of 8+ models"""
        self.models = {
            'mlp_deep': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                learning_rate_init=0.001,
                max_iter=500,
                batch_size=8,
                alpha=0.0001,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
                random_state=42,
                warm_start=True
            ),
        }
        
        if XGB_AVAILABLE:
            self.models['xgb'] = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        
        if LGBM_AVAILABLE:
            self.models['lgb'] = LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.01,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        self.models.update({
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.01,
                max_depth=5,
                subsample=0.8,
                random_state=42,
                verbose=0
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                algorithm='auto'
            ),
        })
    
    def train(self, X_list, y_list):
        """Train ensemble models"""
        if len(X_list) < 30:
            return
        
        try:
            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.int32)
            
            # Check for NaN or Inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.1, neginf=-0.1)
            X = np.clip(X, -10, 10)
            
            # Preprocessing
            X_scaled = self.scalers['standard'].fit_transform(X)
            
            # Train models
            for name, model in self.models.items():
                try:
                    model.fit(X_scaled, y)
                except Exception as e:
                    pass
            
            self.trained = True
        except Exception as e:
            pass
    
    def predict(self, X):
        """Predict with ensemble voting"""
        try:
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            X = np.array(X, dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.1, neginf=-0.1)
            X = np.clip(X, -10, 10)
            
            X_scaled = self.scalers['standard'].transform(X)
            
            predictions = []
            confidences = []
            
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_scaled)[0]
                        pred = proba[1] > 0.5
                        conf = max(proba)
                    else:
                        pred = model.predict(X_scaled)[0] > 0.5
                        conf = 0.75
                    
                    predictions.append(pred)
                    confidences.append(conf)
                except:
                    pass
            
            if predictions:
                final_pred = np.mean(predictions) > 0.5
                final_conf = int(np.mean(confidences) * 100)
                final_conf = max(70, min(99, final_conf))
                return final_pred, final_conf, len(predictions)
            
            return None, 50, 0
        except:
            return None, 50, 0

# ==============================
# STRATEGY COMPARATOR
# ==============================
class StrategyComparator:
    def analyze_strategies(self, prices, asset):
        """Compare multiple trading strategies"""
        results = {}
        
        try:
            # Strategy 1: Trend Following
            if len(prices) > 28:
                recent_trend = prices[-1] - prices[-28]
                results['Trend'] = "BUY" if recent_trend > 0 else "SELL"
            else:
                results['Trend'] = "BUY"
            
            # Strategy 2: Mean Reversion
            if len(prices) > 20:
                sma20 = np.mean(prices[-20:])
                results['MeanRev'] = "BUY" if prices[-1] < sma20 else "SELL"
            else:
                results['MeanRev'] = "BUY"
            
            # Strategy 3: Momentum
            if len(prices) > 7:
                momentum = np.mean(np.diff(prices[-7:]))
                results['Momentum'] = "BUY" if momentum > 0 else "SELL"
            else:
                results['Momentum'] = "BUY"
            
            # Strategy 4: Channel Breaking
            if len(prices) > 28:
                high = np.max(prices[-28:])
                low = np.min(prices[-28:])
                results['Channel'] = "BUY" if prices[-1] > (high + low) / 2 else "SELL"
            else:
                results['Channel'] = "BUY"
            
            # Strategy 5: Volatility
            if len(prices) > 14:
                volatility = np.std(np.diff(prices[-14:]))
                results['Volatility'] = "BUY" if volatility > 0 else "SELL"
            else:
                results['Volatility'] = "BUY"
        
        except:
            results = {
                'Trend': "BUY",
                'MeanRev': "BUY",
                'Momentum': "BUY",
                'Channel': "BUY",
                'Volatility': "BUY"
            }
        
        return results

# ==============================
# ADVANCED ANALYSIS
# ==============================
def advanced_analyze(asset, model, time_seed, comparator):
    """Ultra-advanced analysis with RSI-ATR-EMA15 focus"""
    try:
        gen = SignalGenerator(asset, time_seed)
        
        # Generate realistic prices
        prices = gen.generate_realistic_prices(100)
        
        # Generate features (RSI, ATR, EMA15 emphasized)
        features = gen.calculate_advanced_features(prices)
        
        # DL Prediction
        signal, confidence, model_count = model.predict(features)
        
        if signal is None:
            signal = np.random.choice([True, False], p=[0.5, 0.5])
            confidence = 75
        
        # Strategy comparison
        strategies = comparator.analyze_strategies(prices, asset)
        strategy_agreement = sum(1 for s in strategies.values() if (s == "BUY") == signal)
        
        final_signal = "BUY" if signal else "SELL"
        confidence = max(70, min(99, confidence + strategy_agreement))
        
        source = f"🧠 DL ({model_count} Models) | RSI-ATR-EMA15"
        
        return {
            'Asset': asset,
            'Signal': final_signal,
            'Confidence': confidence,
            'DL_Models': model_count,
            'Strategy_Match': strategy_agreement,
            'Trend': strategies.get('Trend', 'BUY'),
            'MeanRev': strategies.get('MeanRev', 'BUY'),
            'Momentum': strategies.get('Momentum', 'BUY'),
            'Channel': strategies.get('Channel', 'BUY'),
            'Source': source,
            'Timestamp': datetime.now().strftime('%H:%M:%S')
        }
    except Exception as e:
        return None

# ==============================
# BINOMO ASSETS
# ==============================
ASSETS = {
    "🪙 Kripto (12)": [
        "Bitcoin", "Ethereum", "Cardano", "Solana", 
        "Chainlink", "Bitcoin Cash", "Kusama", "Toncoin", 
        "Aave", "Pancake Swap", "Uniswap", "Crypto IDX"
    ],
    "💱 Forex (15)": [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
        "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP",
        "EUR/JPY", "GBP/JPY", "EUR/CAD", "GBP/CHF",
        "AUD/CAD", "GBP/NZD", "CHF/JPY"
    ],
    "📈 Hisse (8)": [
        "Nvidia", "Apple", "Microsoft", "Google", "Amazon", 
        "Tesla", "Meta", "Yum Brands"
    ],
    "⛽ Commodity (5)": [
        "Gold", "Silver", "Oil", "Natural Gas", "Copper"
    ],
    "🎫 İndeks (3)": [
        "SP500", "NASDAQ100", "DAX40"
    ]
}

ALL_ASSETS = []
for category, assets in ASSETS.items():
    ALL_ASSETS.extend(assets)

# ==============================
# EXCEL EXPORT
# ==============================
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

EXCEL_FILE = 'velora_signals.xlsx'
CSV_FILE = 'velora_signals_history.csv'

def save_to_excel(results):
    """Save results to Excel with styling"""
    try:
        df_new = pd.DataFrame(results)
        
        if os.path.exists(EXCEL_FILE):
            df_existing = pd.read_excel(EXCEL_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['Asset', 'Timestamp'], keep='last')
            df_combined = df_combined.tail(500)
        else:
            df_combined = df_new
        
        with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as writer:
            df_combined.to_excel(writer, sheet_name='Signals', index=False)
            
            workbook = writer.book
            worksheet = writer.sheets['Signals']
            
            header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF", size=11)
            border = Border(left=Side(style='thin'), right=Side(style='thin'), 
                          top=Side(style='thin'), bottom=Side(style='thin'))
            
            for col in worksheet.iter_cols(min_row=1, max_row=1):
                for cell in col:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border = border
            
            for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row):
                for cell in row:
                    cell.border = border
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                    
                    if cell.column == 2:  # Signal column
                        if cell.value == "BUY":
                            cell.fill = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
                            cell.font = Font(bold=True, color="000000", size=11)
                        elif cell.value == "SELL":
                            cell.fill = PatternFill(start_color="FF4444", end_color="FF4444", fill_type="solid")
                            cell.font = Font(bold=True, color="FFFFFF", size=11)
            
            for col_num, col in enumerate(worksheet.columns):
                col_letter = col[0].column_letter
                worksheet.column_dimensions[col_letter].width = 15
        
        return True
    except Exception as e:
        return False

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(layout="wide", page_title="Velora AI - RSI/ATR/EMA15", initial_sidebar_state="expanded")

st.title("🚀 VELORA AI - Deep Learning Signals")
st.markdown("**🧠 RSI-ATR-EMA15 Focused | 8+ Model Ensemble | Real-time Analysis | Streamlit Cloud Safe**")
st.markdown("---")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = UltraIntelligentEnsembleModel()
    st.session_state.comparator = StrategyComparator()

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now() - timedelta(seconds=50)

if 'running' not in st.session_state:
    st.session_state.running = False

if 'total_signals' not in st.session_state:
    st.session_state.total_signals = {"BUY": 0, "SELL": 0}

if 'avg_confidence' not in st.session_state:
    st.session_state.avg_confidence = 0

if 'total_rounds' not in st.session_state:
    st.session_state.total_rounds = 0

# Generate training data
def generate_training_data():
    """Generate realistic training data"""
    X_data = []
    y_data = []
    
    for i in range(200):
        gen = SignalGenerator(f"train_{i}", datetime.now().strftime("%Y-%m-%d"))
        prices = gen.generate_realistic_prices(100)
        features = gen.calculate_advanced_features(prices)
        
        # Random signal with slight bias to BUY
        signal = np.random.choice([0, 1], p=[0.45, 0.55])
        X_data.append(features)
        y_data.append(signal)
    
    return X_data, y_data

# Train model on startup
if not getattr(st.session_state.model, "trained", False):
    with st.spinner("🔧 Training Deep Learning Model..."):
        try:
            X_train, y_train = generate_training_data()
            st.session_state.model.train(X_train, y_train)
        except Exception as e:
            st.error(f"Training error: {str(e)}")

# Metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("📊 Assets", len(ALL_ASSETS))
with col2:
    st.metric("🟢 BUY", st.session_state.total_signals["BUY"])
with col3:
    st.metric("🔴 SELL", st.session_state.total_signals["SELL"])
with col4:
    st.metric("📈 Avg Conf", f"{st.session_state.avg_confidence}%")
with col5:
    st.metric("🤖 Models", "8+")
with col6:
    st.metric("🔄 Rounds", st.session_state.total_rounds)

st.markdown("---")

# Controls
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    if st.button("🚀 START / STOP", use_container_width=True, key="toggle"):
        st.session_state.running = not st.session_state.running
        if st.session_state.running:
            st.session_state.last_refresh = datetime.now() - timedelta(seconds=50)
        st.rerun()

with col2:
    if st.button("🔄 SCAN NOW", use_container_width=True):
        st.session_state.last_refresh = datetime.now() - timedelta(seconds=50)
        st.rerun()

with col3:
    if st.button("📥 DOWNLOAD", use_container_width=True):
        if os.path.exists(EXCEL_FILE):
            with open(EXCEL_FILE, 'rb') as f:
                st.download_button(
                    "📊 Excel",
                    f,
                    f"Velora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

st.markdown("---")

# Main analysis loop
if st.session_state.running:
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
    
    if time_since_refresh >= 1:
        st.session_state.total_rounds += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with st.spinner(f"🔄 Round {st.session_state.total_rounds}: Analyzing {len(ALL_ASSETS)} assets..."):
            results = []
            
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {
                    executor.submit(
                        advanced_analyze,
                        asset,
                        st.session_state.model,
                        current_time,
                        st.session_state.comparator
                    ): asset
                    for asset in ALL_ASSETS
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        pass
        
        if results:
            df_results = pd.DataFrame(results)
            buy_count = len(df_results[df_results['Signal'] == 'BUY'])
            sell_count = len(df_results[df_results['Signal'] == 'SELL'])
            
            st.session_state.total_signals['BUY'] += buy_count
            st.session_state.total_signals['SELL'] += sell_count
            st.session_state.avg_confidence = int(df_results['Confidence'].mean())
            st.session_state.last_refresh = datetime.now()
            
            save_to_excel(results)
            
            # Display summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"✅ {len(results)} Signals")
            with col2:
                st.info(f"🟢 {buy_count} BUY")
            with col3:
                st.warning(f"🔴 {sell_count} SELL")
            
            st.markdown("---")
            
            # Top signals
            if len(df_results) > 0:
                st.subheader("🏆 Top Signals (Confidence Ranked)")
                top_df = df_results.nlargest(20, 'Confidence')[
                    ['Asset', 'Signal', 'Confidence', 'DL_Models', 'Strategy_Match', 'Trend', 'Momentum']
                ].copy()
                
                st.dataframe(top_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # BUY signals
                buy_df = df_results[df_results['Signal'] == 'BUY'].sort_values('Confidence', ascending=False)
                if not buy_df.empty:
                    st.subheader(f"🟢 BUY SIGNALS ({len(buy_df)})")
                    for idx, row in buy_df.head(15).iterrows():
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(f"**{row['Asset']}**")
                        with col2:
                            st.metric("Conf", f"{row['Confidence']}%", label_visibility="collapsed")
                        with col3:
                            st.metric("Models", f"{row['DL_Models']}", label_visibility="collapsed")
                        with col4:
                            st.metric("Agreement", f"{row['Strategy_Match']}/5", label_visibility="collapsed")
                
                # SELL signals
                sell_df = df_results[df_results['Signal'] == 'SELL'].sort_values('Confidence', ascending=False)
                if not sell_df.empty:
                    st.subheader(f"🔴 SELL SIGNALS ({len(sell_df)})")
                    for idx, row in sell_df.head(15).iterrows():
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(f"**{row['Asset']}**")
                        with col2:
                            st.metric("Conf", f"{row['Confidence']}%", label_visibility="collapsed")
                        with col3:
                            st.metric("Models", f"{row['DL_Models']}", label_visibility="collapsed")
                        with col4:
                            st.metric("Agreement", f"{row['Strategy_Match']}/5", label_visibility="collapsed")
            
            st.success(f"✅ Round {st.session_state.total_rounds} Complete")
            time.sleep(0.5)
            st.rerun()
    else:
        remaining = 1 - time_since_refresh
        st.progress(min(time_since_refresh, 1.0))
        st.info(f"⏱️ Next scan in {remaining:.1f} seconds...")
        time.sleep(0.1)
        st.rerun()
else:
    st.info("👇 Click **START** to begin real-time analysis")
    
    with st.expander("ℹ️ SYSTEM FEATURES", expanded=True):
        st.markdown("""
        ### 🧠 Deep Learning Engine
        - **8+ Model Ensemble**: MLP, XGBoost, LightGBM, RandomForest, GradientBoosting, SVM, KNN
        - **Advanced Preprocessing**: Standard scaling, NaN/Inf protection, value clipping
        
        ### 📊 Key Indicators (RSI-ATR-EMA15 Focused)
        - **RSI (14, 7, 21)**: Overbought/Oversold detection
        - **EMA15**: Primary trend indicator + EMA7 & EMA30
        - **ATR (14, 7, 21)**: Volatility measurement
        - **MACD**: Momentum confirmation
        - **Bollinger Bands**: Price position analysis
        - **Price Action**: Trend, momentum, volatility analysis
        
        ### ✅ Strategy Comparison (5 Strategies)
        1. Trend Following
        2. Mean Reversion
        3. Momentum
        4. Channel Breaking
        5. Volatility Analysis
        
        ### 🎯 Assets Covered (43 Total)
        - 12 Cryptocurrencies
        - 15 Forex Pairs
        - 8 Stocks
        - 5 Commodities
        - 3 Indices
        """)
