import yfinance as yf
import pandas as pd
import ta
import time
import os
from datetime import datetime, timedelta
from pandas_datareader import data as web
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
import difflib
import xml.etree.ElementTree as ET
import requests
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping
# import matplotlib.font_manager as fm
import tensorflow as tf
from tensorflow.keras import backend as K


# ------------------------
# 설정
# ------------------------
tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK-B", "TSLA", "JPM", "AVGO", "ADBE", "NFLX", "CRM", "AMD", "SMCI", "NOW", "ASML", "MSCI", "CDNS"]
start_date = "2019-01-01"
end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

OUTPUT_DIR = "data"
MERGED_FILE = os.path.join(OUTPUT_DIR, "stock_macro_merged.csv")
PREDICTED_FILE = os.path.join(OUTPUT_DIR, "predicted_with_scores.csv")
SIMULATION_FILE_SIMPLE_FORMATTED = os.path.join(OUTPUT_DIR, "simulation_result_simple.csv")

FEATURE_COLUMNS = [
    'Close', 'Return', 'MA_5', 'MA_20', 'RSI', 'MACD_diff', 'Volume_Change',
    '기준금리', '실업률', '소비자물가지수(CPI)', '근원 소비자물가지수',
    '개인소비지출', '소비자심리지수', '신규 실업수당청구건수',
    '10년 국채 수익률', '경기선행지수', '국내총생산'
]

macro_indicators = {
    "GDP": "국내총생산",
    "FEDFUNDS": "기준금리",
    "UNRATE": "실업률",
    "CPIAUCSL": "소비자물가지수(CPI)",
    "CPILFESL": "근원 소비자물가지수",
    "PCE": "개인소비지출",
    "UMCSENT": "소비자심리지수",
    "ICSA": "신규 실업수당청구건수",
    "GS10": "10년 국채 수익률",
    "USSLIND": "경기선행지수"
}

# ------------------------
# 1단계: 주가 + 거시지표 수집
# ------------------------
def get_stock_data(ticker):
    try:
        print(f"[Stock] {ticker} 수집 중...")
        time.sleep(0.5)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            return pd.DataFrame()
        df["Ticker"] = ticker
        df["Return"] = df["Close"].pct_change()
        df["MA_5"] = df["Close"].rolling(5).mean()
        df["MA_20"] = df["Close"].rolling(20).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
        df["MACD_diff"] = ta.trend.MACD(df["Close"]).macd_diff()
        df["Volume_Change"] = df["Volume"].pct_change()
        df = df.ffill().bfill()
        df.reset_index(inplace=True)
        return df
    except:
        return pd.DataFrame()

def get_macro_data():
    macro_data = pd.DataFrame()
    for code, name in macro_indicators.items():
        try:
            time.sleep(0.5)
            df = web.DataReader(code, 'fred', start_date, end_date)
            df.rename(columns={code: name}, inplace=True)
            df = df.resample('MS').first().ffill()
            macro_data = df if macro_data.empty else macro_data.join(df, how="outer")
        except:
            continue
    return macro_data.rename_axis("Date").reset_index()

def update_stock_and_macro_data():
    print("[1단계] 데이터 수집 시작")
    stock_data_list = [get_stock_data(t) for t in tickers]
    stock_data_list = [df for df in stock_data_list if not df.empty]
    if not stock_data_list:
        print("❌ 주가 수집 실패")
        return None

    all_stock = pd.concat(stock_data_list, ignore_index=True)
    all_stock["Month"] = pd.to_datetime(all_stock["Date"]).dt.to_period("M").dt.to_timestamp()

    macro_data = get_macro_data()
    macro_data["Month"] = pd.to_datetime(macro_data["Date"]).dt.to_period("M").dt.to_timestamp()
    merged_df = pd.merge(all_stock, macro_data.drop(columns=["Date"]), on="Month", how="left")
    merged_df.drop(columns=["Month"], inplace=True)
    merged_df.ffill(inplace=True)
    merged_df["UpdatedAt"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_df.to_csv(MERGED_FILE, index=False)
    print(f"[1단계] 저장 완료 → {MERGED_FILE}")
    return merged_df

# ------------------------
# 2단계: AI 모델 예측
# ------------------------
np.random.seed(42)
tf.random.set_seed(42)

# 모델 생성 함수 정의
def build_gb_1d():
    return GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=4, subsample=0.8)

def build_gb_20d():
    return GradientBoostingRegressor(n_estimators=150, learning_rate=0.04, max_depth=6, subsample=0.9)

def build_dense_lstm(input_shape):
    K.clear_session()
    model = Sequential([
        LSTM(128, activation='tanh', input_shape=input_shape),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def predict_ai_scores(df):
    print("[2단계] AI 예측 시작")

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.loc[:, ~df.columns.duplicated()]

    # 수익률 및 타겟 생성
    df["Target_1D"] = df.groupby("Ticker")["Close"].shift(-1)
    df["Target_20D"] = df.groupby("Ticker")["Close"].shift(-20)
    df["Return_1D"] = (df["Target_1D"] - df["Close"]) / df["Close"]
    df["Return_20D"] = (df["Target_20D"] - df["Close"]) / df["Close"]
    df["Return"] = df.groupby("Ticker")["Close"].pct_change()
    df = df.ffill().bfill()

    # 훈련 데이터 분리
    train_df = df[df["Date"] <= pd.to_datetime("2024-12-31")].copy()
    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    y_train_1d = train_df["Return_1D"]
    y_train_20d = train_df["Return_20D"]

    # 모델 학습
    gb_1d = build_gb_1d()
    gb_1d.fit(X_train, y_train_1d)

    gb_20d = build_gb_20d()
    gb_20d.fit(X_train, y_train_20d)

    # Dense-LSTM 훈련
    SEQUENCE_LENGTH = 10
    scaler = MinMaxScaler()
    X_lstm_train, y_lstm_train = [], []

    for ticker in train_df["Ticker"].unique():
        temp_df = train_df[train_df["Ticker"] == ticker].copy()
        X_temp = temp_df[FEATURE_COLUMNS].fillna(0).values
        y_temp = temp_df["Return_1D"].values
        X_scaled = scaler.fit_transform(X_temp)

        for i in range(SEQUENCE_LENGTH, len(X_scaled)):
            X_lstm_train.append(X_scaled[i - SEQUENCE_LENGTH:i])
            y_lstm_train.append(y_temp[i])

    X_lstm_train = np.array(X_lstm_train)
    y_lstm_train = np.array(y_lstm_train)

    dense_lstm_model = build_dense_lstm(X_lstm_train.shape[1:])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    dense_lstm_model.fit(X_lstm_train, y_lstm_train, epochs=30, batch_size=16, validation_split=0.1, callbacks=[early_stop], verbose=1)

    # 예측 시작
    test_dates = df[df["Date"] >= pd.to_datetime("2025-05-01")]["Date"].drop_duplicates().sort_values()
    all_preds = []

    for current_date in test_dates:
        test_df = df[df["Date"] == current_date].copy()
        if test_df.empty:
            continue

        test_df[FEATURE_COLUMNS] = test_df[FEATURE_COLUMNS].fillna(method='ffill').fillna(method='bfill').fillna(0)
        if test_df[FEATURE_COLUMNS].isnull().values.any():
            print(f"⚠️ {current_date.date()} → 여전히 NaN 있음, 예측 스킵")
            continue

        test_df["Predicted_Return_GB_1D"] = gb_1d.predict(test_df[FEATURE_COLUMNS]) * 4
        test_df["Predicted_Return_GB_20D"] = gb_20d.predict(test_df[FEATURE_COLUMNS])

        lstm_preds = []
        valid_rows = []

        for _, row in test_df.iterrows():
            ticker = row["Ticker"]
            date = row["Date"]
            past_window = df[(df["Ticker"] == ticker) & (df["Date"] < date)].sort_values("Date").tail(SEQUENCE_LENGTH)

            if len(past_window) < SEQUENCE_LENGTH:
                continue

            past_feats = past_window[FEATURE_COLUMNS].fillna(0)
            scaled_feats = scaler.transform(past_feats)
            input_seq = np.expand_dims(scaled_feats, axis=0)
            pred = dense_lstm_model.predict(input_seq, verbose=0)[0][0]
            lstm_preds.append(pred)  # pred * 30 제거
            valid_rows.append(row)

        if not valid_rows:
            continue

        test_df = pd.DataFrame(valid_rows)
        test_df["Predicted_Return_Dense_LSTM"] = lstm_preds

        all_preds.append(test_df)
        print(f"✅ {current_date.date()} 예측 완료 - {len(test_df)}종목")

    if not all_preds:
        print("❌ 예측 결과 없음. 시뮬레이션 불가")
        return pd.DataFrame(columns=["Date", "Ticker", "Close", "Return_1D"])  # ← 빈 컬럼 포함해 반환
    
    result_df = pd.concat(all_preds, ignore_index=True)
    
    # 예측 종가 계산
    result_df["예측종가_GB_1D"] = result_df["Close"] * (1 + result_df["Predicted_Return_GB_1D"])
    result_df["예측종가_GB_20D"] = result_df["Close"] * (1 + result_df["Predicted_Return_GB_20D"])
    result_df["예측종가_Dense_LSTM"] = result_df["Close"] * (1 + result_df["Predicted_Return_Dense_LSTM"])
    result_df = pd.merge(result_df, df[["Date", "Ticker", "Return_1D"]], on=["Date", "Ticker"], how="left")

    # ✅ 결측치 보정만 수행
    result_df["Return_1D"] = result_df["Return_1D"].ffill().bfill()


    result_df.to_csv(PREDICTED_FILE, index=False)
    print(f"[2단계] 전체 예측 결과 저장 완료 → {PREDICTED_FILE}")
    return result_df
# ------------------------
SIMULATION_FILE_SIMPLE_FORMATTED = "data/simulation_result_simple.csv"

def simulate_combined_trading_simple_formatted(df):
    print("[3단계] 통합 모의투자 시작 (누적 보유 + 조건부 부분매도)")

    initial_capital = 10000
    portfolios = {
        "GB_1D": {"capital": initial_capital, "holding": {}},
        "GB_20D": {"capital": initial_capital, "holding": {}},
        "Dense-LSTM": {"capital": initial_capital, "holding": {}},
    }
    history = []
    TRADE_AMOUNT = 2000

    df_sorted = df.sort_values(by=["Date", "Ticker"]).copy()
    df_sorted = df_sorted.fillna(method='ffill').fillna(method='bfill')
    df_sorted["Date"] = pd.to_datetime(df_sorted["Date"]).dt.tz_localize(None)
    df_sorted = df_sorted[df_sorted["Date"] >= pd.to_datetime("2024-05-01")]

    if df_sorted.empty:
        print("  - 시뮬레이션할 데이터가 없습니다.")
        return pd.DataFrame()

    print(f"  - {df_sorted['Date'].min().date()} 부터 시뮬레이션 실행 중...")

    for date, date_df in df_sorted.groupby("Date"):
        for model, score_col in zip(
            portfolios.keys(),
            ["Predicted_Return_GB_1D", "Predicted_Return_GB_20D", "Predicted_Return_Dense_LSTM"]
        ):
            portfolio = portfolios[model]
            current_holdings = list(portfolio["holding"].keys())

            for ticker in current_holdings:
                holding_info = portfolio["holding"][ticker]
                holding_stock_data = date_df[date_df["Ticker"] == ticker]

                if not holding_stock_data.empty:
                    current_price = holding_stock_data.iloc[0]["Close"]
                    holding_score = holding_stock_data.iloc[0][score_col]
                    holding_value = holding_info["shares"] * current_price

                    if holding_score <= -0.02:
                        shares_to_sell_value = min(TRADE_AMOUNT, holding_value)
                        shares_to_sell = int(shares_to_sell_value // current_price)

                        if shares_to_sell == 0 and holding_value > 0:
                            shares_to_sell = 1
                        shares_to_sell = min(shares_to_sell, holding_info["shares"])

                        if shares_to_sell > 0:
                            sell_price = current_price
                            sell_amount = shares_to_sell * sell_price * 0.999
                            portfolio["capital"] += sell_amount
                            portfolio["holding"][ticker]["shares"] -= shares_to_sell

                            buy_price = holding_info["buy_price"]
                            total_asset_after_sell = portfolio["capital"] + sum(
                                h["shares"] * current_price for h in portfolio["holding"].values()
                            )

                            history.append({
                                "날짜": date,
                                "모델": model,
                                "티커": ticker,
                                "예측 수익률": holding_score * 10000,
                                "현재가": sell_price,
                                "매수(매도)": f"SELL ({shares_to_sell}주)",
                                "잔여 현금": portfolio["capital"],
                                "총 자산": total_asset_after_sell
                            })

                            if portfolio["holding"][ticker]["shares"] <= 0:
                                del portfolio["holding"][ticker]

            top_candidates = date_df[date_df[score_col] > 0.01].sort_values(by=score_col, ascending=False).head(2)

            for _, row in top_candidates.iterrows():
                ticker = row["Ticker"]
                score = row[score_col]

                if portfolio["capital"] >= TRADE_AMOUNT:
                    buy_price = row["Close"]
                    buy_value_to_spend = min(TRADE_AMOUNT, portfolio["capital"] * 0.99)
                    shares = int(buy_value_to_spend // (buy_price * 1.001))

                    if shares > 0:
                        cost = shares * buy_price * 1.001
                        portfolio["capital"] -= cost

                        if ticker in portfolio["holding"]:
                            portfolio["holding"][ticker]["shares"] += shares
                        else:
                            portfolio["holding"][ticker] = {
                                "shares": shares,
                                "buy_price": buy_price
                            }

                        total_asset_after_buy = portfolio["capital"] + sum(
                            h["shares"] * buy_price for h in portfolio["holding"].values()
                        )

                        history.append({
                            "날짜": date,
                            "모델": model,
                            "티커": ticker,
                            "예측 수익률": score * 10000,
                            "현재가": buy_price,
                            "매수(매도)": f"BUY ({shares}주)",
                            "잔여 현금": portfolio["capital"],
                            "총 자산": total_asset_after_buy
                        })

    result_df = pd.DataFrame(history)
    if not result_df.empty:
        result_df = result_df[["날짜", "모델", "티커", "예측 수익률", "현재가", "매수(매도)", "잔여 현금", "총 자산"]]
        result_df = result_df.merge(
            df_sorted[["Date", "Ticker", "Return_1D"]].rename(columns={"Date": "날짜", "Ticker": "티커"}),
            on=["날짜", "티커"],
            how="left"
        )
        result_df["실제 수익률"] = result_df["Return_1D"] * 10000
        result_df["예측 방향 일치"] = (result_df["예측 수익률"] * result_df["실제 수익률"]) > 0
        result_df["예측 정확도(%)"] = result_df.apply(
            lambda row: round(
                max(0.0, (1 - abs(row["예측 수익률"] - row["실제 수익률"]) / (abs(row["실제 수익률"]) + 1e-6))) * 100,
                2
            ),
            axis=1
        )

        os.makedirs("data", exist_ok=True)
        result_df.to_csv(SIMULATION_FILE_SIMPLE_FORMATTED, index=False, encoding="utf-8-sig")

    final_assets = {}
    for model, port in portfolios.items():
        holding_summary = {}
        total_holding_value = 0
        for ticker, info in port["holding"].items():
            latest_price_row = df_sorted[df_sorted["Ticker"] == ticker].sort_values(by="Date", ascending=False).head(1)

            if not latest_price_row.empty:
                current_price = latest_price_row.iloc[0]["Close"]
                shares = info["shares"]
                holding_value = shares * current_price
                holding_summary[ticker] = {
                    "보유 수량": shares,
                    "현재가": round(current_price, 2),
                    "평가 금액": round(holding_value, 2)
                }
                total_holding_value += holding_value

        total_asset = total_holding_value + port["capital"]
        final_assets[model] = {
            "현금 잔액": round(port["capital"], 2),
            "총 자산": round(total_asset, 2),
            "보유 종목 수": len(port["holding"]),
            "보유 종목": holding_summary
        }

    return result_df, final_assets

def export_final_portfolios(final_assets):
    os.makedirs("data", exist_ok=True)
    for model, info in final_assets.items():
        rows = []
        for ticker, details in info["보유 종목"].items():
            rows.append({
                "Ticker": ticker,
                "Shares": details["보유 수량"],
                "Current Price": details["현재가"],
                "Evaluation Value": details["평가 금액"]
            })
        if rows:
            df_model = pd.DataFrame(rows)
            df_model.to_csv(f"data/final_portfolio_{model}.csv", index=False)



# 4단계: 시각화 (간단한 시뮬레이션 결과로는 시각화가 제한될 수 있습니다)
# ------------------------
def plot_prediction_vs_actual(df, model_name, ticker):
    """
    Compare predicted vs actual close prices for a ticker-model combination.
    """
    df = df[df["Ticker"] == ticker].copy()
    df = df[df["Date"] >= pd.to_datetime("2025-05-01")].sort_values("Date")

    pred_col = f"예측종가_{model_name}"
    if df.empty or pred_col not in df.columns:
        return

    safe_ticker = ticker.replace("-", "_")

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Close"], label="Actual Close", color="blue", linewidth=2)
    plt.plot(df["Date"], df[pred_col], label=f"Predicted Close ({model_name})", color="orange", linestyle="--", linewidth=2)

    plt.title(f"{ticker} - {model_name} Prediction vs Actual", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"charts/predicted_vs_actual_{model_name}_{safe_ticker}.png")
    plt.close()

    print("[Step 4] All charts saved → charts/*.png")

# ------------------------
# 실행
# ------------------------
if __name__ == "__main__":
    merged_df = update_stock_and_macro_data()
    if merged_df is not None:
        merged_df["Date"] = pd.to_datetime(merged_df["Date"])
        latest_date = merged_df["Date"].max().date()
        print(f"[✓] Latest date in collected data: {latest_date}")

        predicted_df = predict_ai_scores(merged_df.copy())

        if not predicted_df.empty:
            # Step 1: Run simulation
            simulation_results_simple, final_assets = simulate_combined_trading_simple_formatted(predicted_df.copy())

            if not simulation_results_simple.empty:
                # Step 2: Add Return_1D, Accuracy info
                simulation_results_simple = simulation_results_simple.merge(
                    predicted_df[["Date", "Ticker", "Return_1D"]].rename(columns={"Date": "날짜", "Ticker": "티커"}),
                    on=["날짜", "티커"],
                    how="left"
                )
                
                simulation_results_simple["Return_1D"] = simulation_results_simple["Return_1D"] * 10000
                simulation_results_simple["Prediction_Match"] = (simulation_results_simple["예측 수익률"] * simulation_results_simple["Return_1D"]) > 0
                simulation_results_simple["Prediction_Accuracy(%)"] = simulation_results_simple["Prediction_Match"].apply(lambda x: 100 if x else 0)

                simulation_results_simple.to_csv("data/simulation_result_simple_with_accuracy.csv", index=False)
                print("[✓] Saved simulation result with accuracy info → simulation_result_simple_with_accuracy.csv")

                # Step 3: Save portfolios
                export_final_portfolios(final_assets)
                print("[✓] Saved final portfolios by model")

                # Step 4: Visualization (prediction vs actual)
                for model in ["GB_1D", "GB_20D", "Dense_LSTM"]:
                    col_name = f"예측종가_{model}"
                    if col_name in predicted_df.columns:
                        for ticker in predicted_df["Ticker"].unique():
                            plot_prediction_vs_actual(predicted_df.copy(), model, ticker)

                print("[✓] Saved prediction vs actual charts → charts/")

            # Preview
            print("\n📊 [Prediction Preview - Last 5 Rows]")
            print(predicted_df.tail(5))

            print("\n📈 [Simulation Result Preview - First 2 Rows]")
            print(simulation_results_simple.head(2).to_string(index=False))

            print("\n💼 [Final Portfolio Summary]")
            for model, info in final_assets.items():
                print(f"\n📌 Model: {model}")
                print(f"  - Total Asset: ${info['총 자산']}")
                print(f"  - Cash Balance: ${info['현금 잔액']}")
                print(f"  - Number of Holdings: {info['보유 종목 수']}")
                if info["보유 종목"]:
                    print("  - Holdings:")
                    for ticker, details in info["보유 종목"].items():
                        print(f"     ▸ {ticker}: {details['보유 수량']} shares, Price=${details['현재가']}, Value=${details['평가 금액']}")
                else:
                    print("  - No holdings.")
        else:
            print("[X] No prediction result, simulation aborted.")
