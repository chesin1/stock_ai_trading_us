from IPython import get_ipython
from IPython.display import display
# %%
from IPython import get_ipython
from IPython.display import display
# %%
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
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# 설정
# ------------------------
tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "TSLA", "UNH", "JPM"]
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
def predict_ai_scores(df):
    print("[2단계] AI 예측 시작")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=FEATURE_COLUMNS + ["Return"]).copy()

    # 혹시 모를 NaN 최신 데이터로 채우기
    df = df.fillna(method='ffill').fillna(method='bfill')


    df["Target_1D"] = df.groupby("Ticker")["Close"].shift(-1)
    df["Target_20D"] = df.groupby("Ticker")["Close"].shift(-20)

    df["Return_1D"] = (df["Target_1D"] - df["Close"]) / df["Close"]
    df["Return_20D"] = (df["Target_20D"] - df["Close"]) / df["Close"]

    latest_date = df["Date"].max().date()
    train_df = df[df["Date"].dt.date < latest_date].copy()
    test_df = df[df["Date"].dt.date == latest_date].copy()

    if test_df.empty:
        print(f"❌ 가장 최근 날짜({latest_date})의 데이터가 없어 예측을 수행할 수 없습니다.")
        return pd.DataFrame()

    train_df_for_training = train_df.dropna(subset=["Target_1D", "Target_20D"])
    X_train = train_df_for_training[FEATURE_COLUMNS]
    y_train_1d = train_df_for_training["Return_1D"]
    y_train_20d = train_df_for_training["Return_20D"]

    gb_1d = GradientBoostingRegressor()
    gb_1d.fit(X_train, y_train_1d)
    test_df["Predicted_Return_GB_1D"] = gb_1d.predict(test_df[FEATURE_COLUMNS])

    gb_20d = GradientBoostingRegressor()
    gb_20d.fit(X_train, y_train_20d)
    test_df["Predicted_Return_GB_20D"] = gb_20d.predict(test_df[FEATURE_COLUMNS])

    print("  - LSTM 모델 학습 시작...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(train_df_for_training[FEATURE_COLUMNS])
    X_lstm_train = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    y_lstm_train = train_df_for_training["Return_1D"].values

    dense_lstm_model = Sequential()
    dense_lstm_model.add(LSTM(64, input_shape=(1, X_scaled.shape[1]), return_sequences=False))
    dense_lstm_model.add(Dense(32, activation='relu'))
    dense_lstm_model.add(Dense(1))
    dense_lstm_model.compile(optimizer='adam', loss='mse')

    dense_lstm_model.fit(X_lstm_train, y_lstm_train, epochs=10, batch_size=16, verbose=0)
    print("  - LSTM 모델 학습 완료")

    print("  - LSTM 모델 예측 시작...")
    X_test_scaled = scaler.transform(test_df[FEATURE_COLUMNS])
    X_lstm_test = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    test_df["Predicted_Return_LSTM"] = dense_lstm_model.predict(X_lstm_test, verbose=0).flatten()
    print("  - LSTM 모델 예측 완료")

    test_df.to_csv(PREDICTED_FILE, index=False)
    print(f"[2단계] 예측 결과 저장 완료 → {PREDICTED_FILE}")
    return test_df

# ------------------------
# 3단계: 통합 모의투자 시뮬레이션 (간단한 형식)
# ------------------------
def simulate_combined_trading_simple_formatted(df):
    print("[3단계] 통합 모의투자 시작 (간단한 형식)")

    initial_capital = 10000
    portfolios = {
        "GB_1D": {"capital": initial_capital, "holding": {}},
        "GB_20D": {"capital": initial_capital, "holding": {}},
        "LSTM": {"capital": initial_capital, "holding": {}},
    }
    history = []
    TRADE_AMOUNT = 2000 # 거래 단위 2000 달러

    df_sorted = df.sort_values(by=["Date", "Ticker"]).copy()

    # 혹시 모를 NaN 최신 데이터로 채우기
    df_sorted = df_sorted.fillna(method='ffill').fillna(method='bfill')


    if df_sorted.empty:
         print("  - 시뮬레이션할 데이터가 없습니다.")
         return pd.DataFrame()

    print(f"  - {df_sorted['Date'].min().date()} 부터 시뮬레이션 실행 중...")

    for date, date_df in df_sorted.groupby("Date"):
        for model, score_col in zip(portfolios.keys(), ["Predicted_Return_GB_1D", "Predicted_Return_GB_20D", "Predicted_Return_LSTM"]):
            portfolio = portfolios[model]
            best_buy_ticker = None
            best_buy_score = -np.inf
            current_holding_ticker = list(portfolio["holding"].keys())[0] if portfolio["holding"] else None

            # 1. 매도 판단 (2000달러 단위 또는 전량)
            if current_holding_ticker:
                holding_info = portfolio["holding"][current_holding_ticker]
                holding_stock_data = date_df[date_df["Ticker"] == current_holding_ticker]

                if not holding_stock_data.empty:
                    current_price = holding_stock_data.iloc[0]["Close"]
                    holding_score = holding_stock_data.iloc[0][score_col]
                    holding_value = holding_info["shares"] * current_price

                    if holding_score <= 0:
                        # 2000달러 어치 매도 또는 전량 매도
                        shares_to_sell_value = min(TRADE_AMOUNT, holding_value)
                        shares_to_sell = int(shares_to_sell_value // current_price)
                        if shares_to_sell == 0 and shares_to_sell_value > 0: # 최소 1주 매도
                             shares_to_sell = 1
                             if shares_to_sell > holding_info["shares"]: # 보유량보다 많으면 보유량만큼만
                                  shares_to_sell = holding_info["shares"]


                        if shares_to_sell > 0:
                            sell_price = current_price
                            sell_amount = shares_to_sell * sell_price
                            portfolio["capital"] += sell_amount
                            portfolio["holding"][current_holding_ticker]["shares"] -= shares_to_sell

                            total_asset_after_sell = portfolio["capital"] + (portfolio["holding"][current_holding_ticker]["shares"] * current_price)

                            history.append({
                                "날짜": date,
                                "모델": model,
                                "티커": current_holding_ticker,
                                "예측 수익률": holding_score,
                                "현재가": sell_price,
                                "매수(매도)": f"SELL ({shares_to_sell}주)",
                                "잔여 현금": portfolio["capital"],
                                "총 자산": total_asset_after_sell
                            })

                            if portfolio["holding"][current_holding_ticker]["shares"] <= 0:
                                del portfolio["holding"][current_holding_ticker]
                                current_holding_ticker = None


            # 2. 매수 판단 (현재 보유 종목이 없거나 매도 후 자금이 있을 때)
            if portfolio["capital"] >= TRADE_AMOUNT: # 2000달러 이상 자본금이 있을 때 매수 고려
                for idx, row in date_df.iterrows():
                    # 현재 보유 종목이 아니면서 예측 수익률이 높은 종목 찾기
                    if (not portfolio["holding"] or row["Ticker"] != list(portfolio["holding"].keys())[0]):
                         score = row[score_col]
                         if score > 0.01 and score > best_buy_score:
                             best_buy_score = score
                             best_buy_ticker = row["Ticker"]


                # 매수 실행
                if best_buy_ticker:
                     buy_stock_data = date_df[date_df["Ticker"] == best_buy_ticker]
                     if not buy_stock_data.empty:
                          buy_price = buy_stock_data.iloc[0]["Close"]
                          # 2000달러 어치 매수 (자본금이 2000 미만이면 가능한 만큼)
                          buy_value_to_spend = min(TRADE_AMOUNT, portfolio["capital"] * 0.99) # 자본금의 99% 또는 2000달러
                          shares = int(buy_value_to_spend // (buy_price * 1.001)) # 거래 수수료 고려

                          if shares > 0:
                              cost = shares * buy_price * 1.001
                              portfolio["capital"] -= cost

                              # 기존 보유 종목에 추가하거나 새로 보유 시작
                              if best_buy_ticker in portfolio["holding"]:
                                  portfolio["holding"][best_buy_ticker]["shares"] += shares
                                  # 평균 매수 단가 업데이트 (선택 사항, 현재는 단순 합산)
                                  # new_total_cost = portfolio["holding"][best_buy_ticker]["shares"] * portfolio["holding"][best_buy_ticker]["buy_price"] + cost
                                  # new_total_shares = portfolio["holding"][best_buy_ticker]["shares"] + shares
                                  # portfolio["holding"][best_buy_ticker]["buy_price"] = new_total_cost / new_total_shares
                              else:
                                  # 전량 매도 후 다른 종목을 매수하는 경우, 기존 holding은 비어있음
                                  if not portfolio["holding"]:
                                      portfolio["holding"][best_buy_ticker] = {"shares": shares, "buy_price": buy_price}
                                  else:
                                      # 이론적으로 이 경우는 발생하지 않아야 하지만, 혹시 모를 상황 대비
                                      print(f"경고: {date} {model} 이미 보유 중인 종목({list(portfolio['holding'].keys())[0]})이 있지만, 다른 종목({best_buy_ticker}) 매수 시도.")
                                      pass # 여기서는 매수하지 않음


                              total_asset_after_buy = portfolio["capital"] + (portfolio["holding"][best_buy_ticker]["shares"] * buy_price)


                              history.append({
                                  "날짜": date,
                                  "모델": model,
                                  "티커": best_buy_ticker,
                                  "예측 수익률": best_buy_score,
                                  "현재가": buy_price,
                                  "매수(매도)": f"BUY ({shares}주)",
                                  "잔여 현금": portfolio["capital"],
                                  "총 자산": total_asset_after_buy
                              })


    result_df = pd.DataFrame(history)

    if not result_df.empty:
        result_df = result_df[["날짜", "모델", "티커", "예측 수익률", "현재가", "매수(매도)", "잔여 현금", "총 자산"]]
        result_df.to_csv(SIMULATION_FILE_SIMPLE_FORMATTED, index=False)
        print(f"[3단계] 시뮬레이션 결과 저장 완료 → {SIMULATION_FILE_SIMPLE_FORMATTED}")
    else:
        print("[3단계] 시뮬레이션 결과 없음")

    return result_df

# 4단계: 시각화 (간단한 시뮬레이션 결과로는 시각화가 제한될 수 있습니다)
# ------------------------
def visualize_trades_simple(df, sim_df_simple):
    print("[4단계] 시각화 시작")
    os.makedirs("charts", exist_ok=True)
    df["Date"] = pd.to_datetime(df["Date"])

    if sim_df_simple.empty:
         print("  - 시각화할 시뮬레이션 결과가 없습니다.")
         return

    sim_df_simple["날짜"] = pd.to_datetime(sim_df_simple["날짜"])

    for ticker in df["Ticker"].unique():
        fig, ax = plt.subplots(figsize=(12, 6))
        stock_df = df[df["Ticker"] == ticker].sort_values(by="Date")
        ax.plot(stock_df["Date"], stock_df["Close"], label="Close Price", alpha=0.6)

        for model in sim_df_simple["모델"].unique():
            trades = sim_df_simple[(sim_df_simple["티커"] == ticker) & (sim_df_simple["모델"] == model)].copy()

            if trades.empty:
                 continue

            trades = pd.merge(trades, stock_df[["Date", "Close"]].rename(columns={"Close": "Actual_Close"}), left_on="날짜", right_on="Date", how="left")

            if 'Actual_Close' in trades.columns:
                 buys = trades[trades["매수(매도)"].str.contains("BUY", na=False)]
                 sells = trades[trades["매수(매도)"].str.contains("SELL", na=False)]

                 ax.scatter(buys["날짜"], buys["Actual_Close"], label=f"{model} BUY", marker="^", color="green", zorder=5)
                 ax.scatter(sells["날짜"], sells["Actual_Close"], label=f"{model} SELL", marker="v", color="red", zorder=5)
            else:
                 print(f"경고: 시각화 중 '{ticker}' 종목에 대해 'Actual_Close' 컬럼을 찾을 수 없습니다. 해당 종목의 시각화가 정확하지 않을 수 있습니다.")


        ax.set_title(f"{ticker} - AI Trading Signals (Simple Sim - {model})") # 모델별로 시각화 제목에 표시
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"charts/{ticker}_trades_simple_{model}.png") # 모델별 파일명
        plt.close()
    print("[4단계] 시각화 완료 → charts/*.png")


# ------------------------
# 실행
# ------------------------
if __name__ == "__main__":
    merged_df = update_stock_and_macro_data()
    if merged_df is not None:
        merged_df["Date"] = pd.to_datetime(merged_df["Date"])
        latest_date = merged_df["Date"].max().date()
        print(f"📅 수집된 데이터의 가장 최근 날짜: {latest_date}")

        predicted_df = predict_ai_scores(merged_df.copy())

        if not predicted_df.empty:
            simulation_results_simple = simulate_combined_trading_simple_formatted(predicted_df.copy())

            if not simulation_results_simple.empty:
                 # 간단한 시뮬레이션 결과로는 복잡한 시각화는 어렵습니다.
                 # 거래 시점만 표시하는 시각화 함수를 사용합니다.
                 visualize_trades_simple(merged_df.copy(), simulation_results_simple.copy())

            print("\n📊 [예측 결과 미리보기 - 마지막 2행]")
            print(predicted_df.tail(2))

            print("\n📈 [시뮬레이션 결과 미리보기 - 처음 2행]")
            print(simulation_results_simple.head(2))

            print("\n💰 [최종 포트폴리오 자산 현황]")
            if not simulation_results_simple.empty:
                # 각 모델의 마지막 거래 시점의 총 자산
                final_capitals = simulation_results_simple.groupby("모델")["총 자산"].last()
                print(final_capitals)
            else:
                print("시뮬레이션 결과가 없습니다.")