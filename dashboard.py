import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# === FUNKCJE ===

def get_price_data(crypto_id="bitcoin", days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    response = requests.get(url, params=params)
    data = response.json()["prices"]
    df = pd.DataFrame(data, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

def add_moving_averages(df, short=3, long=7):
    df["MA_short"] = df["price"].rolling(window=short).mean()
    df["MA_long"] = df["price"].rolling(window=long).mean()
    return df

def calculate_rsi(df, period=14):
    delta = df["price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def generate_signal(df):
    last = df.dropna().iloc[-1]
    trend = ""
    if last["MA_short"] > last["MA_long"]:
        trend = "📈 KUPUJ"
    elif last["MA_short"] < last["MA_long"]:
        trend = "📉 SPRZEDAJ"
    else:
        trend = "⏸ OBSERWUJ"
    
    rsi = last["RSI"]
    if rsi > 70:
        rsi_msg = f"RSI: {rsi:.2f} 🔴 PRZEKUPIONY"
    elif rsi < 30:
        rsi_msg = f"RSI: {rsi:.2f} 🟢 PRZESPRZEDANY"
    else:
        rsi_msg = f"RSI: {rsi:.2f} ✅ OK"
    
    return trend, rsi_msg

def predict_next_price(df):
    df = df.dropna()
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["price"].values
    model = LinearRegression().fit(X, y)
    next_day = np.array([[len(df)]])
    prediction = model.predict(next_day)[0]
    return round(prediction, 2)

# === STREAMLIT UI ===

st.set_page_config(page_title="Crypto Dashboard", layout="centered")
st.title("📊 Crypto Trend Dashboard")

cryptos = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Solana": "solana",
    "Cardano": "cardano"
}

selected = st.selectbox("Wybierz kryptowalutę:", list(cryptos.keys()))
crypto_id = cryptos[selected]

with st.spinner("Pobieranie danych..."):
    df = get_price_data(crypto_id)
    df = add_moving_averages(df)
    df = calculate_rsi(df)

trend, rsi_msg = generate_signal(df)
prediction = predict_next_price(df)

st.subheader(f"🔍 Analiza: {selected}")
st.markdown(f"**Sygnał trendu:** {trend}")
st.markdown(f"**RSI:** {rsi_msg}")
st.markdown(f"**🧠 Prognozowana cena jutro:** `${prediction}`")

st.line_chart(df[["price", "MA_short", "MA_long"]].dropna())

if st.button("🔔 Wyślij alert na Telegrama"):
    # wpisz swój token i chat_id
    token = "TWÓJ_BOT_TOKEN"
    chat_id = "TWÓJ_CHAT_ID"
    message = f"{selected.upper()} | {trend} | {rsi_msg} | Prognoza: ${prediction}"
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                  data={"chat_id": chat_id, "text": message})
    st.success("Alert wysłany!")
