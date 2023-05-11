from pathlib import Path
import streamlit as st
import pandas as pd
import openai
import os
import re
from dotenv import load_dotenv
from collect_data import DataCollector
from models import ZooVARMAX, ZooProphet, ZooETS, ZooSARIMAX
load_dotenv(".env")
openai.api_key = os.environ["openai-api-key"]


def get_ticker_symbol(name: str) -> str:
    raw_resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Company:Tesla\nTicker:TSLA\n\nCompany:Apple\nTicker:AAPL\n\nCompany:{name}\nTicker:"},
        ]
    )["choices"][0]["message"]["content"]
    matches = re.findall(r"[A-Z]{4,}", raw_resp)
    if not matches:
        return ""
    return matches[0]


st.subheader("Data")
company_name = st.text_input(label="Company name", value="")
if company_name != "":
    with st.spinner("Retrieving ticker symbol..."):
        ticker = get_ticker_symbol(name=company_name)
    if ticker != "":
        # try to collect data
        with st.spinner("Checking data availability..."):
            try:
                collector = DataCollector(
                    ticker_symbol=ticker,
                    fred_api_key=os.environ["fred-api-key"]
                )
                collector.collect(
                    world_series_idx=["T10YIE", "SP500", "SOFR180DAYAVG", "IHLIDXUS", "OBMMIC15YF"],
                    output_dir="data"
                )
                st.success("Collected data!")
            except Exception as e:
                st.error("Error while handling data")
st.divider()

dataset_paths = [str(p) for p in Path("data").rglob("*.csv")]
dataset_names = [p.split("/")[-1].split(".")[0] for p in dataset_paths]
selected_dataset = st.selectbox("Downloaded data", options=dataset_names)
dataset_idx = dataset_names.index(selected_dataset)
df = pd.read_csv(dataset_paths[dataset_idx])
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
st.write(df)

st.subheader("Forecasting")
algorithms = st.multiselect(
    label="Algorithms",
    options=["ETS", "Prophet", "SARIMAX", "VARMAX"]
)
alg_data = {}
zoo = {
    "ETS": ZooETS,
    "Prophet": ZooProphet,
    "SARIMAX": ZooSARIMAX,
    "VARMAX": ZooVARMAX,
}
res = df["Close"]
for algorithm in algorithms:
    with st.spinner(f"Processing {algorithm} for {dataset_names[dataset_idx]}..."):
        # if model does not exist, then train and save it
        model_path = f"weights/{dataset_names[dataset_idx]}/{algorithm.lower()}.ts"
        if Path(model_path).exists():
            model = zoo[algorithm].from_weights(
                weights_dir="weights",
                company_name=dataset_names[dataset_idx]
            )
        else:
            model = zoo[algorithm]()
            model.train(data=df)
            model.save(weights_dir="weights", company_name=dataset_names[dataset_idx])

        alg_data[algorithm] = model.predict(days=90)

        # create chart dataframe for plotting
        res = pd.concat([res, alg_data[algorithm]], axis=1)

res.columns = ["Close"] + algorithms
st.line_chart(res)






