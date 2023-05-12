# Advanced Time Series Forecasting for Stock Prices

## Introduction

Welcome to my advanced machine learning project repository! The project aims to provide accurate and insightful financial data forecasting using advanced time series models.

Here, you'll find code and resources for various forecasting methodologies, including SARIMAX, ETS, Prophet, VARMAX, DeepAR, and Temporal Fusion Transformer. Each model offers unique strengths, resulting in a comprehensive approach to predicting stock prices.

A highlight of this project is the interactive dashboard. By entering a company name, users can receive a three-month forecast of its stock prices. The dashboard integrates different forecasting methods, presenting predictions in a clear and user-friendly format.

Whether you're an experienced data scientist, a financial professional, or a student delving into machine learning, I hope this repository serves as a valuable and inspiring resource.

Feel free to reach out with any questions or suggestions. Happy forecasting!

## Project Structure

```
├── datasets             <- Folder for storing datasets of company's stock prices
├── ipynb                <- Python notebooks with training code
├── weights              <- Folder with serialized models
|
├── .gitignore           <- List of ignored files and directories
├── app.py               <- Streamlit dashboard app
├── models.py            <- Model classes for inference within app
└── collect_data.py      <- Code for getting a multivariate dataset of company's stocks
```

## Instructions

1. Create `.env` file with following variables: `openai-api-key` and `fred-api-key`
2. Run the Streamlit app with `streamlit run app.py`

## Metrics

All models were tested against random walk on Tesla's stock prices:

![image](https://github.com/Dmmc123/transform-ts/assets/54360024/05c2f2f5-80b9-44ca-8347-9ca567b4b131)
