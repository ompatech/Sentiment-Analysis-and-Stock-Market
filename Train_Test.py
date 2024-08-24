import pandas as pd
import yfinance as yf
from tqdm import tqdm

apple_news = pd.read_csv('apple_finbert.csv')
amazon_news = pd.read_csv('amazon_finbert.csv')
microsoft_news = pd.read_csv('microsoft_finbert.csv')
netflix_news = pd.read_csv('netflix_finbert.csv')

def clean_news(df):
    df = df.drop(['Unnamed: 0'], axis=1)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.groupby('DATE').agg({'IMPACT_SCORE': 'mean'}).reset_index()
    return df

def get_stock(comp):
    stock = yf.download(comp)
    stock = stock.loc['2020-12-29':'2022-01-10']
    stock = stock.reset_index()
    stock['Date'] = pd.to_datetime(stock['Date'])
    stock = stock.rename(columns={'Date': 'DATE'})
    return stock

def combine(news, stock):
    combined_df = pd.merge(news, stock, on='DATE', how='inner')
    combined_df['stock_pattern'] = (combined_df['Close'].shift(-1) >= combined_df['Open']).astype(int)
    combined_df = combined_df.drop(['Open', 'Low', 'High', 'Close', 'Adj Close', 'Volume'], axis=1)
    return combined_df

companies = [apple_news, amazon_news, microsoft_news, netflix_news]
companies_name = ['AAPL', 'AMZN', 'MSFT', 'NFLX']

for i in tqdm(range(len(companies))):
    cleaned_news = clean_news(companies[i])
    stock = get_stock(companies_name[i])
    combined_df = combine(cleaned_news, stock)
    filename = companies_name[i] + '_Train_Test.csv'
    combined_df.to_csv(filename, index=False)