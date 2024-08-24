from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from scipy.special import softmax 
import numpy as np
import torch
from tqdm import tqdm
from timeit import default_timer as timer
import warnings
labels = ['positive', 'negative', 'neutral']
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def process_batch(df_batch, tokenizer, model, labels, softmax):
    def sentiment(text):
        text = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = model(**text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        l = scores.tolist()
        return labels[l.index(max(l))]

    def score(text):
        text = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = model(**text)
        scores = output[0][0].detach().numpy()
        score = softmax(scores)
        return (score[0] - score[1]) / (score[2] + score[0] + score[1])
    
    # Apply functions to the batch
    df_batch['sentiment'] = df_batch['Abstract'].apply(sentiment)
    df_batch['impact_score'] = df_batch['Abstract'].apply(score)
    
    return df_batch

def t_wt(df, tokenizer, model, labels, softmax, batch_size=1000):
    df = df.copy()
    num_batches = (len(df) + batch_size - 1) // batch_size  # Calculate number of batches

    # Initialize an empty DataFrame to store results
    result_df = pd.DataFrame()
    
    for i in tqdm(range(num_batches), desc='Processing batches'):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(df))
        df_batch = df.iloc[start:end]
        processed_batch = process_batch(df_batch, tokenizer, model, labels, softmax)
        result_df = pd.concat([result_df, processed_batch], ignore_index=True)

    return result_df

def clean(df):
    df = df.drop(['Publisher', 'Complete_Title', 'Abstract', 'Website', 'Stock', 'Exact_Time'], axis=1)
    df = df.rename(columns={'Time': 'DATE', 'sentiment': 'SENTIMENT', 'impact_score': 'IMPACT_SCORE'})
    return df



apple = pd.read_csv('AAPL_news.csv')
amazon = pd.read_csv('AMZN_news.csv')
microsoft = pd.read_csv('MSFT_news.csv')
netflix = pd.read_csv('NFLX_news.csv')


companies = [apple, amazon, microsoft, netflix]
comp_name = ['apple', 'amazon', 'microsoft', 'netflix']
for i in range(len(companies)):
    start_time = timer()
    comp_new = t_wt(companies[i], tokenizer, model, labels, softmax, batch_size=32)
    end_time = timer()
    comp_clean = clean(comp_new)
    filename = comp_name[i] + '_finbert.csv'
    comp_clean.to_csv(filename)
    print(f"/nProcessing {comp_name[i]} news: " + str(end_time - start_time) + ' seconds')

print("ALL THE FILES ARE DONE!!!!!")
