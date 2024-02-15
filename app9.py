import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")
import re
import numpy as np
import os
import streamlit as st


@st.cache_data
def load_data():
    df = pd.read_csv(file)
    return df


file = "Rec_sys_content.csv"
df = load_data()

df.isnull().sum(axis = 0)
df.dropna().reset_index(inplace = True)

count_vectorizer = CountVectorizer()

tfidf_vec = TfidfVectorizer( analyzer='word', ngram_range=(1,3))


df['Description'] = df['Product Name'] + ' ' +df['Description']

unique_df = df.drop_duplicates(subset=['Description'], keep='first')
#unique_df1 = df1.drop_duplicates(subset=['Objeto'], keep='first')

unique_df['desc_lowered'] = unique_df['Description'].apply(lambda x: x.lower())
#unique_df1['Objeto'] = unique_df1['Objeto'].apply(lambda x: x.lower())

unique_df['desc_lowered'] = unique_df['desc_lowered'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
#unique_df1['Objeto'] = unique_df1['Objeto'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

desc_list = list(unique_df['desc_lowered'])
#desc_list1 = list(unique_df1['Objeto'])

unique_df= unique_df.reset_index(drop=True)
#unique_df1= unique_df1.reset_index(drop=True)

unique_df.reset_index(inplace=True)
#unique_df1.reset_index(inplace=True)


def find_euclidean_distances(sim_matrix, index, n=10): 
    # Getting Score and Index
    result = list(enumerate(sim_matrix[index]))
    # Sorting the Score and taking top 10 products
    sorted_result = sorted(result,key=lambda x:x[1],reverse=False)[1:10+1]
    # Mapping index with data
    similar_products =  [{'value': unique_df.iloc[x[0]]['Product Name'], 'score' : round(x[1], 2)} for x in sorted_result]
    return similar_products

def find_similarity(cosine_sim_matrix, index, n=10):
    # calculate cosine similarity between each vectors
    result = list(enumerate(cosine_sim_matrix[index]))
    # Sorting the Score
    sorted_result = sorted(result,key=lambda x:x[1],reverse=True)[1:n+1]
    similar_products =  [{'value': unique_df.iloc[x[0]]['Product Name'], 'score' : round(x[1], 2)} for x in sorted_result]
    return similar_products

def find_manhattan_distance(sim_matrix, index, n=10):   
    # Getting Score and Index
    result = list(enumerate(sim_matrix[index]))
    # Sorting the Score and taking top 10 products
    sorted_result = sorted(result,key=lambda x:x[1],reverse=False)[1:10+1]
    # Mapping index with data
    similar_products =  [{'value': unique_df.iloc[x[0]]['Product Name'], 'score' : round(x[1], 2)} for x in sorted_result]
    return similar_products


def get_recommendation_tfidf(product_id, df, similarity, n=10):

    row = df.loc[df['Product Name'] == product_id]
    index = list(row.index)[0]
    description = row['desc_lowered'].loc[index]
    tfidf_matrix = tfidf_vec.fit_transform(desc_list)

    if similarity == "cosine":
        sim_matrix = cosine_similarity(tfidf_matrix)
        products = find_similarity(sim_matrix , index)

    elif similarity == "manhattan":
        sim_matrix = manhattan_distances(tfidf_matrix)
        products = find_manhattan_distance(sim_matrix , index)

    else:
        sim_matrix = euclidean_distances(tfidf_matrix)
        products = find_euclidean_distances(sim_matrix , index)

    return products


opt = ["cosine","manhattan","euclidean"]

st.sidebar.text_input("Your name", key="product_id")

st.sidebar.selectbox('Algoritmo', key = 'similarity',options = opt)


st.write(pd.DataFrame(get_recommendation_tfidf(st.session_state.product_id,unique_df, similarity = st.session_state.similarity )))


 