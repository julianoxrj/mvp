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
def load_data1():
    df0 = pd.read_excel(file)
    return df0

file = "treino_final0.xlsx"
df1 = load_data1()
df1.isnull().sum(axis = 0)
df1.dropna().reset_index(inplace = True)


count_vectorizer = CountVectorizer()

tfidf_vec = TfidfVectorizer( analyzer='word', ngram_range=(1,3))

unique_df1 = df1.drop_duplicates(subset=['Objeto'], keep='first')

unique_df1['Objeto'] = unique_df1['Objeto'].apply(lambda x: x.lower())

unique_df1['Objeto'] = unique_df1['Objeto'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

desc_list1 = list(unique_df1['Objeto'])

unique_df1= unique_df1.reset_index(drop=True)

unique_df1.reset_index(inplace=True)



def find_euclidean_distances1(sim_matrix, index, n=10): 
    # Getting Score and Index
    result = list(enumerate(sim_matrix[index]))
    # Sorting the Score and taking top 10 products
    sorted_result = sorted(result,key=lambda x:x[1],reverse=False)[1:10+1]
    # Mapping index with data
    similar_products =  [{'value': unique_df1.iloc[x[0]]['Objeto'], 'score' : round(x[1], 2)} for x in sorted_result]
    return similar_products

def find_similarity1(cosine_sim_matrix, index, n=10):
    # calculate cosine similarity between each vectors
    result = list(enumerate(cosine_sim_matrix[index]))
    # Sorting the Score
    sorted_result = sorted(result,key=lambda x:x[1],reverse=True)[1:n+1]
    similar_products =  [{'value': unique_df1.iloc[x[0]]['Objeto'], 'score' : round(x[1], 2)} for x in sorted_result]
    return similar_products

def find_manhattan_distance1(sim_matrix, index, n=10):   
    # Getting Score and Index
    result = list(enumerate(sim_matrix[index]))
    # Sorting the Score and taking top 10 products
    sorted_result = sorted(result,key=lambda x:x[1],reverse=False)[1:10+1]
    # Mapping index with data
    similar_products =  [{'value': unique_df1.iloc[x[0]]['Objeto'], 'score' : round(x[1], 2)} for x in sorted_result]
    return similar_products





def find_similarity1(cosine_sim_matrix, index, n=10):

    result = list(enumerate(cosine_sim_matrix[index]))
    sorted_result = sorted(result,key=lambda x:x[1],reverse=True)[1:n+1]
    similar_products =  [{'value': unique_df1.iloc[x[0]]['Objeto'], 'score' : round(x[1], 2)} for x in sorted_result]
    return similar_products

def get_recommendation_tfidf_editais(edital_id, df1, similarity = "cosine", n=10):
    row1 = df1.loc[df1['Objeto'] == edital_id]
    index1 = list(row1.index)[0]
    description1 = row1['Objeto'].loc[index1]
    tfidf_matrix1 = tfidf_vec.fit_transform(desc_list1)    
    
    if similarity == "cosine":
        sim_matrix = cosine_similarity(tfidf_matrix1)
        products1 = find_similarity1(sim_matrix , index1)

    elif similarity == "manhattan":
        sim_matrix = manhattan_distances(tfidf_matrix1)
        products1 = find_manhattan_distance1(sim_matrix , index1)

    else:
        sim_matrix = euclidean_distances(tfidf_matrix1)
        products1 = find_euclidean_distances1(sim_matrix , index1)
    
    return products1


opt = ["cosine","manhattan","euclidean"]

st.sidebar.text_area("Your name", key="edital_id")

st.sidebar.selectbox('Algoritmo', key = 'similarity',options = opt)





st.write(pd.DataFrame(get_recommendation_tfidf_editais(st.session_state.edital_id, unique_df1, similarity = st.session_state.similarity )))
