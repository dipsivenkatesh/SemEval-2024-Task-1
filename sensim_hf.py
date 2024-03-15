import re
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
# pip install -U sentence-transformers
# from sentence_transformers import SentenceTransformer, util
import json
import requests

plt.style.use('ggplot')

hf_token = "SECRET"
headers = {"Authorization": f"Bearer {hf_token}"}


def get_similarity_scores_huggingface(lang, model_id):
    file_path = f"data/public_test/{lang}/{lang}_test.csv"
    df_str_rel = pd.read_csv(file_path)
    df_str_rel['Text'].values
    df_str_rel['Split_Text'] = df_str_rel['Text'].apply(lambda x: x.split("\n"))
    def dice_score(s1,s2):
        s1 = s1.lower()
        s1_split = re.findall(r"\w+|[^\w\s]", s1, re.UNICODE)
        s2 = s2.lower()
        s2_split = re.findall(r"\w+|[^\w\s]", s2, re.UNICODE)
        dice_coef = len(set(s1_split).intersection(set(s2_split))) / (len(set(s1_split)) + len(set(s2_split)))
        return round(dice_coef, 2)

    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"

    def embed_list(text_list):
        response = requests.post(api_url, headers=headers, json={"inputs": text_list, "options":{"wait_for_model":True}})
        return response.json()

    def get_embedded_lists():
        s1_list = []
        s2_list = []
        for sen in df_str_rel["Split_Text"]:
            s1_list.append(sen[0])
            s2_list.append(sen[1])
        s1_embedded_list = embed_list(s1_list)
        s2_embedded_list = embed_list(s2_list)
        return s1_embedded_list, s2_embedded_list

    def spearman_similarity(l1, l2):
        spearman_corr, p_value = spearmanr(l1, l2)
        return spearman_corr

    def sen_tran_similarity(s1, s2):
        #Compute embedding for both lists
        l1 = s1
        l2 = s2
        s_sim = spearman_similarity(l1, l2)
        s_sim = round(s_sim, 2)
        return s_sim

    # print("getting embedded lists")
    s1_embedded_list, s2_embedded_list = get_embedded_lists()
    if isinstance(s1_embedded_list, dict) or isinstance(s2_embedded_list, dict):
        e1 = None
        e2 = None
        if isinstance(s1_embedded_list, dict):
            e1 = s1_embedded_list.get("error")
        if isinstance(s2_embedded_list, dict):
            e2 = s2_embedded_list.get("error")
        if e1 is not None or e2 is not None:
            error = {
                "error_1": e1,
                "error_2": e2
            }
            return error
    # print("got embedded lists")
    embedded_list_size = len(s1_embedded_list)
    # for index,row in df_str_rel.iterrows():
    s_sim_pred_scores = []
    for i in range(embedded_list_size):
        s1, s2 = s1_embedded_list[i], s2_embedded_list[i]
        s_sim = sen_tran_similarity(s1, s2)
        s_sim_pred_scores.append(s_sim)

    df_str_rel['Pred_Score'] = s_sim_pred_scores
    print(df_str_rel.head())
    df_str_rel[['PairID', 'Pred_Score']].to_csv(f'answers/pred_{lang}.csv', index=False)
