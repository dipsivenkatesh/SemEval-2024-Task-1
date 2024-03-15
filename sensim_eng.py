from openai import OpenAI
import pandas as pd
import re
import numpy as np
from scipy.stats import spearmanr, pearsonr
import json

client = OpenAI()

def get_similarity_scores_openai(lang, model_id):
    file_path = f"data/public_test/{lang}/{lang}_test.csv"
    df_str_rel = pd.read_csv(file_path)
    df_str_rel['Text'].values
    df_str_rel['Split_Text'] = df_str_rel['Text'].apply(lambda x: x.split("\n"))
   

    def embed_list(text_list, model_id):
        response = client.embeddings.create(
            model = model_id,
            input=text_list
        )
        return json.loads(response.model_dump_json())
        
    def get_embedded_lists_1(model_id):
        s1_list = []
        s2_list = []
        for sen in df_str_rel["Split_Text"][:2048]:
            s1_list.append(sen[0])
            s2_list.append(sen[1])
        s1_embedded_list = embed_list(s1_list, model_id)
        s2_embedded_list = embed_list(s2_list, model_id)
        return s1_embedded_list, s2_embedded_list
    def get_embedded_lists_2(model_id):
        s1_list = []
        s2_list = []
        for sen in df_str_rel["Split_Text"][2048:]:
            s1_list.append(sen[0])
            s2_list.append(sen[1])
        s1_embedded_list = embed_list(s1_list, model_id)
        s2_embedded_list = embed_list(s2_list, model_id)
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

    s1_embedded_list_obj_1, s2_embedded_list_obj_1 = get_embedded_lists_1(model_id)
    s1_embedded_list_1 = [obj.get("embedding") for obj in s1_embedded_list_obj_1.get("data")]
    s2_embedded_list_1 = [obj.get("embedding") for obj in s2_embedded_list_obj_1.get("data")]
    s1_embedded_list_obj_2, s2_embedded_list_obj_2 = get_embedded_lists_2(model_id)
    s1_embedded_list_2 = [obj.get("embedding") for obj in s1_embedded_list_obj_2.get("data")]
    s2_embedded_list_2 = [obj.get("embedding") for obj in s2_embedded_list_obj_2.get("data")]
    s1_embedded_list = s1_embedded_list_1 + s1_embedded_list_2
    s2_embedded_list = s2_embedded_list_1 + s2_embedded_list_2
    s_sim_pred_scores = []
    embedded_list_size = len(s1_embedded_list)
    # for index,row in df_str_rel.iterrows():
    for i in range(embedded_list_size):
        s1, s2 = s1_embedded_list[i], s2_embedded_list[i]
        s_sim = sen_tran_similarity(s1, s2)
        s_sim_pred_scores.append(s_sim)

    df_str_rel['Pred_Score'] = s_sim_pred_scores
    print(df_str_rel.head())
    df_str_rel[['PairID', 'Pred_Score']].to_csv(f'answers/pred_{lang}.csv', index=False)


