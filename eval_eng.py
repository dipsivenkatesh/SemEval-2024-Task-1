from sensim_eng import get_similarity_scores_openai

lang_model_map_eng = {
    "eng": "text-embedding-3-large"
}

for lang, model in lang_model_map_eng.items():
    print(model, lang)
    file_path = f"data/public_test/{lang}/{lang}_test.csv"
    get_similarity_scores_openai(lang=lang, model_id=model)

print("evaluations done for english")