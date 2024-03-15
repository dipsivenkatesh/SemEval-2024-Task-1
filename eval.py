from sensim_hf import get_similarity_scores_huggingface
from sensim import get_similarity_scores_openai

lang_model_map = {
    "amh": "sentence-transformers/LaBSE",
    "arq": "text-embedding-3-large",
    "ary": "text-embedding-3-large",
    "eng": "sentence-transformers/all-MiniLM-L12-v2",
    "esp": "sentence-transformers/LaBSE",
    "hau": "sentence-transformers/LaBSE",
    "kin": "text-embedding-3-large",
    "mar": "sentence-transformers/LaBSE",
    "tel": "sentence-transformers/LaBSE"
}
# lang_model_map_eng = {
#     "eng": "text-embedding-3-large"
# }

for lang, model in lang_model_map.items():
    print(model, lang)
    file_path = f"data/public_test/{lang}/{lang}_test.csv"
    if model == "text-embedding-3-large":
        get_similarity_scores_openai(lang=lang, model_id=model)
    else:
        get_similarity_scores_huggingface(lang=lang, model_id=model)

print("evaluations done for all except english")