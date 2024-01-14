import torch
from sentence_transformers import SentenceTransformer


class SBERT_FEATURE_EXTRACTOR:

    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name) # load model 

    def get_text_features(self, text):
        if self.device == "cpu":
            return self.model.encode(text, convert_to_tensor=True).data.flatten().numpy()
        return self.model.encode(text, convert_to_tensor=True).data.cpu().flatten().numpy()


AVAILABLE_SBERT_FEATURE_EXTRACTORS = {
    'xlm-distilroberta-base': lambda: SBERT_FEATURE_EXTRACTOR(model_name='xlm-r-distilroberta-base-paraphrase-v1'),
    'distilroberta-base': lambda: SBERT_FEATURE_EXTRACTOR(model_name='paraphrase-distilroberta-base-v2'),
    'stsb-roberta-base': lambda: SBERT_FEATURE_EXTRACTOR(model_name='stsb-roberta-base-v2'),
    'stsb-mpnet-base': lambda: SBERT_FEATURE_EXTRACTOR(model_name='stsb-mpnet-base-v2'),
    'all-MiniLM-L6': lambda: SBERT_FEATURE_EXTRACTOR(model_name='all-MiniLM-L6-v2'),
    'all-MiniLM-L12': lambda: SBERT_FEATURE_EXTRACTOR(model_name='all-MiniLM-L12-v2'),
    'all-mpnet-base': lambda: SBERT_FEATURE_EXTRACTOR(model_name='all-mpnet-base-v2'),
    'multi-QA-distilbert': lambda: SBERT_FEATURE_EXTRACTOR(model_name='multi-qa-distilbert-cos-v1'),
    'multi-QA MiniLM-L6': lambda: SBERT_FEATURE_EXTRACTOR(model_name='multi-qa-MiniLM-L6-cos-v1')
}
