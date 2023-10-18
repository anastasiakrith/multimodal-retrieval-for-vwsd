import requests
from io import BytesIO
from PIL import Image
import numpy as np

from modules.vl_transformers import CLIP, ALIGN
from modules.captioners import BLIP, GIT
from modules.metrics import cosine_similarity, euclidean_distance, manhattan_distance
from modules.llms import GPT_3


AVAILABLE_FEATURE_EXTRACTORS = {
    'clip': lambda: CLIP(large=False, text_features=True),
    'clip': lambda: CLIP(large=True, text_features=True)
    'align': lambda: ALIGN(),
}

AVAILABLE_METRICS = {
    'cosine': cosine_similarity,
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance
}

AVAILABLE_CAPTIONERS = {
    'blip': lambda strategy: BLIP(strategy=strategy, large=False),
    'blip_large': lambda strategy: BLIP(strategy=strategy, large=True),
    'git': lambda strategy: GIT(strategy=strategy, large=False),
    'git_large': lambda strategy: GIT(strategy=strategy, large=True),
}

AVAILABLE_LLMS = {
    'gpt-3': lambda: GPT_3()
}

AVAILABLE_PROMPT_TEMPLATES = {
    'exact': lambda x: f"{x} ",
    'what_is': lambda x: f"What is {x}?",
    'meaning_of': lambda x: f"What is the meaning of {x}?",
    'describe': lambda x: f"Describe {x}.",
}


class TextRetrievalModule:

    def __init__(self, captioner, strategy, feature_extractor, metric, llm=None, prompt_template=None):

        # Feature Extractors #
        if vl_transformer not in AVAILABLE_FEATURE_EXTRACTORS:
            raise ValueError(f"Invalid VL transformer: {feature_extractor}. Should be one of {AVAILABLE_FEATURE_EXTRACTORS.keys()}")

        self.feature_extractor = AVAILABLE_FEATURE_EXTRACTORS[feature_extractor]()

        # Captioner #
        if captioner not in AVAILABLE_CAPTIONERS:
            raise ValueError(f"Invalid Captioner: {captioner}. Should be one of {AVAILABLE_CAPTIONERS.keys()}")
        if strategy not in ['greedy', 'beam']:
            raise ValueError(f"Invalid captioner strategy: {strategy}. Should be one of ['greedy', 'beam']")

        self.captioner = AVAILABLE_CAPTIONERS[captioner](strategy)
        self.strategy = strategy

        # LLM or None #
        if llm is not None and llm not in AVAILABLE_LLMS:
            raise ValueError(f"Invalid LLM: {llm}. Should be one of {AVAILABLE_LLMS.keys()}")

        self.llm = AVAILABLE_LLMS[llm] if llm is not None else llm

        # Prompt or None #
        if self.llm is not None and prompt_template not in AVAILABLE_PROMPT_TEMPLATES:
            raise ValueError(f"Invalid prompt template: {prompt_template}. Should be one of {AVAILABLE_PROMPT_TEMPLATES.keys()}")

        self.prompt = AVAILABLE_PROMPT_TEMPLATES[prompt_template] if self.llm is not None else None

        # Similarity/Distance metric #
        if metric not in AVAILABLE_METRICS:
            raise ValueError(f"Invalid metric: {metric}. Should be one of {AVAILABLE_METRICS.keys()}")

        if metric == 'cosine':
            self.similarity_metric = AVAILABLE_METRICS[metric]
            self.distance_metric = None
        else:
            self.similarity_metric = None
            self.distance_metric = AVAILABLE_METRICS[metric]


    def run(self, given_phrase, target_word, images):

        ################################
        # Retrieve Captions for Images #
        ################################
        captions = [self.captioner.run(image) for image in images]

        ###########################
        # Enhanced text if needed #
        ###########################
        text = given_phrase
        if self.llm is not None:
            text = self.llm.completion(self.prompt(given_phrase))

        ####################    
        # Extract features #
        ####################
        # Captions #
        captions_features = []
        if self.strategy == 'greedy':
            for caption in captions:
                try:
                    captions_features.append(self.feature_extractor.get_text_features(caption))
                except:
                    captions_features.append([])
        else:
            for cap_group in captions:
                tmp = []
                for caption in cap_group:
                    try:
                        tmp.append(self.feature_extractor.get_text_features(caption))
                    except:
                        tmp.append([])
                captions_features.append(tmp)
        
        # Phrase #
        text_features = self.feature_extractor.get_text_features(text) 
        
        ####################
        # Calculate Metric #
        ####################
        if self.similarity_metric is not None:
            # Calculate similarity #
            similarities = []
            if self.strategy == "greedy":
                similarities = np.array([self.similarity_metric(cap_feat, text_features) if len(cap_feat) > 0 else 0 for img_feat in captions_features])
            else:
                for cap_group in captions_features:
                    if len(cap_group) == 0:
                        similarities.append(0)
                    else:
                        similarities.append(
                            np.max([self.similarity_metric(cap_feat, text_features) for cap_feat in cap_group])
                        )
                similarities = np.array(similarities)

            return {'ordered_pred_images': np.argsort(similarities)[::-1]}

        else:
            # Calculate distance #
            distances = []
            if self.strategy == "greedy":
                distances = np.array([self.distance_metric(cap_feat, text_features) if len(cap_feat) > 0 else 0 for img_feat in captions_features])
            else:
                for cap_group in captions_features:
                    if len(cap_group) == 0:
                        distances.append(0)
                    else:
                        distances.append(
                            np.max([self.distance_metric(cap_feat, text_features) for cap_feat in cap_group])
                        )
                distances = np.array(distances)

            return  {'ordered_pred_images': np.argsort(distances)}