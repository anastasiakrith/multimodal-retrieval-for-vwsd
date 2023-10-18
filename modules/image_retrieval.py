import requests
from io import BytesIO
from PIL import Image
import numpy as np

from modules.vl_transformers import CLIP, ALIGN
from modules.metrics import cosine_similarity, euclidean_distance, manhattan_distance

class WikipediaModule:

    def __init__(self, max_images=10):
        self.max_images = max_images


    def get_wikipedia_page_titles(self, context, target_word):

        session = requests.Session()
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": context
        }
        response = session.get(url=url, params=params)
        json = response.json()
        titles = []
        for x in json['query']['search']:
            titles.append(x['title'])
        
        if len(titles) > 0:
            return titles
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": target_word
        }
        response = session.get(url=url, params=params)
        json = response.json()
        titles = []
        for x in json['query']['search']:
            titles.append(x['title'])
            
        return titles


    def get_image_url(self, title):

        query = 'http://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles='
        try:
            api_res = requests.get(query + '_'.join(title.split())).json()
            first_part = api_res['query']['pages']
            # this is a way around not knowing the article id number
            for key, value in first_part.items():
                if (value['original']['source']):
                    data = value['original']['source']
                    return data
        except Exception as exc:
            print(exc)
            print("Partial URL: " + '_'.join(title.split()))
            data = None
        return data

    
    def retrieve(self, context, target_word):

        titles = self.get_wikipedia_page_titles(context, target_word)
        images_list = []
        for title in titles:
            image_url = self.get_image_url(title)
            if image_url is not None:
                response = requests.get(image_url)
                if response.ok:
                    images_list.append(Image.open(BytesIO(response.content)))
            
            if len(images_list) >= self.max_images:
                break

        return images_list[:self.images_list]



class WikidataModule:

    def __init__(self, max_images=10):
        self.max_images = max_images

    def search_entity_by_text(self, text):

        def fetch_wikidata_api(params):
            url = 'https://www.wikidata.org/w/api.php'
            try:
                return requests.get(url, params=params)
            except:
                return 'There was and error'


        # What text to search for
        query = text
        # Which parameters to use
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'search': query,
            'language': 'en'
        }
    
        response = fetch_wikidata_api(params)
        if not response.ok:
            return None
        data = response.json()
        return data["search"][0]["id"]

    def search_image_by_entity(self, entity):

        endpoint = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"
        query = """
                SELECT ?image 
                WHERE {
                    wd:"""+ entity +""" wdt:P18 ?image
                }
                """
        
        response = requests.get(endpoint, params={"format": "json", "query": query})
        if not response.ok:
            return []
        response = response.json()
        images = [record['image']['value'] for record in response['results']['bindings']]
        return images


    def retrieve(self, context, target_word):

        wikidata_entities = []
        for search_str in [target_word, context]:

            wikidata_entity = self.search_entity_by_text(search_str)
            if (not wikidata_entity) or (wikidata_entity in wikidata_entities):
                continue
            wikidata_entities.append(wikidata_entity)
            
        images_list = []
        for entity in wikidata_entities:
            image_urls = self.search_image_by_entity(wikidata_entity)
            for image_url in image_urls:
                response = requests.get(image_url)
                if response.ok:
                    images_list.append(Image.open(BytesIO(response.content)))
              
            if len(images_list) >= self.max_images:
                break
        
        return images_list[:self.images_list]
            

AVAILABLE_VL_TRANSFORMERS_FOR_IMG_RETRIEVAL = {
    'clip': lambda: CLIP(large=False),
    'align': lambda: ALIGN(),
}

AVAILABLE_WIKI = {
    'wikidata': lambda: WikidataModule(),
    'wikipedia': lambda: WikipediaModule()
}

AVAILABLE_METRICS = {
    'cosine': cosine_similarity,
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance
}


class ImageRetrievalModule:

    def __init__(self, wiki, vl_transformer, metric):

         # VL Transformer #
        if vl_transformer not in AVAILABLE_VL_TRANSFORMERS_FOR_IMG_RETRIEVAL:
            raise ValueError(f"Invalid VL transformer: {vl_transformer}. Should be one of {AVAILABLE_VL_TRANSFORMERS_FOR_IMG_RETRIEVAL.keys()}")

        self.vl_transformer = AVAILABLE_VL_TRANSFORMERS[vl_transformer]()

        # Wiki Source for Image Retrieval #
        if wiki not in AVAILABLE_WIKI:
            raise ValueError(f"Invalid Wiki: {wiki}. Should be one of {AVAILABLE_WIKI.keys()}")

        self.wiki = AVAILABLE_WIKI[wiki]()

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

        # Retrieve Images from Wikipedia/Wikidata #
        retrieved_images = self.wiki.retrieve(given_phrase, target_word)

        # Extract features #
        retrieved_images_features = []
        for image in retrieved_images:
            try:
                retrieved_images_features.append(self.vl_transformer.get_image_features(image))
            except:
                pass

        given_images_features = []
        for image in images:
            try:
                given_images_features.append(self.vl_transformer.get_image_features(image))
            except:
                given_images_features.append([])


        if self.similarity_metric is not None:
            # Calculate similarity #
            similarities = []
            for img_feat in given_images_features:
                similarities.append(
                    np.max([self.similarity_metric(img_feat, retrieved_feat) if len(img_feat) > 0 else 0 for retrieved_feat in retrieved_images_features])
                )
            similarities = np.array(similarities)
            return {'ordered_pred_images': np.argsort(similarities)[::-1]}

        else:
            # Calculate distance #
            distances = []
            for img_feat in given_images_features:
                distances.append(
                    np.min([self.distance_metric(img_feat, retrieved_feat) if len(img_feat) > 0 else 0 for retrieved_feat in retrieved_images_features])
                )
            return {'ordered_pred_images': np.argsort(np.array(distances))}