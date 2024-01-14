import torch

from transformers import CLIPProcessor, CLIPModel, CLIPTextModelWithProjection, AutoTokenizer, AlignProcessor, AlignModel, BlipForImageTextRetrieval, AutoProcessor, BlipModel

class CLIP:
    """
    Wrapper for openai/clip-vit-{}-patch{} from hugging-face
    """
    def __init__(self, large=False, text_features=False):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.large = large
        if not self.large:
            self.load_from = "openai/clip-vit-base-patch32"
        else:
            self.load_from = "openai/clip-vit-large-patch14"
        if text_features:
            self.model = CLIPTextModelWithProjection.from_pretrained(self.load_from).to(self.device)
            self.processor = AutoTokenizer.from_pretrained(self.load_from)
        else:
            self.model = CLIPModel.from_pretrained(self.load_from).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.load_from)
    
    def run(self, text, images):

        model_input = self.processor(text=text, images=images, return_tensors="pt", padding=True).to(self.device)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**model_input)
        similarity_score = model_output.logits_per_image.flatten()
        if self.device != "cpu":
            similarity_score = similarity_score.cpu()
        probs = similarity_score.softmax(dim=0)
        
        return {'probs': probs.tolist(), 'similarity_score': similarity_score.tolist()}


    def get_image_features(self, image):
    
        model_input = self.processor(images=image, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            image_features = self.model.get_image_features(**model_input)
        if self.device == "cpu":
            return image_features.flatten().numpy()      
        return image_features.flatten().cpu().numpy()     


    def get_text_features(self, text):
    
        model_input = self.processor(text=text, padding=True, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**model_input)
        if self.device == "cpu":
            return  model_output.text_embeds[0].numpy() 
        return  model_output.text_embeds[0].cpu().numpy()  

class ALIGN:
    """
    Wrapper for kakaobrain/align-base from hugging-face
    """

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_from = "kakaobrain/align-base"
        self.model = AlignModel.from_pretrained(self.load_from).to(self.device)
        self.processor = AlignProcessor.from_pretrained(self.load_from)

    def run(self, text, images):

        model_input = self.processor(text=text, images=images, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**model_input)
        similarity_score = model_output.logits_per_image.flatten()
        if self.device != "cpu":
            similarity_score = similarity_score.cpu()
        probs = similarity_score.softmax(dim=0)
        
        return {'probs': probs.tolist(), 'similarity_score': similarity_score.tolist()}


    def get_image_features(self, image):
        
        model_input = self.processor(images=image, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            image_features = self.model.get_image_features(**model_input)
        if self.device == "cpu":
            return image_features.flatten().numpy()
        return image_features.flatten().cpu().numpy()
    
    def get_text_features(self, text):
        
        model_input = self.processor(text=text, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            text_features = self.model.get_text_features(**model_input)
        if self.device == "cpu":
            return text_features.flatten().numpy()
        return text_features.flatten().cpu().numpy()

class BLIP_COCO:
    """
    Wrapper for Salesforce/blip-itm-{}-coco from hugging-face
    """
    def __init__(self, large=False):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.large = large
        if not self.large:
            self.load_from = "Salesforce/blip-itm-base-coco"
        else:
            self.load_from = "Salesforce/blip-itm-large-coco"
        self.model = BlipForImageTextRetrieval.from_pretrained(self.load_from).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.load_from)
    
    def run(self, text, images):

        model_input = self.processor(text=text, images=images, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**model_input)
        similarity_score = model_output.itm_score[:, 1].flatten()
        if self.device != "cpu":
            similarity_score = similarity_score.cpu()
        probs = similarity_score.softmax(dim=0).flatten()
        
        return  {'probs': probs.tolist(), 'similarity_score': similarity_score.tolist()}


class BLIP_FLICKR:
    """
    Wrapper for Salesforce/blip-itm-{}}-flickr from hugging-face
    """
    def __init__(self, large=False):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.large = large
        if not self.large:
            self.load_from = "Salesforce/blip-itm-base-flickr"
        else:
            self.load_from = "Salesforce/blip-itm-large-flickr"
        self.model = BlipForImageTextRetrieval.from_pretrained(self.load_from).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.load_from)
    
    def run(self, text, images):

        model_input = self.processor(text=text, images=images, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**model_input)
        similarity_score = model_output.itm_score[:, 1].flatten()
        if self.device != "cpu":
            similarity_score = similarity_score.cpu()
        probs = similarity_score.softmax(dim=0).flatten()
        
        return  {'probs': probs.tolist(), 'similarity_score': similarity_score.tolist()}


class BLIP_CAPTIONING:
    """
    Wrapper for Salesforce/blip-image-captioning-base from hugging-face
    """
    def __init__(self):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_from = "Salesforce/blip-image-captioning-base"
        self.model = BlipModel.from_pretrained(self.load_from).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.load_from)
    
    def get_text_features(self, text):

        model_input = self.processor(text=text, padding=True, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            text_features = self.model.get_text_features(**model_input)
        if self.device == "cpu":
            return text_features.flatten().numpy()
        return text_features.flatten().cpu().numpy()