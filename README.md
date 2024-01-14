# Language Models and Multimodal Retrieval for Visual Word Sense Disambiguation (VWSD)

## Install

```
git clone https://github.com/anastasiakrith/multimodal-retrieval-for-vwsd.git
cd multimodal-retrieval-for-vwsd
```

### Setting up (virtualenv)

On the project folder run the following commands:

1. ```$ virtualenv env```    to create a virtual environment
2. ```$ source venv/bin/activate``` to activate the environment
3. ```$ pip install -r requirements.txt``` to install packages
4. Create a ```.env``` file with the environmental variables. The project needs a ```OPENAI_API_KEY``` with the API key corresponding to your openai account, and optionally a ```DATASET_PATH``` corresponding to the absolute path of [VWSD dataset](https://raganato.github.io/vwsd/).


## Running the project

### VL Retrieval
```
python vl_retrieval_eval.py -llm "gpt-3.5" -vl "clip" -baseline -penalty 
```

### QA Retrieval
```
python qa_retrieval_eval.py -llm "gpt-3.5" -captioner "git" -strategy "greedy" -prompt "no_CoT" -zero_shot
```

### Image-to-Image Retrieval
```
python image_retrieval_eval.py -vl "clip" -wiki "wikipedia" -metric "cosine"
```

### Text-to-Text Retrieval
```
python text_retrieval_eval.py -captioner "git" -strategy "greedy" -extractor "clip" -metric "cosine"
```


## Acknowledgement
The implementation relies on resources from [openai-api](https://platform.openai.com/docs/api-reference) and [hugging-face transformers](https://github.com/huggingface/transformers).
