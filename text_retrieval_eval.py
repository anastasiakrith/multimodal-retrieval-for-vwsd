import argparse
from tqdm import tqdm

from modules.text_retrieval import TextRetrievalModule
from modules.text_retrieval import AVAILABLE_FEATURE_EXTRACTORS, AVAILABLE_METRICS, AVAILABLE_CAPTIONERS, AVAILABLE_LLMS, AVAILABLE_PROMPT_TEMPLATES
from modules.dataset import Dataset
from modules.metrics import ScoreModule


def run_text_retrieval(llm, prompt, captioner, captioner_strategy, feature_extractor, metric, dataset_path):

    dataset = Dataset(base_dir=dataset_path)
    test_dataset = dataset.test_dataloader()

    text_retrieval = TextRetrievalModule(captioner=captioner, strategy=captioner_strategy, feature_extractor=feature_extractor, metric=metric, llm=llm, prompt_template=prompt)
    score = ScoreModule(approach='text_retrieval')

    for i in tqdm(range(len(test_dataset))):
        retrieval = text_retrieval.run(given_phrase=test_dataset[i]['given_phrase'], images=test_dataset[i]['images'])
        score.add(golden_image_index=test_dataset[i]['gold_image_index'], predictions=retrieval['ordered_pred_images'])

    print(f'Accuracy Score: {score.accuracy_score()}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-llm", "--llm_model", default=None, help=f"Choose LLM model. Options: {AVAILABLE_LLMS.keys()} or None")
    parser.add_argument("-prompt", "--prompt", default=None, help=f"Choose prompt. Options: {AVAILABLE_PROMPT_TEMPLATES.keys()} or None")
    parser.add_argument("-captioner", "--captioner", default="git", help=f"Choose Captioner model. Options: {AVAILABLE_CAPTIONERS.keys()}")
    parser.add_argument("-strategy", "--captioner_strategy", default="greedy", help=f"Choose Captioner strategy. Options: 'greedy', 'beam'")
    parser.add_argument("-extractor", "--feature_extractor", default="clip", help=f"Choose Feature Extractor. Options: {AVAILABLE_FEATURE_EXTRACTORS.keys()}")
    parser.add_argument("-metric", "--metric", default="cosine", help=f"Choose Similarity/Distance metric. Options: {AVAILABLE_METRICS.keys()}")
    parser.add_argument("-dataset_path", "--dataset_path", default=None, help=f"Set dataset path.")

    args = parser.parse_args()

    run_text_retrieval(
        llm=args.llm_model,
        prompt=args.prompt,
        captioner=args.captioner,
        captioner_strategy=args.captioner_strategy,
        feature_extractor=args.feature_extractor,
        metric=args.metric,
        dataset_path=args.dataset_path
    )
