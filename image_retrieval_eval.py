import argparse
from tqdm import tqdm

from modules.image_retrieval import ImageRetrievalModule
from modules.image_retrieval import AVAILABLE_VL_TRANSFORMERS_FOR_IMG_RETRIEVAL, AVAILABLE_WIKI, AVAILABLE_METRICS
from modules.dataset import Dataset
from modules.metrics import ScoreModule

def run_image_retrieval(vl_transformer, wiki, metric, dataset_path=None):

    dataset = Dataset(base_dir=dataset_path)
    test_dataset = dataset.test_dataloader()
    
    image_retrieval = ImageRetrievalModule(wiki=wiki, vl_transformer=vl_transformer, metric=metric)
    score = ScoreModule(approach='image_retrieval')

    for i in tqdm(range(len(test_dataset))):
        retrieval = image_retrieval.run(given_phrase=test_dataset[i]['given_phrase'], target_word=test_dataset[i]['word'], images=test_dataset[i]['images'])
        score.add(golden_image_index=test_dataset[i]['gold_image_index'], predictions=retrieval['ordered_pred_images'])
    
    print(f'Accuracy Score: {score.accuracy_score()}')
    print(f'MRR Score: {score.mrr_score()}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-vl", "--vl_transformer", default="clip", help=f"Choose VL transformer model. Options: {AVAILABLE_VL_TRANSFORMERS_FOR_IMG_RETRIEVAL.keys()}")
    parser.add_argument("-wiki", "--wiki", default=None, help=f"Choose wiki source. Options: {AVAILABLE_WIKI.keys()}")
    parser.add_argument("-metric", "--metric", default="cosine", help=f"Choose Similarity/Distance metric. Options: {AVAILABLE_METRICS.keys()}")
    parser.add_argument("-dataset_path", "--dataset_path", default=None, help=f"Set dataset path.")

    args = parser.parse_args()

    run_image_retrieval(
        vl_transformer=args.vl_transformer,
        wiki=args.wiki,
        metric=args.metric,
        dataset_path=args.dataset_path
    )
