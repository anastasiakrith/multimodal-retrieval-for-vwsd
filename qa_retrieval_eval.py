import argparse
from tqdm import tqdm

from modules.qa import QAModule
from modules.qa import AVAILABLE_LLMS, AVAILABLE_CAPTIONERS, AVAILABLE_QA_PROMPT_TEMPLATES
from modules.dataset import Dataset
from modules.metrics import ScoreModule

def get_k_default_value(captioner, captioner_strategy):
    if captioner_strategy == 'greedy': # all greedy
        return 5
    if captioner == 'blip-large': # blip-large beam
        return 1
    return 2 # git-large 


def run_qa_retrieval(llm, captioner, captioner_strategy, prompt, dataset_path):

    dataset = Dataset(base_dir=dataset_path)

    test_dataset = dataset.test_dataloader()

    qa = QAModule(llm=llm, captioner=captioner, strategy=captioner_strategy, prompt_template=prompt)
    score = ScoreModule(approach='qa_retrieval') 
           
    for i in tqdm(range(len(test_dataset))):
        prediction = qa.run(given_phrase=test_dataset[i]['given_phrase'], images=test_dataset[i]['images'])
        score.add(golden_image_index=test_dataset[i]['gold_image_index'], predictions=prediction)

    print(f'Accuracy Score: {score.accuracy_score()}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-llm", "--llm_model", default="gpt-3.5", help=f"Choose LLM model. Options: {AVAILABLE_LLMS.keys()}")
    parser.add_argument("-captioner", "--captioner", default="git", help=f"Choose Captioner model. Options: {AVAILABLE_CAPTIONERS.keys()}")
    parser.add_argument("-strategy", "--captioner_strategy", default="greedy", help=f"Choose Captioner strategy. Options: 'greedy', 'beam'")
    parser.add_argument("-prompt", "--prompt", default='no_CoT', help=f"Choose prompt. Options: {AVAILABLE_QA_PROMPT_TEMPLATES.keys()}")
    parser.add_argument("-dataset_path", "--dataset_path", default=None, help=f"Set dataset path.")

    args = parser.parse_args()

    run_qa_retrieval(
        llm=args.llm_model,
        captioner=args.captioner,
        captioner_strategy=args.captioner_strategy,
        prompt=args.prompt,
        dataset_path=args.dataset_path
    )
