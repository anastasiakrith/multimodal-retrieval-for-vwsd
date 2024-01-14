import re

from modules.llms import GPT_3_5
from modules.captioners import BLIP, GIT
from modules.prompts import no_CoT_prompt, think_prompt, CoT_prompt

AVAILABLE_LLMS = {
    'gpt-3.5': lambda: GPT_3_5(),
}

AVAILABLE_CAPTIONERS = {
    'blip': lambda strategy: BLIP(strategy=strategy, large=False),
    'blip_large': lambda strategy: BLIP(strategy=strategy, large=True),
    'git': lambda strategy: GIT(strategy=strategy, large=False),
    'git_large': lambda strategy: GIT(strategy=strategy, large=True),
}

AVAILABLE_QA_PROMPT_TEMPLATES = {
    'no_CoT': no_CoT_prompt,
    'CoT': think_prompt,
}


class QAModule:

    def __init__(self, llm, captioner, strategy, prompt_template):

        if llm not in AVAILABLE_LLMS:
            raise ValueError(f"Invalid LLM: {llm}. Should be one of {AVAILABLE_LLMS.keys()}")
        
        if captioner not in AVAILABLE_CAPTIONERS:
            raise ValueError(f"Invalid LLM: {captioner}. Should be one of {AVAILABLE_CAPTIONERS.keys()}")

        if strategy not in ['greedy', 'beam']:
            raise ValueError(f"Invalid captioner strategy: {strategy}. Should be one of ['greedy', 'beam']")
        
        if prompt_template not in AVAILABLE_QA_PROMPT_TEMPLATES:
            raise ValueError(f"Invalid prompt template: {prompt_template}. Should be one of {AVAILABLE_QA_PROMPT_TEMPLATES.keys()}")

        self.strategy = strategy
        self.llm = AVAILABLE_LLMS[llm]()
        self.captioner = AVAILABLE_CAPTIONERS[captioner](strategy)
        self.CoT_flag = prompt_template == 'CoT' 
        self.prompt = AVAILABLE_QA_PROMPT_TEMPLATES[prompt_template]

    def parse_answer(self, answer):
        pred = re.findall(r'\(A\)|\(B\)|\(C\)|\(D\)|\(E\)|\(F\)|\(G\)|\(H\)|\(I\)|\(J\)', answer)
        if len(pred) == 1:
            mapping = {'(A)': 0, '(B)':1, '(C)': 2, '(D)': 3, '(E)': 4, '(F)': 5, '(G)': 6, '(H)': 7, '(I)': 8, '(J)': 9}
            return mapping[pred[0]]
        
        pred = re.search(r'A|B|C|D|E|F|G|H|I|J', answer)
        if len(pred) > 0:
            mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9}
            return mapping[pred[0]]
        return None


    def run(self, given_phrase, images):

        captions_list = [self.captioner.run(img) for img in images]
        llm_prompt = self.prompt(given_phrase=given_phrase, captions_list=captions_list, strategy=self.strategy)
        llm_answer = self.llm.completion(llm_prompt)
        if self.CoT_flag:
            llm_prompt = CoT_prompt(given_phrase=given_phrase, captions_list=captions_list, strategy=self.strategy, llm_response=llm_answer)
            llm_answer = self.llm.completion(llm_prompt)
        return self.parse_answer(llm_answer)