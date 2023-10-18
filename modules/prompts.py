
letters_list = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)']

def no_CoT_prompt(given_phrase, captions_list, strategy, gold_image_index=None):
    
    if gold_image_index is None:
        final_answer = ""
    else:
        # in few-shot setting
        if strategy == "greedy":
            final_answer = f"{letters_list[gold_image_index]} {captions_list[gold_image_index]}"
        elif strategy == "beam":
            final_answer = f"{letters_list[gold_image_index]} [{', '.join(captions_list[gold_image_index])}]"


    if strategy == "greedy":
        answer_choices = ', '.join([f"{letter} {caption}" for letter, caption in zip(letters_list, captions_list)])
        return f"What is the most appropriate caption for the {given_phrase}?\nAnswer Choices: {answer_choices}\nA:{final_answer}"
    elif strategy == "beam":
        answer_choices = ', '.join([f"{letter} [{', '.join(captions)}]" for letter, captions in zip(letters_list, captions_list)])
        return f"What is the most appropriate group of captions for the {given_phrase}?\nAnswer Choices: {answer_choices}\nA:{final_answer}"


def think_prompt(given_phrase, captions_list, strategy):
    return no_CoT_prompt(given_phrase, captions_list, strategy) + " Let\'s think step by step."

def CoT_prompt(given_phrase, captions_list, strategy, llm_response):
    return f"{think_prompt(given_phrase, captions_list, strategy)}\n{llm_response} Therefore, among A through J the answer is "
