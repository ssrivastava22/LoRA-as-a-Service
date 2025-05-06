from datasets import load_dataset, Dataset
import json

def tokenize(text, tokenizer):
    return tokenizer(text["text"], truncation=True, padding="max_length", max_length=512)

def prepare_dataset(dataset_path, tokenizer):

    with open(dataset_path, 'r') as f:
        lines = [json.loads(line) for line in f]
    
    prompts = [ex["prompt"] + tokenizer.eos_token for ex in lines]
    responses = [ex["response"] + tokenizer.eos_token for ex in lines]

    combined_data = [{"text": prompt + response} for prompt, response in zip(prompts, responses)]
    dataset = Dataset.from_list(combined_data)
    dataset = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    return dataset