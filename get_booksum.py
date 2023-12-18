from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

def generate_summary(chapter_text, tokenizer, model, device):
    inputs = tokenizer([chapter_text], max_length=1024, return_tensors='pt', truncation=True)
    inputs = inputs.to(device)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def add_machine_summaries_to_split(split, tokenizer, model, device, skip_rate=5):
    machine_summaries = []
    for i, entry in enumerate(split):
        if i % skip_rate == 0:
            chapter_text = entry['chapter']
            machine_summary = generate_summary(chapter_text, tokenizer, model, device)
        else:
            machine_summary = None  
        machine_summaries.append(machine_summary)
    
    return split.add_column("machine_summary", machine_summaries)


dataset = load_dataset("kmfoda/booksum")
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

skip_rate = 5 
for split in dataset.keys():
    dataset[split] = add_machine_summaries_to_split(dataset[split], tokenizer, model, device, skip_rate)

dataset['train'].to_csv('train_with_summaries.csv', index=False)
dataset['validation'].to_csv('validation_with_summaries.csv', index=False)
dataset['test'].to_csv('test_with_summaries.csv', index=False)
