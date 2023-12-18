import argparse
import torch
import pandas as pd
from transformers import pipeline as pipelineHF
from transformers import AutoTokenizer
from summary_qg import extract_qa_pairs

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fast', help="Use the smaller and faster versions of the models", action='store_true')
args = parser.parse_args()


df = pd.read_csv('train_with_summaries.csv')

qg_model = "valhalla/t5-small-qa-qg-hl" if args.fast else "valhalla/t5-base-qa-qg-hl"
sum_model = "sshleifer/distilbart-cnn-6-6" if args.fast else "facebook/bart-large-cnn"


qg = pipelineHF("multitask-qa-qg", model=qg_model)
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# GPU setup
if torch.cuda.is_available():
    qg = qg.to('cuda')

# Data processing
data = []
for _, row in df.iterrows():
    summary_text = row['machine_summary']  

    qa_pairs = extract_qa_pairs(tokenizer, qg, summarizer, summary_text)

    for pair in qa_pairs:
        data.append([row['book_id'], row['chapter'], pair['question'], pair['answer']])

output_df = pd.DataFrame(data, columns=['BookID', 'Chapter', 'Question', 'Answer'])
output_df.to_csv('out_with_qa.csv', index=False)
