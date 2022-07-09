import pandas as pd
import csv
from random import randint
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")

df = pd.read_csv('raw.csv')

def correct_words(string_sequence):
    corrected_strings = [''.join(sentence).replace('\t', '').replace('\n', '').replace(' ', '') for sentence in string_sequence]
    return corrected_strings

# Remove unknown tokens from snippets
df['snippet'] = correct_words(list(df['snippet']))

# Tokenize and clean '' from snippets because they are None
clean_snippet_tokens = []
for sequence in list(sample['snippet']):
    tokens = tokenizer.tokenize(sequence)
    clean_tokens = [token for token in tokens if token != '']
    clean_snippet_tokens.append(clean_tokens)

# add <mask> to tokens
mask_map = {}
for snippet_tokens in clean_snippet_tokens:
    if len(snippet_tokens) > 512:
        continue
    try:
        random_int = randint(0, len(snippet_tokens)-1)
    except (TypeError, ValueError):
        continue
    masked_token = snippet_tokens[random_int]
    masked_snippet = snippet_tokens.copy()
    masked_snippet[random_int] = '<mask>'
    mask_map[masked_token] = masked_snippet

# Join tokens
masked_snippets = []
for snippet_tokens in mask_map.values():
    masked_snippets.append(''.join(snippet_tokens))

# Create a data dictionary
data_dict = {'word': mask_map.keys(), 'snippet': masked_snippets}

# Turn it into Dataframe and save to csv
masked_df = pd.DataFrame(data_dict)
masked_df.to_csv('masked_df.csv')