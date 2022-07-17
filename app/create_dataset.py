from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset('csv', data_files='raw.csv')
# Define model name
model_checkpoint = 'huggingface/CodeBERTa-small-v1'
# Define a tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Split dataset
dataset_split = dataset['train'].train_test_split(test_size=.1,train_size=.9)
# Edit columns
dataset_split = dataset_split.rename_column('snippet', 'text')
dataset_split = dataset_split.remove_columns('language')


def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# Use batched=True to activate fast multithreading!
tokenized_datasets = dataset_split.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

chunk_size = 128 

def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# Save the dataset
lm_datasets.save_to_disk('lm_datasets.hf')