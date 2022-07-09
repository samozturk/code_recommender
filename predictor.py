from pprint import pprint
import string
# # Piece by piece
# from transformers import AutoTokenizer, AutoModel


# sequence = ["func getData(path string) <mask> {}", 'fmt.Println("Hello")']


# checkpoint = "huggingface/CodeBERTa-small-v1"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# # Tokenize
# inputs = tokenizer(sequence, padding=True, truncation=True, return_tensors="pt")
# pprint(inputs)

# #
# model = AutoModel.from_pretrained(checkpoint)
# outputs = model(**inputs)

# pprint(outputs)

## PIPELINE
from transformers import pipeline, TextClassificationPipeline, RobertaForSequenceClassification, RobertaTokenizer, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


sequence = ["func getData(path string) <mask> {}", 'fmt.<mask>("Hello")']


def predict_word(sequence:str) -> "list[str]" :
    '''Predict the masked words. Mask token:<mask>

    Args:
        sequence (list[str]): Sequence of code snippets include masked tokens
        return_sequence (bool): True if you want whole sequence, false if you only want predicted word
    Returns:
        Returns predicted words 
    '''
    prediction = []
    fill_mask = pipeline(
    "fill-mask",
    model="huggingface/CodeBERTa-small-v1",
    # tokenizer="huggingface/CodeBERTa-small-v1",
    tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1", model_max_len=512, padding=True, truncation='drop_rows_to_fit')

)
    outputs = fill_mask(sequence)
    for output in outputs:
        prediction.append(output)

    return prediction






def predict_language(text):
    CODEBERTA_LANGUAGE_ID = "huggingface/CodeBERTa-language-id"
    pipeline = TextClassificationPipeline(
                            model = RobertaForSequenceClassification.from_pretrained(CODEBERTA_LANGUAGE_ID),
                            tokenizer = RobertaTokenizer.from_pretrained(CODEBERTA_LANGUAGE_ID)
                        )
    return pipeline(text)[0]

# a = predict_language("def my_func(): \n hello", text_classification_pipeline)
# print(a)

from transformers import AutoTokenizer, AutoModel

checkpoint = "microsoft/codebert-base"
def embed(raw_inputs, checkpoint=checkpoint):
    '''Function to get embeddings of words using codebert.

    Args:
        raw_inputs (list[str]): input words in format of list of strings
        checkpoint (str, optional): huggingface checkpoint. Defaults to checkpoint.
    Returns:
        embedding vector (numpy.ndarray): embedding of raw input. shape will be (raw_input_size, 768)

    '''
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

    model = AutoModel.from_pretrained(checkpoint)

    outputs = model(**inputs)
    return (outputs.last_hidden_state[0][1:-1]).detach().numpy()


def get_sentence_embedding(sentences):
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    # Sentences we want sentence embeddings for

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings