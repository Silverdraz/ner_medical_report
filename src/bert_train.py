"""
------------------------------------
Train the Bert model for token classification using the processed hugging face datasets with tokens and ner_tags feature columns.
Both the model and tokenizer are saved for use during inference. Very importantly, since the tokenization may split words into their subword
forms, it is imperative for the model to align the labels with this splitting of subwords, especially during training.
"""


#Import statements
import json #read json files
import datasets #Huggingface datasets module
from transformers import BertTokenizerFast #Tokenizer
from transformers import AutoModelForTokenClassification #Bert Model
from transformers import DataCollatorForTokenClassification #Data Collator for padding, etc
import evaluate #evaluation metric during train/val
import numpy as np # numpy arrays
from transformers import TrainingArguments, Trainer #Training module

#Global Constants
TRAIN_DS_PATH = r"..\data\train_ds.hf" #Path to HF train data
VALID_DS_PATH = r"..\data\valid_ds.hf" #Path to HF Validation data
SAVE_MODEL_PATH = r"..\models\hf\bert_ner_model" #Path to saved model
SAVED_TOKENIZER_PATH = r"..\models\hf\bert_tokenizer" #Path to saved tokenizer
#label_list = []

def main():
    #load the prepared huggingface data
    train_ds, valid_ds = load_data()
    #Initialise the variables
    tokenizer,model,data_collator,args = initialize_vars()

    #Align tokens for loss function
    tokenized_datasets_train, tokenized_datasets_valid = map_dataset(train_ds,valid_ds)
   
    label_list = train_ds.features["ner_tags"].feature.names

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        seqeval = evaluate.load("seqeval")
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    #Prepare for training
    trainer = Trainer( 
        model, 
        args, 
        train_dataset=tokenized_datasets_train, 
        eval_dataset=tokenized_datasets_valid, 
        data_collator=data_collator, 
        tokenizer=tokenizer, 
        compute_metrics=compute_metrics 
    ) 
    trainer.train()

    #Save the parameters of the model for inference
    save_models(model,tokenizer)

    #Store the mappings of int2str and str2int in the model for reproducibility 
    store_configs(label_list)


def initialize_vars():
    """Initialize the Bert tokenizers, Bert model, data_collator for padding and training args

    Returns:
        tokenizer: bert tokenizer for tokenizing split words
        model: Bert Model for training
        data_collator: to ensure batches consist of the same lengths across rows by padding
        args: training args
    """
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=82)
    data_collator = DataCollatorForTokenClassification(tokenizer) 
    args = TrainingArguments( 
        r"..\models\hf\train_output",
        evaluation_strategy = "epoch", 
        learning_rate=5e-5, 
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16, 
        num_train_epochs=20, 
        weight_decay=0.01, 
    )
    return tokenizer,model,data_collator,args


def load_data():
    """load the train dataset and validation dataset

    Returns:
        train_ds: Loaded train huggingface dataset
        valid_ds: Loaded validation huggingface dataset
    """
    train_ds = datasets.load_from_disk(TRAIN_DS_PATH) 
    valid_ds = datasets.load_from_disk(VALID_DS_PATH)    
    return train_ds, valid_ds

def map_dataset(train_ds,valid_ds):
    """Align the train dataset and validation dataset as tokenized sentences can have
    different lengths from the original sequences.

    Returns:
        tokenized_datasets_train: Aligned samples in train dataset
        tokenized_datasets_valid: Aligned samples in validation dataset
    """
    tokenized_datasets_train = train_ds.map(tokenize_and_align_labels, batched=True)
    tokenized_datasets_valid = valid_ds.map(tokenize_and_align_labels, batched=True)
    return tokenized_datasets_train, tokenized_datasets_valid
  
def tokenize_and_align_labels(examples, label_all_tokens=True): 
    """
    Function to tokenize and align labels with respect to the tokens. This function is specifically designed for
    Named Entity Recognition (NER) tasks where alignment of the labels is necessary after tokenization.

    Args:
        examples: A dictionary containing the tokens and the corresponding NER tags.
                  - "tokens": list of words in a sentence.
                  - "ner_tags": list of corresponding entity tags for each word.
                     
        label_all_tokens: A flag to indicate whether all tokens should have labels. 
                          If False, only the first token of a word will have a label, 
                          the other tokens (subwords) corresponding to the same word will be assigned -100.

    Returns:
        tokenized_inputs: A dictionary containing the tokenized inputs and the corresponding labels aligned with the tokens.
    """
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True) 
    labels = [] 
    for i, label in enumerate(examples["ner_tags"]): 
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        # word_ids() => Return a list mapping the tokens
        # to their actual word in the initial sentence.
        # It Returns a list indicating the word corresponding to each token. 
        previous_word_idx = None 
        label_ids = []
        # Special tokens like `<s>` and `<\s>` are originally mapped to None 
        # We need to set the label to -100 so they are automatically ignored in the loss function.
        for word_idx in word_ids: 
            if word_idx is None: 
                # set –100 as the label for these special tokens
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            elif word_idx != previous_word_idx:
                # if current word_idx is != prev then its the most regular case
                # and add the corresponding token   
                #print(label)              
                label_ids.append(label[word_idx]) 
            else: 
                # to take care of sub-words which have the same word_idx
                # set -100 as well for them, but only if label_all_tokens == False
                label_ids.append(label[word_idx] if label_all_tokens else -100) 
                # mask the subword representations after the first subword
                 
            previous_word_idx = word_idx 
        labels.append(label_ids) 
    tokenized_inputs["labels"] = labels 
    return tokenized_inputs 


def save_models(model,tokenizer):
    """
    Save the parameters for the model and tokenizer
    """
    #Save the model and tokenizer
    model.save_pretrained(SAVE_MODEL_PATH)
    tokenizer.save_pretrained(r"..\models\hf\bert_tokenizer")

def store_configs(label_list):
    """Store the specific int2str and str2int mapping for ClassLabel feature 

    Returns:
        tokenized_datasets_train: Aligned samples in train dataset
        tokenized_datasets_valid: Aligned samples in validation dataset
    """
    #For storing in the config as state
    id2label = {
        str(i): label for i,label in enumerate(label_list)
    }
    label2id = {
        label: str(i) for i,label in enumerate(label_list)
    }

    #Update the config keys
    config = json.load(open(r"..\models\hf\bert_ner_model\config.json"))
    config["id2label"] = id2label
    config["label2id"] = label2id
    json.dump(config, open(r"..\models\hf\bert_ner_model\config.json","w"))    

if __name__ == "__main__":
    main()
    

