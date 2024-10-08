"""
------------------------------------
Converts entities data from spacy offset to iob for huggingface training. The tokens (using spacy tokenizer - split by whitespaces and punctuations)
and the IOB tags are aligned for token classification in Bert. 
"""

#Import statements
import os #path module
import spacy #nlp linguistic framework
from spacy.util import filter_spans #overlap tokens
import spacy.training # IOB format for NER
from spacy.training.iob_utils import doc_to_biluo_tags # IOB format for NER
from datasets import Dataset, Sequence, ClassLabel, Value # Huggingface datasets features

#Global Constants
DATA_PATH = r"..\data\MACCROBAT2018" #Path to raw json data
TRAIN_HF_PATH = r"..\data\train_ds.hf" #Path to train dataset for huggingface
VALID_HF_PATH = r"..\data\valid_ds.hf" #Path to validation dataset for huggingface

def main():
    train_span_data,val_span_data = create_train_val()
    #Annot.ann, file.txt
    file_name_list = os.listdir(DATA_PATH)
    nlp = spacy.blank("en")
    # train_db = DocBin()
    # valid_db = DocBin()
    count_index = 0
    total_ner_tag_list = []
    for file in file_name_list:
        if file.endswith(".txt"):
            continue
        full_file_name = os.path.join(DATA_PATH,file)
        full_file_text = full_file_name.replace(".ann",".txt")
        with open(full_file_text,encoding="utf-8") as f:
            #Read the contents of the file into a variable
            doc_text = f.read()
        doc = nlp.make_doc(doc_text)
        ents = []
        with open(full_file_name,encoding="utf-8") as f:
            f,ents,doc = read_annot_file(f,ents,doc)
        filtered_ents = filter_spans(ents)
        doc.ents = filtered_ents
        #spans_spacy_list = spans_list(ents)

        token_list, ner_tag_list, total_ner_tag_list = hf_data_structure(doc,total_ner_tag_list)
        count_index = add_to_data(train_span_data,
                                  val_span_data,
                                  token_list,
                                  ner_tag_list,
                                  count_index)
    create_hf_datasets(train_span_data,val_span_data,total_ner_tag_list)


def create_train_val():
    """Creates the dict structure for training and validation using Bert

    Returns:
        train_span_data: a dict containing tokens and ner_tags keys for bert training
        val_span_data: a dict containing tokens and ner_tags keys for bert validation
    """
    train_span_data = {
            "tokens": [],
            "ner_tags": []
    }
    val_span_data = {
        "tokens": [],
        "ner_tags": []
    }
    return train_span_data, val_span_data

def read_annot_file(f,ents,doc):
    """Extract all the entities and create spans from these entities to be appended to doc.ent

    Args:
        f: binary text file
        ents: List to store span entities
        doc: spacy doc to create span

    Returns:
        f: binary text file
        ents: List to store span entities
        doc: spacy doc to create span
    """
    #Read the contents of the file into a variable
    names = f.read()
    #Annot file has a new line for each annotation
    split_files = names.split("\n")
    for annot in split_files:
        #T is for tags
        if annot.startswith("T"):
            #Split to extract start,end,entities
            splitted_annot = annot.split("\t")
            entity = splitted_annot[1]
            text = splitted_annot[2]
            extract_entity = entity.split(" ")
            entity_tag = extract_entity[0].strip()
            start_index = int(extract_entity[1].strip())
            #Some multiple annotations in the same line; Extract the first
            if ";" in extract_entity[2]:
                end_index = int(extract_entity[2].split(";")[0])
            else:
                end_index = int(extract_entity[2])
            span = doc.char_span(start_index, end_index, entity_tag)
            if span is None:
                print("Skipping Entity")
            else:
                #Append to add to doc
                ents.append(span)
    return f,ents,doc


def hf_data_structure(doc,total_ner_tag_list):
    """Retrieve all the relevant info (tokens and ner tags)

    Args:
        doc: spacy doc object for text/tokens
        total_ner_tag_list: List to store all ner tags

    Returns:
        token_list: the tokens (split into words with punctuations separated (default spacy tokenization))
        ner_tag_list: ner iob tags for the respective tokens
        total_ner_tag_list: All ner tags across the docs which will be made into a set afterwards
    """
    biluo_tags = doc_to_biluo_tags(doc)
    iob_tags = spacy.training.biluo_to_iob(biluo_tags)
    token_list = []
    ner_tag_list  = []
    for token, iob_tag in zip(doc,iob_tags):
        token_list.append(token.text)
        ner_tag_list.append(iob_tag)
        total_ner_tag_list.append(iob_tag)   
    return token_list, ner_tag_list, total_ner_tag_list

def add_to_data(train_span_data,
                val_span_data,
                token_list,
                ner_tag_list,
                count_index):
    """Adds the list of per row tokens and list of ner tags to the training and validation dataset

    Args:
        train_span_data: a dict containing tokens and ner_tags keys for bert training
        val_span_data: a dict containing tokens and ner_tags keys for bert validation
        token_list: the tokens (split into words with punctuations separated (default spacy tokenization))
        ner_tag_list: ner iob tags for the respective tokens
        count_index: counter for train-validation split

    Returns:
        count_index: counter for train-validation split
    """
    if count_index <= (0.85 * len(os.listdir(DATA_PATH))//2):
        train_span_data["tokens"].append(token_list)
        train_span_data["ner_tags"].append(ner_tag_list)
    else:
        val_span_data["tokens"].append(token_list)
        val_span_data["ner_tags"].append(ner_tag_list)
    count_index = count_index +1
    return count_index

def create_hf_datasets(train_span_data,val_span_data,total_ner_tag_list):
    """Create specific dataset type (huggingface dataset type). Columns/Features are prepared for training

    Args:
        train_span_data: a dict containing tokens and ner_tags keys for bert training
        val_span_data: a dict containing tokens and ner_tags keys for bert validation
        total_ner_tag_list: All ner tags across the docs which will be made into a set afterwards
    """
    train_ds = Dataset.from_dict(train_span_data)
    valid_ds = Dataset.from_dict(val_span_data)
    train_ds = train_ds.cast_column("ner_tags", Sequence(ClassLabel(names=list(set(total_ner_tag_list)))))
    train_ds = train_ds.cast_column("tokens",Sequence(Value("string")))
    train_ds.save_to_disk(TRAIN_HF_PATH)
    valid_ds = valid_ds.cast_column("ner_tags", Sequence(ClassLabel(names=list(set(total_ner_tag_list)))))
    valid_ds = valid_ds.cast_column("tokens",Sequence(Value("string")))
    valid_ds.save_to_disk(VALID_HF_PATH)


if __name__ == "__main__":
    main()
    


