"""
------------------------------------
Ingest the raw text and annotated data from .txt and .ann files. Since the respective individual .txt and .ann files are clustered together, 
the iteration is customised in this specific ordering. Creates training_data.spacy and validation_data.spacy files for training and validation.
Training and Validation datasets are prepared. Although it is ideal to have a test set, this project aims to showcase the fine-tuning process of 
spacy v3 as well as fine-tuning huggingface datasets. Test set can be KIV due to the small rows in dataset.

Code for training spacy v3
# python -m spacy init fill-config base_config.cfg config.cfg
# python -m spacy train config.cfg --output .\models\spacy
"""

#Import statements
import spacy #nlp linguistic framework
from spacy.tokens import DocBin #custom dataset
from spacy.util import filter_spans #Handles overlapping tokens
import os #path module

#Global Constants
DATA_PATH = r"..\data\MACCROBAT2018" #Path to raw json data
TRAIN_DATA_PATH = r"..\data\training_data.spacy" #Path to spacy training dataset
VAL_DATA_PATH = r"..\data\validation_data.spacy" #Path to spacy training dataset

def main():
    #Annot.ann, file.txt
    file_name_list = os.listdir(DATA_PATH)
    #Create a blank pipeline
    nlp = spacy.blank("en")
    #Custom data
    train_db = DocBin()
    valid_db = DocBin()
    count_index = 0
    for file in file_name_list:
        #To access both .txt and .ann in a single loop; only access both at .ann loop
        if file.endswith(".txt"):
            continue
        full_file_name = os.path.join(DATA_PATH,file)
        #Retrieve the doc text
        full_file_text = full_file_name.replace(".ann",".txt")
        with open(full_file_text,encoding="utf-8") as f:
            #Read the contents of the file into a variable
            text = f.read()
        doc = nlp.make_doc(text)
        ents = []
        #Retrieve the entities
        with open(full_file_name,encoding="utf-8") as f:
            ents, doc = read_annot_file(f,ents,doc)
        filtered_ents = filter_spans(ents)
        doc.ents = filtered_ents
        doc,train_db,valid_db,count_index = add_data_doc(doc,train_db,valid_db,count_index)
    save_files(train_db,valid_db)

def read_annot_file(f,ents,doc):
    """Extract all the entities and create spans from these entities to be appended to doc.ent

    Args:
        f: binary text file
        ents: List to store span entities
        doc: spacy doc to create span

    Returns:
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
            start_index = int(extract_entity[1])
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
    return ents,doc

def add_data_doc(doc,train_db,valid_db,count_index):
    """Adds the doc to the respective training dataset or validation dataset. Acts as a train-test split function

    Args:
        doc: spacy doc with text and entities
        train_db: training dataset
        valid_db: validation dataset
        count_index: counter for train-validation split

    Returns:
        doc: spacy doc with text and entities
        train_db: training dataset
        valid_db: validation dataset
        count_index: counter for train-validation split
    """
    #Train-test split ratio of 0.85
    if count_index <= (0.85 * len(os.listdir(DATA_PATH))//2):
        train_db.add(doc)
    else:
        valid_db.add(doc)
    count_index = count_index +1
    return doc,train_db,valid_db,count_index


def save_files(train_db,valid_db):
    """Saves the training dataset and validation dataset for model training

    Args:
        train_db: training dataset
        valid_db: validation dataset

    """
    train_db.to_disk(TRAIN_DATA_PATH)
    valid_db.to_disk(VAL_DATA_PATH)

if __name__ == "__main__":
    main()
    


