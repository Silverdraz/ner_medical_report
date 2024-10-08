"""
------------------------------------
Applying the data preprocessing, model building, model comparison and evaluation stages in the machine learning pipeline
"""
# python -m spacy init fill-config base_config.cfg config.cfg
# python -m spacy train config.cfg --output .\output
#Import statements
import spacy #nlp linguistic framework
import json #read json files
from spacy.tokens import DocBin #custom data
from tqdm import tqdm #progress bar
from spacy.util import filter_spans #overlap tokens
import os
import random

import streamlit as st
import spacy
from spacy import displacy

#Global Constants
DATA_PATH = r"..\data\MACCROBAT2018" #Path to raw json data

#unique_annots = []
def unique_annotations():
    #Annot.ann, file.txt
    file_name_list = os.listdir(DATA_PATH)
    nlp = spacy.blank("en")
    ents = []
    for file in file_name_list:
        if file.endswith(".txt"):
            continue
        full_file_name = os.path.join(DATA_PATH,file)
        full_file_text = full_file_name.replace(".ann",".txt")
        with open(full_file_text,encoding="utf-8") as f:
            #Read the contents of the file into a variable
            text = f.read()
        doc = nlp.make_doc(text)
        #ents = []
        with open(full_file_name,encoding="utf-8") as f:
            #Read the contents of the file into a variable
            names = f.read()
            split_files = names.split("\n")
            for annot in split_files:
                if annot.startswith("T"):
                    splitted_annot = annot.split("\t")
                    entity = splitted_annot[1]
                    text = splitted_annot[2]
                    extract_entity = entity.split(" ")
                    entity_tag = extract_entity[0].strip()
                    start_index = int(extract_entity[1])
                    if ";" in extract_entity[2]:
                        end_index = int(extract_entity[2].split(";")[0])
                    else:
                        end_index = int(extract_entity[2])
                    if entity_tag != entity_tag.strip():
                        print(entity_tag,"this is here")
                    span = doc.char_span(start_index, end_index, entity_tag)
                    if span is None:
                        print("Skipping Entity")
                    else:
                        ents.append(entity_tag)
    return ents

def read_annot_file(f,ents,doc):
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

def main():
    #Load the final model
    nlp = spacy.load(r"..\models\spacy\model-last")
    #Retrieve the annotations for colors
    unique_annots = unique_annotations()
    #Retrieve only unique annotations
    set_annots = set(unique_annots)
    unique_colors = []
    for index in range(0,len(set_annots)):
        color = "%06x" % random.randint(0, 0xFFFFFF)
        color = "#" + color
        unique_colors.append(color)
    dictionary = dict(zip(set_annots, unique_colors))
    options = {"ents": list(set_annots),
               "colors": dictionary}
    text_area = st.text_area("Text to input:", None)
    if text_area is not None:
        doc = nlp(text_area)
        ent_html = displacy.render(doc, style="ent", jupyter=False,options=options)
        # Display the entity visualization in the browser:
        st.markdown(ent_html, unsafe_allow_html=True)
  

if __name__ == "__main__":
    main()
    


