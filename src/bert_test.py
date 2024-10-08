from transformers import pipeline
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline
import json
import streamlit as st
import random
import spacy
from spacy import displacy
import os

#Global Constants
DATA_PATH = r"..\data\MACCROBAT2018" #Path to raw json data

unique_annots = []
def unique_annotations():
    #Annot.ann, file.txt
    file_name_list = os.listdir(DATA_PATH)
    #print(file_name_list)
    nlp = spacy.blank("en")
    for file in file_name_list:
        if file.endswith(".txt"):
            continue
        full_file_name = os.path.join(DATA_PATH,file)
        full_file_text = full_file_name.replace(".ann",".txt")
        with open(full_file_text,encoding="utf-8") as f:
            #Read the contents of the file into a variable
            text = f.read()
        doc = nlp.make_doc(text)
        ents = []
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
                    #print(start_index,end_index,entity_tag)
                    if entity_tag != entity_tag.strip():
                        print(entity_tag,"this is here")
                    span = doc.char_span(start_index, end_index, entity_tag)
                    if span is None:
                        print("Skipping Entity")
                    else:
                        unique_annots.append(entity_tag)
    return unique_annots


def main():
    model_fine_tuned = AutoModelForTokenClassification.from_pretrained(r"..\models\hf\bert_ner_model")
    tokenizer = AutoTokenizer.from_pretrained(r"..\models\hf\bert_tokenizer")
    nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer,device="cuda",grouped_entities=True)

    text_area = st.text_area("Text to input:", None)
    unique_annots = unique_annotations()
    set_annots = set(unique_annots)
    unique_colors = []
    for index in range(0,len(set_annots)):
        color = "%06x" % random.randint(0, 0xFFFFFF)
        color = "#" + color
        unique_colors.append(color)
    dictionary = dict(zip(set_annots, unique_colors))
    options = {"ents": list(set_annots),
               "colors": dictionary}
    print(dictionary)
    if text_area is not None:
        nlp_spacy = spacy.blank("en")
        doc = nlp_spacy.make_doc(text_area)
        doc_hf = nlp(text_area)
        print(doc_hf,"this is doc")
        ents = []
        for span_dict in doc_hf:
            span = doc.char_span(span_dict["start"], span_dict["end"], span_dict["entity_group"])
            if span is None:
                print("Skipping Entity")
            else:
                #convert_to_bil.append((start_index,end_index,entity_tag))
                ents.append(span)
        doc.ents = ents
        # Take the text from the input field and render the entity html.
        # Note that style="ent" indicates entities.
        ent_html = displacy.render(doc, style="ent", jupyter=False,options=options)
        #ent_html = displacy.serve(doc, style="ent")
        # Display the entity visualization in the browser:
        st.markdown(ent_html, unsafe_allow_html=True)
       
  
if __name__ == "__main__":
    main()
    

