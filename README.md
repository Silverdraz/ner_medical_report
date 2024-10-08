# Named Entity Recognition on patient case reports
**The objective of this NLP project is to showcase the fine-tuning process of both Spacy V3 & Bert Model from Hugging Face. Very critically, the data has to be formatted appropriately for both Spacy and HuggingFace and the conversion of NER label offsets from Spacy formatting to IOB formatting of Hugging Face. Specifically, for transformer architectures or Bert Model, the tokens and NER labels have to be both aligned, especially when tokens can be split into subwords. The primary objective was to understand the required data formatting for the models and to also understand the fine-tuning process of NER, a sequence tagging tasak, unlike the common text classification.**

## Screenshot of the Web Application
![alt text](Screenshot_25-9-2024_05847_localhost.jpeg)


## Scoring Metric Choice
**f1 score** is utilised as the scoring metric as it is possible that there can be an imbalance of the entities (NER tags), given that for this NER dataset, the class labels are many. EDA could have been performed on the target class (NER tags) distribution, though the objective of the project is to showcase the data formatting process and fine-tuning process on custom datasets.

## Modelling Approaches 
**-------Model Architecture Approaches--------**

1. **Spacy V3**
Spacy model (non-transformer spacy model) was chosen as a baseline algorithm and achieved a performance of about 56% when f1 score is used as an evaluation metric. Training and validation data was split 85%-15%. However, the samples/annotations in the sample is very small and ideally should have been larger.


2. **Hugging Face - Bert Model**
BertForTokenClassification model was fine-tuned on the custom NER dataset and had a f1 score of approximately 65%, an improvement over the 56% reported by spacy baseline. Instead of using a spacy transformer, the objective was to fine-tune on the custom medical dataset using hugging face and to demostrate fine-tuning on sequence tagging tasks, instead of simply a text classification.

**-------Future Improvements--------**
1. Check the distribution of the target labels.
2. Plot visualisation graphs to check for overfitting
3. Custom layers on top of general Bert Model for NER finetuning.
4. Merge spacy and huggingface bert streamlit into a single module and have a dropdown to indicate the choice of model