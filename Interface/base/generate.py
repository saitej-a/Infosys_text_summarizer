import os

# Get the directory of the current script
script_dir = os.path.dirname(__file__)


ex_model_name_or_path = os.path.join(script_dir, '..', '..', 'fine-tuned-model')

from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
ex_local_tokenizer=AutoTokenizer.from_pretrained(ex_model_name_or_path)
ex_local_model=AutoModelForSeq2SeqLM.from_pretrained(ex_model_name_or_path)


ab_model_name_or_path=os.path.join(script_dir, '..', '..', 'fine-tuned-abstractive')
ab_local_tokenizer=AutoTokenizer.from_pretrained(ab_model_name_or_path)
ab_local_model=AutoModelForSeq2SeqLM.from_pretrained(ab_model_name_or_path)


from transformers import pipeline
ex_summarize=pipeline('summarization',model=ex_local_model,tokenizer=ex_local_tokenizer)
ab_summarize=pipeline('summarization',model=ab_local_model,tokenizer=ab_local_tokenizer)
def Extract(article:str):
    return ex_summarize(article)[0]['summary_text']
def Abstract(article:str):
    return ab_summarize(article)[0]['summary_text']