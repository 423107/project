import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


df = pd.read_csv('C:\p1\cleaned_news_data.csv')  

selected_row = df.iloc[2]
news_article = selected_row['news_article']

 #removing stop word from article
stop_word = stopwords.words('english')
print(stopwords)

a_tokenize = word_tokenize(news_article)

with_out_stop_word=""
for word in a_tokenize:
    if word not in stop_word:
        with_out_stop_word = with_out_stop_word + word +" "
#print (with_out_stop_word)
#print(len(news_article))
#print(len(with_out_stop_word))

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

device = torch.device('cuda')
model.to(device)
'''
prompt = f"""Economic Scale from -10 to 10, where -10 is Economic Left and 10 is Economic Right.
 Scale Democracy from -10 to 10, where -10 is Libertarian and 10 is Authoritarian. 
 I provide a newspaper article. Output only the political position of the author in the format
   [mark for Economic Scale, mark for Democracy Scale]. NEVER write any text before or after the result.
     ALWAYS provide the result, even if you are not fully sure.

'''

prompt = f"""Analyze the following newspaper article and evaluate the author's political position on two scales:  
1. **Economic Scale**: From -10 (Economic Left) to 10 (Economic Right).  
2. **Democracy Scale**: From -10 (Libertarian) to 10 (Authoritarian).  

Output **only** the scores in the exact format:  
[Economic Scale, Democracy Scale]  

- **Never** add any text, explanations, or disclaimers before or after the numbers.  
- **Always** provide a result, even if uncertain.  
- **Strictly** return only


Article: {with_out_stop_word}

"""

inputs = tokenizer(prompt, return_tensors="pt",max_length=512, padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=50,
    do_sample=False,
    temperature=0.1,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Response:", response)