# LLM - Bielik - Fine tuning - part I

## Introduction

The rise of the Large Language Models are definetely a game changer in approach to AI on various markets and branches - we can observe increased popularity of LLMs in academia, business and agriculture. Working with LLMs became a must-do task for all R&D or business departments in order to acknowledge and adapt upcoming possibilities, to stay competetive and gain business advantage. There are many players who invested bilions of USD to build their own Large Language Model and took a part in recent markets challenges by introducing their own solution trained on a different data set or to serve a different task (specialized for example in generation of poems or making a summary). At the beginning, as usual when it comes to innovation at that scale, we are thinking of endless possibilities, what kind of new opportunities we reach without putting much focus on other important factors like costs, maintanence, security, stability. Currently, we have multiple models which are production ready - many work has been done at those fields also making those LLMs more reliable and secured. One of the outcomes from this LLM race is that people noticed that it is good idea to look at smaller models, more specialized in one or two tasks, not the general ones. This specialization also is related to language supported. LLMs are broadly trained on english language data, due to the fact that more than half of websites content is written in English. Many models, paid and open source ones, have different level of quality when it comes when using them with local language. Quite often it occurs that offered quality is not enough to prove the business value for specific use case. This pushed some parties to create smaller models or suggested the finetuning with data in local language. 

## Goal
The goal of this notebook is to finetune a Polish language based LLM called Bielik, developed by SpeakLeash organization (<model page>) with the legal documents. Data is specifically chosen for this task - legal documents or, law as a general concept, are strictly connected to the particular langauge. Primary concept of legal documents is to be in line with local (polish in this case) regulations. One of the approaches is to translate documents to english and then used by the LLM. However, legal documents such as purchase agreements or any other agreements are designed by lawyers who understand the legal language nuances. They need to construct documents that are reliable, understandable by many and embedded in Polish law regulations. This requires from LLM to be more specific on many levels - starting from using prevalent data in one language and finetuning it to be able to work with more detailed task. 

## Table of contents

This notebook covers multiple steps:
1. Loading and splitting data into chunks
2. Generating new data (question and answers) for the future finetuning
3. Evaluating generated data
4. Selecting the data based on metrics 
5. Preparing the output for FT


## Loading and working with data

Data source: https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU20140000827/U/D20140827Lj.pdf 


```python
import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv, find_dotenv
import pymupdf
import uuid
from tqdm.notebook import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.external_requests.openai_req import OpenAIReq
from src.external_requests.prompting import PromptBuilder
from src.evaluation.distance_based import DistanceEvaluator
from src.evaluation.helper import SimilarityMatrix
import tiktoken
```


```python
%load_ext autoreload
%autoreload 2
%reload_ext autoreload
```


```python
_ = load_dotenv(find_dotenv())
```

For this use case I am using commonly used splitter - RecursiveCharacterTextSplitter from langchain and pymudf library for parsing the PDF. Document is well formatted and doesn't contain more complex structures like graphs or tables. 


```python
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
```

```python
path = 'src/data/ustawa_o_prawach_konsumenta.pdf'  # local file system
all_data = pymupdf.open(path)
```


```python
all_chunks = []
for document in all_data:
    text_page = document.get_textpage().extractText()
    chunks_per_page = splitter.split_text(text_page)
    all_chunks.append(chunks_per_page)
        
len(all_chunks)
```




    44




```python
def get_number_of_tokens(text:str) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    n_tokens = len(encoding.encode(text))
    return n_tokens
```

### Generating new data (questions) with a support of GPT 3.5

For this task the GPT 3.5 was chosen. GPT has proven itself as a great tool for text generation in many languages and also for assessing own work (this part will be covered later in the article). What is more, it has proven also it's abilities to generate structurised content. In the below code snippet there is one important remark. As presented, there are two similar variables - max_tokens and model_max_tokens. ChatGPT model support tokens limitation during an api call. However, this limitation ca be tricky as it should be considered rather as a cost guard than concise content guard. Setting model's max tokens to low number results in sentence truncation and losing information. That is the reason the second variable was introduced - max_tokens. Max_tokens variable is passed to the prompt for the model as a part of an instruction to model to instruct the model to generate succinct yet comprehensive answer within the given token limit. 


```python
model_version = 'gpt-4o-mini'
```


```python
all_data = []
max_tokens = 200
model_max_tokens = int(max_tokens + (max_tokens*0.20))
for i, page in tqdm(enumerate(all_chunks)):
    page_id = str(uuid.uuid4())
    for chunk in enumerate(page):
        nth_chunk, text_chunk = chunk
        generated_questions = OpenAIReq(model=model_version).call_api(prompt=PromptBuilder.generate_questions(n_questions=3, text=chunk, tokens=max_tokens),
                                                   temperature=0.2,max_tokens=model_max_tokens)
        data = {'chunk_id': f'{page_id}_{nth_chunk}', 
                'text_chunk': text_chunk,
                'page_id': page_id,
                'generated_questions': generated_questions}
        all_data.append(data)
```


    0it [00:00, ?it/s]



```python
df = pd.DataFrame.from_dict(data=all_data)
```

Let's do some data cleaning and validating. We need a helper method to do required data preparation with validation on null values:


```python
def process_df_with_questions(data: pd.DataFrame) -> pd.DataFrame:
    data['questions_splitted'] = data['generated_questions'].str.split('\n')
    data = data.explode('questions_splitted')
    data = data[~data['questions_splitted'].isna()]
    data['questions_splitted']  = data['questions_splitted'].replace(r'^\s*$', np.nan, regex=True)
    return data

df = process_df_with_questions(data=df)
```


```python
df = df[~df['questions_splitted'].isna()]
df['questions_splitted'].isna().sum()
```




    0




```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chunk_id</th>
      <th>text_chunk</th>
      <th>page_id</th>
      <th>generated_questions</th>
      <th>questions_splitted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>99778fb5-7045-450c-9607-3c89babc9a05_0</td>
      <td>©Kancelaria Sejmu \n \n \n \ns. 1/44 \n \n2024...</td>
      <td>99778fb5-7045-450c-9607-3c89babc9a05</td>
      <td>- Jakie są główne obowiązki przedsiębiorcy wob...</td>
      <td>- Jakie są główne obowiązki przedsiębiorcy wob...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>99778fb5-7045-450c-9607-3c89babc9a05_0</td>
      <td>©Kancelaria Sejmu \n \n \n \ns. 1/44 \n \n2024...</td>
      <td>99778fb5-7045-450c-9607-3c89babc9a05</td>
      <td>- Jakie są główne obowiązki przedsiębiorcy wob...</td>
      <td>- Jakie zasady i tryb dotyczą zawierania umów ...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>99778fb5-7045-450c-9607-3c89babc9a05_0</td>
      <td>©Kancelaria Sejmu \n \n \n \ns. 1/44 \n \n2024...</td>
      <td>99778fb5-7045-450c-9607-3c89babc9a05</td>
      <td>- Jakie są główne obowiązki przedsiębiorcy wob...</td>
      <td>- Jakie są szczegółowe przepisy dotyczące praw...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>99778fb5-7045-450c-9607-3c89babc9a05_1</td>
      <td>przedsiębiorstwa; \n3) \nzasady i tryb wykonan...</td>
      <td>99778fb5-7045-450c-9607-3c89babc9a05</td>
      <td>- Jakie są główne zasady dotyczące prawa odstą...</td>
      <td>- Jakie są główne zasady dotyczące prawa odstą...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>99778fb5-7045-450c-9607-3c89babc9a05_1</td>
      <td>przedsiębiorstwa; \n3) \nzasady i tryb wykonan...</td>
      <td>99778fb5-7045-450c-9607-3c89babc9a05</td>
      <td>- Jakie są główne zasady dotyczące prawa odstą...</td>
      <td>- W jaki sposób ustawa reguluje zawieranie umó...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>242</th>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937_0</td>
      <td>©Kancelaria Sejmu \n \n \n \ns. 44/44 \n \n202...</td>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937</td>
      <td>- Jakie informacje powinny być zawarte w formu...</td>
      <td>- Czy formularz odstąpienia od umowy można wys...</td>
    </tr>
    <tr>
      <th>242</th>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937_0</td>
      <td>©Kancelaria Sejmu \n \n \n \ns. 44/44 \n \n202...</td>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937</td>
      <td>- Jakie informacje powinny być zawarte w formu...</td>
      <td>- Jakie są terminy na odstąpienie od umowy spr...</td>
    </tr>
    <tr>
      <th>243</th>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937_1</td>
      <td>następujących towarów(*) umowy dostawy następu...</td>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937</td>
      <td>- Jakie towary lub usługi mogą być objęte umow...</td>
      <td>- Jakie towary lub usługi mogą być objęte umow...</td>
    </tr>
    <tr>
      <th>243</th>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937_1</td>
      <td>następujących towarów(*) umowy dostawy następu...</td>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937</td>
      <td>- Jakie towary lub usługi mogą być objęte umow...</td>
      <td>- Jakie informacje są wymagane od konsumenta p...</td>
    </tr>
    <tr>
      <th>243</th>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937_1</td>
      <td>następujących towarów(*) umowy dostawy następu...</td>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937</td>
      <td>- Jakie towary lub usługi mogą być objęte umow...</td>
      <td>- Czy podpis konsumenta jest obowiązkowy w prz...</td>
    </tr>
  </tbody>
</table>
<p>732 rows × 5 columns</p>
</div>



### Fetching answers for generated questions based on the text chunk
Next step is also focused on generation, but in this case we are focusing on answers to the previously generated questions. As in previous example, we also introduce a token limitation for the model, and it is passed to the prompt as part of the instruction. In this case we want to limit the answer length in order to get concise and precise answer as it is quite often expected from LLM to deliver brief answers. To support that, we set the temperature as low as possible.


```python
answer_tokens = 200
df[f'answer_{answer_tokens}'] = df.apply(lambda x: OpenAIReq(model=model_version).call_api(prompt=PromptBuilder.get_answers_based_on_text_and_questions(text=x['text_chunk'], question= x['questions_splitted'], tokens=answer_tokens), temperature=0.0), axis=1)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chunk_id</th>
      <th>text_chunk</th>
      <th>page_id</th>
      <th>generated_questions</th>
      <th>questions_splitted</th>
      <th>answer_200</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>99778fb5-7045-450c-9607-3c89babc9a05_0</td>
      <td>©Kancelaria Sejmu \n \n \n \ns. 1/44 \n \n2024...</td>
      <td>99778fb5-7045-450c-9607-3c89babc9a05</td>
      <td>- Jakie są główne obowiązki przedsiębiorcy wob...</td>
      <td>- Jakie są główne obowiązki przedsiębiorcy wob...</td>
      <td>Główne obowiązki przedsiębiorcy wobec konsumen...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>99778fb5-7045-450c-9607-3c89babc9a05_0</td>
      <td>©Kancelaria Sejmu \n \n \n \ns. 1/44 \n \n2024...</td>
      <td>99778fb5-7045-450c-9607-3c89babc9a05</td>
      <td>- Jakie są główne obowiązki przedsiębiorcy wob...</td>
      <td>- Jakie zasady i tryb dotyczą zawierania umów ...</td>
      <td>Zasady i tryb zawierania umów na odległość i p...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>99778fb5-7045-450c-9607-3c89babc9a05_0</td>
      <td>©Kancelaria Sejmu \n \n \n \ns. 1/44 \n \n2024...</td>
      <td>99778fb5-7045-450c-9607-3c89babc9a05</td>
      <td>- Jakie są główne obowiązki przedsiębiorcy wob...</td>
      <td>- Jakie są szczegółowe przepisy dotyczące praw...</td>
      <td>Prawo konsumenta do odstąpienia od umowy regul...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>99778fb5-7045-450c-9607-3c89babc9a05_1</td>
      <td>przedsiębiorstwa; \n3) \nzasady i tryb wykonan...</td>
      <td>99778fb5-7045-450c-9607-3c89babc9a05</td>
      <td>- Jakie są główne zasady dotyczące prawa odstą...</td>
      <td>- Jakie są główne zasady dotyczące prawa odstą...</td>
      <td>Główne zasady dotyczące prawa odstąpienia od u...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>99778fb5-7045-450c-9607-3c89babc9a05_1</td>
      <td>przedsiębiorstwa; \n3) \nzasady i tryb wykonan...</td>
      <td>99778fb5-7045-450c-9607-3c89babc9a05</td>
      <td>- Jakie są główne zasady dotyczące prawa odstą...</td>
      <td>- W jaki sposób ustawa reguluje zawieranie umó...</td>
      <td>Ustawa reguluje zawieranie umów na odległość d...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>242</th>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937_0</td>
      <td>©Kancelaria Sejmu \n \n \n \ns. 44/44 \n \n202...</td>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937</td>
      <td>- Jakie informacje powinny być zawarte w formu...</td>
      <td>- Czy formularz odstąpienia od umowy można wys...</td>
      <td>Formularz odstąpienia od umowy można wysłać w ...</td>
    </tr>
    <tr>
      <th>242</th>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937_0</td>
      <td>©Kancelaria Sejmu \n \n \n \ns. 44/44 \n \n202...</td>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937</td>
      <td>- Jakie informacje powinny być zawarte w formu...</td>
      <td>- Jakie są terminy na odstąpienie od umowy spr...</td>
      <td>Zgodnie z obowiązującymi przepisami, konsument...</td>
    </tr>
    <tr>
      <th>243</th>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937_1</td>
      <td>następujących towarów(*) umowy dostawy następu...</td>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937</td>
      <td>- Jakie towary lub usługi mogą być objęte umow...</td>
      <td>- Jakie towary lub usługi mogą być objęte umow...</td>
      <td>Według podanego tekstu, umowa dostawy może obe...</td>
    </tr>
    <tr>
      <th>243</th>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937_1</td>
      <td>następujących towarów(*) umowy dostawy następu...</td>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937</td>
      <td>- Jakie towary lub usługi mogą być objęte umow...</td>
      <td>- Jakie informacje są wymagane od konsumenta p...</td>
      <td>Zgodnie z przedstawionym wzorem, wymagane info...</td>
    </tr>
    <tr>
      <th>243</th>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937_1</td>
      <td>następujących towarów(*) umowy dostawy następu...</td>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937</td>
      <td>- Jakie towary lub usługi mogą być objęte umow...</td>
      <td>- Czy podpis konsumenta jest obowiązkowy w prz...</td>
      <td>Nie, podpis konsumenta nie jest obowiązkowy w ...</td>
    </tr>
  </tbody>
</table>
<p>732 rows × 6 columns</p>
</div>




```python
df.to_csv('data/data_26_01_2025.csv', index=False, sep=';')
```

To validate if GPT "listened" to our instruction with token limitation, let's count tokens and plot the histogram. As shown, the model generated answer mostly within the range of 100-300 tokens.   


```python
df['n_tokens_ans_200'] = df['answer_200'].apply(lambda x: get_number_of_tokens(text=x))
pd.plotting.hist_series(df['n_tokens_ans_200'])
```




    <Axes: >




    
![png](llm-bielik-ft-data-generation-part-I_files/llm-bielik-ft-data-generation-part-I_26_1.png)
    


## Verifing the quality of generated answers and questions based on selected metrics
To verify the quality of our text generation in relation to text chunk there are couple metrics we can use:
- Coverage 
- Coherence (Consistency)
- Diversity (based on Word Mover Distance)
- Relevance
- Global relevance

Majority of the metrics are based on the LLM own capability to assess it's own work. This has been proven and it is often used in various evaluation frameworks like RAGAS.
Here metrics are chosen carefully based on article from Microsoft -  https://arxiv.org/abs/2401.08406

Briefly, we can describe the metrics in the way prompt instruction is constructed:
- coverage - Your task is to rate from 1 to 5 if the question can be extracted from the context.
- coherence - Your task is to rate from 1 to 5 the coherence (fluency) of the question and provide explanation.
- relevance - Your task is to rate from 1 to 5 if the provided question could be asked by a lawyer then rate 5, and rate 1 if not based on the question and context.
- global relevance - Your task is to rate from 1 to 5 if the provided question could be asked by a lawyer then rate 5, and rate 1 if not based on the question.
- diversity - not LLM based metric. To calculate this we used Word Mover Distance (WMD). 



```python
df['coverage_rating'] = df.apply(lambda x: OpenAIReq(model=model_version).call_api(prompt=PromptBuilder.get_coverage(question=x['questions_splitted'], context=x['text_chunk'])), axis=1)
pd.plotting.hist_series(df['coverage_rating'])
```




    <Axes: >




    
![png](llm-bielik-ft-data-generation-part-I_files/llm-bielik-ft-data-generation-part-I_28_1.png)
    



```python
dist_evaluator = DistanceEvaluator()
```


```python
df[f'wmd_question_answer_{answer_tokens}'] = df.apply(lambda x: dist_evaluator.get_diversity_by_wmd(x['questions_splitted'], x[f'answer_{answer_tokens}']), axis=1)
```


```python
pd.plotting.hist_series(df['wmd_question_answer_200'])
```




    <Axes: >




    
![png](llm-bielik-ft-data-generation-part-I_files/llm-bielik-ft-data-generation-part-I_31_1.png)
    



```python
df['wmd_chunk_question'] = df.apply(lambda x: dist_evaluator.get_diversity_by_wmd(x['text_chunk'], x['questions_splitted']), axis=1)
```


```python
pd.plotting.hist_series(df['wmd_chunk_question'])
```




    <Axes: >




    
![png](llm-bielik-ft-data-generation-part-I_files/llm-bielik-ft-data-generation-part-I_33_1.png)
    



```python
df['coherence_pl'] = df['questions_splitted'].apply(lambda x: OpenAIReq(model=model_version).call_api(prompt=PromptBuilder.get_coherence_pl(x)))
```


```python
#df[['coherence_rate_pl', 'coherence_expl_pl']] = df.coherence_pl.str.split(',', n=1,  expand=True)
df[['relevance_score', 'relevance_expl']] = df.relevance.str.split(',', n=1,  expand=True)

```


```python
df[['relevance_score', 'relevance_expl']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>relevance_score</th>
      <th>relevance_expl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>```json\n{\n  "score": 5</td>
      <td>\n  "explanation": "Pytanie dotyczące głównych...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>```json\n{\n  "score": 5</td>
      <td>\n  "explanation": "Pytanie dotyczące zasad i ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>```json\n{\n  "score": 5</td>
      <td>\n  "explanation": "Pytanie dotyczące szczegół...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>```json\n{\n  "score": 5</td>
      <td>\n  "explanation": "Pytanie jest precyzyjne i ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>```json\n{\n  "score": 4</td>
      <td>\n  "explanation": "Pytanie dotyczące regulacj...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>727</th>
      <td>```json\n{\n  "score": 4</td>
      <td>\n  "explanation": "Pytanie dotyczące możliwoś...</td>
    </tr>
    <tr>
      <th>728</th>
      <td>```json\n{\n  "score": 5</td>
      <td>\n  "explanation": "Pytanie dotyczące terminów...</td>
    </tr>
    <tr>
      <th>729</th>
      <td>```json\n{\n  "score": 4</td>
      <td>\n  "explanation": "Pytanie jest dobrze sformu...</td>
    </tr>
    <tr>
      <th>730</th>
      <td>```json\n{\n  "score": 4</td>
      <td>\n  "explanation": "Pytanie jest dobrze sformu...</td>
    </tr>
    <tr>
      <th>731</th>
      <td>```json\n{\n  "score": 4</td>
      <td>\n  "explanation": "Pytanie dotyczące obowiązk...</td>
    </tr>
  </tbody>
</table>
<p>732 rows × 2 columns</p>
</div>




```python
df = pd.read_excel('data/full_26_01_2025.xlsx')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chunk_id</th>
      <th>text_chunk</th>
      <th>page_id</th>
      <th>generated_questions</th>
      <th>questions_splitted</th>
      <th>answer_200</th>
      <th>n_tokens_ans_200</th>
      <th>coverage_rating</th>
      <th>wmd_question_answer_200</th>
      <th>wmd_chunk_question</th>
      <th>coherence_pl</th>
      <th>coherence_rate_pl</th>
      <th>coherence_expl_pl</th>
      <th>relevance</th>
      <th>global_relevance</th>
      <th>relevance_score</th>
      <th>relevance_expl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>99778fb5-7045-450c-9607-3c89babc9a05_0</td>
      <td>©Kancelaria Sejmu \n \n \n \ns. 1/44 \n \n2024...</td>
      <td>99778fb5-7045-450c-9607-3c89babc9a05</td>
      <td>- Jakie są główne obowiązki przedsiębiorcy wob...</td>
      <td>- Jakie są główne obowiązki przedsiębiorcy wob...</td>
      <td>Główne obowiązki przedsiębiorcy wobec konsumen...</td>
      <td>325</td>
      <td>5</td>
      <td>0.174457</td>
      <td>0.249545</td>
      <td>{\n  "score": 4,\n  "explanation": "Tekst jest...</td>
      <td>4</td>
      <td>\n  "explanation": "Tekst jest zrozumiały i kl...</td>
      <td>5</td>
      <td>4</td>
      <td>```json\n{\n  "score": 5</td>
      <td>\n  "explanation": "Pytanie dotyczące głównych...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>99778fb5-7045-450c-9607-3c89babc9a05_0</td>
      <td>©Kancelaria Sejmu \n \n \n \ns. 1/44 \n \n2024...</td>
      <td>99778fb5-7045-450c-9607-3c89babc9a05</td>
      <td>- Jakie są główne obowiązki przedsiębiorcy wob...</td>
      <td>- Jakie zasady i tryb dotyczą zawierania umów ...</td>
      <td>Zasady i tryb zawierania umów na odległość i p...</td>
      <td>234</td>
      <td>5</td>
      <td>0.185090</td>
      <td>0.220349</td>
      <td>{\n  "score": 4,\n  "explanation": "Tekst jest...</td>
      <td>4</td>
      <td>\n  "explanation": "Tekst jest zrozumiały i kl...</td>
      <td>5</td>
      <td>5</td>
      <td>```json\n{\n  "score": 5</td>
      <td>\n  "explanation": "Pytanie dotyczące zasad i ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>99778fb5-7045-450c-9607-3c89babc9a05_0</td>
      <td>©Kancelaria Sejmu \n \n \n \ns. 1/44 \n \n2024...</td>
      <td>99778fb5-7045-450c-9607-3c89babc9a05</td>
      <td>- Jakie są główne obowiązki przedsiębiorcy wob...</td>
      <td>- Jakie są szczegółowe przepisy dotyczące praw...</td>
      <td>Prawo konsumenta do odstąpienia od umowy regul...</td>
      <td>259</td>
      <td>5</td>
      <td>0.199740</td>
      <td>0.276844</td>
      <td>{\n  "score": 4,\n  "explanation": "Oceniłem t...</td>
      <td>4</td>
      <td>\n  "explanation": "Oceniłem tekst na 4, ponie...</td>
      <td>5</td>
      <td>4</td>
      <td>```json\n{\n  "score": 5</td>
      <td>\n  "explanation": "Pytanie dotyczące szczegół...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>99778fb5-7045-450c-9607-3c89babc9a05_1</td>
      <td>przedsiębiorstwa; \n3) \nzasady i tryb wykonan...</td>
      <td>99778fb5-7045-450c-9607-3c89babc9a05</td>
      <td>- Jakie są główne zasady dotyczące prawa odstą...</td>
      <td>- Jakie są główne zasady dotyczące prawa odstą...</td>
      <td>Główne zasady dotyczące prawa odstąpienia od u...</td>
      <td>328</td>
      <td>5</td>
      <td>0.176067</td>
      <td>0.179241</td>
      <td>{\n  "score": 4,\n  "explanation": "Tekst jest...</td>
      <td>4</td>
      <td>\n  "explanation": "Tekst jest zrozumiały i kl...</td>
      <td>5</td>
      <td>4</td>
      <td>```json\n{\n  "score": 5</td>
      <td>\n  "explanation": "Pytanie jest precyzyjne i ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>99778fb5-7045-450c-9607-3c89babc9a05_1</td>
      <td>przedsiębiorstwa; \n3) \nzasady i tryb wykonan...</td>
      <td>99778fb5-7045-450c-9607-3c89babc9a05</td>
      <td>- Jakie są główne zasady dotyczące prawa odstą...</td>
      <td>- W jaki sposób ustawa reguluje zawieranie umó...</td>
      <td>Ustawa reguluje zawieranie umów na odległość d...</td>
      <td>237</td>
      <td>5</td>
      <td>0.187992</td>
      <td>0.243796</td>
      <td>{\n  "score": 4,\n  "explanation": "Tekst jest...</td>
      <td>4</td>
      <td>\n  "explanation": "Tekst jest spójny, poniewa...</td>
      <td>4</td>
      <td>5</td>
      <td>```json\n{\n  "score": 4</td>
      <td>\n  "explanation": "Pytanie dotyczące regulacj...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>727</th>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937_0</td>
      <td>©Kancelaria Sejmu \n \n \n \ns. 44/44 \n \n202...</td>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937</td>
      <td>- Jakie informacje powinny być zawarte w formu...</td>
      <td>- Czy formularz odstąpienia od umowy można wys...</td>
      <td>Formularz odstąpienia od umowy można wysłać w ...</td>
      <td>143</td>
      <td>4</td>
      <td>0.157546</td>
      <td>0.228782</td>
      <td>{\n  "score": 4,\n  "explanation": "Oceniłem t...</td>
      <td>4</td>
      <td>\n  "explanation": "Oceniłem tekst na 4, ponie...</td>
      <td>4</td>
      <td>4</td>
      <td>```json\n{\n  "score": 4</td>
      <td>\n  "explanation": "Pytanie dotyczące możliwoś...</td>
    </tr>
    <tr>
      <th>728</th>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937_0</td>
      <td>©Kancelaria Sejmu \n \n \n \ns. 44/44 \n \n202...</td>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937</td>
      <td>- Jakie informacje powinny być zawarte w formu...</td>
      <td>- Jakie są terminy na odstąpienie od umowy spr...</td>
      <td>Zgodnie z obowiązującymi przepisami, konsument...</td>
      <td>195</td>
      <td>4</td>
      <td>0.194363</td>
      <td>0.239988</td>
      <td>{\n  "score": 3,\n  "explanation": "Tekst jest...</td>
      <td>3</td>
      <td>\n  "explanation": "Tekst jest dość klarowny i...</td>
      <td>5</td>
      <td>4</td>
      <td>```json\n{\n  "score": 5</td>
      <td>\n  "explanation": "Pytanie dotyczące terminów...</td>
    </tr>
    <tr>
      <th>729</th>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937_1</td>
      <td>następujących towarów(*) umowy dostawy następu...</td>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937</td>
      <td>- Jakie towary lub usługi mogą być objęte umow...</td>
      <td>- Jakie towary lub usługi mogą być objęte umow...</td>
      <td>Według podanego tekstu, umowa dostawy może obe...</td>
      <td>160</td>
      <td>5</td>
      <td>0.245816</td>
      <td>0.325671</td>
      <td>{\n  "score": 4,\n  "explanation": "Tekst jest...</td>
      <td>4</td>
      <td>\n  "explanation": "Tekst jest spójny i klarow...</td>
      <td>4</td>
      <td>4</td>
      <td>```json\n{\n  "score": 4</td>
      <td>\n  "explanation": "Pytanie jest dobrze sformu...</td>
    </tr>
    <tr>
      <th>730</th>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937_1</td>
      <td>następujących towarów(*) umowy dostawy następu...</td>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937</td>
      <td>- Jakie towary lub usługi mogą być objęte umow...</td>
      <td>- Jakie informacje są wymagane od konsumenta p...</td>
      <td>Zgodnie z przedstawionym wzorem, wymagane info...</td>
      <td>166</td>
      <td>5</td>
      <td>0.168693</td>
      <td>0.304204</td>
      <td>{\n  "score": 4,\n  "explanation": "Oceniłem t...</td>
      <td>4</td>
      <td>\n  "explanation": "Oceniłem tekst na 4, ponie...</td>
      <td>4</td>
      <td>4</td>
      <td>```json\n{\n  "score": 4</td>
      <td>\n  "explanation": "Pytanie jest dobrze sformu...</td>
    </tr>
    <tr>
      <th>731</th>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937_1</td>
      <td>następujących towarów(*) umowy dostawy następu...</td>
      <td>e4264747-fe26-4f97-9c85-3d2f11caf937</td>
      <td>- Jakie towary lub usługi mogą być objęte umow...</td>
      <td>- Czy podpis konsumenta jest obowiązkowy w prz...</td>
      <td>Nie, podpis konsumenta nie jest obowiązkowy w ...</td>
      <td>174</td>
      <td>2</td>
      <td>0.183296</td>
      <td>0.245831</td>
      <td>{\n  "score": 4,\n  "explanation": "Tekst jest...</td>
      <td>4</td>
      <td>\n  "explanation": "Tekst jest zrozumiały i kl...</td>
      <td>4</td>
      <td>4</td>
      <td>```json\n{\n  "score": 4</td>
      <td>\n  "explanation": "Pytanie dotyczące obowiązk...</td>
    </tr>
  </tbody>
</table>
<p>732 rows × 17 columns</p>
</div>




```python
def get_score(x):
    digits = [int(s) for s in x.split() if s.isdigit()]
    return digits[0] if len(digits) > 0 else 0
```


```python
df['coherence_rate_pl'] = df['coherence_rate_pl'].apply(get_score)
df['global_relevance'] = df['global_relevance'].apply(get_score)
df['relevance'] = df['relevance_score'].apply(get_score)
```


```python
df['relevance'].value_counts(dropna=False)
```




    relevance
    4    407
    5    320
    2      3
    1      1
    3      1
    Name: count, dtype: int64




```python
df[['coherence_rate_pl', 'coherence_expl_pl']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coherence_rate_pl</th>
      <th>coherence_expl_pl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>\n  "explanation": "Tekst jest zrozumiały i kl...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>\n  "explanation": "Tekst jest zrozumiały i kl...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>\n  "explanation": "Oceniłem tekst na 4, ponie...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>\n  "explanation": "Tekst jest zrozumiały i kl...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>\n  "explanation": "Tekst jest spójny, poniewa...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>727</th>
      <td>4</td>
      <td>\n  "explanation": "Oceniłem tekst na 4, ponie...</td>
    </tr>
    <tr>
      <th>728</th>
      <td>3</td>
      <td>\n  "explanation": "Tekst jest dość klarowny i...</td>
    </tr>
    <tr>
      <th>729</th>
      <td>4</td>
      <td>\n  "explanation": "Tekst jest spójny i klarow...</td>
    </tr>
    <tr>
      <th>730</th>
      <td>4</td>
      <td>\n  "explanation": "Oceniłem tekst na 4, ponie...</td>
    </tr>
    <tr>
      <th>731</th>
      <td>4</td>
      <td>\n  "explanation": "Tekst jest zrozumiały i kl...</td>
    </tr>
  </tbody>
</table>
<p>732 rows × 2 columns</p>
</div>




```python
df['relevance'] = df.apply(lambda x: OpenAIReq(model=model_version).call_api(prompt=PromptBuilder.get_relevance_pl(question=x['questions_splitted'], context=x['text_chunk'])),axis=1)
```


```python
df['global_relevance'] = df.apply(lambda x: OpenAIReq().call_api(prompt=PromptBuilder.get_global_relevance(question=x['questions_splitted'])),axis=1)
```

## Evaluation


```python
reporting = SimilarityMatrix(data=df)
sim_matrix = reporting.create_similarity_matrix_for_wmd(on_column='questions_splitted')
```

    732it [03:34,  3.42it/s]
    


```python
sim_matrix.shape
```




    (729, 729)



#### Relevance anf global relevance


```python
pd.plotting.hist_series(df['relevance'])
```




    <Axes: >




    
![png](llm-bielik-ft-data-generation-part-I_files/llm-bielik-ft-data-generation-part-I_48_1.png)
    



```python
df['global_relevance'].value_counts(normalize=True)
```




    global_relevance
    4    0.629781
    5    0.351093
    3    0.017760
    1    0.001366
    Name: proportion, dtype: float64




```python
pd.plotting.hist_series(df['global_relevance'])
```




    <Axes: >




    
![png](llm-bielik-ft-data-generation-part-I_files/llm-bielik-ft-data-generation-part-I_50_1.png)
    


#### Coverage summary


```python
df['coverage_rating'].value_counts(normalize=True)
```




    coverage_rating
    5    0.903005
    4    0.079235
    3    0.008197
    2    0.006831
    1    0.002732
    Name: proportion, dtype: float64




```python
pd.plotting.hist_series(df['coverage_rating'])
```




    <Axes: >




    
![png](llm-bielik-ft-data-generation-part-I_files/llm-bielik-ft-data-generation-part-I_53_1.png)
    


#### Word mover distance


```python
# WMD question <-> chunk
pd.plotting.hist_series(df['wmd_chunk_question'])
```




    <Axes: >




    
![png](llm-bielik-ft-data-generation-part-I_files/llm-bielik-ft-data-generation-part-I_55_1.png)
    



```python
mean_wmd_qch = df['wmd_chunk_question'].mean()
std_wmd_qch = df['wmd_chunk_question'].std()
print(f'Mean for question to chunk for WMD is {mean_wmd_qch} with std as {std_wmd_qch}')
```

    Mean for question to chunk for WMD is 0.2046351021518607 with std as 0.055642205389882675
    


```python
# WMD question <-> answer
pd.plotting.hist_series(df['wmd_question_answer_200'])
```




    <Axes: >




    
![png](llm-bielik-ft-data-generation-part-I_files/llm-bielik-ft-data-generation-part-I_57_1.png)
    



```python
mean_wmd_qa = df['wmd_question_answer_200'].mean()
std_wmd_qa = df['wmd_question_answer_200'].std()
print(f'Mean for question to answer for WMD is {mean_wmd_qa} with std as {std_wmd_qa}')
```

    Mean for question to answer for WMD is 0.17849177597334837 with std as 0.04319934620213855
    


```python
# WMD question <-> question
std, mean = reporting.get_summary_for_wmd(sim_matrix)
print(f'Mean for question to question for WMD is {mean} with std as {std}')
```

    Mean for question to question for WMD is 0.24440089371260362 with std as 0.06065276852233461
    



#### Coherence rate


```python
df['coherence_rate_pl'].value_counts(normalize=True)
```




    coherence_rate_pl
    4    0.829235
    3    0.121585
    2    0.035519
    5    0.013661
    Name: proportion, dtype: float64




```python
pd.plotting.hist_series(df['coherence_rate_pl'])
```




    <Axes: >




    
![png](llm-bielik-ft-data-generation-part-I_files/llm-bielik-ft-data-generation-part-I_63_1.png)
    



```python
pd.plotting.hist_series(df['coherence_rate'])
```




    <Axes: >




    
![png](llm-bielik-ft-data-generation-part-I_files/llm-bielik-ft-data-generation-part-I_64_1.png)
    


### Evaluation Summary
From the results we can conclude that GPT prepared for us quite good dataset - majority of values are high (low in terms of similarity). The only outlier in those results is the coherence score which can be taken for further investigation and analysis. Both polish and english version are not generating the best scores - they are placed somewhere in the upper middle (3 and 4 rating).

Overall, the results have proven that generated question with support of GPT model are good enough to use them as the input instructions for the model. Now, question value pairs will be transformed into the proper structure to be later on used for fine tuning.   

## Data preparation for FT


```python
df.to_excel('data/full_26_01_2025.xlsx', index=False)
```


```python
# SELECT THE DATA
df = pd.read_excel('data/full_26_01_2025.xlsx')
```


```python
selected_data = df[df['coherence_rate_pl'] > 3]
selected_data = selected_data[selected_data['relevance'] > 3]
```


```python
df = selected_data
```


```python
df.questions_splitted
```




    0      - Jakie są główne obowiązki przedsiębiorcy wob...
    1      - Jakie zasady i tryb dotyczą zawierania umów ...
    2      - Jakie są szczegółowe przepisy dotyczące praw...
    3      - Jakie są główne zasady dotyczące prawa odstą...
    4      - W jaki sposób ustawa reguluje zawieranie umó...
                                 ...                        
    726    - Jakie informacje powinny być zawarte w formu...
    727    - Czy formularz odstąpienia od umowy można wys...
    729    - Jakie towary lub usługi mogą być objęte umow...
    730    - Jakie informacje są wymagane od konsumenta p...
    731    - Czy podpis konsumenta jest obowiązkowy w prz...
    Name: questions_splitted, Length: 614, dtype: object




```python
qa_df_200 = df[['questions_splitted', 'answer_200']]
qa_df_200 = qa_df_200.rename(columns={'questions_splitted': 'question', 'answer_200': 'answer'})

qa_df_200 = qa_df_200.sample(frac=1)

now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
with open(f'ft_output_data/data_ft_{now}_hq.jsonl', "w", encoding='utf8') as f:
    f.write(qa_df_200.to_json(orient='records', lines=True, force_ascii=False))
```

## Summary
As our last piece of code is executed, we can finish first part of this article - data preparation. I hope you enjoyed it as much as me! Let's move on to the second part of the finetuning polish LLM model with our generated data. 

See ya! 

## References and sources
1. https://arxiv.org/abs/2401.08406
2. https://www.statista.com/statistics/262946/most-common-languages-on-the-internet/
3. https://medium.com/@bijit211987/the-rise-of-small-language-models-efficient-customizable-cb48ddee2aad
4. https://python.langchain.com/docs/how_to/recursive_text_splitter/
5. https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html
