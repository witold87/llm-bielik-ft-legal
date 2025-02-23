# Fine-tuning Bielik - Polish LLM from Speakleash - how to guide

## Goal
The goal of this repository is to show in two steps how to fine-tune the Bielik model from Speakleash on a custom dataset. 
The first step is to generate the data for fine-tuning, and the second step is to fine-tune the model.

## TL;DR
1. The model was fine-tuned on a custom dataset generated using the GPT-4o model from OpenAI.
2. The data was validated by using the LLM-based and distance-based metrics.
3. Fine-tuned model achieved comparable accuracy to the groundtruth and outperformed the original (base) models (7B and 11B).


## Code
Code and notebooks are here:
- [Data generation](https://github.com/witold87/llm-bielik-ft-legal/blob/main/src/llm-bielik-ft-data-generation-part-I.ipynb)
- [Fine-tuning](https://github.com/witold87/llm-bielik-ft-legal/blob/main/src/llm-bielik-ft-fine-tuning-part-II.ipynb)
    

## Table of contents

Those notebook cover the following topics:
1. Loading and splitting data into chunks
2. Generating new data (question and answers) for the future fine-tuning using GPT-4o model from OpenAI
3. Evaluating generated data using LLM-based and distance-based (Word Mover Distance) metrics.
4. Selecting the data based on metrics 
5. Preparing the output for FT
6. Loading the data with proper format
7. Preparing the model for fine-tuning
8. Fine-tuning the model
9. Running the model against the validation set and evaluating the results in comparison to the original model and groundtruth


## Sources
- [Speakleash](https://huggingface.co/speakleash/Bielik-11B-v2.3-Instruct) - Bielik model
- [OpenAI](https://openai.com) - GPT-4o model
- [Hugging Face](https://huggingface.co) - Transformers library
- Other sources are present in the notebooks
