# Infosys-Text-Summarization
This is the project about Text summarization. We are building model which converts larger texts into smaller summary likely heading

# Step 1
## Finding the Suitable Dataset: 

https://cs.nyu.edu/~kcho/DMQA/
This dataset contains the documents and accompanying questions from the news articles of CNN and from Daily Mail. There are approximately 90k documents and 380k questions in CNN.There are approximately 197k documents and 879k questions in DM.


## Initial design
![Initial Design](<Initial Design.jpeg>)

## Dataset

We reduced the data into 40mb (1000 records) which will be also useful when we are training the pre-trained models

![Plot of dataset](<output.png>)



Firstly i would like to develop extractive model
for this extractive model i want to use pre-trained BERT model as the base model ,it is the advanced model for text extraction
and iam gonna feed dataset into the custom model based on Bert model
for this custom model we use bert tokeniser which is developed by the google

## Changing the model

we changed the model from BERT to T5-small Because we faced so many problems while training the model or fine-tuning the model.
 We almost spent 4 days on it to resolve the errors but we are couldn't solve it, i don't know why!

## T5-small 

The T5-small model is a versatile and efficient variant of the T5 model family, suitable for a wide range of NLP tasks while being resource-friendly. Its smaller size allows for faster processing and deployment in resource-constrained environments, making it an excellent choice for many practical applications where speed and efficiency are prioritized over maximum performance.

The T5-small model is a smaller version of the T5 (Text-to-Text Transfer Transformer) model, developed by Google Research. T5 is designed to treat all NLP tasks as text-to-text problems, allowing it to use the same model, objective, training procedure, and decoding process for various tasks such as translation, summarization, and question answering.

 ### Key Features of T5-Small:
#### Architecture:

The T5-small model is based on the Transformer architecture, specifically the encoder-decoder structure used in sequence-to-sequence tasks.
It has 60 million parameters, which is significantly fewer than the larger T5 models, making it more efficient and faster to run, though less powerful in terms of performance.
#### Model Size:

The T5 family includes various sizes such as T5-small, T5-base, T5-large, T5-3B (3 billion parameters), and T5-11B (11 billion parameters). T5-small is the most compact and lightweight variant.
#### Training:

T5 models are pre-trained on the C4 (Colossal Clean Crawled Corpus) dataset, a large collection of web pages, using a text-to-text framework. This means they are trained to convert any input text into a desired output text.
The pre-training involves a span-corruption task, where random spans of text are replaced with a special mask token, and the model learns to predict the missing spans.
#### Use Cases:

Despite its smaller size, T5-small can still be effectively fine-tuned for a variety of downstream tasks such as:
Text classification
Question answering
Text summarization
Translation
Text generation
It is particularly useful for scenarios where computational resources are limited or when a quick inference is required.
#### Performance:

While T5-small offers the advantages of efficiency and speed, it may not match the performance of larger models on more complex tasks or datasets. However, it strikes a good balance between resource usage and performance for many practical applications.


## Evaluating the model

The ROUGE score is a powerful tool for evaluating the performance of text generation models. By providing various metrics to assess the overlap between generated and reference texts, it helps quantify how well a model performs in tasks like summarization and translation. Despite its limitations, ROUGE remains a crucial metric in the field of NLP.

Here are the results of ROUGE

rouge1: 0.5435...
rouge2: 0.2868...
rougeL: 0.4135...
rougeLsum: 0.4918...

Actually the score is not that bad. There are 2 reasons why my score is not very good
--> Used insufficient data to train the model because to process the training i don't have GPU in my computer
--> The model is not well tuned because i don't have enough time to tune the model
--> The dataset we used is CNN/Dailymails, In this dataset most of the summaries (highlights) is in the abstractive format we used thr extractive method to train

### Results


![ROUGE Score](<output2.png>)