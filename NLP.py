# Author : Svitlana Kramar
# All Libraries required for this lab are listed below. The libraries pre-installed on Skills Network Labs are commented.
# !mamba install -qy pandas==1.3.4 numpy==1.21.4 seaborn==0.9.0 matplotlib==3.5.0 scikit-learn==0.20.1
# Note: If your environment doesn't support "!mamba install", use "!pip install"

import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModel

#Sentiment-Analysis

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("Having three long haired, heavy shedding dogs at home, I was pretty skeptical that this could hold up to all the hair and dirt they trek in, but this wonderful piece of tech has been nothing short of a godsend for me! ")

#Topic-Classification

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "Exploratory Data Analysis is the first course in Machine Learning Program that introduces learners to the broad range of Machine Learning concepts, applications, challenges, and solutions, while utilizing interesting real-life datasets",
    candidate_labels=["art", "natural science", "data analysis"],
)

#Text Generator

generator = pipeline("text-generation", model="gpt2")
generator("This course will teach you")

generator = pipeline("text-generation", model="distilgpt2")
generator(
    "This course will teach you",
    max_length=30,
    num_return_sequences=2,
)

#masked language modeling

unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("This course will teach you all about <mask> models.", top_k=4)

#NER(Name Entity Recognition)

ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("My name is Roberta and I work with IBM Skills Network in Toronto")

del ner

#questioning-answering

qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "Which name is also used to describe the Amazon rainforest in English?"
context = "The Amazon rainforest, also known in English as Amazonia or the Amazon Jungle."
qa_model(question = question, context = context)

#text-summarization

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    """
Exploratory Data Analysis is the first course in Machine Learning Program that introduces learners to the broad range of Machine Learning concepts, applications, challenges, and solutions, while utilizing interesting real-life datasets. So, what is EDA and why is it important to perform it before we dive into any analysis?
EDA is a visual and statistical process that allows us to take a glimpse into the data before the analysis. It lets us test the assumptions that we might have about the data, proving or disproving our prior believes and biases. It lays foundation for the analysis, so our results go along with our expectations. In a way, it’s a quality check for our predictions.
As any data scientist would agree, the most challenging part in any data analysis is to obtain a good quality data to work with. Nothing is served to us on a silver plate, data comes in different shapes and formats. It can be structured and unstructured, it may contain errors or be biased, it may have missing fields, it can have different formats than what an untrained eye would perceive. For example, when we import some data, very often it would contain a time stamp. To a human it is understandable format that can interpreted. But to a machine, it is not interpretable, so it needs to be told what that means, the data needs to be transformed into simple numbers first. There are also different date-time conventions depending on a country (i.e., Canadian versus USA), metric versus imperial systems, and many other data features that need to be recognized before we start doing the analysis. Therefore, the first step before performing any analysis – is get really aquatinted with your data!
This course will teach you to ‘see’ and to ‘feel’ the data as well as to transform it into analysis-ready format. It is introductory level course, so no prior knowledge is required, and it is a good starting point if you are interested in getting into the world of Machine Learning. The only thing that is needed is some computer with internet, your curiosity and eagerness to learn and to apply acquired knowledge.  If you live in Canada, you might be interested about gasoline prices in different cities or if you are an insurance actuary you need to analyze the financial risks that you will take based on your clients information. Whatever is the case, you will be able to do your own analysis, and confirm or disprove some of the existing information.
The course contains videos and reading materials, as well as well as a lot of interactive practice labs that learners can explore and apply the skills learned. It will allow you to use Python language in Jupyter Notebook, a cloud-based skills network environment that is pre-set for you with all available to be downloaded packages and libraries. It will introduce you to the most common visualization libraries such as Pandas, Seaborn, and Matplotlib to demonstrate various EDA techniques with some real-life datasets.

"""
)

del summarizer

#Translation

en_fr_translator = pipeline("translation_en_to_fr", model="t5-small")
en_fr_translator("How old are you?")

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("La science des données est la meilleure.")

###################################################################################################

#Excercise

#For sentiment analysis, we can also use a specific model that is better suited to our use case by providing the name of the model. For example, if we want a sentiment analysis model for tweets, we can specify the following model id: "cardiffnlp/twitter-roberta-base-sentiment". This model has been trained on ~58M tweets and fine-tuned for sentiment analysis with the "TweetEval" benchmark. The output labels for this model are: 0 -> Negative; 1 -> Neutral; 2 -> Positive.
#In this Exercise, use "cardiffnlp/twitter-roberta-base-sentiment" model pre-trained on tweets data, to analyze any tweet of choice. Optionally, use the default model (used in Example 1) on the same tweet, to see if the result will change.

specific_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
data = "Artificial intelligence are already causing friction in the workforce."
specific_model(data)

#Output
[{'label': 'LABEL_0', 'score': 0.6556357145309448}]

#Topic Classfication
#In this Exercise, use any sentence of choice to classify it under any classes/ topics of choice. Use "zero-shot-classification" and specify the model="facebook/bart-large-mnli".

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "I love travelling and learning new cultures",
    candidate_labels=["art", "education", "travel"],
)

#Output
{'sequence': 'I love travelling and learning new cultures',
 'labels': ['travel', 'education', 'art'],
 'scores': [0.9902299642562866, 0.005778191145509481, 0.003991859499365091]}

#Text Generation
#In this Exercise, use 'text-generator' and 'gpt2' model to complete any sentence. Define any desirable number of returned sentences.

generator = pipeline('text-generation', model = 'gpt2')
generator("Hello, I'm a language model", max_length = 30, num_return_sequences=3)

#Output
[{'generated_text': "Hello, I'm a language model designer for an organization working to build the next generation of social and linguistic data services.\n\nI do also have"},
 {'generated_text': "Hello, I'm a language model researcher at a think tank in Stockholm and an expert on the Swedish Internet infrastructure (so don't call my research a"},
 {'generated_text': 'Hello, I\'m a language modeler, but I have this huge focus. It\'s called the "language modeler" and it\'s basically what'}]

#NER (Name Entity Recognition)
#In this Exercise, use any sentence of choice to extract entities: person, location and organization, using Name Entity Recognition task, specify model as "Jean-Baptiste/camembert-ner".

nlp = pipeline("ner", model="Jean-Baptiste/camembert-ner", grouped_entities=True)
example = "Her name is Anjela and she lives in Seoul."

ner_results = nlp(example)
print(ner_results)

#Output
[{'entity_group': 'PER', 'score': 0.94814444, 'word': 'Anjela', 'start': 11, 'end': 18}, {'entity_group': 'LOC', 'score': 0.99861133, 'word': 'Seoul', 'start': 35, 'end': 41}]

#Questioning and answering
#In this Exercise, use any sentence and a question of choice to extract some information, using "distilbert-base-cased-distilled-squad" model.

question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question_answerer(
    question="Which lake is one of the five Great Lakes of North America?",
    context="Lake Ontario is one of the five Great Lakes of North America. It is surrounded on the north, west, and southwest by the Canadian province of Ontario, and on the south and east by the U.S. state of New York, whose water boundaries, along the international border, meet in the middle of the lake.",
)

#Output
{'score': 0.9834363460540771, 'start': 0, 'end': 12, 'answer': 'Lake Ontario'}

#Text Summarization
#In this Exercise, use any document/paragraph of choice and summarize it, using "sshleifer/distilbart-cnn-12-6" model.

ummarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6",  max_length=59)
summarizer(
    """
Lake Superior in central North America is the largest freshwater lake in the world by surface area and the third-largest by volume, holding 10% of the world's surface fresh water. The northern and westernmost of the Great Lakes of North America, it straddles the Canada–United States border with the province of Ontario to the north, and the states of Minnesota to the northwest and Wisconsin and Michigan to the south. It drains into Lake Huron via St. Marys River and through the lower Great Lakes to the St. Lawrence River and the Atlantic Ocean.
"""
)

#Output
[{'summary_text': " Lake Superior is the largest freshwater lake in the world by surface area . It holds 10% of the world's surface fresh water . It straddles the Canada–U.S. border with the province of Ontario to the north . It drains into Lake Huron via St. Marys River and through the lower Great Lakes to the St. Lawrence River and the Atlantic Ocean ."}]

#Translation
#In this Exercise, use any sentence of choice to translate English to German. The translation model you can use is "translation_en_to_de".

translator = pipeline("translation_en_to_de", model="t5-small")
print(translator("New York is my favourite city", max_length=40))

#Output
[{'translation_text': 'New York ist meine Lieblingsstadt'}]
