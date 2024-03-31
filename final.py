import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers import pipeline
import numpy as np
import streamlit as st 

import spacy
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.parsers.plaintext import PlaintextParser 
from sumy.nlp.tokenizers import Tokenizer 
from sumy.utils import get_stop_words 

from spacy.lang.pt.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import en_core_web_sm


# NLP Pkgs

# def set_seed(seed):
#   torch.manual_seed(seed)
#   if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# set_seed(42)

# model = T5ForConditionalGeneration.from_pretrained('t5-small')
# tokenizer = T5Tokenizer.from_pretrained('t5-small')
# # device = torch.device('cpu')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print ("device ",device)
# model = model.to(device)
# # fastpunct = FastPunct('en')
# tokenizer2 = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
# model2 = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
# model2 = model2.to(device)

# Function for Sumy Summarization
def sumy_summarizer(text):
  return summarize(text)


def punctuation(text):
  text_correct = text.lower()
  final = fastpunct.punct([text_correct])
  return str(final[0])


def summarization_spacy(text):

        
    nlp = en_core_web_sm.load()
    
    
    doc = nlp(text)


    corpus = [sent.text.lower() for sent in doc.sents ]
    
    cv = CountVectorizer(stop_words=list(STOP_WORDS))   
    cv_fit=cv.fit_transform(corpus)    
    word_list = cv.get_feature_names();    
    count_list = cv_fit.toarray().sum(axis=0)    

    
    word_frequency = dict(zip(word_list,count_list))

    val=sorted(word_frequency.values())

    # Check words with higher frequencies
    higher_word_frequencies = [word for word,freq in word_frequency.items() if freq in val[-3:]]
    # print("\nWords with higher frequencies: ", higher_word_frequencies)

    # gets relative frequencies of words
    higher_frequency = val[-1]
    for word in word_frequency.keys():  
        word_frequency[word] = (word_frequency[word]/higher_frequency)


    # SENTENCE RANKING: the rank of sentences is based on the word frequencies
    sentence_rank={}
    for sent in doc.sents:
        for word in sent :       
            if word.text.lower() in word_frequency.keys():            
                if sent in sentence_rank.keys():
                    sentence_rank[sent]+=word_frequency[word.text.lower()]
                else:
                    sentence_rank[sent]=word_frequency[word.text.lower()]
            else:
                continue

    top_sentences=(sorted(sentence_rank.values())[::-1])
    top_sent=top_sentences[:3]

    # Mount summary
    summary=[]
    for sent,strength in sentence_rank.items():  
        if strength in top_sent:
            summary.append(sent)

    summary = str(summary[0])+str(summary[1])+str(summary[2])
    # return orinal text and summary
    return summary

def answergen(context, question):
    # Load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', return_token_type_ids=True)
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

    encoding = tokenizer.encode_plus(question, context, return_tensors="pt")  

    # Extract input ID and attention mask
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # Model prediction
    outputs = model(input_ids, attention_mask=attention_mask)
    start_scores, end_scores = outputs.start_logits, outputs.end_logits  # 正确获取start_scores和end_scores

    # Extract the tag ID of the answer
    ans_tokens = input_ids[0, torch.argmax(start_scores) : torch.argmax(end_scores)+1]  # 索引input_ids以提取答案
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)

    # Convert answer's tokens to string
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer

  
def summary_t5(text):

  model = T5ForConditionalGeneration.from_pretrained('t5-small')
  tokenizer = T5Tokenizer.from_pretrained('t5-small')

  preprocess_text = text.strip().replace("\n","")
  t5_prepared_Text = "summarize: "+preprocess_text
  # print ("original text preprocessed: \n", preprocess_text)

  tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


  # summmarize 
  summary_ids = model.generate(tokenized_text,
                                      num_beams=4,
                                      no_repeat_ngram_size=2,
                                      min_length=50,
                                      max_length=70,
                                      early_stopping=True)

  output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  return output



# define sentiment analysis
def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
    results = sentiment_pipeline(text)
    return results


def greedy_decoding (inp_ids,attn_mask):
  greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
  Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
  return Question.strip().capitalize()


def beam_search_decoding (inp_ids,attn_mask,model,tokenizer):
  # model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_boolean_questions')
  # tokenizer = T5Tokenizer.from_pretrained('t5-small')

  beam_output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                               num_beams=10,
                               num_return_sequences=3,
                               no_repeat_ngram_size=2,
                               early_stopping=True
                               )
  Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
               beam_output]
  return [Question.strip().capitalize() for Question in Questions]

# Streanlit Interface 
def main():
    """ 🪄 Magic Site - NLP with Text """

    # Title
    st.title("🪄 Magic App - NLP with Text")
    st.subheader("Want to summarize and detect the emotion of text in 1 min? You come to the right place!")

    # Description
    st.subheader("📌 How to use this app? Guide with an Example")
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 10px; border-radius: 10px;">
      <p><b>Example Text:</b><br>
      Job Position: Data Analyst Internship<br><br>
      Your profile:<br>
      You are looking for an internship from July 2024 for 6 months.<br>
      Fluent in French and English.<br>
      You have skills in SQL, Excel, DAX, Python, and Power BI.<br>
      Written and verbal communication skills in English, with the ability to operate in a multicultural and multilingual environment.<br>
      Strong analytical skills to identify patterns and insights, and provide actionable recommendations.<br>
      A curious and investigative mind, always ready to dive deeper into data.<br>
      Feel excited? Come to join us!
    </div>
    """, unsafe_allow_html=True)

    # Functions
    st.subheader("📌 Functions Explanation")
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 10px; border-radius: 10px;">
      <p><b>•  Answer Extraction</b><br>
      Write your text and then type the question that you want to get from the text.<br>

      <p><b>Example Answer:</b><br> 
        sql , excel , dax , python , and power bi 
                
      <p><b> •  Sentiment Analysis</b><br>  
      Write your text and then click analyze, it will show the sentiment of the text with score.<br>

      <p><b>Example Answer:</b><br> 
        Sentiment: POSITIVE with a score of 0.9985
                
      <p><b>•  Text Summarization</b><br> 
      Write your text and then get a summary of it with key points.<br>

      <p><b>Example Answer:</b><br>
        you are looking for an internship from July 2024 for 6 months.Fluent in French and English.You have skills in SQL, Excel, DAX, Python, and Power BI.A curious and investigative mind, always ready to dive deeper into data.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("---")
  # Tokenization
    if st.checkbox("Answer Extraction"):
        st.subheader("Extract Answer from text")

        context =  st.text_area("Type your text here")
        question= st.text_input('Question: ', None)

        if st.button("Extract"):
            answer = answergen(context, question)
            st.success(answer)

    if st.checkbox("Sentiment Analysis"):
        st.subheader("Analyze the sentiment of your text")
        user_text = st.text_area("Enter Text", "Type Here...")
        if st.button("Analyze"):
            sentiment_result = analyze_sentiment(user_text)
            st.success(f"Sentiment: {sentiment_result[0]['label']} with a score of {sentiment_result[0]['score']:.4f}")

# Summarization
    if st.checkbox("Show Text Summarization"):
        st.subheader("Summarize Your Text")
        message = st.text_area("Enter Text", "Type Here ..")
        if st.button("Summarize"):
          st.text("Using T5 for summarization...")
          summary_result = summary_t5(message)
          st.success(summary_result)
  


# Sidebar
st.sidebar.subheader(":question: About App")
if st.sidebar.button('Welcome to our NLP toolbox'):
    st.sidebar.write("This app is designed to empower users with a suite of natural language processing (NLP) tools, enabling you to extract insights from text effortlessly."),
    st.sidebar.write("Whether you're analyzing sentiments, summarizing articles, or extracting answers from passages, our app provides intuitive and powerful functionality to meet your NLP needs.")
  
  
st.sidebar.subheader(":female-technologist: Team Members")
st.sidebar.markdown("WANG Lin, TANG Shuhui, WANG Yu")

st.sidebar.subheader(":bulb: Inspired By")
st.sidebar.markdown("[Project Reference - Github](https://github.com/parthplc/TeachEasy)")
st.sidebar.info("Thanks to: Vaibhav, Meghav, Kritika")
  
  

if __name__ == '__main__':
   main()