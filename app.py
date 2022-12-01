import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("./distilbartTokenizer_v1")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import nltk
nltk.download('punkt')
from nltk import tokenize

from spacy.lang.en.examples import sentences 
nlp = spacy.load("en_core_web_sm")
from bs4 import BeautifulSoup #to help with the web scraping
import requests #make http request to the web and 
def extr_summ(doc, n):
    doc = nlp(doc)

    keyword = []
    stopwords = list(STOP_WORDS)
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    for token in doc:
        if(token.text in stopwords or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            keyword.append(token.text)

    freq_word = Counter(keyword)

    max_freq = Counter(keyword).most_common(1)[0][1]
    for word in freq_word.keys():  
            freq_word[word] = (freq_word[word]/max_freq)
    freq_word.most_common(5)

    sent_strength={}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent]+=freq_word[word.text]
                else:
                    sent_strength[sent]=freq_word[word.text]

    summarized_sentences = nlargest(n, sent_strength, key=sent_strength.get)

    final_sentences = [ w.text for w in summarized_sentences ]
    summary = ' '.join(final_sentences)
    return summary

@st.cache(allow_output_mutation=True)
def load_summarizer():
    # model = pipeline("summarization")
    # model = AutoModelForSeq2SeqLM.from_pretrained("./distilbartModel_v1")
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

    return model


def generate_chunks(inp_str):
    max_chunk = 500
    inp_str = inp_str.replace('.', '.<eos>')
    inp_str = inp_str.replace('?', '?<eos>')
    inp_str = inp_str.replace('!', '!<eos>')
    
    sentences = inp_str.split('<eos>')
    current_chunk = 0 
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1: 
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])
    return chunks

def get_text_from_URL(URL):
    # URL = "https://towardsdatascience.com/a-bayesian-take-on-model-regularization-9356116b6457"
    # URL = "https://hackernoon.com/will-the-game-stop-with-gamestop-or-is-this-just-the-beginning-2j1x32aa"
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = soup.find_all(['h1', 'p'])
    text = [result.text for result in results]
    ARTICLE = ' '.join(text)
    print(ARTICLE)
    return ARTICLE

def gen_summary(model, long_text, len_chunks, summ_range):
  model_gen_summ = []
  for idx, i in enumerate(long_text):
    ARTICLE_TO_SUMMARIZE = i
    # print('\nOriginal Text : \n', ARTICLE_TO_SUMMARIZE)
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt', truncation=True)
    # inputs.to(device)
    # model.to(device)
    # Generate Summary
    n = len(inputs['input_ids'][0])
    min = summ_range[0]
    max = summ_range[1]
    # summary_ids = model.generate(inputs['input_ids'], do_sample = True, num_beams=3, min_length=100, max_length=100, num_return_sequences=5, temperature=1.0, top_k = 50, top_p = 0.5)
    summary_ids = model.generate(inputs['input_ids'], min_length=min // len_chunks, max_length=max // len_chunks,temperature=2.0, top_k = 50, top_p = 0.2, repetition_penalty = 10.0)  # Experiment here
    print(f'\nModel Summary {idx+1}:')

    modelsumm = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    model_gen_summ.append(''.join(modelsumm))

    print(''.join(modelsumm))

  return model_gen_summ

def main():
    print('\nWaiting for input')
    st.title("Automated Text Summarization")
    activities = ["Using Text","Using URL"]
    tab1, tab2= st.tabs(activities)
    summarizer = load_summarizer()
    
    with tab1:
        min = 50
        max = 100
        do_sample = False
        option = st.selectbox(
        'Select',
        ['Simple', 'Advanced'])
        if option == 'Advanced':
            with st.form("my_form"):
                min, max = st.select_slider(
                'Select a range of length',
                options=[i for i in range(10, 501, 10)],
                value=(10, 100),)
                do_sample = st.checkbox("Use sampling")

                # Every form must have a submit button.
                submitted = st.form_submit_button("Submit")
                if submitted:
                    st.write("slider", slider_val, "checkbox", checkbox_val)
        

        sentence = st.text_area('Please type your article :', height=400)
        button_txt = st.button("Summarize text")

        sents_count = tokenize.sent_tokenize(sentence)
        
        # do_sample = st.sidebar.checkbox("Do sample", value=False)
        with st.spinner("Generating Summary.."):
            if button_txt and sentence:
                # print(url[:500])
                if len(sents_count) < 6:
                    print("Extractive Summarization", len(sents_count))
                    text = extr_summ(sentence, len(sents_count) // 2)
                else:
                    print("Abstractive Summarization")
                    chunks = generate_chunks(sentence)
                    len_chunks = len(chunks)
                    print(f'Chunks length : {len_chunks}')
                    # chunks = extr_summ(sentence, len(sentence) )
                    # print(f'Extractive Summary : \n{chunks}\n')
                    # res = summarizer(chunks, min_length=100, max_length=200,temperature=2.0, top_k = 50, top_p = 0.2, repetition_penalty = 10.0)
                    res = gen_summary(summarizer, chunks, len_chunks, (min, max))
                    text = ' '.join(res)
                    # print(f'\nAbstractive Summary : \n{text}\n')
                    # st.write(result[0]['summary_text'])
                st.write(text)
    
    with tab2:
        sentence = st.text_area('Please paste your URL :', height=30)
        #Example = "https://towardsdatascience.com/a-bayesian-take-on-model-regularization-9356116b6457"
        # https://medium.com/@randylaosat/a-beginners-guide-to-machine-learning-dfadc19f6caf
        button_URL = st.button("Summarize URL")

        # max_url = st.sidebar.slider('Select max', 50, 500, step=10, value=150)
        # min_url = st.sidebar.slider('Select min', 10, 450, step=10, value=50)
        # do_sample = st.sidebar.checkbox("Do sample", value=False)
        with st.spinner("Generating Summary.."):
            if button_URL and sentence:
                try:
                    url = get_text_from_URL(sentence)
                    sents_count = tokenize.sent_tokenize(url)
                except ValueError:
                    st.error("Please enter a valid input")
                    return
                # print(sents_count)
                # if len(sents_count) < 6:
                #     print("Extractive Summarization", len(sents_count))
                #     text = extr_summ(sentence, len(sents_count))
                # else:
                print("Abstractive Summarization")
                chunks = generate_chunks(url)
                print(f'Chunks length : {len(chunks)}')
                # chunks = extr_summ(sentence, len(sentence) )
                # print(f'Extractive Summary : \n{chunks}\n')
                res = summarizer(chunks, )
                text = ' '.join([summ['summary_text'] for summ in res])
                # print(f'\nAbstractive Summary : \n{text}\n')
                # st.write(result[0]['summary_text'])
                st.write(text)

if __name__ == '__main__':
    main()
