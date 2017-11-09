from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Lsa
from sumy.summarizers.edmundson import EdmundsonSummarizer as Edmundson
from sumy.summarizers.lex_rank import LexRankSummarizer as LexRank
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.evaluation.rouge import rouge_n, rouge_1, rouge_2
from sumy.models.dom import Sentence, Paragraph, ObjectDocumentModel
import nltk
from nltk.corpus import stopwords as stpwords
from itertools import product
from nltk.corpus import wordnet as wn
from flask import Flask, render_template, request, json
from urllib.request import urlopen
from bs4 import BeautifulSoup as BS
import html5lib

app = Flask(__name__)

@app.route("/")
def main():
    keys = keyword('boot_load')
    summs = query('boot_load')
    return render_template('index.html', key_1 = keys[0], key_2 = keys[1], key_3 = keys[2], p_1 = summs[0], p_2 = summs[1], p_3 = summs[2])

#constants for summary
LANGUAGE = "english"
SENTENCES_COUNT = 4

#calculate ROUGE_N values for n = 1, 2, 3
def calc_value(eval_sentences, ref_sentences):
    n_1 = rouge_1(eval_sentences, ref_sentences)
    n_2 = rouge_2(eval_sentences, ref_sentences)
    n_3 = rouge_n(eval_sentences, ref_sentences, 3)

    #print('n1 '+ str(n_1) +'\n')
    #print('n2 ' + str(n_2) + '\n')
    #print('n3 ' + str(n_3) + '\n')
    #print('avg ' + str((n_1+n_2+n_3)/3))

    return (n_1+n_2+n_3)/3

def max_r_value(Lsa_eval, Ed_eval, Lex_eval, ref):
    Ed = calc_value(Ed_eval, ref)
    Lsa = calc_value(Lsa_eval, ref)
    Lex = calc_value(Lex_eval, ref)
    list=[Ed, Lsa, Lex]
    max = 0
    #returns index of max avg rogue_n val 0=Ed, 1=Lsa, 2=Lex
    for i in range(len(list)):
        if list[i] > max:
            max = list[i]
            maxIndex = i
    #print("Maximum average Rouge test value " + str(max))
    return maxIndex

@app.route('/keywordCall', methods=['POST'])
def keywordCall():
    # read the posted values from the UI
    input = request.form['inputSearch']
    keys = keyword(str(input))
    summs = query(keys)
    return render_template('index.html', key_1 = str(keys[0]), key_2 = str(keys[1]), key_3 = str(keys[2]), p_1 = str(summs[0]), p_2 = str(summs[1]), p_3 = str(summs[2]))

def keyword(input):
    if input == 'boot_load':
        return [' ', ' ', ' ']
    # temp sentence
    temp_s = input
    # timer
    #start = timeit.default_timer()
    # tokenize
    token_s = nltk.word_tokenize(temp_s)
    # remove stopwords
    stopWords = set(stpwords.words('english'))
    print(stopWords)
    #filtered_tokens = [word for word in token_s if word not in stopWords]
    filtered_tokens = []
    for word in token_s:
        if word not in stopWords:
            filtered_tokens.append(word)
    # ini print
    print('tokens:', token_s)
    print('filtered:', filtered_tokens)

    # dummy var
    prob_list = []
    comp_list = []
    top_3 = []

    # loops through for comparisons
    filtered_iter = iter(filtered_tokens)
    for i in range(len(filtered_tokens)):
        total = 0
        elem = next(filtered_iter)

        # loops through phrase with each keyword
        allsyns1 = set(ss for word in token_s for ss in wn.synsets(word))
        allsyns2 = set(ss for word in elem for ss in wn.synsets(word))
        full_list = [(wn.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2)]
        score_list = iter(full_list)

        # get avg score
        for j in range(len(full_list)):
            total += float(next(score_list)[0])
        print('word:', elem, '| weighted avg:', total / len(full_list))

        # store dummy
        prob_list.append(float(total / len(full_list)))
        comp_list.append(elem)

    # grab min(3, #of elements)
    for i in range(min(3, len(prob_list))):
        curr_max = prob_list.index(max(prob_list))
        top_3.append(comp_list[curr_max])

        prob_list.pop(curr_max)
        comp_list.pop(curr_max)


        # prints top 3
    print('TOP:', top_3)
    #return top keywords into html
    # fill in keywords list so len == 3, to fill html with something
    if len(top_3) == 1:
        top_3.append(' ')
    if len(top_3) == 2:
        top_3.append(' ')
    return top_3

def query(keywords):
    if keywords == 'boot_load':
        return [' ', ' ', ' ']
    #checks wikipedia for an article about the keyword
    #uses beautiful soup to see if the following string is in the html
    # if it is in the html - it signifies that no page was found for
    #the corresponding keyword, in this case return "no articles" to html
    #in the case an article is found, summarize and inject in html
    urls = []
    sums = []
    count = 0
    for i in keywords:
        #if a null keyword is detected, was inserted because no more keywords to read
        if i == ' ':
            #fill in sums list so len == 3, to fill html with something
            if len(sums) == 1:
                sums.append(' ')
            if len(sums) == 2:
                sums.append(' ')
            return sums
        print('Keyword: '+i+'\n')
        urls.append('https://en.wikipedia.org/wiki/'+i)
        page = urlopen(urls[count])
        soup = BS(page, 'html.parser')
        string = soup.find('b')
        # check if a null wikipedia page
        if not string == "Wikipedia does not have an article with this exact name.":
            #return summary
            print(count)
            sums.append(summary(urls[count]))
        else:
            print(count)
            sums.append("No articles found")

        count +=1

    # fill in sums list so len == 3, to fill html with something
    if len(sums) == 1:
        sums.append(' ')
    if len(sums) == 2:
        sums.append(' ')
    return sums

def summary(article_url):
    url = article_url
    #url = "https://udarajay.com/applied-machine-learning-the-less-confusing-guide/"
    # url = "http://www.encyclopedia.com/plants-and-animals/plants/plants/cabbage"
    # url = "http://www.encyclopedia.com/medicine/diseases-and-conditions/pathology/accident"
    # url = "http://www.encyclopedia.com/earth-and-environment/atmosphere-and-weather/atmospheric-and-space-sciences-atmosphere/air"
    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))

    # create a list of reference sentences to calculate ROUGE_N scores
    ref_sentences = []
    trim_ref_sentences = []
    for paragraph in parser._article.main_text:
        for sections in paragraph:
            for sentences in sections:
                try:
                    if len(sentences) > 35:
                        # trim off super short - likely a few word sentences
                        ref_sentences.append(sentences)
                except TypeError:
                    # catch type errors caused by annotated text ie h1, b, etc
                    print("Calculating...")
                    continue
    trim_ref_sentences.extend(Sentence(s, Tokenizer(LANGUAGE)) for s in ref_sentences)

    # or for plain text files
    # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    # define summarizers for the summarizing methods being used
    summarizer_Lsa = Lsa(stemmer)
    summarizer_Lsa.stop_words = get_stop_words(LANGUAGE)
    summary_Lsa = summarizer_Lsa(parser.document, SENTENCES_COUNT)

    summarizer_LexRank = LexRank()
    summary_LexRank = summarizer_LexRank(parser.document, SENTENCES_COUNT)

    summarizer_Edmundson = Edmundson(stemmer)
    summarizer_Edmundson.null_words = get_stop_words(LANGUAGE)
    summarizer_Edmundson.bonus_words = parser.significant_words
    summarizer_Edmundson.stigma_words = parser.stigma_words
    summary_Edmundson = summarizer_Edmundson(parser.document, SENTENCES_COUNT)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print('\n')
    print('Lsa summary\n')
    # print summaries
    summary_Lsa_trim = []
    for sentence in summary_Lsa:
        # trim off super short - likely a few word sentences
        if len(sentence._text) > 20:
            print(sentence)
            summary_Lsa_trim.append(sentence)

    # calc rouge_n scores
    calc_value(summary_Lsa_trim, trim_ref_sentences)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print('\n')
    print('LexRank summary\n')
    summary_LexRank_trim = []
    for sentence in summary_LexRank:
        # trim off super short - likely a few word sentences
        if len(sentence._text) > 20:
            print(sentence)
            summary_LexRank_trim.append(sentence)

    # calc rouge_n scores
    calc_value(summary_LexRank_trim, trim_ref_sentences)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print('\n')
    print('Edmundson summary\n')
    summary_Edmundson_trim = []
    for sentence in summary_Edmundson:
        # trim off super short - likely a few word sentences
        if len(sentence._text) > 20:
            print(sentence)
            summary_Edmundson_trim.append(sentence)

    # calc rouge_n scores
    calc_value(summary_Edmundson_trim, trim_ref_sentences)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print('\n')
    # returns index of max 0=Ed, 1=Lsa, 2=Lex
    models = {0: "Edmundson Model", 1: "Lsa Model", 2: "LexRank Model"}
    best_summary = max_r_value(summary_Lsa_trim, summary_LexRank_trim, summary_Edmundson_trim, trim_ref_sentences)
    print(models.get(best_summary) + ' is the best model according to an average of the Rouge_3, 2 and 1 tests')

    #return the summary of the best model
    if(best_summary == 0):
        return summary_Edmundson_trim
    elif(best_summary == 1):
        return summary_Lsa_trim
    elif(best_summary==2):
        return summary_LexRank_trim

if __name__ == "__main__":
    app.run()
