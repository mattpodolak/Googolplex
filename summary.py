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
from flask import Flask
from datetime import datetime
app = Flask(__name__)

@app.route('/')
def homepage():
    the_time = datetime.now().strftime("%A, %d %b %Y %l:%M %p")

    return """
    <h1>Hello heroku</h1>
    <p>It is currently {time}.</p>

    <img src="http://loremflickr.com/600/400">
    """.format(time=the_time)


LANGUAGE = "english"
SENTENCES_COUNT = 4

#calculate ROUGE_N values for n = 1, 2, 3
def calc_value(eval_sentences, ref_sentences):
    n_1 = rouge_1(eval_sentences, ref_sentences)
    n_2 = rouge_2(eval_sentences, ref_sentences)
    n_3 = rouge_n(eval_sentences, ref_sentences, 3)

    print('n1 '+ str(n_1) +'\n')
    print('n2 ' + str(n_2) + '\n')
    print('n3 ' + str(n_3) + '\n')
    print('avg ' + str((n_1+n_2+n_3)/3))

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
    print("Maximum average Rouge test value " + str(max))
    return maxIndex



if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
    url = "http://www.encyclopedia.com/plants-and-animals/plants/plants/potato"
    #url = "http://www.encyclopedia.com/plants-and-animals/plants/plants/cabbage"
    #url = "http://www.encyclopedia.com/medicine/diseases-and-conditions/pathology/accident"
    #url = "http://www.encyclopedia.com/earth-and-environment/atmosphere-and-weather/atmospheric-and-space-sciences-atmosphere/air"
    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))

    #create a list of reference sentences to calculate ROUGE_N scores
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
                    #catch type errors caused by annotated text ie h1, b, etc
                    print("typeError")
                    continue
    trim_ref_sentences.extend(Sentence(s, Tokenizer(LANGUAGE)) for s in ref_sentences)

    # or for plain text files
    # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    #define summarizers for the summarizing methods being used
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

    #store summaries in a text  file
    #list_of_sums = [summary_Lsa, summary_LexRank, summary_Edmundson]
    #f = open('summarized.txt', 'w')
    #for t in list_of_sums:
    #    line = ' '.join(str(x) for x in t)
    #    f.write(line + '\n')
    #f.close()

    #create new shorter summaries
    #parser = PlaintextParser.from_file("summarized.txt", Tokenizer(LANGUAGE))
    #SENTENCES_COUNT = 3

    #define summarizers for the summarizing methods being used
    #summarizer_Lsa = Lsa(stemmer)
    #summarizer_Lsa.stop_words = get_stop_words(LANGUAGE)
    #summary_Lsa = summarizer_Lsa(parser.document, SENTENCES_COUNT)

    #summarizer_LexRank = LexRank()
    #summary_LexRank = summarizer_LexRank(parser.document, SENTENCES_COUNT)

    #summarizer_Edmundson = Edmundson(stemmer)
    #summarizer_Edmundson.null_words = get_stop_words(LANGUAGE)
    #summarizer_Edmundson.bonus_words = parser.significant_words
    #summarizer_Edmundson.stigma_words = parser.stigma_words
    #summary_Edmundson = summarizer_Edmundson(parser.document, SENTENCES_COUNT)

    #print summaries
    summary_Lsa_trim = []
    for sentence in summary_Lsa:
        #trim off super short - likely a few word sentences
        if len(sentence._text)>20:
            print(sentence)
            summary_Lsa_trim.append(sentence)

    #calc rouge_n scores
    calc_value(summary_Lsa_trim, trim_ref_sentences)

    print('\n')
    summary_LexRank_trim = []
    for sentence in summary_LexRank:
        #trim off super short - likely a few word sentences
        if len(sentence._text)>20:
            print(sentence)
            summary_LexRank_trim.append(sentence)

    #calc rouge_n scores
    calc_value(summary_LexRank_trim, trim_ref_sentences)

    print('\n')
    summary_Edmundson_trim = []
    for sentence in summary_Edmundson:
        #trim off super short - likely a few word sentences
        if len(sentence._text)>20:
            print(sentence)
            summary_Edmundson_trim.append(sentence)

    #calc rouge_n scores
    calc_value(summary_Edmundson_trim, trim_ref_sentences)

    # returns index of max 0=Ed, 1=Lsa, 2=Lex
    models = {0: "Edmundson Model", 1: "Lsa Model", 2:"LexRank Model"}
    best_summary = max_r_value(summary_Lsa_trim, summary_LexRank_trim, summary_Edmundson_trim, trim_ref_sentences)
    print(models.get(best_summary) + ' is the best model according to an average of the Rouge_3, 2 and 1 tests')