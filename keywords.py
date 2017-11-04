import nltk
from nltk.stem import*
from nltk.corpus import stopwords
from itertools import product
from nltk.corpus import wordnet as wn
from collections import Counter

def similarity_test(sentence_raw, token_sample):
    # tokenize+tag raw sentence and token
    sentence = nltk.pos_tag((nltk.word_tokenize(sentence_raw)))
    token = nltk.pos_tag(nltk.word_tokenize(token_sample))

def main():

    # temp sentence
    temp_s = """sometimes rain is good but i like summer better decent"""

    # tokenize
    token_s = nltk.word_tokenize(temp_s)

    # tagged
    #tagged_s = nltk.pos_tag(token_s)
    # lemmatizer
    #lemmatizer = WordNetLemmatizer()
    #tokens = lemmatizer.lemmatize(token_s)

    # remove stopwords
    stopWords = set(stopwords.words('english'))
    filtered_tokens = [word for word in token_s if word not in stopWords]
    filtered_tokens = []
    for word in token_s:
        if word not in stopWords:
            filtered_tokens.append(word)

    print('tokens:', token_s)
    print('filtered:', filtered_tokens)

    #print('tags:', tagged_s)
    #print('lemma:', tokens)

    # probablity tests
    #print('test:', nltk.pos_tag(nltk.word_tokenize('poop')))

    #
    prob_list = []
    comp_list = []
    top3 = []
    for words, keys in product(token_s, filtered_tokens):
        synwords = wn.synsets(word)
        synkeys = wn.synsets(keys)
        for sensew, sensek in product(synwords, synkeys):
            siml = wn.wup_similarity(sensew, sensek)
            if siml != None and siml not in prob_list:
                prob_list.append(siml)
                comp_list.append(tuple((words, keys, sensek, sensew)))
                #comp_list.append(tuple((synwords, synkeys)))
                #comp_list.append(tuple((sensew, sensek)))
                #similarity_list.append((siml, tuple((synwords, synkeys))))

    # pop max prob then pop index of max 3 times
    for i in range(min(3, len(filtered_tokens))):
        curr_max = prob_list.index(max(prob_list))
        top3.append(tuple((prob_list[curr_max], comp_list[curr_max])))
        prob_list.pop(curr_max)
        comp_list.pop(curr_max)

    for item in top3:
        print(item)


main()
