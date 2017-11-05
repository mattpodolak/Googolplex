import nltk
from nltk.corpus import stopwords
from itertools import product
from nltk.corpus import wordnet as wn
# import timeit

def main():
    # temp sentence
    temp_s = """why is the sky so blue and not red"""
    # timer
    #start = timeit.default_timer()
    # tokenize
    token_s = nltk.word_tokenize(temp_s)
    # remove stopwords
    stopWords = set(stopwords.words('english'))
    filtered_tokens = [word for word in token_s if word not in stopWords]
    filtered_tokens = []
    for word in token_s:
        if word not in stopWords:
            filtered_tokens.append(word)
    # ini print
    print('tokens:', token_s)
    print('filtered:', filtered_tokens)
    # loops through for comparisons
    filtered_iter = iter(filtered_tokens)
    for i in range(len(filtered_tokens)):
        total = 0
        elem = next(filtered_iter)
        allsyns1 = set(ss for word in token_s for ss in wn.synsets(word))
        allsyns2 = set(ss for word in elem for ss in wn.synsets(word))
        full_list = [(wn.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2)]
        score_list = iter(full_list)
        for j in range(len(full_list)):
            total += float(next(score_list)[0])
        print('word:', elem, 'weighted avg:', total/len(full_list))

    #stop = timeit.default_timer()
    #print('time:', stop - start, 's')

main()
