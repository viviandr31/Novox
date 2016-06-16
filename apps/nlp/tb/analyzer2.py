import nltk
# import nltk.data
from docx import *
import csv
import re
import string
import scipy
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import operator

# path of the MS word document
# infile = 'Test3-SEP-article.docx'

# path of txt document which contains functional words and punctuations we
# want, can be modified
# unigram = 'apps/nlp/tb/unigram.txt'

# path of txt document which contains bigram words we want, can be modified
# bigram = 'apps/nlp/tb/bigram.txt'

# path of txt document which contains POS tags we want, can be modified
# posfile = 'apps/nlp/tb/pos.txt'

# path of the output CSV document
# outpath_count = 'media/count_vector.csv'
# outpath_norm = 'media/norm_vector.csv'

examplewords = 'apps/nlp/tb/examplewords.txt'
unnecessary = 'apps/nlp/tb/unnecessary.txt'
author = 'apps/nlp/tb/author.txt'
zscore_vector = 'apps/nlp/tb/zscore_vector.csv'


examplewords_f = open(examplewords, 'r')
examplewords_text = examplewords_f.read()
examplewords_list = examplewords_text.split('\n')

unnecessary_f = open(unnecessary, 'r')
unnecessary_text = unnecessary_f.read()
unnecessary_list = unnecessary_text.split('\n')


# read the words in the word document


def getWord(document):
    doc = []
    for paragraph in document.paragraphs:
        if len(paragraph.text.split()) > 10:
            doc.append(paragraph.text)
    return doc


# tag pos, bi-pos, example word, unnecessary word in the paragraph
def tag(text):
    # print text
    # lower, tokenize, and POS the text; may need some recoding of special words
    # pattern = re.compile('.*[^a-z0-9].*')
    # print text
    text_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    text_transf = []
    for pair in text_tagged:
        if pair[0] == 'you' or pair[0] == 'You':
            text_transf.append((pair[0], 'you'))
        if pair[0] == 'i' or pair[0] == 'I':
            text_transf.append((pair[0], 'i'))
        if pair[1] == 'CC' or pair[1] == 'DT' or pair[1] == 'IN' or pair[1] == 'PRP' or pair[1] == 'PRP$' or pair[
            1] == 'RP' or pair[1] == 'UH':
            text_transf.append((pair[0], 'FunctionWords'))
        if pair[1] == 'JJ' or pair[1] == 'JJR' or pair[1] == 'JJS':
            text_transf.append((pair[0], 'Adjectives'))
        if pair[1] == 'NN' or pair[1] == 'NNS' or pair[1] == 'NNP' or pair[1] == 'NNPS':
            text_transf.append((pair[0], 'Nouns'))
        if pair[1] == 'RB' or pair[1] == 'RBR' or pair[1] == 'RBS':
            text_transf.append((pair[0], 'Adverbs'))
        if pair[1] == 'VB' or pair[1] == 'VBG' or pair[1] == 'VBP' or pair[1] == 'VBZ':
            text_transf.append((pair[0], 'Verbs'))
        if pair[1] == 'VBD':
            text_transf.append((pair[0], 'PastVerbs'))
        if pair[1] == 'VBN':
            text_transf.append((pair[0], 'ppVerbs'))
        if pair[1] == 'WDT' or pair[1] == 'WP' or pair[1] == 'WP$' or pair[1] == 'WRB':
            text_transf.append((pair[0], 'whWord'))
        if pair[1] == 'EX':
            text_transf.append((pair[0], 'Existential'))
        if pair[1] == 'FW':
            text_transf.append((pair[0], 'ForeignWord'))
        if pair[1] == 'LS':
            text_transf.append((pair[0], 'ListMarker'))
        if pair[1] == 'MD':
            text_transf.append((pair[0], 'Modal'))
        if pair[1] == 'PDT':
            text_transf.append((pair[0], 'Predeterminer'))
        if pair[1] == 'POS':
            text_transf.append((pair[0], 'Possessive'))
        if pair[1] == 'SYM':
            text_transf.append((pair[0], 'Symbol'))
        if pair[1] == 'TO':
            text_transf.append((pair[0], 'To'))
        if pair[1] == 'CD':
            text_transf.append((pair[0], 'CardinalNumber'))
        if pair[0] in set(string.punctuation):
            text_transf.append((pair[0], pair[0]))
        if pair[0] == '``' or pair[0] == "''":
            text_transf.append((pair[0], '"'))
    text_transf = tagwords(text, examplewords_list, 'example_words', text_transf)
    text_transf = tagwords(text, unnecessary_list, 'unnecessary_words', text_transf)
    return text_transf


def tagwords(text, features, featurename, textlist):
    # taglist = []
    newtext = ' ' + re.sub(r'[,.;:?!()""\[\]{}<>]', ' ', text).lower()
    #print 'newtext', newtext
    for word in features:
        if ' ' + word + ' ' in newtext:
            textlist.append((word, featurename))
    #print 'tagwords', textlist
    return textlist


def analyze2(infile):
    print 'processing %s' % infile

    # read the vector into dataframe
    df = pd.read_csv(zscore_vector, sep=',')

    # to compute difference between two adjacent paragraphs
    df_sub = df.ix[:, 1:len(df.columns) - 2]
    df_sub['AuthorChange'] = df['Author']
    vector_diff = df_sub.values
    df_new = pd.DataFrame(columns=list(df_sub))

    for i in range(1, len(vector_diff)):
        df_new.loc[i - 1] = vector_diff[i - 1] - vector_diff[i]

    # add paragraph index of two adjacent paragraphs
    df_new['PreviousParagraph'] = [i for i in range(1, len(df))]
    df_new['NextsParagraph'] = [i for i in range(2, len(df) + 1)]

    # compute the cosine similarity
    vector_feature = df.ix[:, 1:len(df.columns) - 2].values
    cos_matrix = cosine_similarity(vector_feature)
    cos = []
    for i in range(len(df) - 1):
        cos.append(cos_matrix[i][i + 1])

    df_new['CosineSimilarity'] = cos

    # write features whose absolute difference between 2 adjacent paragraphs greater than 2
    bigdiff_feature = []
    feature_count = len(df_new.ix[:, 0:len(df_new.columns) - 4].values[0])
    for i in range(len(df_new)):
        # print 'new'
        wordlist = {}
        words = []
        for j in range(feature_count):
            if abs(df_new.ix[:, 0:len(df_new.columns) - 4].values[i][j]) >= 3:
                wordlist[list(df_new)[j]] = abs(df_new.ix[:, 0:len(df_new.columns) - 4].values[i][j])
        templist = sorted(wordlist.items(), key=operator.itemgetter(1), reverse=True)
        for pair in templist:
            words.append(pair[0])
            # print pair[0],pair[1]
        bigdiff_feature.append(words)

    df_new['BigDiffFeatures'] = ['|'.join(diff) for diff in bigdiff_feature]
    print 'big difference feature ', bigdiff_feature

    # format the html for showing the big difference feature
    show_feature = []
    para = 0
    for features in bigdiff_feature:
        s = []
        for feature in features:
            classname = feature + '%s' % para
            str = '<span onMouseOver="setfeaturecolor(\'%s\')" onmouseout="onMouseOut(\'%s\')" class ="%s">%s</span>' % (classname, classname, classname, feature)
            s.append(str)
        s = ', '.join(s)
        show_feature.append(s)
        para +=1

    # output the df_new as a csv file

    # count the incorrectly predicted case
    incor_count = 0
    for i in range(len(df_new)):
        if df_new['AuthorChange'].values[i] != 0 and df_new['CosineSimilarity'].values[i] >= 0:
            print df_new['AuthorChange'].values[i], df_new['CosineSimilarity'].values[i]
            incor_count += 1
        if df_new['AuthorChange'].values[i] == 0 and df_new['CosineSimilarity'].values[i] < 0:
            print df_new['AuthorChange'].values[i], df_new['CosineSimilarity'].values[i]
            incor_count += 1

    print 'incorrect: ', incor_count

    accuracy = 1 - (float(incor_count) / len(df_new))

    print 'accuracy: ', accuracy

    #
    # read the MS word document into a string list
    document = Document(infile)
    doc = getWord(document)
    newdoc = []
    for i in range(len(doc) - 1):
        newdoc.append(doc[i] + ' ' + doc[i + 1])
    # print newdoc

    tagged_text = [tag(line) for line in newdoc]
    # print tagged_text

    words = []
    for m in range(len(bigdiff_feature)):
        dic = {}
        for tags in bigdiff_feature[m]:
            dic[tags] = []
            if len(tags.split()) == 2:
                for i in range(len(tagged_text[m])):
                    if tagged_text[m][i][1] == tags.split()[0] and tagged_text[m][i + 1][1] == tags.split()[1]:
                        dic[tags].append(tagged_text[m][i][0] + ' ' + tagged_text[m][i + 1][0])
            if len(tags.split()) == 1:
                for a, b in tagged_text[m]:
                    if b == tags and a not in dic[tags]:
                        dic[tags].append(a)
        words.append(dic)

    para = 0
    for word_dict in words:
        dkeys = word_dict.keys()
        #print "paragraph %s" % para
        # print doc[para]
        for dkey in dkeys:
            for hiword in word_dict[dkey]:
                if hiword == '.':
                    continue
                classname = dkey + '%s' % para
                doc[para] = re.sub(r'\b%s\b' % hiword, '<span class ="%s" ><b>%s</b></span>' % (classname, hiword), doc[para], flags=re.UNICODE)
                doc[para+1] = re.sub(r'\b%s\b' % hiword, '<span class ="%s"><b>%s</b></span>' % (classname, hiword), doc[para+1], flags=re.UNICODE)
        para = para + 1

    alerts = []
    for i in range(len(bigdiff_feature)):
        content = '<div class="alert"> <span class="closebtn" onclick="this.parentElement.style.display=\'none\';">&times;</span>Paragraph above is very different on %s from the paragraph below .</div>' % show_feature[i]
        alerts.append(content)

    result =[]
    for i in range(len(words)):
        result.append(doc[i])
        if (words[i]!={}):
            result.append(alerts[i])
    result.append(doc[i+1])

    print words

    return result
