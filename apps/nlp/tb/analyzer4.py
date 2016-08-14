import nltk
import nltk.data
from nltk.corpus import brown
import string
from docx import *
import csv
import re
import pandas as pd
import os
import pickle
import scipy
import numpy
import math
import unidecode
import json
from sklearn.metrics.pairwise import cosine_similarity
import operator
from django.conf import settings
from os.path import join


# features besides pos, exmaple words, unnecessary words
otherfeature = ['TotalWords', 'AvgWordLength', 'AvgSentLength', 'ARI', 'boldwords', 'italicwords', 'bulletstyle',
                'Capwords', 'Text', 'Author']

# examplewords = 'apps/nlp/tb/examplewords.txt'
# unnecessary = 'apps/nlp/tb/unnecessary.txt'
# author = 'apps/nlp/tb/author.txt'
# zscore_vector = 'apps/nlp/tb/zscore_vector.csv'
#
#
# examplewords_f = open(examplewords, 'r')
# examplewords_text = examplewords_f.read()
# examplewords_list = examplewords_text.split('\n')

# read the words in the word document
def getWord(document):
    doc = []
    for paragraph in document.paragraphs:
        # paragraph with less than 10 words will not be read
        if len(paragraph.text.split()) > 10:
            # unicode
            doc.append([unidecode.unidecode(paragraph.text)])
            # to check whether the paragraph is in bullet
            doc[len(doc) - 1].append(paragraph.style.name)
            # hold a space for bold and italic
            doc[len(doc) - 1].append(0)
            doc[len(doc) - 1].append(0)
            # to check how many bold or italic words in the paragraph
            for run in paragraph.runs:
                if run.bold:
                    doc[len(doc) - 1][2] += len(nltk.word_tokenize(run.text))
                if run.italic:
                    doc[len(doc) - 1][3] += len(nltk.word_tokenize(run.text))
    return doc


# transformed the document into tagged document
def tag(paragraph):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    # separate the paragraph into sentence list
    sent_list = sent_detector.tokenize(paragraph.strip())
    # POS tag each sentence [[(word,tag),(word,tag)...],#next sentence[(word,tag)...]..]
    tagged_paragraph_sent = [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in sent_list]
    tagged_paragraph = []
    # join the tagged sentences into a paragraph. Some features need sentence list while some needs paragraph.
    for i in tagged_paragraph_sent:
        tagged_paragraph += i
    return tagged_paragraph_sent, tagged_paragraph


# transform the tagged document into a ['word',['tag1','tag2'...]] style
def tag_word(text_tagged):
    word_tagged_list = []
    for pair in text_tagged:
        word_tagged_list.append([pair[0], [pair[1]]])
    return word_tagged_list


# Transform the text into paragraphs composed of only POS tags 'NN DT JJ...'
def pos(text_tagged):
    text_transf = ''
    # ignore the tagged puncations. Only have tags for words
    p = re.compile('\w')
    for pair in text_tagged:
        if pair[0] not in ['.', '?', '!', ':', ';', ',', '-', '(', ')', '[', ']', '{', '}', '<', '>', '``', '\'', '/']:
            text_transf += pair[1] + ' '
    return text_transf


# Transform the text into POS vector
def pos_uni_features(text, pos_tags):
    features = {}
    # create a freq dist of the POS tags in a paragraph NN:12, JJ:11...
    unifdist = nltk.FreqDist(nltk.word_tokenize(text))
    for tag in pos_tags:
        features['%s' % tag] = unifdist[tag]
    return features


# Transform the text given into bigram vector
# def pos_bi_features(text,bipos):
#	features = {}
#	bifdist = nltk.FreqDist(nltk.bigrams(nltk.word_tokenize(text)))
#	for word in bipos:
#		features['%s' % word] = bifdist[word.split()[0],word.split()[1]]
#	return features

# Transform the text into word vector
def word_features(text, wordlist):
    features = {}
    # create unigram and bigram freq dist of the words in a paragraph
    # some features have two words, so bigram freq dist
    unifdist = nltk.FreqDist(nltk.word_tokenize(text.lower()))
    bifdist = nltk.FreqDist(nltk.bigrams(nltk.word_tokenize(text.lower())))
    for word in wordlist:
        if len(nltk.word_tokenize(word)) == 1:
            features['%s' % word] = unifdist[word]
        if len(nltk.word_tokenize(word)) == 2:
            features['%s' % word] = bifdist[nltk.word_tokenize(word)[0], nltk.word_tokenize(word)[1]]
        # for features with 3 or more words, just search in the paragraph
        if len(nltk.word_tokenize(word)) > 2:
            features['%s' % word] = text.lower().count(word)
    return features


def bieber(text_tagged,word_tag_list):
	features = {}
	thatComplement = 0
	adjComplement = 0
	publicVerb = 0
	privateVerb = 0
	firstPersonPronouns = 0
	possibilityModal = 0
	downtoners = 0
	contractionwords = 0
	# bieber feature words are unigram
	for i in range(len(text_tagged)):
		if text_tagged[i][0].lower() in public_verbs:
			publicVerb+=1
			word_tag_list[i][1].append('publicVerb')
		if text_tagged[i][0].lower() in private_verbs:
			privateVerb+=1
			word_tag_list[i][1].append('privateVerb')
		if text_tagged[i][0].lower() in ['i','me','my','mine','myself','we','our','ours','us','ourselves']:
			firstPersonPronouns+=1
			word_tag_list[i][1].append('firstPersonPronouns')
		if text_tagged[i][0].lower() in ['may','might','could','must']:
			possibilityModal+=1
			word_tag_list[i][1].append('possibilityModal')
		if text_tagged[i][0].lower() in ['hardly','slightly','barely','just','somewhat']:
			downtoners+=1
			word_tag_list[i][1].append('downtoners')
		if text_tagged[i][0].lower() in ish_words:
			downtoners+=1
			word_tag_list[i][1].append('downtoners')
		if text_tagged[i][0].lower() in ['n\'t','\'s','\'re','\'ve','\'d','\'ll','\'m']:
			contractionwords+=1
			word_tag_list[i][1].append('contractionwords')
	# bieber feature words are bigram like 'have to'
	for i in range(len(text_tagged)-1):
		if text_tagged[i][1] in ['VB','VBD','VBG','VBN','VBP','VBZ'] and text_tagged[i+1][0].lower() == 'that' and text_tagged[i+1][1] == 'IN':
			thatComplement+=1
			# add word tag see tag_word()
			word_tag_list[i][1].append('thatComplement')
			word_tag_list[i+1][1].append('thatComplement')
		if text_tagged[i][0].lower() in ['have','has','had','having'] and text_tagged[i+1][0].lower() == 'to':
			possibilityModal+=1
			word_tag_list[i][1].append('possibilityModal')
			word_tag_list[i+1][1].append('possibilityModal')
		if text_tagged[i][0].lower() + ' ' + text_tagged[i+1][0].lower() in ['a bit','a little','only just''kind of','sort of','little bit','tiny bit']:
			downtoners+=1
			word_tag_list[i][1].append('downtoners')
			word_tag_list[i+1][1].append('downtoners')
		if text_tagged[i][1] in ['JJ','JJR','JJS'] and text_tagged[i+1][0].lower() == 'that' and text_tagged[i+1][1] == 'IN':
			adjComplement+=1
			word_tag_list[i][1].append('adjComplement')
			word_tag_list[i+1][1].append('adjComplement')
	# bieber feature words are trigram like "tell him that"
	for i in range(len(text_tagged)-2):
		if text_tagged[i][1] in ['VB','VBD','VBG','VBN','VBP','VBZ'] and text_tagged[i+1][1] in ['NN','NNS','NNP','RB','PRP','RP'] and text_tagged[i+2][0].lower() == 'that' and text_tagged[i+2][1] == 'IN':
			thatComplement+=1
			word_tag_list[i][1].append('thatComplement')
			word_tag_list[i+1][1].append('thatComplement')
			word_tag_list[i+2][1].append('thatComplement')
        # make into feature list
	features['thatComplement'] = thatComplement
	features['adjComplement'] = adjComplement
	features['publicVerb'] = publicVerb
	features['privateVerb'] = privateVerb
	features['firstPersonPronouns'] = firstPersonPronouns
	features['possibilityModal'] = possibilityModal
	features['downtoners'] = downtoners
	features['contractionwords'] = contractionwords
	return features


# detect agentless passive voice
def agentless(text_tagged_sent, word_tag_list):
    features = {}
    passiveVoice = 0
    for j in range(len(text_tagged_sent)):
        # join words into a sentence to check if there is word 'by'
        temp_sent = []
        for pair in text_tagged_sent[j]:
            temp_sent.append(pair[0].lower())
        # for sentence without 'by', check if there is passive voice
        if 'by' not in temp_sent:
            # check passive voice that 'be' + past participle
            for i in range(len(text_tagged_sent[j]) - 1):
                if text_tagged_sent[j][i][0].lower() in ['is', 'am', 'are', 'was', 'were', 'been', 'be', 'being'] and \
                                text_tagged_sent[j][i + 1][1] == 'VBN':
                    # print text_tagged_sent[j][i][0].lower(),text_tagged_sent[j][i+1][1]
                    passiveVoice += 1
                    for c in range(len(word_tag_list) - 1):
                        if word_tag_list[c][0] == text_tagged_sent[j][i][0] and text_tagged_sent[j][i + 1][1] in \
                                word_tag_list[c + 1][1]:
                            word_tag_list[c][1].append('agentlessPassive')
                            word_tag_list[c + 1][1].append('agentlessPassive')
            # check passive voice that 'be' + some one word + past participle like 'is not booked'
            for i in range(len(text_tagged_sent[j]) - 2):
                if text_tagged_sent[j][i][0].lower() in ['is', 'am', 'are', 'was', 'were', 'been', 'be', 'being'] and \
                                text_tagged_sent[j][i + 2][1] == 'VBN':
                    # print text_tagged_sent[j][i][0].lower(),text_tagged_sent[j][i+1][1]
                    passiveVoice += 1
                    for c in range(len(word_tag_list) - 1):
                        if word_tag_list[c][0] == text_tagged_sent[j][i][0] and text_tagged_sent[j][i + 2][1] in \
                                word_tag_list[c + 2][1]:
                            word_tag_list[c][1].append('agentlessPassive')
                            word_tag_list[c + 2][1].append('agentlessPassive')
    features['agentlessPassive'] = passiveVoice
    return features


# Compute average word length per paragraph
def avg_word_length(text):
    totalchar = 0
    # drop the punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # count how many words in a paragraph
    wordcount = len(text.split())
    # sum up number of characters in each word
    for word in text.split():
        totalchar += len(word)
    return float(totalchar) / float(wordcount)


# Ccompute average sentence length per paragraph
def ave_sent_length(text):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sent_list = sent_detector.tokenize(text.strip())
    sentcount = len(sent_list)
    sentwords = len(text.split())
    return float(sentwords) / float(sentcount)


# Compute ARI readibility score
def ari_score(text):
    return 4.71 * avg_word_length(text) + (0.5 * ave_sent_length(text)) - 21.43


# Output a CSV document of the count vector
def cvswrite_count(featuresets1, featuresets2, featuresets3, text, outpath):
    fp = open(outpath, "wb")
    writer = csv.writer(fp)
    # pattern = re.compile('.*[^a-z].*')
    # write the feature names
    featurenames = sorted(featuresets1[0].keys()) + sorted(featuresets2[0].keys()) + sorted(
        featuresets3[0].keys()) + otherfeature
    writer.writerow(featurenames)
    # write values for each paragraph as a row
    for i in range(0, len(featuresets1)):
        featureline = []
        # write values of features1
        for key in sorted(featuresets1[0].keys()):
            featureline.append(str(featuresets1[i][key]))
        # write values of features2
        for key in sorted(featuresets2[0].keys()):
            featureline.append(str(featuresets2[i][key]))
        # write values features3
        for key in sorted(featuresets3[0].keys()):
            featureline.append(str(featuresets3[i][key]))
        # write word count for each paragraph
        featureline.append(len(text[i][0].split()))
        # write average word length
        featureline.append(avg_word_length(text[i][0]))
        # write average sentence length
        featureline.append(ave_sent_length(text[i][0]))
        # write ARI
        featureline.append(ari_score(text[i][0]))
        # write style features
        featureline.append(text[i][2])
        featureline.append(text[i][3])
        featureline.append('List' in text[i][1])
        featureline.append(sum(1 for word in nltk.word_tokenize(text[i][0]) if word.isupper() and len(word) > 1) > 0)
        # write content of the paragraph
        featureline.append(text[i][0].encode('utf-8'))
        # write author
        featureline.append(0)
        writer.writerow(featureline)
    fp.close()

def analyze3(infile):

    import cProfile, pstats, StringIO
    pr = cProfile.Profile()
    pr.enable()
    # all the POS tags
    postags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
               'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
               'VBZ', 'WDT', 'WP', 'WP$', 'WRB']


    tags=postags + ['thatComplement','publicVerb', 'privateVerb', 'firstPersonPronouns', 'possibilityModal','downtoners','contractionwords','agentlessPassive']
    global public_verbs, private_verbs, ish_words


    # bipos_file = open('bipos.txt','r')
    # bipos_text = bipos_file.read()
    # bipos = bipos_text.split('\n')

    # read the word list
    # word_file = open('wordlist.txt','r')
    word_file = join(settings.BASE_DIR, 'apps/nlp/tb/wordlist.txt')
    word_filef = open(word_file, 'rU')
    word_text = word_filef.read()
    wordlist = word_text.splitlines()
    wordlist = list(set(wordlist))
    #print wordlist


    # read the privateverbs.txt, containing the private verbs
    prverb_file = join(settings.BASE_DIR, 'apps/nlp/tb/privateverbs.txt')
    prverb_filef = open(prverb_file, 'r')
    prverb_text = prverb_filef.read()
    private_verbs = prverb_text.splitlines()
    private_verbs = list(set(private_verbs))
    # print len(private_verbs)

    # read the publicverbs.txt, containing the public verbs
    puberb_file = join(settings.BASE_DIR, 'apps/nlp/tb/publicverbs.txt')
    puberb_filef = open(puberb_file, 'r')
    puberb_text = puberb_filef.read()
    public_verbs = puberb_text.splitlines()
    public_verbs = list(set(public_verbs))
    # print len(public_verbs)

    # read the ishwords.txt, containing the ish downtoners
    ish_file = join(settings.BASE_DIR, 'apps/nlp/tb/ishwords.txt')
    ish_filef = open(ish_file, 'r')
    ish_text = ish_filef.read()
    ish_words = ish_text.splitlines()
    ish_words = list(set(ish_words))

    # generate brown words freq dist

    brown_pos_fdist = json.load(open(join(settings.BASE_DIR, "apps/nlp/tb/brown_pos_fdist.txt")))
    brown_word_fdist = json.load(open(join(settings.BASE_DIR, "apps/nlp/tb/brown_word_fdist.txt")))

    # start processing each document

    outpath_count = join(settings.MEDIA_ROOT, 'count_vector.csv')

    # read the MS word document into a string list
    document = Document(infile)
    doc = getWord(document)

    # tag the paragraphs in a document
    text_tagged_sent = []
    text_tagged = []
    for line, style, bold, italic in doc:
        text_tagged_sent.append(tag(line)[0])
        text_tagged.append(tag(line)[1])
    # transform the tagged paragraph into ['word',['tag1','tag2'...]] style
    word_tagged_list = [tag_word(text) for text in text_tagged]

    # bieber features
    featuresets3 = []
    for k in range(len(text_tagged)):
        if len(text_tagged[k]) == len(word_tagged_list[k]):
            featuresets3.append(bieber(text_tagged[k], word_tagged_list[k]))
    # agentless passive voice feature
    featuresets3_sub = []
    for k in range(len(text_tagged)):
        featuresets3_sub.append(agentless(text_tagged_sent[k], word_tagged_list[k]))

    # join the two featuresets above
    if len(featuresets3) == len(featuresets3_sub):
        for i in range(0, len(featuresets3)):
            featuresets3[i].update(featuresets3_sub[i])

    # other features
    postext = [(pos(text)) for text in text_tagged]
    featuresets1 = [(pos_uni_features(line, postags)) for line in postext]
    featuresets2 = [(word_features(line, wordlist)) for line, style, bold, italic in doc]

    # if all the featuresets have equal length with length of the document, then write the csv file
    if len(featuresets1) == len(featuresets2) and len(featuresets1) == len(featuresets3):
        cvswrite_count(featuresets1, featuresets2, featuresets3, doc, outpath_count)

        # drop the feature that occurs only once in a document
    df_ratio = pd.read_csv(outpath_count, sep=',')
    cols = list(df_ratio)
    for col in cols[0:len(cols) - 2]:
        if df_ratio[col].sum() < 2:
            df_ratio = df_ratio.drop(col, 1)

    # weigh the vector based on freq in brown corpus and normalize based on paragraph length
    cols = list(df_ratio)
    for col in cols:
        if col in brown_pos_fdist.keys():
            df_ratio[col] = (df_ratio[col] * 100 / df_ratio['TotalWords']) / (math.log(brown_pos_fdist[col] + 1) + 0.5)
        if col in brown_word_fdist.keys():
            df_ratio[col] = (df_ratio[col] * 100 / df_ratio['TotalWords']) / (math.log(brown_word_fdist[col] + 1) + 0.5)
        if col == 'boldwords' or col == 'italicwords':
            df_ratio[col] = df_ratio[col] = df_ratio[col] * 100 / df_ratio['TotalWords']
    # start the index in csv from 1
    df_ratio.index += 1
    # df_ratio.to_csv(path + '/' + filename + '/' + 'ratio_vector.csv')

    # z score
    for col in cols[0:len(cols) - 2]:
        if df_ratio[col].std(ddof=0) != 0:
            df_ratio[col] = (df_ratio[col] - df_ratio[col].mean()) / df_ratio[col].std(ddof=0)
        else:
            df_ratio[col] = 0

    # df_ratio.to_csv(path + '/' + filename + '/' + 'zscore_vector.csv')

    # to compute difference between two adjacent paragraphs
    df_sub = df_ratio.ix[:, 0:len(df_ratio.columns) - 2]
    # get the feature about whether there is an author change between two adjacent paragraphs
    df_sub['AuthorChange'] = df_ratio['Author']
    # the compute the difference between two paragraphs on each feature
    vector_diff = df_sub.values
    df_new = pd.DataFrame(columns=list(df_sub))

    for i in range(1, len(vector_diff)):
        df_new.loc[i - 1] = vector_diff[i - 1] - vector_diff[i]

    # add paragraph index of two adjacent paragraphs
    df_new['PreviousParagraph'] = [i for i in range(1, len(df_ratio))]
    df_new['NextsParagraph'] = [i for i in range(2, len(df_ratio) + 1)]

    # compute the cosine similarity
    vector_feature = df_ratio.ix[:, 0:len(df_ratio.columns) - 2].values
    cos_matrix = cosine_similarity(vector_feature)
    cos = []
    for i in range(len(df_ratio) - 1):
        cos.append(cos_matrix[i][i + 1])

    df_new['CosineSimilarity'] = cos

    # find features whose absolute difference between 2 adjacent paragraphs greater than 3
    # and there is a cosine similarity smaller than 0
    bigdiff_feature = []
    bigdiff_dic = []
    feature_count = len(df_new.ix[:, 0:len(df_new.columns) - 4].values[0])
    # words = []
    print len(df_new), len(text_tagged)
    for i in range(0, len(df_new)):
        # print 'new'
        dic = {}
        bigdiffwords = []
        for j in range(feature_count):
            if abs(df_new.ix[:, 0:len(df_new.columns) - 4].values[i][j]) >= 3 and df_new['CosineSimilarity'][i] < 0:
                dic[list(df_new)[j]] = [df_new.ix[:, 0:len(df_new.columns) - 4].values[i][j]]
                # if list(df_new)[j] in postags:
                # for m in range(len(text_tagged[i])):
                # if text_tagged[i][m][1] == list(df_new)[j] and text_tagged[i][m][0] not in dic[list(df_new)[j]]:
                # dic[list(df_new)[j]].append(text_tagged[i][m][0])
                # for n in range(len(text_tagged[i+1])):
                # if text_tagged[i+1][n][1] == list(df_new)[j] and text_tagged[i+1][n][0] not in dic[list(df_new)[j]]:
                # dic[list(df_new)[j]].append(text_tagged[i+1][n][0])
                # else:
                # if list(df_new)[j] not in dic[list(df_new)[j]]:
                # dic[list(df_new)[j]].append(list(df_new)[j])
        # words.append(dic)
        # sort the big features based on the magnitude of abs difference
        templist = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
        for pair in templist:
            bigdiffwords.append(pair[0])
            # print pair[0],pair[1]
        bigdiff_feature.append(bigdiffwords)
        bigdiff_dic.append(dic)
    # print the big diff features into a feature
    df_new['BigDiffFeatures'] = ['|'.join(diff) for diff in bigdiff_feature]

    # para = 0
    # for word_dict in word_tagged_list:
    #     dkeys = word_dict.keys()
    #     # print "paragraph %s" % para
    #     # print doc[para]
    #     for dkey in dkeys:
    #         for hiword in word_dict[dkey]:
    #             if hiword == '.':
    #                 continue
    #             classname = dkey + '_' + '%s' % para
    #             doc1[para] = re.sub(r'\b%s\b' % hiword, '<span class ="%s">%s</span>' % (classname, hiword), doc1[para],
    #                                 flags=re.UNICODE)
    #             a = para + 1
    #             classname = dkey + '_' + '%s' % a
    #             doc1[para + 1] = re.sub(r'\b%s\b' % hiword, '<span class ="%s">%s</span>' % (classname, hiword),
    #                                     doc1[para + 1], flags=re.UNICODE)
    #     para = para + 1


    #find the word in the bigdiff_feature and add html in it
    para =0
    for i in range(len(bigdiff_feature)):
        if not bigdiff_feature[i]:
            para = para +1
        else:
            for word in bigdiff_feature[i]:
                if word in tags:
                    for pair in word_tagged_list[i]:
                        if word in pair[1]:
                            classname = word + '_' + '%s' % para
                            if len(pair[0]) < 20:
                                pair[0] = '<span class ="%s">%s</span>' % (classname, pair[0])
                            else:
                                s = pair[0].split('>')
                                w = s[1].split('<')
                                preclass = s[0].split('"')
                                pair[0] = '<span class ="%s %s">%s</span>' % (preclass[1], classname, w[0])
                    for pair in word_tagged_list[i + 1]:
                        if word in pair[1]:
                            a = para + 1
                            classname = word + '_' + '%s' % a
                            if len(pair[0]) < 20:
                                pair[0] = '<span class ="%s">%s</span>' % (classname, pair[0])
                            else:
                                s = pair[0].split('>')
                                w = s[1].split('<')
                                preclass = s[0].split('"')
                                pair[0] = '<span class ="%s %s">%s</span>' % (preclass[1], classname, w[0])
                else:
                    for pair in word_tagged_list[i]:
                        if word == pair[0].lower():
                            classname = word + '_' + '%s' % para
                            if len(pair[0]) < 20:
                                pair[0] = '<span class ="%s">%s</span>' % (classname, pair[0])
                            else:
                                s = pair[0].split('>')
                                w = s[1].split('<')
                                preclass = s[0].split('"')
                                pair[0] = '<span class ="%s %s">%s</span>' % (preclass[1], classname, w[0])
                    for pair in  word_tagged_list[i + 1]:
                        if word == pair[0].lower():
                            a = para + 1
                            classname = word + '_' + '%s' % a
                            if len(pair[0]) < 20:
                                pair[0] = '<span class ="%s">%s</span>' % (classname, pair[0])
                            else:
                                s = pair[0].split('>')
                                w = s[1].split('<')
                                preclass = s[0].split('"')
                                pair[0] = '<span class ="%s %s">%s</span>' % (preclass[1], classname, w[0])
            para = para + 1

    #join the word into paragraph
    doc1 = []
    punctuations = '''''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for i in range(len(word_tagged_list)):
        text = ''
        for pair in word_tagged_list[i]:
            match = re.search('(\')(\w)+', pair[0])
            if pair[0] not in punctuations:
                if match:
                    text += pair[0]
                    print 'match', match
                    print 'pair[0]', pair[0]
                else:
                    if pair[0] == '``':
                        pair[0] = '\'\''
                    text += " " + pair[0]
            else:
                text += pair[0]
        doc1.append(text)
    #print 'doc1', doc1


    #create alert boxes
    alerts = []
    para = 0
    feature_dictionary = {'CC':'Coordinatiing conjunction', 'CD':'Cardinal number', 'DT':'Deteminer', 'EX':'Existential there', 'FW':'Foreign word', 'IN':'Preposition or Subordinating conjunction',
                          'JJ':'Adjective', 'JJR':'Comparative Adjective', 'JJS':'Superlative Adjective', 'LS':'List item marker', 'MD':'Modal', 'NN':'Noun, sigualr or mass', 'NNS':'Plural Noun',
                          'NNP':'Proper noun, singular', 'NNPS':'Proper noun, plural', 'PDT':'Predeterminer', 'POS':'Possessive ending', 'PRP':'Personal pronoun', 'PRP$':'Possessive Pronoun', 'RB':'Adverb',
                          'RBR':'Comparative Adverb', 'RBS':'Superlative Adverb', 'RP':'Particle', 'SYM':'Symbol', 'TO':'to', 'UH':'Interjection', 'VB':'Verb', 'VBD':'Past Tense Verb', 'VBG':'Verb, Verb, gerund or present participle',
                          'VBN':'Verb, past participle', 'VBP':'Verb, non-3rd person singular present', 'VBZ':'Verb, 3rd person singular present', 'WDT':'Wh-determiner', 'WP':'Wh-pronoun', 'WP$':'Possessive wh-pronoun', 'WRB':'Wh-adverb'
                        }
    for word_dict in bigdiff_dic:
        s = []
        strm =[]
        strl =[]
        content = ''
        dkeys = word_dict.keys()
        for dkey in dkeys:
            hiword = word_dict[dkey]
            if dkey == 'TotalWords':
                if hiword[0]>0:
                    str = 'The paragraph above has many more words than the one below. '
                else:
                    str = 'The paragraph above has many fewer words than the one below. '
                s.append(str)
            elif dkey == 'AvgSentLength':
                if hiword[0] > 0:
                    str = 'The paragraph above has longer sentences than the one below. '
                else:
                    str = 'The paragraph above has shorter sentences than the one below. '
                s.append(str)
            elif dkey == 'AvgWordLength':
                if hiword[0] > 0:
                    str = 'The paragraph above has longer words than the one below. '
                else:
                    str = 'The paragraph above has shorter words than the one below. '
                s.append(str)
            else:
                classname = dkey + '_' + '%s' % para
                featureclass = classname + 'f'
                if dkey in feature_dictionary:
                    keycontent= feature_dictionary[dkey]
                    if hiword[0] > 0:
                        strmore = '<span onMouseOver="setfeaturecolor(\'%s\')" onmouseout="onMouseOut(\'%s\')" id ="%s"><u style="color:#005ce6">%s</u></span> ' % (
                            classname, classname, featureclass, keycontent)
                        strm.append(strmore)
                    else:
                        strless = '<span onMouseOver="setfeaturecolor(\'%s\')" onmouseout="onMouseOut(\'%s\')" id ="%s"><u style="color:#005ce6">%s</u></span> ' % (
                            classname, classname, featureclass, keycontent)
                        strl.append(strless)
                else:
                    if hiword[0] > 0:
                        strmore = '<span onMouseOver="setfeaturecolor(\'%s\')" onmouseout="onMouseOut(\'%s\')" id ="%s"><u style="color:#005ce6">%s</u></span> ' % (
                            classname, classname, featureclass, dkey)
                        strm.append(strmore)
                    else:
                        strless = '<span onMouseOver="setfeaturecolor(\'%s\')" onmouseout="onMouseOut(\'%s\')" id ="%s"><u style="color:#005ce6">%s</u></span> ' % (
                            classname, classname, featureclass, dkey)
                        strl.append(strless)
        strmore = ', '.join(strm)
        strless = ', '.join(strl)
        str = ''
        if strmore:
            str = 'The paragraph above has many more %s than the one below.' % strmore
        if strless:
            str += ' The paragraph above has many less %s than the one below. ' % strless
        s.append(str)
        s = ' '.join(s)
        if s:
            content = '<div class="alert"> %s </div>' % s
        alerts.append(content)
        para += 1


    result = []
    for i in range(len(bigdiff_feature)):
        result.append(doc1[i])
        if (bigdiff_feature[i] != {}):
            result.append(alerts[i])
    result.append(doc1[i + 1])

    pr.disable()
    # s = StringIO.StringIO()
    opath = join(settings.MEDIA_ROOT, 'a.dmp')
    s = open(opath, 'w')
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    s.close()
    # print s.getvalue()

    return result

    # result_file.close()
