import nltk
import nltk.data
from docx import *
import csv
import re


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

# read the words in the word document


def getWord(document):
    doc = []
    for paragraph in document.paragraphs:
        if not paragraph.text == '':
            doc.append(paragraph.text)
    return doc

# Transform the text into tags like "NN VBN JJ RB"


def pos(text):
    # lower, tokenize, and POS the text; may need some recoding of special words
    # pattern = re.compile('.*[^a-z0-9].*')
    text_tagged = nltk.pos_tag(nltk.word_tokenize(text.lower()))
    text_transf = ''
    for pair in text_tagged:
        text_transf = text_transf + pair[1] + ' '
    return text_transf

# Transform the text given into unigram vector


def unigram_features(text, word_features):
    features = {}
    unifdist = nltk.FreqDist(nltk.word_tokenize(text.lower()))
    for word in word_features:
        features['%s' % word] = unifdist[word]
    return features

# Calculate bigram frequency distribution


def bigramDist(text):
    # create a new empty frequency distribution
    biDist = nltk.FreqDist()
    # loop through the words in order, looking at all pairs of words
    temp = nltk.word_tokenize(text.lower())
    for i in range(1, len(temp)):
        biword = temp[i - 1] + ' ' + temp[i]
        biDist.inc(biword)
    return biDist

# Transform the text given into bigram vector


def bigram_features(text, word_features):
    features = {}
    bifdist = nltk.FreqDist(nltk.bigrams(nltk.word_tokenize(text.lower())))
    for word in word_features:
        features['%s' % word] = bifdist[
            nltk.word_tokenize(word)[0], nltk.word_tokenize(word)[1]]
    return features

# compute average word length per paragraph


def avg_word_length(text):
    totalchar = 0
    wordcount = len(text.split())
    for word in text.split():
        totalchar += len(word)
    return float(totalchar) / float(wordcount)

# compute average sentence length per paragraph


def ave_sent_length(text):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sent_list = sent_detector.tokenize(text.strip())
    sentcount = len(sent_list)
    sentwords = len(text.split())
    return float(sentwords) / float(sentcount)

# Output a CSV document of the count vector


def cvswrite_count(featuresets1, featuresets2, featuresets3, text, outpath):
    fp = open(outpath, "wb")
    writer = csv.writer(fp)
    pattern = re.compile('.*[^a-z].*')
    # write the feature names
    featurenames = sorted(featuresets1[0].keys()) + sorted(featuresets2[0].keys()) + sorted(
        featuresets3[0].keys()) + ['Total Words', 'Avg Word Length', 'Avg Sent Length', 'Text']
    writer.writerow(featurenames)
    # write values for each paragraph as a row
    if len(featuresets1) == len(featuresets2) and len(featuresets2) == len(text):
        for i in range(0, len(featuresets1)):
            featureline = []
            # write values of functional words and punctuations
            for key in sorted(featuresets1[0].keys()):
                featureline.append(str(featuresets1[i][key]))
            # write values of POS tags
            for key in sorted(featuresets2[0].keys()):
                featureline.append(str(featuresets2[i][key]))
            # write values of bigram words
            for key in sorted(featuresets3[0].keys()):
                featureline.append(str(featuresets3[i][key]))
            # write word count for each paragraph
            featureline.append(len(text[i].split()))
            # write average word length
            featureline.append(avg_word_length(text[i]))
            # write average sentence length
            featureline.append(ave_sent_length(text[i]))
            # write content of the paragraph
            featureline.append(text[i].encode('utf-8'))
            writer.writerow(featureline)
    fp.close()

# Output the normalized vector: word count * 1000 / total words


def cvswrite_norm(featuresets1, featuresets2, featuresets3, text, outpath):
    fp = open(outpath, "wb")
    writer = csv.writer(fp)
    pattern = re.compile('.*[^a-z].*')
    # write the feature names
    featurenames = sorted(featuresets1[0].keys()) + sorted(featuresets2[0].keys()) + sorted(
        featuresets3[0].keys()) + ['Total Words', 'Avg Word Length', 'Avg Sent Length', 'Text']
    writer.writerow(featurenames)
    # write values for each paragraph as a row
    if len(featuresets1) == len(featuresets2) and len(featuresets2) == len(text):
        for i in range(0, len(featuresets1)):
            featureline = []
            # write values of functional words and punctuations
            for key in sorted(featuresets1[0].keys()):
                featureline.append(
                    str(featuresets1[i][key] * 1000 / len(text[i].split())))
            # write values of POS tags
            for key in sorted(featuresets2[0].keys()):
                featureline.append(
                    str(featuresets2[i][key] * 1000 / len(text[i].split())))
            # write values of bigram words
            for key in sorted(featuresets3[0].keys()):
                featureline.append(
                    str(featuresets3[i][key] * 1000 / len(text[i].split())))
            # write word count for each paragraph
            featureline.append(len(text[i].split()))
            # write average word length
            featureline.append(avg_word_length(text[i]))
            # write average sentence length
            featureline.append(ave_sent_length(text[i]))
            # write content of the paragraph
            featureline.append(text[i].encode('utf-8'))
            writer.writerow(featureline)
    fp.close()

def analyze(infile, unigram_text, bigram_text, pos_text, outpath_norm, outpath_count):
    print 'processing %s' % infile
    # read the functional words and punctuations we want
    # unigram_f = open(unigram, 'r')
    # unigram_text = unigram_f.read()
    unigram_words = nltk.wordpunct_tokenize(unigram_text)
    # read the bigram features we want
    # bigram_f = open(bigram, 'r')
    # bigram_text = bigram_f.read()
    bigram_words = bigram_text.split('\n')
    # read the POS tags we want
    # pos_f = open(posfile, 'r')
    # pos_text = pos_f.read()
    postags = pos_text.split()
    # read the MS word document into a string list
    document = Document(infile)
    doc = getWord(document)
    # Transform the text into tags
    postext = [(pos(line)) for line in doc]
    # Transform into 3 vectors, one for functional words and punctuations, one
    # for POS tags, one for bigram features
    featuresets1 = [(unigram_features(line, unigram_words)) for line in doc]
    featuresets2 = [(unigram_features(line, postags)) for line in postext]
    featuresets3 = [(bigram_features(line, bigram_words)) for line in doc]
    # Output a CSV document of the vector
    cvswrite_count(
        featuresets1, featuresets2, featuresets3, doc, outpath_count)
    cvswrite_norm(featuresets1, featuresets2, featuresets3, doc, outpath_norm)
