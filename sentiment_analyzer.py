from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import json


file = open('yelp_academic_dataset_review.json', 'r')
sid = SentimentIntensityAnalyzer()

def trainClassifier():
    n_instances = 100
    subj_docs = [(sent, 'subj')
                 for sent in subjectivity.sents(categories='subj')[:n_instances]]
    obj_docs = [(sent, 'obj')
                for sent in subjectivity.sents(categories='obj')[:n_instances]]
    # Each document is represented by a tuple (sentence, label). The sentence
    # is tokenized, so it is represented by a list of strings:
    print(subj_docs[0])
    print()
    print(obj_docs[0])

    train_subj_docs = subj_docs[:80]
    test_subj_docs = subj_docs[80:100]
    train_obj_docs = obj_docs[:80]
    test_obj_docs = obj_docs[80:100]
    training_docs = train_subj_docs + train_obj_docs
    testing_docs = test_subj_docs + test_obj_docs
    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words(
        [mark_negation(doc) for doc in training_docs])

    unigram_feats = sentim_analyzer.unigram_word_feats(
        all_words_neg, min_freq=4)
    print(len(unigram_feats))
    sentim_analyzer.add_feat_extractor(
        extract_unigram_feats, unigrams=unigram_feats)

    training_set = sentim_analyzer.apply_features(training_docs)
    test_set = sentim_analyzer.apply_features(testing_docs)

    trainer = NaiveBayesClassifier.train
    classifier = sentim_analyzer.train(trainer, training_set)
    # Training classifier
    for key, value in sorted(sentim_analyzer.evaluate(test_set).items()):
        print('{0}: {1}'.format(key, value))

    print(sentim_analyzer.classify(getNextReview()))

def getNextReview():
    line = file.readline()
    review = json.loads(line)['text']
    return review

def getSentiment(text):
    print(text)
    ss = sid.polarity_scores(text)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
        print()


#trainClassifier()

getSentiment(getNextReview())