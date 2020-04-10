import json
from collections import defaultdict
from monkeylearn import MonkeyLearn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import argparse


# This script is meant to be run after generation of aspects. [Let us say that we have aspect1 ... aspectn]

def extract_sentiment(path):


    ml = MonkeyLearn('f3f1dcd812efb690eb1a7e4095616bfb288fa744')

    model_id = 'cl_pi3C7JiL'
    #result = ml.classifiers.classify(model_id, data)

    #print(result.body)

    sentiment_analyzer = SentimentIntensityAnalyzer()

    with open(path+".sentences.txt") as fp:
        fs = open(path+".sentiment.txt", "w+")
        while True:
            line = fp.readline()
            if not line:
                break
            fs.write(json.dumps(sentiment_analyzer.polarity_scores(line))+"\n")
        fs.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "reviews_Electronics_5.json"

    extract_sentiment(path)