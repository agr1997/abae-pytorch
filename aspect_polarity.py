import json
from collections import defaultdict
from monkeylearn import MonkeyLearn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import argparse
import pickle

class AspectStats():
    def __init__(self):
        self.count = 0.0
        self.compound = 0.0         # Compound sentiment extracted by vader

    def update(self, stats):
        self.count += 1.0
        self.compound += stats['compound']

    def get_stats(self):
        return {'count':int(self.count), 'compound': self.compound, 'polarity': self.compound/self.count}

def get_aspect_polarities(path, aspect_filename, output_filename):

    aspect_file = open(aspect_filename, 'r')

    aspects = {}
    aspect_stats = {}

    for line in aspect_file.readlines():
        aspects.update(json.loads(line))

    for aspect in aspects.keys():
        aspect_stats[aspect] = AspectStats()

    aspect_term_categories = defaultdict(list)

    for aspect_category in aspects.keys():
        for aspect_term in aspects[aspect_category]:
            aspect_term_categories[aspect_term].append(aspect_category)
    
    aspect_term_categories = dict(aspect_term_categories)

    fp = open(path+".txt", "r")
    sf = open(path+".sentiment.txt", "r")

    while True:
        line = fp.readline()
        if not line:
            break
        sentiment = json.loads(sf.readline())
        words = line.split()
        aspects_to_update = set()
        for word in words:
            if word in aspect_term_categories.keys():
                for item in aspect_term_categories[word]:
                    aspects_to_update.add(item)
        for aspect_to_update in aspects_to_update:
            aspect_stats[aspect_to_update].update(sentiment)


    aspect_file.close()
    fp.close()
    sf.close()

    with open(path+'.polarity.pkl', 'wb') as fp:
        pickle.dump(aspect_stats, fp)

    with open(output_filename, 'w') as fp:
        for asp in aspect_stats:
            fp.write(json.dumps({asp: aspect_stats[asp].get_stats()})+'\n')


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--aspect-file", "-af", dest="aspect_file", type=str, default="aspects.txt",
                        help="Path to the aspects file generated after running main.py.")

    parser.add_argument("--data-file", "-d", dest="path", type=str, default="vacuum.json",
                        help="Path to the original dataset.")

    parser.add_argument("--output-file", "-o", dest="output_file", type=str, default="aspect_sentiments.txt",
                        help="Path of the output file where the aspects and their sentiments will be stored.")

    args = parser.parse_args()
    path = args.path
    aspect_filename = args.aspect_file
    output_filename = args.output_file

    get_aspect_polarities(path, aspect_filename, output_filename)