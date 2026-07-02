import os.path
import re
from os import makedirs,remove
from re import compile,findall,split
from .config import OptionConf
import random
from itertools import product


NEGATIVE_FILE_CANDIDATES = ('ZERO_indices.txt', 'zero1.txt', 'zero.txt')


def resolve_optional_dataset_file(dataset_dir, candidates):
    for filename in candidates:
        path = os.path.join(dataset_dir, filename)
        if os.path.exists(path):
            return path
    return None


def sample_negative_records(positive_records, count):
    left_ids = sorted({record[0] for record in positive_records})
    right_ids = sorted({record[1] for record in positive_records})
    positives = {(record[0], record[1]) for record in positive_records if float(record[2]) > 0}
    available = len(left_ids) * len(right_ids) - len(positives)
    if available < count:
        print('Cannot generate %d negative samples from %d candidate pairs.' % (count, available))
        exit(-1)

    negatives = set()
    while len(negatives) < count:
        pair = (random.choice(left_ids), random.choice(right_ids))
        if pair in positives or pair in negatives:
            continue
        negatives.add(pair)
    return [[left, right, 0.0] for left, right in negatives]


class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def writeFile(dir,file,content,op = 'w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir+file,op) as f:
            f.writelines(content)

    @staticmethod
    def deleteFile(filePath):
        if os.path.exists(filePath):
            remove(filePath)


    @staticmethod
    def loadDataSet(conf, file, bTest=False):
        trainingData = []
        testData = []
        ratingConfig = OptionConf(conf['ratings.setup'])
        if not bTest:
            print('loading training data...')
        else:
            print('loading test data...')
        with open(file) as f:
            ratings = f.readlines()
        # ignore the headline
        if ratingConfig.contains('-header'):
            ratings = ratings[1:]
        # order of the columns
        order = ratingConfig['-columns'].strip().split()
        delim = ' |,|\t'
        if ratingConfig.contains('-delim'):
            delim=ratingConfig['-delim']
        for lineNo, line in enumerate(ratings):
            hda = split(delim,line.strip())
            if not bTest and len(order) < 2:
                print('The rating file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            try:
                herbId = hda[int(order[0])]
                diseaseId = hda[int(order[1])]
                if len(order)<3:
                    rating = 1 #default value
                else:
                    rating  = hda[int(order[2])]
            except ValueError:
                print('Error! Have you added the option -header to the rating.setup?')
                exit(-1)
            if bTest:
                testData.append([herbId, diseaseId, float(rating)])
            else:

                trainingData.append([herbId, diseaseId, float(rating)])

        dataset_dir = os.path.dirname(os.path.abspath(file))
        negative_count = len(testData if bTest else trainingData)
        zero_path = resolve_optional_dataset_file(dataset_dir, NEGATIVE_FILE_CANDIDATES)
        if zero_path:
            with open(zero_path) as f:
                ratings = f.readlines()
            # ignore the headline
            if ratingConfig.contains('-header'):
                ratings = ratings[1:]
            # order of the columns
            order = ratingConfig['-columns'].strip().split()
            delim = ' |,|\t'
            if ratingConfig.contains('-delim'):
                delim=ratingConfig['-delim']
            if negative_count > len(ratings):
                print('The negative file %s has only %d rows, but %d negatives are required.' %
                      (zero_path, len(ratings), negative_count))
                exit(-1)
            new_ratings=random.sample(ratings,negative_count)
            negative_records = []
            for lineNo, line in enumerate(new_ratings):
                hda = split(delim,line.strip())
                if not bTest and len(order) < 2:
                    print('The rating file is not in a correct format. Error: Line num %d' % lineNo)
                    exit(-1)
                try:
                    herbId = hda[int(order[0])]
                    diseaseId = hda[int(order[1])]
                    if len(order)<3:
                        rating = 0 #default value
                    else:
                        rating  = hda[int(order[2])]
                except ValueError:
                    print('Error! Have you added the option -header to the rating.setup?')
                    exit(-1)
                negative_records.append([herbId, diseaseId, float(rating)])
        else:
            print('No negative file found in %s. Generating %d negatives from positives.' %
                  (dataset_dir, negative_count))
            negative_records = sample_negative_records(testData if bTest else trainingData, negative_count)


        # with open('./dataset/TCMSP/zero1.txt') as f:
        #     ratings = f.readlines()
        #     # ignore the headline
        # if ratingConfig.contains('-header'):
        #      ratings = ratings[1:]
        # # order of the columns
        # order = ratingConfig['-columns'].strip().split()
        # delim = ' |,|\t'
        # if ratingConfig.contains('-delim'):
        #     delim=ratingConfig['-delim']
        # new_ratings=random.sample(ratings,56103)

        # with open('./dataset/Symmap/zero.txt') as f:
        #     ratings = f.readlines()
        # # ignore the headline
        # if ratingConfig.contains('-header'):
        #     ratings = ratings[1:]
        # # order of the columns
        # order = ratingConfig['-columns'].strip().split()
        # delim = ' |,|\t'
        # if ratingConfig.contains('-delim'):
        #     delim=ratingConfig['-delim']
        # new_ratings=random.sample(ratings,38043)

        for herbId, diseaseId, rating in negative_records:
            if bTest:
                testData.append([herbId, diseaseId, float(rating)])
            else:
                trainingData.append([herbId, diseaseId, float(rating)])

        if bTest:
            return testData
        else:
            return trainingData

    @staticmethod
    def loadHerbList(filepath):
        herbList = []
        with open(filepath) as f:
            for line in f:
                herbList.append(line.strip().split()[0])
        return herbList


