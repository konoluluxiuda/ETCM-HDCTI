from util.config import OptionConf
from util.dataSplit import *
from multiprocessing import Process, Manager
from util.io import FileIO
from time import strftime, localtime, time
import os
# import pandas as pd
import numpy as np
import mkl

from util.reproducibility import set_global_seed


class HDR(object):
    def __init__(self, config):
        self.trainingData = []  # training data
        self.testData = []  # testData
        self.measure = []
        self.config = config
        self.protocol = config['experiment.protocol'].strip().lower() if config.contains('experiment.protocol') else 'legacy'
        self.strictFolds = None
        self.strictManifest = None
        self.ratingConfig = OptionConf(config['ratings.setup'])
        if self.config.contains('evaluation.setup'):
            self.evaluation = OptionConf(config['evaluation.setup'])
            if self.protocol == 'strict':
                if not self.evaluation.contains('-cv'):
                    raise ValueError('Strict protocol currently requires evaluation.setup=-cv K.')
                k = int(self.evaluation['-cv'])
                self.strictFolds, self.strictManifest = DataSplit.prepareStrictFolds(
                    config, config['datapath'], k
                )
                print('Strict protocol seed: %d' % self.strictManifest['seed'])
            elif self.protocol == 'legacy':
                self.trainingData = FileIO.loadDataSet(config, config['datapath'])
            else:
                raise ValueError('Unknown experiment.protocol: %s' % self.protocol)
        else:
            print('Wrong configuration of evaluation!')
            exit(-1)

        print('Reading data and preprocessing...')



    def execute(self):
        # import the model module
        importStr = 'from ' + self.config['model.name'] + ' import ' + self.config['model.name']
        exec(importStr)
        if self.evaluation.contains('-cv'):
            k = int(self.evaluation['-cv'])
            if k < 2 or k > 10:  # limit to 2-10 fold cross validation
                print("k for cross-validation should not be greater than 10 or less than 2")
                exit(-1)
            mkl.set_num_threads(max(1, mkl.get_max_threads() // k))
            use_multiprocessing = True
            if self.config.contains('gpu.multiprocessing'):
                use_multiprocessing = self.config['gpu.multiprocessing'].lower() in ('1', 'true', 'yes', 'on')
            if self.protocol == 'strict' and use_multiprocessing:
                print('Strict protocol currently runs folds serially to preserve deterministic TensorFlow state.')
                use_multiprocessing = False
            # CUDA/TensorFlow is fragile after fork in WSL. Keep GPU training in one process by default.
            if not use_multiprocessing:
                mDict = {}
            else:
                manager = Manager()
                mDict = manager.dict()
            i = 1
            tasks = []
            dataset_dir = os.path.dirname(os.path.abspath(self.config['datapath']))
            folds = self.strictFolds if self.protocol == 'strict' else DataSplit.crossValidation(
                self.trainingData, k, path=dataset_dir
            )
            base_seed = int(self.config['random.seed']) if self.config.contains('random.seed') else 2026
            for train, test in folds:
                fold = '[' + str(i) + ']'
                recommender = self.config['model.name'] + "(self.config,train,test,fold)"
                if not use_multiprocessing:
                    fold_seed = base_seed + i - 1
                    set_global_seed(fold_seed, reset_tensorflow_graph=True)
                    print('Fold %d random seed: %d' % (i, fold_seed))
                    mDict[i] = eval(recommender).execute()
                    i += 1
                    continue
                # create the process
                p = Process(target=run, args=(mDict, eval(recommender), i))
                tasks.append(p)
                i += 1
            if use_multiprocessing:
                # start the processes
                for p in tasks:
                    p.start()
                    if not self.evaluation.contains('-p'):
                        p.join()
                # wait until all processes are completed
                if self.evaluation.contains('-p'):
                    for p in tasks:
                        p.join()
            # compute the average and standard deviation of k-fold cross validation
            self.measure = [mDict[i] for i in range(1, k + 1) if i in mDict]
            res = []
            if not self.measure:
                print('No fold metrics were returned.')
                return
            if len(self.measure) != k:
                print('Warning: expected %d folds but received metrics from %d folds.' % (k, len(self.measure)))
            for i in range(len(self.measure[0])):
                measure = self.measure[0][i].split(':')[0]
                values = []
                for j in range(len(self.measure)):
                    values.append(float(self.measure[j][i].split(':')[1]))
                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0.0
                res.append('%s:%.6f(±%.6f)\n' % (measure, mean, std))
            # output result
            currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
            outDir = OptionConf(self.config['output.setup'])['-dir']
            fileName = self.config['model.name'] + '@' + currentTime + '-' + str(k) + '-fold-cv' + '.txt'
            FileIO.writeFile(outDir, fileName, res)
            print('The result of %d-fold cross validation:\n%s' % (k, ''.join(res)))
        else:

            recommender = self.config['model.name'] + '(self.config,self.trainingData,self.testData)'
            eval(recommender).execute()

def run(measure, algor, order):
    measure[order] = algor.execute()
