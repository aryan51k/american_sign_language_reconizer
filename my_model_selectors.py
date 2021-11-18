import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        bic = []
        #Loop through possible states
        for x in range(self.min_n_components, self.max_n_components+1):
            try:
                hmm_model = self.base_model(x)
                likelihood = hmm_model.score(self.X, self.lengths)
                p = x**2 + (2 * (x * len(self.X[0]))) - 1
                bic_score = -2 * likelihood + p * np.log(len(self.X))
                bic.append((bic_score, hmm_model))
            except:
                pass
        if not bic:
            return None
        sc, model = min(bic, key=lambda x: x[0])
        return model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        dic_scores = []

        for x in range(self.min_n_components, self.max_n_components+1):
            try:
                hmm_model = self.base_model(x)
                current_score = hmm_model.score(self.X, self.lengths)
                other_words_score = []
                other_average_score = np.mean([hmm_model.score(*self.hwords[word]) for word in self.words if word != self.this_word])
                dic_scores.append((current_score - other_average_score, hmm_model))
            except:
                pass
        if not dic_scores:
            return None
        _, model = max(dic_scores, key=lambda x: x[0])
        return model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        logl_model_list = []
        splits = KFold(n_splits=3)
        for components in range(self.min_n_components, self.max_n_components+1):
            try:
                hmm_model = GaussianHMM(n_components = components, covariance_type = "diag", n_iter = 1000, random_state = self.random_state, verbose = False)
                log_l_list = []
                for cv_train_idx, cv_test_idx in splits.split(self.sequences):
                    train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                    hmm_model.fit(train_X, train_lengths)
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    log_l = hmm_model.score(test_X, test_lengths)
                    log_l_list.append(log_l)
                logl_model_list.append((np.mean(log_l_list), hmm_model))
            except:
                continue
        if logl_model_list:
            _, model = max(logl_model_list, key=lambda x: x[0])
            return model
        else:
            return None
