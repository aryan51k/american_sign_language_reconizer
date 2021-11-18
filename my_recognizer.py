import warnings
import operator
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    #Loop through ids
    for ids in range(0, len(test_set.get_all_Xlengths())):
        #Get params to score the model
        X, Xlengths = test_set.get_item_Xlengths(ids)
        prob = {}
        #Iterate through dict to cache the score
        for word, model in models.items():
            try:
                prob[word] =  model.score(X, Xlengths)
            except:
                continue
        if prob:
            #Append scores to scores list
            probabilities.append(prob)
            #Store the word with the highest estimate
            best_est = max(list(prob.items()), key = operator.itemgetter(1))[0]
            guesses.append(best_est)
    return probabilities, guesses
