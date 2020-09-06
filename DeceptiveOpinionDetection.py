import numpy as np
import os
import sys
import nltk
from nltk import word_tokenize

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

nltk.download('punkt')

def create_data_list(basepath):
    k_fold_data = []
    for i in range(1, k+1):
        data = []
        path = basepath + str(i)
        with os.scandir(path) as entries:
            for entry in entries:
                with open(entry, encoding="utf8") as file:
                    data.append(file.readline())
        k_fold_data.append(data)
    return k_fold_data


def build_ngram_plus_dict(k_fold_data, ngram_plus_dict, feature_type):
    ngram_plus_set = set(ngram_plus_dict)

    for data in k_fold_data:
        for text in data:
            tokens = word_tokenize(text.lower())
            
            # adding unigrams
            for unigram in tokens:
                if not unigram in ngram_plus_set:
                    ngram_plus_set.add(unigram)
                    ngram_plus_dict.append(unigram)

            # adding bigrams
            if feature_type != UNIGRAM:
                for bigram in list(nltk.bigrams(tokens)):
                    if not bigram in ngram_plus_set:
                        ngram_plus_dict.append(bigram)
                        ngram_plus_set.add(bigram)

            # adding trigrams
            if feature_type == TRIGRAM_PLUS:
                for trigram in list(nltk.trigrams(tokens)):
                    if not trigram in ngram_plus_set:
                        ngram_plus_dict.append(trigram)
                        ngram_plus_set.add(trigram)

                    
    return ngram_plus_dict


def create_document_vector(data, ngram_plus_reverse_dict, feature_type):
    reviews = np.empty((0, len(ngram_plus_reverse_dict)), dtype = np.float64)

    for text in data:
        review = np.zeros((1, len(ngram_plus_reverse_dict)), dtype = np.float64)
        tokens = word_tokenize(text.lower())
        
        for unigram in tokens:
            review[0][ngram_plus_reverse_dict[unigram]]+=1

        if feature_type != UNIGRAM:
            for bigram in list(nltk.bigrams(tokens)):
                review[0][ngram_plus_reverse_dict[bigram]]+=1

        if feature_type == TRIGRAM_PLUS:
            for trigram in list(nltk.trigrams(tokens)):
                review[0][ngram_plus_reverse_dict[trigram]]+=1
            
        review = preprocessing.normalize(review, norm='l2')
        reviews = np.append(reviews, review, axis = 0)
    
    return reviews


def svm(k_fold_data_true, k_fold_data_deceptive, ngram_plus_reverse_dict, feature_type):
    
    params_grid = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    test_accuracy_sum = 0

    # 0 -> true, 1 -> deceptive
    for i in range(k):
        print('\n\n Fold', str(i+1), ' as Test Dataset. Document Vector creation starts...')

        test_x = create_document_vector(k_fold_data_true[i], ngram_plus_reverse_dict, feature_type)
        test_y = np.zeros(len(k_fold_data_true[i]))
        
        test_x = np.append(test_x, create_document_vector(k_fold_data_deceptive[i], ngram_plus_reverse_dict, feature_type), axis = 0)
        test_y = np.append(test_y, np.ones(len(k_fold_data_deceptive[i])))
        
        
        train_x = None
        train_y = None
        isFirst = True
        
        for j in range(k):
            if i == j:
                continue
                
            true_reviews = create_document_vector(k_fold_data_true[j], ngram_plus_reverse_dict, feature_type)
            deceptive_reviews = create_document_vector(k_fold_data_deceptive[j], ngram_plus_reverse_dict, feature_type)
            
            if isFirst == True:
                isFirst = False
                train_x = true_reviews
                train_y = np.zeros(true_reviews.shape[0])
                
                train_x = np.append(train_x, deceptive_reviews, axis=0)
                train_y = np.append(train_y, np.ones(deceptive_reviews.shape[0]))
            else:
                train_x = np.append(train_x, true_reviews, axis=0)
                train_y = np.append(train_y, np.zeros(true_reviews.shape[0]))
                
                train_x = np.append(train_x, deceptive_reviews, axis=0)
                train_y = np.append(train_y, np.ones(deceptive_reviews.shape[0]))

        print('FinishedDocument Vector creation.')
        print('Shape-> Train_x: ', train_x.shape, ' Train_y: ', train_y.shape, ' Test_x: ', test_x.shape, ' Test_y: ', test_y.shape)
        print('Running SVM (Linear) classifier...')
        svm_model = GridSearchCV(SVC(), params_grid, cv=5)
        svm_model.fit(train_x, train_y)

        # View the training accuracy score
        print('Best score for training data:', svm_model.best_score_,"\n")

        # View the best parameters for the model found using grid search
        print('Best C:',svm_model.best_estimator_.C,"\n")

        best_model = svm_model.best_estimator_
        
        test_y_predicted = best_model.predict(test_x)
        print(classification_report(test_y, test_y_predicted))
        print("Training set score for SVM: %f" % best_model.score(train_x , train_y))
        curr_fold_test_score = best_model.score(test_x, test_y)
        print("Test  set score for SVM: %f" % curr_fold_test_score)
        test_accuracy_sum+= curr_fold_test_score

    print('Final avg. accuracy of test data after ' + str(k) + '-nested cross validation: ', test_accuracy_sum/k)


def main(feature_type):
    print('Using ' + feature_type + ' as feature_type to create document vector')
    rootpath = 'data/positive_polarity/'
    basepaths = [rootpath + 'deceptive_from_MTurk/fold', rootpath + 'truthful_from_TripAdvisor/fold']

    # k_fold_data -> list of list ([<review_text>])
    print('Loading Dataset...')
    k_fold_data_deceptive = create_data_list(basepaths[0])
    k_fold_data_true = create_data_list(basepaths[1])
    print('Finished Dataset Loading.')

    print('Universal ngram_plus dictionary creation starts...')
    ngram_plus_dict = []
    ngram_plus_dict = build_ngram_plus_dict(k_fold_data_true, ngram_plus_dict, feature_type)
    ngram_plus_dict = build_ngram_plus_dict(k_fold_data_deceptive, ngram_plus_dict, feature_type)
    ngram_plus_reverse_dict = {v: k for k, v in enumerate(ngram_plus_dict)}  # 46753
    print('Finished Universal ngram_plus dictionary creation.')

    svm(k_fold_data_true, k_fold_data_deceptive, ngram_plus_reverse_dict, feature_type)
    pass


k = 5
UNIGRAM = "unigram"
BIGRAM_PLUS = "bigram_plus"
TRIGRAM_PLUS = "trigram_plus"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Call the script in this format: DeceptiveOpinionDetection.py --feature_type <unigram/bigram_plus/trigram_plus>")
        print("Example -> python DeceptiveOpinionDetection.py --feature_type bigram_plus")
        print("Example -> python DeceptiveOpinionDetection.py --feature_type unigram")
        print("Example -> python DeceptiveOpinionDetection.py --feature_type trigram_plus")
        exit(1)

    feature_type = sys.argv[2].lower()
    if feature_type not in [UNIGRAM, BIGRAM_PLUS, TRIGRAM_PLUS]:
        print("feature_type should be one of the elements from the list [ ", UNIGRAM, BIGRAM_PLUS, TRIGRAM_PLUS , "]")
        exit(1)
    main(feature_type)



