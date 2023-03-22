import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support
from sklearn import neighbors

def weights_sum_test(conf_scores, class_num, alpha=0.8):    
    '''
    This function performs rescaling and softmax transformation of confidence scores. 
    
    It uses "numpy" library as "np" to calculate softmax. 
    
    Input:  conf_scores - the list of confidence scores
            class_num - the integer number of classes
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8 
    Output: the list of transformed confidence scores
    '''
    conf_scores_T = conf_scores.T
    # Rescale confident scores 
    conf_scores_T_rescale = [[(conf_scores_T[k][i]-0.5)/(alpha) for i in range(len(conf_scores_T[k]))] for k in range(class_num)]
    conf_scores_T_rescale_sum = [sum(conf_scores_T_rescale[k]) for k in range(class_num)]
    # Normalize confident scores 
    res = [np.exp(conf_scores_T_rescale_sum[k])/sum([np.exp(conf_scores_T_rescale_sum[k]) for k in range(class_num)]) for k in range(class_num)]
    return res

def knn_method(train_data, y, test_data, vector_name, NNeighbour, method):
    '''
    This function performs kNN method. 
    
    It uses "numpy" library as "np" and "neighbors" library. 
    
    Input:  train_data - pandas DataFrame with train data that contains features 'vector_name'
            y - pandas Series or list, golden labels of train_data instances 
            test_data - pandas DataFrame with test data that contains features 'vector_name'
            vector_name - the string with a name of feature vector in train_data and test_data
            NNeighbours - int number, which represents amount of neighbours 'k' 
            method - the string with value "labels" or "conf_scores". The first option returns predicted labels for test    
                instances, the second - the list of confidence scores corresponding to all prediction classes for each test instance.
    Output: y_pred_res - the list of predicted labels (or confidence scores).
    '''   
    # Transform data into numpy matrix 
    train_ind = list(train_data.index)
    test_ind = list(test_data.index)
    X = np.zeros((len(train_data), len(train_data[vector_name][train_ind[0]])))
    for m in range(len(train_ind)):
        for j in range(len(train_data[vector_name][train_ind[m]])):
            X[m][j] = train_data[vector_name][train_ind[m]][j]
    X_test = np.zeros((len(test_data), len(test_data[vector_name][test_ind[0]])))
    for k in range(len(test_data)):
        for j in range(len(test_data[vector_name][test_ind[0]])):
            X_test[k][j] = test_data[vector_name][test_ind[k]][j]
            
    # initialize NearestNeighbor classifier
    knn = neighbors.KNeighborsClassifier(n_neighbors=NNeighbour)
    # train model
    knn.fit(X, y)
    
    # apply for test data
    if method == 'labels':
        res = knn.predict(X_test)
    if method == 'conf_scores':
        res = knn.predict_proba(X_test)  
        
    return res

def knn_ensemble_labels(train_data, y, test_data, features_names, NNeighbours):
    '''
    This function performs an ensemble of kNN models based on predicted labels. 
    
    It uses "numpy" library as "np". 
    
    Input:  train_data - pandas DataFrame with train data that contains features 'vector_names'
            y - pandas Series or list, golden labels of train_data instances 
            test_data - pandas DataFrame with test data that contains features 'vector_names'
            features_names - the list of strings, names of features vectors in train_data and test_data
            NNeighbours - the list of int numbers, where each number represent amount of neighbours 'k' that will be used 
                to perform FRNN OWA classification method for the corresponded feature vector. Lenghts of 'features_names' and 'NNeighbours' lists should be equal.
    Output: y_pred_output - the list of predicted labels.
    '''  
    y_pred = []
    
    for j in range(len(features_names)):  
        vector_name = features_names[j]
        res = knn_method(train_data, y, test_data, vector_name, NNeighbours[j], 'labels')
        y_pred.append(res)
        
    # The voting function to obtain the ensembled label - we used mean 
    y_pred_res = np.mean(y_pred, axis=0)
    y_pred_output = [round(i) for i in y_pred_res]
    return y_pred_output

def knn_ensemble_conf_scores(train_data, y, test_data, features_names, NNeighbours, alpha=0.8):
    '''
    This function performs an ensemble of kNN models based on predicted confidence scores. 
    
    It uses "numpy" library as "np". 
    
    Input:  train_data - pandas DataFrame with train data that contains features 'vector_names'
            y - pandas Series or list, golden labels of train_data instances 
            test_data - pandas DataFrame with test data that contains features 'vector_names'
            features_names - the list of strings, names of features vectors in train_data and test_data
            NNeighbours - the list of int numbers, where each number represent amount of neighbours 'k' that will be used 
                to perform FRNN OWA classification method for the corresponded feature vector. Lenghts of 'features_names' and 'NNeighbours' lists should be equal.
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8 
    Output: y_pred_res - the list of predicted confidence scores.
    '''  
    y_pred = []
    
    # Calculate number of classes
    class_num = len(set(y))
    # Create and fill 3D array
    conf_scores_all = np.zeros((len(features_names), len(test_data), class_num))
    
    for j in range(len(features_names)):  
        vector_name = features_names[j]
        
        result = knn_method(train_data, y, test_data, vector_name, NNeighbours[j], 'conf_scores')
        
        # Check for NaNs 
        for k in range(len(result)):
            if np.any(np.isnan(result[k])):
                result[k] = [0 for i in range(class_num)]              
        conf_scores_all[j] = (result)
        
    # Rescale obtained confidence scores 
    rescaled_conf_scores = np.array([weights_sum_test(conf_scores_all[:, k, :], class_num, alpha) for k in range(len(conf_scores_all[0]))])
    # Use the mean voting function to obtain the predicted label 
    y_pred_res = [np.round(6*np.average(np.exp(k)/sum(np.exp(k)), weights=list(set(y)))) for k in rescaled_conf_scores]
    return y_pred_res

def cross_validation_ensemble_knn(df, features_names, class_name, K_fold, k, method, evaluation, alpha=0.8):
    '''
    This function performs cross-validation evaluation for wkNN ensemble.
    
    It uses "numpy" library as "np" for random permutation of list.
    It uses "scipy" library to calculate Pearson Correlation Coefficient.
    It uses "sklearn" library to calculate F1-score.    
    
    Input:  df - pandas DataFrame with features to evaluate 
            features_names - the list of strings, names of features vectors in df
            class_name - the string name of the column of df that contains classes of instances 
            K_fold - the number of folds of cross-validation, we used K_fold = 5
            k - the list of int numbers, where each number represent amount of neighbours 'k' that will be used 
                to perform FRNN OWA classification method for the corresponded feature vector. Lenghts of 'features_names' and 'k' lists should be equal
            method - this string variable defines the output of wkNN approach, it can be 'labels' or 'conf_scores'
            evaluation - the evaluation method's name: could be 'pcc' for Pearson Correlation Coefficient or 'f1' for F1-score
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8 
    Output: The evaluation score as float number: either PCC or F1-score             
    '''
    df[method] = None

    # Cross-validation
    random_indices = np.random.permutation(df.index)
    for i in range(K_fold): 
        
        # Split df on train and test data
        test_data = df.loc[df.index.isin(random_indices[i*len(df.index)//K_fold:(i+1)*len(df.index)//K_fold])]
        train_data = df[~df.index.isin(test_data.index)]
        y = train_data[class_name]
        y_true = test_data[class_name]  
        
        # Apply kNN method for each feature vector depends on specified output type 
        if method == 'labels':
            y_pred_res = knn_ensemble_labels(train_data, y, test_data, features_names, k)
        if method == 'conf_scores':
            y_pred_res = knn_ensemble_conf_scores(train_data, y, test_data, features_names, k, alpha)
        df[method].loc[test_data.index] = y_pred_res
    
    # Evaluation with F1-score or Pearson Correlation Coefficient 
    if evaluation == 'f1':
        p, r, res, support = precision_recall_fscore_support(df[class_name].to_list(), df[method].to_list(), average = "macro")
    elif evaluation == 'pcc':
        res = pearsonr(df[class_name].to_list(), df[method].to_list())[0]
    else:
        print('Wrong evaluation metric was specified! Choose "pcc" or "f1".')
    return res

def evaluate_embedding_with_lexicon(train, test, words, class_col, vector, lexicons_data, lex_number, k, wkNN_method, alpha=0.8):
    '''
    This function performs evaluation with wkNN model for embedding vector combined with lexicon scores.

    Input:  train - pandas DataFrame with train dataset, it should has words, class_col, and vector columns
            test - pandas DataFrame with test dataset, it should has words and vector columns
            words - string, the name of column in data datafrane with text to evaluate
            class_col - string, the name of column in data datafrane with classes 
            vector - strings, name of features (embedding) vectors in data
            lexicons_data - the special list with all lexicons we considered. It has the following format:
                            each element of an array is an array of 5 lements that has following components:
                            - DataFrame with lexicon
                            - string with the name of column from the DataFrame with list of words
                            - int, number of numerical columns with lexicon scores in DataFrame, all columns besides the one with the list of words  
                            - int, the lower margin of the lexicon scores 
                            - int, the upper margin of the lexicon scores 
                            We used following format for 5 lexicons we considered, you can copy it: 
                            lexicons_data = [[vad, 'Word', 3, 0, 1], [emolex, 'Word', 10, 0, 1], [ai, 'term', 4, 0, 1], [anew, 'Word', 6, 0, 9], [warriner, 'Word', 63, 0, 1000]]
            lex_number - int from 1 to 5, number of lexicons from lexicons_data we want to use
            k - int, which represent amount of neighbours 'k' 
            wkNN_method - this string variable defines the output of wkNN approach, it can be 'labels' or 'conf_scores'
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8 
    Output: The evaluation scores: Mean MAE, Std MAE, Pearson             
    '''
    # Concatenate embedding vector with lexicons 

    columns = []
    if lex_number == 1:
        for lex in lexicons_data:
            col = namestr(lex, globals())[0] 
            train[col] = train.apply(lambda x: append_lexicon_scores(x[words], x[vector], lex), axis=1)
            test[col] = test.apply(lambda x: append_lexicon_scores(x[words], x[vector], lex), axis=1)
            columns.add(col)
    elif lex_number == 2:
        for i in range(len(lexicons_data)):
            for j in range(i+1, len(lexicons_data)):
                col = namestr(lexicons_data[i], globals())[0] + "_" + namestr(lexicons_data[j], globals())[0]
                train[col] = train.apply(lambda x: append_two_lexicon_scores(x[words], x[vector], lexicons_data[i], lexicons_data[j]), axis=1)
                test[col] = test.apply(lambda x: append_two_lexicon_scores(x[words], x[vector], lexicons_data[i], lexicons_data[j]), axis=1)
                columns.add(col)
    elif lex_number == 3:
        for i in range(len(lexicons_data)):
            for j in range(i+1, len(lexicons_data)):
                for k in range(j+1, len(lexicons_data)):
                    col = namestr(lexicons_data[i], globals())[0] + "_" + namestr(lexicons_data[j], globals())[0] + "_" + namestr(lexicons_data[k], globals())[0]
                    train[col] = train.apply(lambda x: append_three_lexicon_scores(x[words], x[vector], lexicons_data[i], lexicons_data[j], lexicons_data[k]), axis=1)
                    test[col] = test.apply(lambda x: append_three_lexicon_scores(x[words], x[vector], lexicons_data[i], lexicons_data[j], lexicons_data[k]), axis=1)
                    columns.add(col)
    elif lex_number == 4:
        for i in range(len(lexicons_data)):
            for j in range(i+1, len(lexicons_data)):
                for k in range(j+1, len(lexicons_data)):
                    for l in range(k+1, len(lexicons_data)):
                        col = namestr(lexicons_data[i], globals())[0] + "_" + namestr(lexicons_data[j], globals())[0] + "_" + namestr(lexicons_data[k], globals())[0] + "_" + namestr(lexicons_data[l], globals())[0]
                        train[col] = train.apply(lambda x: append_four_lexicon_scores(x[words], x[vector], lexicons_data[i], lexicons_data[j], lexicons_data[k], lexicons_data[l]), axis=1)
                        test[col] = test.apply(lambda x: append_four_lexicon_scores(x[words], x[vector], lexicons_data[i], lexicons_data[j], lexicons_data[k], lexicons_data[l]), axis=1)
                        columns.add(col)
    elif lex_number == 5:
        columns.append("Vector_all")
        train["Vector_all"] = train.apply(lambda x: append_five_lexicon_scores(x[words], x[vector], lexicons_data[0], lexicons_data[1], lexicons_data[2], lexicons_data[3], lexicons_data[4]), axis=1)
        test["Vector_all"] = test.apply(lambda x: append_five_lexicon_scores(x[words], x[vector], lexicons_data[0], lexicons_data[1], lexicons_data[2], lexicons_data[3], lexicons_data[4]), axis=1)
    else: 
        return "Enter valid lex_number: from 1 to 5!"
            
    # Calculate evaluation scores
    results = []
    for col in columns:
        if wkNN_method == "labels":
            res = knn_ensemble_labels(train, train[class_col], test, col, k)
        elif wkNN_method == "conf_scores":
            res = knn_ensemble_conf_scores(train, train[class_col], test, col, k, alpha)
        else:
            return "Enter valid wkNN_method: labels or conf_scores!"
        results.append(round(res[0], 4), round(res[1], 4), round(res[2], 4))
    return results    
    
def get_neigbours(test_vector, df_train_vectors, feature, k, text_column, class_column): 
    '''
    This function calculates k neirest neighbours to the test_vector using cosine similarity
    
    Input: test_vector - array of numbers
           df_train_vectors - DataFrame with train instances
           feature - name of a column in df_train_vectors with texts' embedding vectors
           k - a number of neirest neighbours 
           text_column - name of a column in df_train_vectors with texts
           class_column - name of a column in df_train_vectors with texts' classes
    Output: list of k neirest neighbours' texts and list with their classes
    '''
    distances = df_train_vectors[feature].apply(lambda x: cosine_relation(x, test_vector))
    top_k = distances.sort_values(ascending=False)[:k]
    df_top_k = df_train_vectors.loc[top_k.index]
    return df_top_k[text_column].to_list(), df_top_k[class_column].to_list()