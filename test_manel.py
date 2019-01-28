import numpy, os, sys
from tslearn.datasets import *
from tslearn.utils import *
from tslearn.neighbors import *
from tslearn.preprocessing import *
from tslearn.piecewise import *
from sklearn.metrics import accuracy_score

#metric_name = "dtw" "euclidean" "min_dist"
#multivar_final_symbolic_representation(X_train,variables,alphabet)
#multivar_final_symbolic_representation(X_test,variables,alphabet)
def test_multivariate_methods(dataset_name,metric_name,segments,alphabet,sax_multivariate_output):

    X_train, y_train, X_test, y_test, variables = read_dataset_from_file(dataset_name)
    X_train = separate_atributes_dataset(X_train,variables)
    X_test = separate_atributes_dataset(X_test,variables)

    if metric_name == "min_dist" and sax_multivariate_output == None:
        X_train = multivariate_normalization(X_train,variables)
        X_test = multivariate_normalization(X_test,variables)
    else:
        X_train = z_normalize(X_train,variables)
        X_test = z_normalize(X_test,variables)

    if metric_name == "min_dist":
        sax_trans = SymbolicAggregateApproximation(n_segments=segments, alphabet_size=alphabet,variables_size=variables, multivariate_output= sax_multivariate_output)
        X_train = sax_trans.fit_transform(X_train,None)
        X_test = sax_trans.fit_transform(X_test,None)

        #print("X_train : ", X_train)
        #print("X_test : ", X_test)
        #write_discretized_dataset([X_train,X_test],[y_train,y_test],"DATASETS/DISCRETIZED/ECG_CUTTED")
        #sys.exit()

        knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric=metric_name, metric_params=alphabet, variables_size=variables,multivariate_output= sax_multivariate_output)
    else:
        knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric=metric_name, variables_size=variables,multivariate_output= sax_multivariate_output)

    knn_clf.fit(X_train, y_train)
    predicted_labels = knn_clf.predict(X_test)
    acc = accuracy_score(y_test, predicted_labels)

    with open("tests_folder/teste!!!.txt", "a") as myfile:
        a = str(acc) + " dataset_name: " + dataset_name + " metric_name: " + metric_name
        if metric_name == "min_dist":
            a += " n_segments: " + str(segments) + " alphabet_size: " + str(alphabet)
            if sax_multivariate_output == True:
                a += " sax_multivariate_ind\n"
            else:
                a += "\n"
        else:
            a += "\n"
        print(a)
        myfile.write(a)

mode = "seveal_tests"

if mode == "several_tests":
    #datasets = []

    #AUSLAN_CUTTED_CONST_ELIMINATED","ArabicDigits_CUTTED_CONST_ELIMINATED
    #"JapaneseVowels_CUTTED_CONST_ELIMINATED"
    #"AUSLAN_CUTTED_CONST_ELIMINATED","JapaneseVowels_CUTTED_CONST_ELIMINATED",
    #"LP1_CONST_ELIMINATED","LP2_CONST_ELIMINATED","LP3_CONST_ELIMINATED",
    # "LP4_CONST_ELIMINATED","LP5_CONST_ELIMINATED",
    # "CharacterTrajectories_CUTTED","uWave","Wafer_CUTTED_CONST_ELIMINATED"

    datasets = ["CharacterTrajectories_CUTTED","uWave","Libras","Pendigits",
    "ECG_CUTTED","AUSLAN_CUTTED_CONST_ELIMINATED","JapaneseVowels_CUTTED_CONST_ELIMINATED",
    "LP1_CONST_ELIMINATED","LP2_CONST_ELIMINATED","LP3_CONST_ELIMINATED","LP4_CONST_ELIMINATED","LP5_CONST_ELIMINATED",
    "Wafer_CUTTED_CONST_ELIMINATED","AtrialFibrilation_FORMATED",
    "Epilepsy_FORMATED","ERing_FORMATED","EthanolConcentration_FORMATED",
    "StandWalkJump_FORMATED"]
    datasets_ts_length = [109,315,45,8,39,45,7,15,15,15,15,15,104,640,206,65,1751,2500]
    methods = ["min_dist"]
    multivar_sax = [True]
    alphabet_list = [5,10,15,20]
    reduction_list = [1/4, 1/2, 3/4,1]
    test_list = []

    for i in range(0,len(datasets)):
        for method_name in methods:
            if method_name == "euclidean" or method_name == "dtw":
                test_list.append([datasets[i],method_name,5,2,None])
            else:
                for multivar in multivar_sax:
                    for a in alphabet_list:
                        for reducted_length in reduction_list:
                            test_list.append([datasets[i],method_name,int(datasets_ts_length[i]*reducted_length),a,multivar])
    for list_element in test_list:
        test_multivariate_methods(list_element[0],list_element[1],list_element[2],list_element[3],list_element[4])
else:
    test_multivariate_methods("DATASETS/SYNTETICS/syntetic_data_2","min_dist",5,10,None)
