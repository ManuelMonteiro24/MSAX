import numpy, os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from tslearn.datasets import *
from tslearn.utils import *
from tslearn.neighbors import *
from tslearn.preprocessing import *
from tslearn.piecewise import *
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, cohen_kappa_score
from sklearn.utils.multiclass import unique_labels
from sklearn.multiclass import OneVsRestClassifier

#metric_name = "dtw" "euclidean" "min_dist"
#multivar_final_symbolic_representation(X_train,variables,alphabet)
#multivar_final_symbolic_representation(X_test,variables,alphabet)
def plot_roc_auc_curve(y_test, y_score,n_classes):
    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    #macro option
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    return


def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    print(y_pred[0])

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def test_multivariate_methods(dataset_name,metric_name,segments,alphabet,sax_multivariate_output):
    start_time = time.time()

    X_train, y_train, X_test, y_test, variables = read_dataset_from_file(dataset_name)
    X_train = separate_atributes_dataset(X_train,variables)
    X_test = separate_atributes_dataset(X_test,variables)

    #condition for diferent normalizations
    if metric_name == "min_dist" and sax_multivariate_output == None:
        X_train = multivariate_normalization(X_train,variables)
        X_test = multivariate_normalization(X_test,variables)
        dataset_str = "DATASETS/NORMALIZED_MULTIVAR/Pendigits"
    else:
    #print("X_train before")
    #print_covar_matrix(X_train,variables)

        X_train = z_normalize(X_train,variables)
        X_test = z_normalize(X_test,variables)
        dataset_str = "DATASETS/NORMALIZED_STD/Pendigits"

    #write_dataset([X_train,X_test],[y_train,y_test], dataset_str)
    #print("X_train after")
    #print_covar_matrix(X_train,variables)


    if metric_name == "min_dist":
        sax_trans = SymbolicAggregateApproximation(n_segments=segments, alphabet_size=alphabet,variables_size=variables, multivariate_output= sax_multivariate_output)
        X_train = sax_trans.fit_transform(X_train,None)
        X_test = sax_trans.fit_transform(X_test,None)

        #print("X_train : ", X_train)
        #print("X_test : ", X_test)

        if sax_multivariate_output == None:
            dataset_str = "DATASETS/DISCRETIZED_MULTIVAR/Pendigits"
            #write_multivar_discretized_dataset([X_train,X_test],[y_train,y_test], dataset_str)
        else:
            dataset_str = "DATASETS/DISCRETIZED/Pendigits"
            #write_dataset([X_train,X_test],[y_train,y_test], dataset_str)


        knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric=metric_name, metric_params=alphabet, variables_size=variables,multivariate_output= sax_multivariate_output)
    else:
        knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric=metric_name, variables_size=variables,multivariate_output= sax_multivariate_output)

    knn_clf.fit(X_train, y_train)
    predicted_labels = knn_clf.predict(X_test)
    acc = accuracy_score(y_test, predicted_labels)
    end_time = time.time()
    #print("Confusion matrix")
    confussion_matrix = confusion_matrix(y_test, predicted_labels)
    #class_names = ['0','1','2','3','4','5','6','7','8','9']
    #plot_confusion_matrix(y_test, predicted_labels, classes=class_names, normalize=False,
    #                  title='Confusion matrix')

    #clf = OneVsRestClassifier(knn_clf)
    #predicted_labels_roc = clf.fit(X_train, y_train)
    #y_score = clf.predict(X_test)
    #plot_roc_auc_curve(y_test, y_score,10)
    #plt.show()

    cohen_value = cohen_kappa_score(y_test, predicted_labels)
    print(cohen_value)
    #tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels)
    #specificity = tn / (tn+fp) TODO IF NEEDED

    class_report=classification_report(y_test,predicted_labels)
    print(class_report)

    #print(confussion_matrix)

    with open("tests_folder/teste_maio_entrega.txt", "a") as myfile:
        a = str(acc) + " dataset_name: " + dataset_name + " metric_name: " + metric_name
        if metric_name == "min_dist":
            a += " n_segments: " + str(segments) + " alphabet_size: " + str(alphabet)
            if sax_multivariate_output == True:
                a += " sax_multivariate_ind\n"
            else:
                a += " time: " + str(end_time-start_time) + "\n"
        else:
            a += " time: " + str(end_time-start_time) + "\n"
        print(a)
        myfile.write(a)
        #myfile.write("Confusion matrix " + "\n" + "\n")
        #myfile.write(str(confussion_matrix) + "\n" + "\n")


####MAIN####
mode = "severa_tests"

if mode == "several_tests":

    #data for multivariate datasets
    #datasets = ["CharacterTrajectories_CUTTED","uWave","Libras","Pendigits",
    #"ECG_CUTTED","AUSLAN_CUTTED_CONST_ELIMINATED","JapaneseVowels_CUTTED_CONST_ELIMINATED",
    #"LP1_CONST_ELIMINATED","LP2_CONST_ELIMINATED","LP3_CONST_ELIMINATED","LP4_CONST_ELIMINATED","LP5_CONST_ELIMINATED",
    #"Wafer_CUTTED_CONST_ELIMINATED","AtrialFibrilation_FORMATED",
    #"Epilepsy_FORMATED","ERing_FORMATED","EthanolConcentration_FORMATED",
    #"StandWalkJump_FORMATED"]
    #datasets_ts_length = [109,315,45,8,39,45,7,15,15,15,15,15,104,640,206,65,1751,2500]

    datasets = ["DATASETS/CharacterTrajectories_CUTTED","DATASETS/Pendigits"]
    datasets_ts_length = [109,8]
    methods = ["min_dist"]
    multivar_sax = [None,True]
    alphabet_list = [3,5,7,10]
    reduction_list = [1/4,1/2,3/4,1]
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

    test_multivariate_methods("DATASETS/Pendigits","min_dist",4,3,True)

#Dataset	Train Size	Test Size	Length	No. of Classes	#atrributes
#Pendigits	300	10692	8	10	2
