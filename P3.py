import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from scipy.stats import norm




def bayes_classifier(data_training, class_training, data_test):

    classes = np.unique(class_training)


    kde_estimator = {}
    for label in classes:

        class_data = data_training[class_training == label]

        kde_object = KernelDensity(kernel='gaussian' , bandwidth=0.5).fit(class_data)

        kde_estimator[label] = kde_object

        #kde_estimator[0] = kde_object0, kde_estimator[1] = kde_object1, kde_estimator[2] = kde_object12


    class_test = np.zeros(data_test.shape[0])
    for i in range(data_test.shape[0]): #Iteration on elements of test data
        test_point = data_test[i]
        probability = np.zeros(len(classes))

        for j, label in enumerate(classes): #Iteration on classes (labels)
            kde = kde_estimator[label]
            probability[j] = np.exp(kde.score_samples([test_point]))[0]

        past_probability = np.bincount(class_training) / len(class_training)
        final_probability = probability * past_probability

        class_test[i] = classes[np.argmax(final_probability)]

    return class_test

##

def naive_bayes(data_training, class_training, data_test):

    classes = np.unique(class_training)

    kde_estimator = {}

    #kde for each class and feature
    for label in classes:

        class_data = data_training[class_training == label]

        kde_estimator[label] = []

        for feature in range(class_data.shape[1]):

            feature_data = class_data[:, feature].reshape(-1, 1)  #reshape to unkonwn number of rows and one column
            kde_object = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(feature_data)

            kde_estimator[label].append(kde_object) #storing the trained estimator


    class_test = np.zeros(data_test.shape[0])

    for i in range(data_test.shape[0]): # on elements of test data

        test_point = data_test[i]
        probability = np.zeros(len(classes))

        for j, label in enumerate(classes): #on classes (labels)
            class_probability = 1.0

            for feature in range(data_test.shape[1]):  #on features
                kde = kde_estimator[label][feature]
                feature_probability = np.exp(kde.score_samples([[test_point[feature]]]))[0]
                class_probability *= feature_probability

            probability[j] = class_probability

        past_probability = np.bincount(class_training) / len(class_training)
        final_probability = probability * past_probability

        class_test[i] = classes[np.argmax(final_probability)] #max

    return class_test

##

def naive_bayes_gaussian(data_training, class_training, data_test):

    class_data = np.unique(class_training)

    #class means and variances
    class_means = []
    class_variances = []

    #means and variances for each feature
    for label in class_data:
        label_data = data_training[class_training == label]
        class_means.append(np.mean(label_data, axis=0))
        class_variances.append(np.var(label_data, axis=0))

    class_test = np.zeros(data_test.shape[0])

    for i in range(data_test.shape[0]):

        test_point = data_test[i]
        #probability for each class
        probabilities = np.zeros(len(class_data))
        for j, label in enumerate(class_data):
            class_mean = class_means[j]
            class_variance = class_variances[j]

            probability = np.prod(norm.pdf(test_point, class_mean, np.sqrt(class_variance)))
            probabilities[j] = probability

        past_probability = np.bincount(class_training) / len(class_training)
        final_probability = probabilities * past_probability
        class_test[i] = class_data[np.argmax(final_probability)] #max

    return class_test

####


iris = load_iris()
data_matrix = iris.data
class_vector = iris.target
num_iterations = 10
accuracy_bayes = 0
accuracy_naive_bayes = 0
accuracy_naive_bayes_g = 0

for i in range(num_iterations):

    #spliting the dataset 50 / 50
    data_train, data_test, class_train, class_test = train_test_split(data_matrix, class_vector, test_size=0.5, stratify=class_vector)

    bayes_classes = bayes_classifier(data_train, class_train, data_test)
    accuracy_bayes += accuracy_score(class_test, bayes_classes)

    n_bayes_classes = naive_bayes(data_train, class_train, data_test)
    accuracy_naive_bayes += accuracy_score(class_test, n_bayes_classes)

    n_bayes_gaussian_classes = naive_bayes_gaussian(data_train, class_train, data_test)
    accuracy_naive_bayes_g += accuracy_score(class_test, n_bayes_gaussian_classes)


#avg
avg_acc_bayes = accuracy_bayes/num_iterations
avg_acc_n_bayes = accuracy_naive_bayes/num_iterations
avg_acc_n_bayes_g = accuracy_naive_bayes_g/num_iterations


print("avg accuracy for Bayes Classifier: ", avg_acc_bayes)
print("avg accuracy for Naive Bayes Classifier: ", avg_acc_n_bayes)
print("avg accuracy for Naive Bayes Classifier Gaussian: ", avg_acc_n_bayes_g)