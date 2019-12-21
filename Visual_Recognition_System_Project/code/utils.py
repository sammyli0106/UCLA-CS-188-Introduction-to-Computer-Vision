import os
import cv2
import numpy as np
import timeit, time
from sklearn import neighbors, svm, cluster, preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
import random
from scipy.spatial import distance

def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'

    train_classes = sorted([dirname for dirname in os.listdir(train_path) if not dirname.startswith('.')],
                           key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path) if not dirname.startswith('.')],
                          key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)
            
    return train_images, test_images, train_labels, test_labels


def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.

    classifier = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
    classifier.fit(train_features, train_labels)
    predicted_categories = classifier.predict(test_features)

    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test feature.

    svm_results = []  # empty list of lists svm_results[cat][text_feature index]

    for cat in range(15):
        train_labels_mod = [1.0 if label == cat else 0.0 for label in train_labels] # make sure this works with decision_function

        if is_linear:
            classifier = svm.SVC(C=svm_lambda, kernel="linear", gamma="scale")
        else:
            classifier = svm.SVC(C=svm_lambda, kernel="rbf", gamma="scale")

        classifier.fit(train_features, train_labels_mod)
        confidence = classifier.decision_function(test_features)

        svm_results.append(confidence)

    predicted_categories = []

    for i, image in enumerate(test_features):
        temp_list = []
        # get specific svm_results according to index of the image
        for s in svm_results:
            temp_list.append(s[i])

        pred_label = temp_list.index(max(temp_list))
        predicted_categories.append(pred_label)

    return np.array(predicted_categories)


def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size]. 
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.

    output_image = cv2.resize(input_image, (target_size, target_size))
    output_image = cv2.normalize(output_image, None, -1, 1, cv2.NORM_MINMAX)
    return output_image


def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    num_pred = len(true_labels)  # total number of predictions
    num_corr = 0                 # number of correct predictions

    for label in zip(true_labels, predicted_labels):
        if label[0] == label[1]:
            num_corr+=1

    accuracy = num_corr/num_pred * 100

    # accuracy is a scalar, defined in the spec (in %)
    return accuracy


def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the 
    # dimension of the feature. each row is a cluster centroid / visual word.

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image.

    descriptors_list = []

    # Extract features from training images
    if feature_type == "sift":
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=20)
        for t in train_images:
            _, descs = sift.detectAndCompute(t, None)

            if descs is None or len(descs) is 1:
                continue

            descriptors_list.extend(descs)

    elif feature_type == "surf":
        surf = cv2.xfeatures2d.SURF_create()
        for t in train_images:
            _, descs = surf.detectAndCompute(t, None)

            if descs is None or len(descs) is 1:
                continue

            temp_list = []
            temp_list.extend(descs)
            temp_list_2 = random.sample(temp_list, 20)
            descriptors_list.extend(temp_list_2)

    elif feature_type == "orb":
        orb = cv2.ORB_create(nfeatures=20)
        for t in train_images:
            _, descs = orb.detectAndCompute(t, None)

            if descs is None or len(descs) is 1:
                continue

            descriptors_list.extend(descs)

    else:
        raise ValueError("Invalid input. Valid values for feature_type are 'sift', 'surf', and 'orb'.")

    descriptors_array = np.asarray(descriptors_list)

    # separate by clustering_type
    if clustering_type is "kmeans":

        kmeans = KMeans(n_clusters=dict_size, random_state=None).fit(descriptors_list)
        vocabulary = kmeans.cluster_centers_

    elif clustering_type is "hierarchical":

        clustering = AgglomerativeClustering(n_clusters=dict_size).fit(descriptors_array)
        labels = clustering.labels_

        clusterCenter = []
        for i in range(0, dict_size):
            clusterDescs = descriptors_array[labels == i]
            clusterMean = np.mean(clusterDescs, axis=0)
            clusterCenter.append(clusterMean)

        clusterCenter = np.asarray(clusterCenter)
        vocabulary = clusterCenter

    else:
        raise ValueError("Invalid input. Valid values for clustering_type are 'kmeans' and 'hierarchical'.")

    return vocabulary


def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary
    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram

    # initalize variables
    descriptors_list = []
    preprocessed_image = []

    # Extract the features from the training images
    if feature_type == "sift":
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=20)
        _, descs = sift.detectAndCompute(image, None)

        if descs is not None:
            descriptors_list.extend(descs)

    elif feature_type == "surf":
        surf = cv2.xfeatures2d.SURF_create()
        _, descs = surf.detectAndCompute(image, None)

        if descs is not None:
            temp_list = []
            temp_list.extend(descs)

            descriptors_list.extend(descs)

    elif feature_type == "orb":
        orb = cv2.ORB_create(nfeatures=20)
        _, descs = orb.detectAndCompute(image, None)

        if descs is not None:
            descriptors_list.extend(descs)

    else:
        raise ValueError("Invalid input. Valid values for feature_type are 'sift', 'surf', and 'orb'.")

    # build the histogram here
    histogram = np.zeros(len(vocabulary))

    # # KNN approach - slow
    # if len(descriptors_list) is not 0:
    #     neigh = KNeighborsClassifier(n_neighbors=1)
    #     neigh.fit(vocabulary, range(len(vocabulary)))
    #     cluster_result = neigh.predict(descriptors_list)
    #     for j in cluster_result:
    #         histogram[j] += 1

    if len(descriptors_list) is not 0:
        dist_matrix = distance.cdist(np.asarray(descriptors_list), vocabulary, "euclidean")
        for i in range(len(descriptors_list)):
            min_label = np.argmin(dist_matrix[i])  # return index of minimum element
            histogram[min_label] += 1              # increment histogram

    preprocessed_image.extend(histogram)
    Bow = np.squeeze(preprocessed_image)

    return Bow


def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors

    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy, 
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds

    classResult = []

    for sz in [8, 16, 32]:  # for each tinyImage size group

        # resize both training features and test features
        start_time = time.time()
        train_features_resized = np.array([imresize(t, sz).flatten() for t in train_features])
        test_features_resized = np.array([imresize(t, sz).flatten() for t in test_features])
        stop_time = time.time()

        # store duration of resizing process
        resize_time = stop_time-start_time

        for num_neighbors in [1, 3, 6]:  # for each specified number of neighbors to consult in KNN

            # run classifier and note duration
            start_time = time.time()
            pred_labels = KNN_classifier(train_features_resized, train_labels, test_features_resized, num_neighbors)
            stop_time = time.time()

            # compute and store accuracy of classifier
            this_acc = reportAccuracy(test_labels, pred_labels)
            classResult.append(this_acc)

            # compute and store total time for both resizing and running classifier
            classResult.append(resize_time + (stop_time-start_time))

    return classResult
    