from utils import *
import argparse

parser = argparse.ArgumentParser(description='CS188.2 - Fall 19 - Homework 1')
parser.add_argument("--tiny", "-t", type=bool, default=True, help='run Tiny Images')
parser.add_argument("--create-path", "-cp", type=bool, default=True, help='create the Results directory')
args = parser.parse_args()

# The argument is included as an idea for debugging, with a few examples in the main. Feel free to modify it or add arguments.
# You are also welcome to disregard this entirely

#############################################################################################################################
# This file contains the main. All the functions you need to write are included in utils. You also need to edit the main.
# The main just gets you started with the data and highlights the high level structure of the project.
# You are free to modify it as you wish - the modifications you are required to make have been marked but you are free to make
# others.
# What you cannot modify is the number of files you have to save or their names. All the save calls are done for you, you
# just need to specify the right data.
#############################################################################################################################


if __name__ == "__main__":

    # use these flags to either run the appropriate section (1) or skip it (0)
    f_run_tinyImages = 1     # run tinyImages
    f_make_vocab = 0         # create vocabularies - if 0, will load from memory
    f_run_computeBOW = 0     # run computeBOW on training and test images
    f_run_KNN_BOW = 1        # run KNN on training and test feature descriptors
    f_run_LinearSVM_BOW = 1  # run SVM with linear kernel on training and test feature descriptors
    f_run_RBF_SVM_BOW = 1    # run SVM with RBF kernel on training and test feature descriptors
    # note that most sections will print to console by default, comment out print statements if not desired

    if args.create_path:
        # To save accuracies, runtimes, voabularies, ...
        if not os.path.exists('Results_ins'):
            os.mkdir('Results_ins')
        SAVEPATH = 'Results_ins/'

    # Load data, the function is written for you in utils
    train_images, test_images, train_labels, test_labels = load_data()

    if args.tiny and f_run_tinyImages:
        print("Running tiny images...")
        # You have to write the tinyImages function
        tinyRes = tinyImages(train_images, test_images, train_labels, test_labels)

        # Split accuracies and runtimes for saving
        for element in tinyRes[::2]:
            # Check that every second element is an accuracy in reasonable bounds
            assert (7 < element and element < 21)
        acc = np.asarray(tinyRes[::2])
        runtime = np.asarray(tinyRes[1::2])

        # Save results
        np.save(SAVEPATH + 'tiny_acc.npy', acc)
        np.save(SAVEPATH + 'tiny_time.npy', runtime)

    # Create vocabularies, and save them in the result directory
    # You need to write the buildDict function
    vocabularies = []
    vocab_idx = []  # If you have doubts on which index is mapped to which vocabulary, this is referenced here
    # e.g vocab_idx[i] will tell you which algorithms/neighbors were used to compute vocabulary i
    # This isn't used in the rest of the code so you can feel free to ignore it

    print("Making vocabularies...")

    for feature in ['sift', 'surf', 'orb']:
        for algo in ['kmeans', 'hierarchical']:
            for dict_size in [20, 50]:
                filename = 'voc_' + feature + '_' + algo + '_' + str(dict_size) + '.npy'

                if f_make_vocab:
                    print('Now making: ' + filename)
                    vocabulary = buildDict(train_images, dict_size, feature, algo)
                    np.save(SAVEPATH + filename, np.asarray(vocabulary))
                else:
                    print('Now loading: ' + filename)
                    vocabulary = np.load(SAVEPATH + filename)

                vocabularies.append(vocabulary) # A list of vocabularies (which are 2D arrays)
                vocab_idx.append(filename.split('.')[0]) # Save the map from index to vocabulary

    # Compute the Bow representation for the training and testing sets
    test_rep = []  # To store a set of BOW representations for the test images (given a vocabulary)
    train_rep = []  # To store a set of BOW representations for the train images (given a vocabulary)
    features = ['sift'] * 4 + ['surf'] * 4 + ['orb'] * 4  # Order in which features were used
    # for vocabulary generation

    if f_run_computeBOW:
        print("Running computeBOW...")

        # You need to write ComputeBow()
        for i, vocab in enumerate(vocabularies):
            for image in train_images: # Compute the BOW representation of the training set
                rep = computeBow(image, vocab, features[i]) # Rep is a list of descriptors for a given image
                train_rep.append(rep)
            np.save(SAVEPATH + 'bow_train_' + str(i) + '.npy', np.asarray(train_rep)) # Save the representations for vocabulary i
            train_rep = [] # reset the list to save the following vocabulary
            for image in test_images: # Compute the BOW representation of the testing set
                rep = computeBow(image, vocab, features[i])
                test_rep.append(rep)
            np.save(SAVEPATH + 'bow_test_' + str(i) + '.npy', np.asarray(test_rep)) # Save the representations for vocabulary i
            test_rep = [] # reset the list to save the following vocabulary

    # # test computeBOW on one image
    # print("Testing...")
    # vocab = np.load(SAVEPATH + 'voc_sift_kmeans_20.npy')
    # rep = computeBow(test_images[0], vocab, "sift")
    # print(rep)

    # Use BOW features to classify the images with a KNN classifier
    # A list to store the accuracies and one for runtimes
    knn_accuracies = []
    knn_runtimes = []

    # Use KNN_Classifier to classify images represented by a BOW, 1, 3, 6 neighbors
    # Dictionary sizes : 20, SIFT, SURF, ORB features
    # Dictionary sizes : 50, SIFT, SURF, ORB features
    # total 36 values, accuracy and runtime

    if f_run_KNN_BOW:
        print("Running KNN on BOW features...")

        # Your code below, eg:
        for i, vocab in enumerate(vocabularies):
            # load the tranRep and testRep
            train_Rep = np.load(SAVEPATH + 'bow_train_' + str(i) + '.npy')
            test_Rep = np.load(SAVEPATH + 'bow_test_' + str(i) + '.npy')

            # KNN Classify
            classify_start_time = time.time()
            predicted_labels = KNN_classifier(train_Rep, train_labels, test_Rep, 9)

            classify_time = time.time() - classify_start_time
            knn_runtimes.append(classify_time)
            accuracy = reportAccuracy(test_labels, predicted_labels)
            knn_accuracies.append(accuracy)
            print('KNN BOW - bow_test_' + str(i) + '.npy Accuracy: %.4f' % accuracy)

        np.save(SAVEPATH+'knn_accuracies.npy', np.asarray(knn_accuracies)) # Save the accuracies in the Results/ directory
        np.save(SAVEPATH+'knn_runtimes.npy', np.asarray(knn_runtimes)) # Save the runtimes in the Results/ directory

        # End code section

    # Use BOW features to classify the images with 15 Linear SVM classifiers
    lin_accuracies = []
    lin_runtimes = []
    svm_runtimes_1 = []

    lin_start_time = 0
    lin_end_time = 0
    lin_time = 0

    if f_run_LinearSVM_BOW:
        print("Running linear SVM on BOW features...")

        # Your code below
        for i in range(12):
            bow_train = np.load(SAVEPATH + 'bow_train_' + str(i) + '.npy')
            bow_test = np.load(SAVEPATH + 'bow_test_' + str(i) + '.npy')

            lin_start_time = time.time()
            # piazza suggest C=10
            lin_labels = SVM_classifier(bow_train, train_labels, bow_test, True, 1)
            lin_end_time = time.time()
            lin_time = lin_end_time - lin_start_time
            svm_runtimes_1.append(lin_time)

            svm_accuracy = reportAccuracy(test_labels, lin_labels)
            print('Linear SVM - bow_test_' + str(i) + '.npy Accuracy: %.4f' % svm_accuracy)

            lin_accuracies.append(svm_accuracy)

        # get the final classify time combined with Bow time
        for i, j in zip(knn_runtimes, svm_runtimes_1):
            final_time = i + j
            lin_runtimes.append(final_time)

        np.save(SAVEPATH+'lin_accuracies.npy', np.asarray(lin_accuracies)) # Save the accuracies in the Results/ directory
        np.save(SAVEPATH+'lin_runtimes.npy', np.asarray(lin_runtimes)) # Save the runtimes in the Results/ directory

        # End code section

    # Use BOW features to classify the images with 15 Kernel SVM classifiers
    rbf_accuracies = []
    rbf_runtimes = []
    svm_runtimes_2 = []

    rbf_start_time = 0
    rbf_end_time = 0
    rbf_time = 0

    if f_run_RBF_SVM_BOW:
        print("Running RBF SVM on BOW features...")

        # Your code below
        for i in range(12):
            bow_train = np.load(SAVEPATH + 'bow_train_' + str(i) + '.npy')
            bow_test = np.load(SAVEPATH + 'bow_test_' + str(i) + '.npy')

            rbf_start_time = time.time()
            # piazza suggests using C=10
            rbf_labels = SVM_classifier(bow_train, train_labels, bow_test, False, 1)
            rbf_end_time = time.time()
            rbf_time = rbf_end_time - rbf_start_time
            svm_runtimes_2.append(rbf_time)

            svm_accuracy = reportAccuracy(test_labels, rbf_labels)
            print('RBF SVM - bow_test_' + str(i) + '.npy Accuracy: %.4f' % svm_accuracy)

            rbf_accuracies.append(svm_accuracy)

        # get the final classify time combined with Bow time
        for i, j in zip(knn_runtimes, svm_runtimes_2):
            final_time = i + j
            rbf_runtimes.append(final_time)

        np.save(SAVEPATH + 'rbf_accuracies.npy', np.asarray(rbf_accuracies))  # Save the accuracies in the Results/ directory
        np.save(SAVEPATH + 'rbf_runtimes.npy', np.asarray(rbf_runtimes))  # Save the runtimes in the Results/ directory

        # End code section
