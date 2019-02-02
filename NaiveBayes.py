import pandas as pd
import os
import collections
import operator


def read_corpora_pu(path):
    # table that contains emails
    global parts
    emails = []
    # table containing 1 if email is ham or 0 if spam
    ham = []

    data_dir = os.listdir(path)
    for w in data_dir:
        if "readme" in w:  # if value "readme" found,
            data_dir.remove(w)  # remove it.
    print(data_dir)

    for i in data_dir:
        if os.path.isdir(path):
            parts = os.listdir(path + "\\" + i)
        for part in parts:

            email_files = os.listdir(path + "\\" + i + "\\" + part)
            for e in email_files:
                # check if this email is spam and insert 0 or 1 to ham
                if "spmsg" in e:
                    ham.append(0)
                else:
                    ham.append(1)

                # now read the email
                path_of_file = path + "\\" + i + "\\" + part
                f = open(path_of_file + "\\" + e, 'r')
                emails.append(f.read())
                f.close()

    print("Dataset was successfully loaded!")

    return emails, ham


def split_dataset(df):
    split_1 = int(0.7 * len(df))
    split_2 = int(0.9 * len(df))
    train = df[:split_1]
    dev = df[split_1:split_2]
    test = df[split_2:]
    return train, dev, test


def preprocess_data():
    # here we need the path for the pu_corpora_public folder
    emails, ham = read_corpora_pu("ABSOLUTE_PATH")

    updated_emails = []
    # we tokenize data
    for message in emails:
        new_message = tokenize_message(message)
        updated_emails.append(new_message)

    # Now we preprocess data
    emails = updated_emails

    df = pd.DataFrame()

    df['Email'] = emails

    df['IsHam'] = ham

    # Here we split the dataframe into train,dev and test sets
    train, dev, test = split_dataset(df)

    return train, test


# it computes the frequency for every word in our data
def compute_bag_of_words(data):
    text_data = data['Email']
    voc_dict = []
    for mes in text_data:
        voc_dict += mes
    voc = collections.Counter(voc_dict)
    sorted_voc = sorted(voc.items(), key=operator.itemgetter(1), reverse=True)

    voc_dict = dict(sorted_voc)
    return voc_dict


def selectAttributes(data, k):
    bow = compute_bag_of_words(data)
    attributes = []
    counter = 0
    # select top k attributes
    for key, value in bow.items():
        if counter == k:
            break
        attributes.append(key)
        counter += 1

    return attributes


def tokenize_message(message):
    words = []
    tokens = []

    for token in message.split():
        tokens.append(token)

    for i in range(3, len(tokens)):
        words.append(tokens[i])
    return words


# here we compute the frequencies of every word given that the message is ham or spam
def compute_freq_for_words(data, topAttr):
    voc_ham = []
    voc_spam = []
    voc = []

    for index, row in data.iterrows():
        example = row['Email']
        if row['IsHam'] == 1:
            voc_ham += example
        else:
            voc_spam += example
        voc += example

    vocT = collections.Counter(voc)
    sorted_voc = sorted(vocT.items(), key=operator.itemgetter(1), reverse=True)

    counterHam = {}
    counterSpam = {}

    attributes = selectAttributes(data, topAttr)

    length_of_attr = len(attributes)

    for w in attributes:

        counterh = 0
        counters = 0
        for mes in voc_ham:
            if w in mes:
                counterh += 1
        counterHam[w] = float(counterh)

        for mes in voc_spam:
            if w in mes:
                counters += 1

        counterSpam[w] = float(counters)  # +1.0)/(len(voc_spam)+length_of_attr)

    return counterSpam, counterHam, attributes, len(voc_ham), len(voc_spam), length_of_attr


def decision_for_example(example, attributes, spam_prop, ham_prop, prop_ham_class, prop_spam_class, lenH, lenS,
                         lenAttr):
    product_spam = 1.0
    product_ham = 1.0

    for w in attributes:
        if w in example:
            product_ham *= (ham_prop.get(w) + 1.0) / (lenH + lenAttr)  # Laplace
            product_spam *= (spam_prop.get(w) + 1.0) / (lenS + lenAttr)  # Laplace
        else:
            product_ham *= 1 - ham_prop.get(w)  # /(lenH+lenAttr)
            product_spam *= 1 - spam_prop.get(w)  # /(lenS+lenAttr)

    product_ham *= prop_ham_class

    product_spam *= prop_spam_class

    if product_ham > product_spam:
        return 1
    else:
        return 0


def run_naive_bayes(topAttr):
    print("Naive Bayes Classifier \nBy: \nTrikalis Christos 3140205\nNtymenos Georgios 3140147\n")
    train, test = preprocess_data()

    prop_ham_class, prop_spam_class = compute_propabilities(train)

    # lists that contain the propability for every word given they belong to a ham or spam message
    spam_prop, ham_prop, attributes, lenVocH, lenVocS, lenAttr = compute_freq_for_words(train, topAttr)

    decisions_train = []
    # for each example
    for index, row in train.iterrows():
        example = row['Email']
        # decision for a particular example
        decision = decision_for_example(example, attributes, spam_prop, ham_prop, prop_ham_class, prop_spam_class,
                                        lenVocH, lenVocS, lenAttr)
        decisions_train.append(decision)
    actual = list(train['IsHam'])
    accuracy, precision, recall, f1 = predict(actual, decisions_train)
    printScores(accuracy, precision, recall, f1, "train")

    plot_validation_scores(actual, decisions_train)

    decisions_test = []
    # for each message
    for index, row in test.iterrows():
        example = row['Email']
        # decision for a particular example
        decision = decision_for_example(example, attributes, spam_prop, ham_prop, prop_ham_class, prop_spam_class,
                                        lenVocH, lenVocS, lenAttr)
        decisions_test.append(decision)
    actual_test = list(test['IsHam'])
    accuracy, precision, recall, f1 = predict(actual_test, decisions_test)
    printScores(accuracy, precision, recall, f1, "test")

    plot_validation_scores(actual, decisions_test)


def compute_propabilities(data):
    ham_c = 0
    spam_c = 0
    total_data = len(data)

    for index, row in data.iterrows():

        if row['IsHam'] == 1:
            ham_c += 1
        else:
            spam_c += 1

    propability_ham = float(ham_c) / total_data
    propability_spam = float(spam_c) / total_data

    return propability_ham, propability_spam


def plot_validation_scores(Y, predictions):
    scoreAccuracy = []
    scoreRecall = []
    scoreF1 = []
    scorePrecision = []

    for i in range(len(Y)):
        accuracy, precision, recall, f1 = predict(Y[:i + 1], predictions[:i + 1])
        scoreAccuracy.append(accuracy)
        scoreRecall.append(recall)
        scoreF1.append(f1)
        scorePrecision.append(precision)


def predict(actual, predictions):
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    zippledList = zip(actual, predictions)

    for real, prediction in zippledList:
        if real == 1 & prediction == 1:
            tp += 1.0
        elif real == 1 & prediction == 0:
            fn += 1.0
        elif real == 0 & prediction == 1:
            fp += 1.0
        else:
            tn += 1.0

    # print("tp+tn+fn+fp = "+str(tp + tn + fp + fn))
    accuracy = (tp + tn) / float(tp + tn + fp + fn)

    if tp == 0.0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:

        recall = float(tp) / float(tp + fn)
        precision = float(tp) / (tp + fp)
        f1 = 2.0 * (recall * precision) / (recall + precision)

    # graph(acc)

    return accuracy, precision, recall, f1


def printScores(acc, pre, re, f1, datatag):
    print("\n\n")
    print('\033[1m' + "Accuracy  of " + datatag + " is : " + "{:.0%}".format(acc))
    print('\033[1m' + "Precision of " + datatag + " is : " + "{:.0%}".format(pre))
    print('\033[1m' + "Recall    of " + datatag + " is : " + "{:.0%}".format(re))
    print('\033[1m' + "f1        of " + datatag + " is : " + "{:.0%}".format(f1))


run_naive_bayes(50)  # k = 50