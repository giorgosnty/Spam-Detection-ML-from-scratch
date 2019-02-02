import pandas as pd
import os
import collections
from math import log
import operator


def read_corpora_pu(path):

    global parts        # table that contains emails
    emails = []
    ham = []            # table containing 1 if email is ham or 0 if spam

    data_dir = os.listdir(path)         #has also "readme.txt"
    for w in data_dir:
        if "readme" in w:               #if value "readme" found,
           data_dir.remove(w)           #remove it.
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
def preprocess_data(topK):
    emails, ham = read_corpora_pu("ABSOLUTE_PATH")

    updated_emails = []
    # we need to tokenize data and compute frequency
    for message in emails:
        new_message = tokenize_message(message)
        updated_emails.append(new_message)


    # Now we need to preprocess data
    emails = updated_emails
    df = pd.DataFrame()

    df['Email'] = emails
    df['IsHam'] = ham

    # Here we split the dataframe into train,dev and test sets
    train, dev, test = split_dataset(df)

    train_emails = train['Email']
    train_ham = train['IsHam']


    #compute bag of words based on train set
    bag_of_words = compute_bag_of_words(train_emails)

    attributes = []
    counter = 0
    #select top 100 for attributes
    #but for faster execution, select top 10
    for key, value in bag_of_words.items():
        if counter==topK:
            break
        attributes.append(key)
        counter+=1

    return train,dev,test,attributes
def predict(query,tree,default =1):
    q = query['Email']  # dont use the label column for predictions
    t = dict(tree)
    tree_keys = list(t.keys())

    for key in tree_keys:
        if key in q:
            list_1 = list(tree[key].values())
            if isinstance(list_1[1], list):
                result = dict(list_1[1])
            else:
                result = list_1[1]

        else:
            list_0 = list(tree[key].values())
            if isinstance(list_0[0], list):
                result = dict(list_0[0])
            else:
                result = list_0[0]

        if isinstance(result, dict):
            return  predict(query, result)
        else:
            return result
    return default
def compute_scores(actual, predictions):
    tp = 0.0
    tn=0.0
    fp=0.0
    fn = 0.0
    tp_list = []
    fn_list = []

    zippledList = zip(actual,predictions)

    for real,prediction in zippledList:
        if real == 1 & prediction == 1:
            tp += 1.0
            tp_list.append(tp)
        elif real == 1 & prediction == 0:
            fn += 1.0
            fn_list.append(fn)
        elif real == 0 & prediction == 1:
            fp += 1.0
        else:
            tn += 1.0



    accuracy = (tp + tn) / float(tp + tn + fp + fn)

    if tp==0.0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        recall = float(tp) / float(tp + fn)
        precision = float(tp) / (tp + fp)
        f1 = 2.0 * (recall * precision) / (recall + precision)
    return accuracy,precision, recall,f1

def tokenize_message(message):
    words = []
    tokens = []

    for token in message.split():
        tokens.append(token)

    for i in range(3, len(tokens)):
        words.append(tokens[i])
    return words
def compute_bag_of_words(emails):

    voc_dict = []
    for mes in emails:
        voc_dict+=mes

    voc = collections.Counter(voc_dict)

    sorted_voc = sorted(voc.items(), key=operator.itemgetter(1),reverse=True)
    voc_dict = dict(sorted_voc)
    return voc_dict
def calculate_entropy(data):
    spam_c = 0
    ham_c = 0
    for index, row in data.iterrows():

        if row['IsHam'] == 1:
            ham_c += 1
        else:
            spam_c += 1

    length = len(data)
    # propability of ham/spam
    if length == 0:
        return 1000000
    else:
        prob_ham = float(ham_c) / len(data)
        prob_spam = float(spam_c) / len(data)

    if (prob_spam == 0) | (prob_ham == 0):
        entropy = 0
        return entropy

    entropy = -prob_ham * log(prob_ham, 2) - prob_spam * log(prob_spam, 2)
    return entropy
def calculate_information_gain(data, attribute):

    entropy_init = calculate_entropy(data)              # entropy as it is in this step

    num_of_data = len(data)

    columns = ['Email','IsHam']

    has_attribute = []
    has_not_attribute =[]

    for index,row in data.iterrows():
        temp = []
        if attribute in row['Email']:
            temp.append(row['Email'])
            temp.append(row['IsHam'])
            has_attribute.append(temp)
        else:
            temp.append(row['Email'])
            temp.append(row['IsHam'])
            has_not_attribute.append(temp)



    #propabilities of the separation based on the attribute
    prop_has_attr = float(len(has_attribute))/float(num_of_data)
    prop_has_not_attr= float(len(has_not_attribute))/float(num_of_data)


    #entropy of the separation parts
    df_has = pd.DataFrame(has_attribute,columns=columns)
    df_has_not = pd.DataFrame(has_not_attribute,columns=columns)

    entropy_has_attr = calculate_entropy(df_has)
    entropy_has_not_attr = calculate_entropy(df_has_not)

    inforrmation_gain= entropy_init - (entropy_has_attr*prop_has_attr+entropy_has_not_attr*prop_has_not_attr)

    return inforrmation_gain
# this is the heuristic function computing the information gain
def select_best_attribbute(data,attributes):

    max_info = -100.0
    best_attribute = ""
    for a in attributes:
        info_gain = calculate_information_gain(data,a)
        if info_gain>max_info:
            max_info = info_gain
            best_attribute = a

    return best_attribute
def splitBasedIn(data,attribute):
    columns = ['Email', 'IsHam']
    subDataPos = pd.DataFrame(columns=columns)
    subDataNeg = pd.DataFrame(columns=columns)

    for index, row in data.iterrows():
        if attribute in row['Email']:
            subDataPos = subDataPos.append(row)
        else:
            subDataNeg = subDataNeg.append(row)

    return subDataPos, subDataNeg

def train_id3(data,default,attributes):

    tree = {}
    if len(data)== 0 :
        print("No more data")
        return default
    elif data_is_pure(data)==1:                        # if every email is ham-only or spam-only
        print("Pure node")
        return default
    elif len(attributes)==0:                            # if attributes vector has no more attributes

        return most_frequent_category(data)
    else:
        bestAttr = select_best_attribbute(data, attributes)
        bestAttr = str(bestAttr)
        print("Best Attribute: "+str(bestAttr))
        tree[bestAttr] = {}
        remaining_attributes = [i for i in attributes if i != bestAttr]
        print(remaining_attributes)

        # here split dataset and recursively call the id3

        subDataPos, subDataNeg = splitBasedIn(data, bestAttr)

        subtreeL = train_id3(subDataPos, most_frequent_category(data), remaining_attributes)
        subtreeR = train_id3(subDataNeg, most_frequent_category(data), remaining_attributes)

        tree[bestAttr]['0'] = subtreeR
        tree[bestAttr]['1'] = subtreeL

    return tree

def data_is_pure(data):


    counterPurity = data['IsHam'].value_counts()                        # counts how many 'IsHam' , and how many from the rest . (here, we have only 2 categories, so its gonna count 'IsHam' and the second)
    counterPurity = counterPurity.sort_index(0,None,True)               # sorts Series object counterPurity by index  (start=0, axis=None ,ascending = True).

    if len(counterPurity) == 1:
        category = counterPurity.first_valid_index()
        if category == 1:
            return 1                                                    # means all values are the same and pure, so return 0 as expected
        else:
            return 0
    else:
        return -1                                                        # else return 1 (not expected)
def most_frequent_category(data):


    lisT = data['IsHam'].value_counts()
    lisT = lisT.sort_index(0,None,True)
    countSpam = lisT[0]
    countHam = len(data) - countSpam
    del lisT
    max_freq =  max(countSpam,countHam)
    if  max_freq == countSpam:
        return 0
    elif max_freq == countHam:
        return 1
def show_AccuracyPrecisionRecallF1(acc,pre,re,f1,datatag):
    print("\n\n")
    print('\033[1m'+"Accuracy  of "+datatag+" is : " + "{:.0%}".format(acc))
    print('\033[1m'+"Precision of "+datatag+" is : " + "{:.0%}".format(pre))
    print('\033[1m'+"Recall    of "+datatag+" is : " + "{:.0%}".format(re))
    print('\033[1m'+"f1        of "+datatag+" is : " + "{:.0%}".format(f1))

def plot_validation_scores(Y,predictions):
    scoreAccuracy = []
    scoreRecall = []
    scoreF1 = []
    scorePrecision = []

    for i in range(len(Y)):
        accuracy, precision, recall, f1 = compute_scores(Y[:i+1],predictions[:i+1])
        #print(accuracy)
        scoreAccuracy.append(accuracy)
        scoreRecall.append(recall)
        scoreF1.append(f1)
        scorePrecision.append(precision)


def main(topAttr):
    print("ID3 Classifier \nBy: \nTrikalis Christos 3140205\nNtymenos Georgios 3140147\n")
    train,dev,test,attributes = preprocess_data(topAttr)

    # id3 algorithm excecution
    tree = train_id3(train, 2, attributes)

    # predict and calculate accuracy, recall, precision
    predictions_test = []
    for i in range(0, len(test)):
        res = predict(test.iloc[i], tree, 1)
        predictions_test.append(res)

    predictions_train = []
    for i in range(0, len(train)):
        res = predict(train.iloc[i], tree, 1)
        predictions_train.append(res)

    Y_test = test["IsHam"]
    Y_train = train["IsHam"]

    accuracy, precision, recall,f1 = compute_scores(list(Y_test), predictions_test)
    show_AccuracyPrecisionRecallF1(accuracy,precision,recall,f1,"test data")

    plot_validation_scores(list(Y_test),predictions_test)

    accuracy_train, precision_train, recall_train, f1_train = compute_scores(list(Y_train), predictions_train)
    show_AccuracyPrecisionRecallF1(accuracy_train, precision_train, recall_train, f1_train, "train data")

    plot_validation_scores(list(Y_train), predictions_train)

main(10)# k = 10