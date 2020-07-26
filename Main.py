#handle, text, lang = en, retweet_count, favourite_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import FpGrowth as fp

# Till hashtag prerequisites, this part of the code is EXECUTED ONLY ONCE for better performance.
"""
data = pd.read_csv("tweets.csv")
keep_columns = ["handle", "text", "lang"]

# removing columns of data which we won't be dealing with and keeping only keep_columns
remove_columns_list = list(set(data.columns)-set(keep_columns))
data.drop(columns = remove_columns_list, inplace = True)

# keeping rows which have tweets in english
data = data[data.lang == "en"]
data.drop(columns = ["lang"], inplace=True)

data.to_csv("twitter.csv")

hillary = data[data.handle == "HillaryClinton"]
hillary.drop(columns = ["handle"], inplace=True)
hillary.to_csv("Hillary.csv")

trump = data[data.handle == "realDonaldTrump"]
trump.drop(columns = ["handle"], inplace=True)
trump.to_csv("Trump.csv")

#prerequisites

"""

pd.set_option('display.max_columns', None)

# data cleaning
clean_data = ['',"will","a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at", "be", "because", "been", "before", "being","below", "between", "both", "but", "by","can", "cant", "cannot", "could", "couldnt", "did","didn't", "do", "does", "doesnt", "doing","dont", "down", "during", "each", "few", "for", "from", "further", "had", "hadnt","has", "hasnt", "have", "havent", "having","he", "hed", "hell", "hes", "her","here", "heres", "hers", "herself", "him", "himself", "his", "how", "how's", "i","id", "ill", "im", "ive", "if","in", "into", "is", "isnt", "it","its", "its", "itself", "lets", "me","more", "most", "mustnt", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them","themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was","wasnt", "we", "wed", "well", "were", "weve", "were", "werent", "what", "whats", "when", "when's", "where", "wheres", "which","while", "who", "whos", "whom", "why","whys", "with", "wont", "would", "wouldnt","you", "youd", "youll", "youre", "youve", "your", "yours","yourself", "yourselves"]
punctuation = list(string.punctuation + "”’â€™")


def clean(x):
    # make lowercase
    x = x.lower()

    #x = x.replace("â€™","") if x.find("â€™") else x
    #x = x.replace("â†","") if x.find("â€™") else x

    # to remove punctuation
    temp = ''
    for i in x:
        if i in punctuation or i.isnumeric():
            continue
        temp += i
    x = temp

    # remove links

    if x.find("http") != -1 :
        pos = x.find("http")
        space_pos = x.find(" ", pos)
        x = x[:pos] #x[0:pos] + x[space_pos:] if space_pos != -1 else x[0:pos]

    # removing common words

    words = x.split(" ")
    temp = ""
    for i in words:
        if i not in clean_data:
            temp += i+" "

    x = temp.strip()
    return x

def preprocessing():
    transactions = []
    for i,j in data.iterrows():
        str = j.text
        str = clean(str)
        data.at[i,'text'] = str.split(" ")
        transactions.append(str.split(" "))
    data.drop(data.columns[[0]], axis = 1, inplace = True)
    data.to_csv("trial.csv")
    return transactions



"""         FP growth algorithm begins from here     """


def support_count(x):
    count = 0
    x = set(x)
    for j in data['text']:
        j = set(j)
        if (x.issubset(j)):
            count += 1
    return count

def fpgrowth(transactions, minsup):

    # generating frequent patterns

    patterns = fp.find_frequent_patterns(transactions, minsup)

    # generating association rules as {(left): ((right), confidence)}

    rules = fp.generate_association_rules(patterns, 0.7)

    rules_data = {}
    rules_data["Antecedent"] = list(rules.keys())
    rules_data["Consequent"] = [rules[i][0] for i in rules_data["Antecedent"]]

    # confidence is conditional probability that a transaction containing the LHS will also contain the RHS.
    rules_data["Confidence"] = [rules[i][1] for i in rules_data["Antecedent"]]


    for i in range(len(rules_data["Confidence"])):
        if rules_data["Confidence"][i]>1:
            rules_data["Confidence"][i]=1


    # calculating support of RHS aka consequent
    # support = count of item i in N transactions/ N transactions

    support = []
    for i in rules_data["Consequent"]:
        a = support_count(i)
        support.append(a*1.0/len(data))

    rules_data['RHS_support'] = support

    # calculating lift

    lift = []
    p = 0
    for i in range(len(rules_data["Confidence"])):
        p = rules_data["Confidence"][i] / rules_data['RHS_support'][i]
        lift.append(p)
    rules_data['Lift'] = lift

    # storing results in .csv file

    results = pd.DataFrame.from_dict(rules_data)
    return results

# running algorithm on Hillary Clinton's tweets

data = pd.read_csv("Hillary.csv")
# data = data.head(50) # to reduce number of rows while testing
transactions = preprocessing()
results = fpgrowth(transactions, 10) # 10 is the minimum support
results.to_csv("Results - Hillary.csv")
print("Results of FP Growth Algorithm for Hillary is saved in file Results - Hillary.csv")

# running algorithm on Donald Trump's tweets

data = pd.read_csv("Trump.csv")
# data = data.head(50) # to reduce number of rows while testing
transactions = preprocessing()
results = fpgrowth(transactions, 10) # 10 is the minimum support
results.to_csv("Results - Trump.csv")
print("Results of FP Growth Algorithm for Trump is saved in file Results - Trump.csv")

