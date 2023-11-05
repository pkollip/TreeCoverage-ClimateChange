import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import os
states_mapping = {
    'Alabama': 1,
    'Alaska': 2,
    'Arizona': 3,
    'Arkansas': 4,
    'California': 5,
    'Colorado': 6,
    'Connecticut': 7,
    'Delaware': 8,
    'Florida': 9,
    'Georgia': 10,
    'Hawaii': 11,
    'Idaho': 12,
    'Illinois': 13,
    'Indiana': 14,
    'Iowa': 15,
    'Kansas': 16,
    'Kentucky': 17,
    'Louisiana': 18,
    'Maine': 19,
    'Maryland': 20,
    'Massachusetts': 21,
    'Michigan': 22,
    'Minnesota': 23,
    'Mississippi': 24,
    'Missouri': 25,
    'Montana': 26,
    'Nebraska': 27,
    'Nevada': 28,
    'New Hampshire': 29,
    'New Jersey': 30,
    'New Mexico': 31,
    'New York': 32,
    'North Carolina': 33,
    'North Dakota': 34,
    'Ohio': 35,
    'Oklahoma': 36,
    'Oregon': 37,
    'Pennsylvania': 38,
    'Rhode Island': 39,
    'South Carolina': 40,
    'South Dakota': 41,
    'Tennessee': 42,
    'Texas': 43,
    'Utah': 44,
    'Vermont': 45,
    'Virginia': 46,
    'Washington': 47,
    'West Virginia': 48,
    'Wisconsin': 49,
    'Wyoming': 50
}

def main():
    #train data
    train = pd.read_csv("/Users/andyle/Desktop/Fall 2023 Classes/PHIL 208/states_training_data - states.csv")
    train = train.drop(columns='Abbreviation')
    label_mapping = {'Bad': 0, 'Decent': 1, 'Good': 2}
    train['Performance (Bad/Decent/Good)'] = train['Performance (Bad/Decent/Good)'].replace(label_mapping)
    train['State'] = train['State'].replace(states_mapping)
    train = train.replace(',','', regex=True)
    #train['Population'] = train['Population'].astype(float)
    #train['Land Area (sq mi)'] = train['Land Area (sq mi)'].astype(float)
    print(train)
    #make X_train
    X_train = train.drop('Performance (Bad/Decent/Good)', axis=1)

    #make y_train
    y_train = train['Performance (Bad/Decent/Good)']

    #test data
    test = pd.read_csv("/Users/andyle/Desktop/Fall 2023 Classes/PHIL 208/states_test_data - states.csv")
    test = test.drop(columns='Abbreviation')
    test['State'] = test['State'].replace(states_mapping)
    test = test.replace(',','', regex=True)
    #test['Population'] = test['Population'].astype(float)
    #test['Land Area (sq mi)'] = test['Land Area (sq mi)'].astype(float)
    X_test = test.drop('Performance (Bad/Decent/Good)', axis=1)
    print(X_train)
    print(y_train)
    print(X_test)
    X_train_np = X_train.to_numpy()
    print(X_train_np)
    y_train_np = y_train.to_numpy()
    print(y_train_np)
    X_test_np = X_test.to_numpy()
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train_np,y_train_np)
    y_pred = knn.predict(X_test_np)
    print("Y_PRED")
    print(y_pred)
    print(len(y_pred))
    test_new = pd.read_csv("/Users/andyle/Desktop/Fall 2023 Classes/PHIL 208/states_test_data - states.csv")
    test_new['Performance (Bad/Decent/Good)'] = y_pred

    reverse_mapping = {0: 'Bad', 1: 'Decent', 2: 'Good'}
    test_new['Performance (Bad/Decent/Good)'] = test_new['Performance (Bad/Decent/Good)'].replace(reverse_mapping)
    test_new.to_csv("/Users/andyle/Desktop/Fall 2023 Classes/PHIL 208/predictions.csv")
    print(test_new)

    if __name__ == '__main__':
        main()
