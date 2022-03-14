import matplotlib.pyplot as plt
from numpy import fabs
import pandas as pd
import numpy as np
import seaborn as sns
def DataVisualization_adult(data):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(3,1)

    # Draw the density map of age under different income levels
    data.age[data.income == ' <=50K'].plot(kind = 'kde', label = '<=50K', ax = axes[0], legend = True, linestyle = '-')
    data.age[data.income == ' >50K'].plot(kind = 'kde', label = '>50K', ax = axes[0], legend = True, linestyle = '--')
    axes[0].set_title('age')

    # Draw the density map of weekly working hours under different income levels
    data['hours-per-week'][data.income == ' <=50K'].plot(kind = 'kde', label = '<=50K', ax = axes[1], legend = True, linestyle = '-')
    data['hours-per-week'][data.income == ' >50K'].plot(kind = 'kde', label = '>50K', ax = axes[1], legend = True, linestyle = '--')
    axes[1].set_title('hours-per-week')

    # Draw the density map of education number under different income levels
    data['education-num'][data.income == ' <=50K'].plot(kind = 'kde', label = '<=50K', ax = axes[2], legend = True, linestyle = '-')
    data['education-num'][data.income == ' >50K'].plot(kind = 'kde', label = '>50K', ax = axes[2], legend = True, linestyle = '--')
    axes[2].set_title('education-num')

    plt.subplots_adjust(wspace = 0.75, hspace = 0.75)
    plt.show()

    # Construct data of the number of people of various race groups under different income levels
    race = pd.DataFrame(data.groupby(by = ['race','income']).aggregate(np.size))
    race = race.reset_index()
    race.rename(columns={'age':'counts'}, inplace=True)
    race.sort_values(by = ['race','counts'], ascending=False, inplace=True)

    # Construct data of the number of relations under different income levels
    relationship = pd.DataFrame(data.groupby(by = ['relationship','income']).aggregate(np.size))
    relationship = relationship.reset_index()
    relationship.rename(columns={'age':'counts'}, inplace=True)
    relationship.sort_values(by = ['relationship','counts'], ascending=False, inplace=True)

    # Construct data on the number of sex at different income levels
    sex = pd.DataFrame(data.groupby(by = ['sex','income']).aggregate(np.size))
    sex = sex.reset_index()
    sex.rename(columns={'age':'counts'}, inplace=True)
    sex.sort_values(by = ['sex','counts'], ascending=False, inplace=True)

    # Construct data on the number of workclass at different income levels
    workclass = pd.DataFrame(data.groupby(by = ['workclass','income']).aggregate(np.size))
    workclass = workclass.reset_index()
    workclass.rename(columns={'age':'counts'}, inplace=True)
    workclass.sort_values(by = ['workclass','counts'], ascending=False, inplace=True)

    # Construct data on the number of marital status at different income levels
    marital_status = pd.DataFrame(data.groupby(by = ['marital-status','income']).aggregate(np.size))
    marital_status = marital_status.reset_index()
    marital_status.rename(columns={'age':'counts'}, inplace=True)
    marital_status.sort_values(by = ['marital-status','counts'], ascending=False, inplace=True)

    # Construct data on the number of occupation at different income levels
    occupation = pd.DataFrame(data.groupby(by = ['occupation','income']).aggregate(np.size))
    occupation = occupation.reset_index()
    occupation.rename(columns={'age':'counts'}, inplace=True)
    occupation.sort_values(by = ['occupation','counts'], ascending=False, inplace=True)

    # Set the frame scale and draw
    plt.figure(figsize = (9,5))
    sns.barplot(x = "race", y = "counts", hue = 'income', data = race)
    plt.show()

    plt.figure(figsize = (9,5))
    sns.barplot(x = "relationship", y = "counts", hue = 'income', data = relationship)
    plt.show()

    plt.figure(figsize = (9,5))
    sns.barplot(x = "sex", y = "counts", hue = 'income', data = sex)
    plt.show()

    plt.figure(figsize = (15,5))
    sns.barplot(x = "workclass", y = "counts", hue = 'income', data = workclass)
    plt.show()

    plt.figure(figsize = (15,5))
    sns.barplot(x = "marital-status", y = "counts", hue = 'income', data = marital_status)
    plt.show()

    plt.figure(figsize = (15,10))
    sns.barplot(x = "occupation", y = "counts", hue = 'income', data = occupation)
    plt.xticks(rotation = 45)
    plt.show()

def DataVisualization_letter(data):
    count = data['lettr'].value_counts()
    plt.title('Letters and corresponding counts')
    plt.plot(count, marker = 'o', markersize=3)
    plt.show()

def result(k_list, train_acc, valid_acc):
    k = k_list
    plt.title('Training accuracy between different K')
    plt.xlabel('Number of K')
    plt.ylabel('Accuracy')
    plt.plot(k, train_acc[0], marker = 'o', markersize = 3)
    plt.plot(k, train_acc[1], marker = 'o', markersize = 3)
    plt.plot(k, train_acc[2], marker = 'o', markersize = 3)
    plt.legend(['1/3 data', '2/3 data', '3/3 data'])
    plt.show()

    plt.title('Valid accuracy between different K')
    plt.xlabel('Number of K')
    plt.ylabel('Accuracy')
    plt.plot(k, valid_acc[0], marker = 'o', markersize = 3)
    plt.plot(k, valid_acc[1], marker = 'o', markersize = 3)
    plt.plot(k, valid_acc[2], marker = 'o', markersize = 3)
    plt.legend(['1/3 data', '2/3 data', '3/3 data'])
    plt.show()





