
# coding: utf-8

# In[1]:

plot(arange(5))


# In[2]:

import pandas as pd
import numpy as np
import matplotlib as plt


# In[15]:

df1 = pd.read_csv("G:/Analytics/Python/datasets/train.csv")


# In[16]:

df1


# In[17]:

df.head(10)


# In[18]:

df.summary()


# In[19]:

df.describe()


# In[21]:

df['Age'].median()


# In[23]:

df['Sex'].unique()


# In[24]:

fig = plt.pyplot.figure()
ax = fig.add_subplot(111)
ax.hist(df['Age'], bins = 10, range = (df['Age'].min(),df['Age'].max()))
plt.pyplot.title('Age distribution')
plt.pyplot.xlabel('Age')
plt.pyplot.ylabel('Count of Passengers')
plt.pyplot.show()


# In[25]:

fig = plt.pyplot.figure()
ax = fig.add_subplot(111)
ax.hist(df['Fare'], bins = 10, range = (df['Fare'].min(),df['Fare'].max()))
plt.pyplot.title('Fare distribution')
plt.pyplot.xlabel('Fare')
plt.pyplot.ylabel('Count of Passengers')
plt.pyplot.show()


# In[27]:

df.boxplot(column='Fare',return_type='axes')


# In[28]:

df.boxplot(column='Fare', by = 'Pclass') # gives box plots for 3 different classes


# In[30]:

df['Pclass'].unique()


# In[31]:

temp1 = df.groupby('Pclass').Survived.count()
temp2 = df.groupby('Pclass').Survived.sum()/df.groupby('Pclass').Survived.count()
fig = plt.pyplot.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Pclass')
ax1.set_ylabel('Count of Passengers')
ax1.set_title("Passengers by Pclass")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Pclass')
ax2.set_ylabel('Probability of Survival')
ax2.set_title("Probability of survival by class")


# In[33]:

df['Survived'].unique


# In[34]:

temp3 = pd.crosstab([df.Pclass, df.Sex], df.Survived.astype(bool)) #get the results in tabular format
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[35]:

temp3


# In[36]:

df.nrow()


# In[37]:

df.count()


# In[42]:

df['Embarked'].unique()


# In[43]:

df.head(10)


# In[45]:

df['Cabin'].unique()


# In[47]:

df['Cabin'].decsrcibe


# In[48]:

sum(df['Cabin'].isnull()) 


# In[49]:

df.head(10)


# In[50]:

df = df.drop(['Ticket','Cabin'], axis=1) 


# In[51]:

df.head(10)


# In[52]:

meanAge = np.mean(df.Age)
df.Age = df.Age.fillna(meanAge)


# In[85]:

def name_extract(word):
  return word.split(',')[1].split('.')[0].strip()


# In[86]:

name_extract("Nasser, Mrs. Nicholas" )


# In[87]:

df2 = pd.DataFrame({'Salutation':df['Name'].apply(name_extract)})


# In[88]:

df2.head(10)


# In[89]:

df = pd.merge(df, df2, left_index = True, right_index = True) # merges on index


# In[90]:

df.head()


# In[95]:

temp1 = df.groupby('Salutation').PassengerId.count()


# In[96]:

temp1


# In[98]:

df.groupby('Salutation').count()


# In[99]:

def group_salutation(old_salutation):
 if old_salutation == 'Mr':
    return('Mr')
 else:
    if old_salutation == 'Mrs':
       return('Mrs')
    else:
       if old_salutation == 'Master':
          return('Master')
       else: 
          if old_salutation == 'Miss':
             return('Miss')
          else:
             return('Others')


# In[100]:

df3 = pd.DataFrame({'New_Salutation':df['Salutation'].apply(group_salutation)})
df = pd.merge(df, df3, left_index = True, right_index = True)


# In[107]:

df3.describe()


# In[108]:

df.head()


# In[115]:

df.groupby('Sex').count()


# In[116]:

df.boxplot(column='Age', by = 'New_Salutation')


# In[118]:

df.head()


# In[124]:

table = df.pivot_table(values='Age', index=['New_Salutation'], columns=['Pclass', 'Sex'], aggfunc=np.median)


# In[122]:

table


# In[127]:

pd.crosstab([df.New_Salutation], [df.Pclass,df.Sex]) #get the results in tabular format


# In[128]:

def fage(x):
    return table[x['Pclass']][x['Sex']][x['New_Salutation']]
# Replace missing values
df['Age'].fillna(df[df['Age'].isnull()].apply(fage, axis=1), inplace=True)


# In[133]:

df.head()


# In[135]:

df.boxplot(column='Age', by = ['Sex','Pclass'])


# In[136]:

df.boxplot(column='Fare', by = ['Pclass'])


# In[137]:

dft = pd.DataFrame(np.random.rand(10,5),columns=list('ABCDE'))


# In[138]:

dft


# In[140]:

dft = pd.DataFrame(np.random.rand(10,5),columns=list('ABCDE'))


# In[141]:

dft.index = list('abcdeflght')


# In[142]:

dft.index


# In[143]:

dft.index = list('abcdeflght')

# Define cutoff value
cutoff = 0.90

for col in dft.columns: 
    # Identify index locations above cutoff
    outliers = dft[col][ dft[col]>cutoff ]

    # Browse through outliers and average according to index location
    for idx in outliers.index:
        # Get index location 
        loc = dft.index.get_loc(idx)

        # If not one of last two values in dataframe
        if loc<dft.shape[0]-2:
            dft[col][loc] = np.mean( dft[col][loc+1:loc+3] )
        else: 
            dft[col][loc] = np.mean( dft[col][loc-3:loc-1] )


# In[144]:

dft


# In[ ]:



