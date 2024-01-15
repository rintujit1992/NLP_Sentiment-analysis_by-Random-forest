#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install spacy')
get_ipython().system('pip install beautifulsoup4')
get_ipython().system('pip install textblob')


# In[2]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[3]:


# disable warning
import warnings
warnings.filterwarnings('ignore')

import pandas as pd


# In[4]:


df=pd.read_csv('https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/twitter_sentiment.csv')
df.head(5)


# In[5]:


df=pd.read_csv('https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/twitter_sentiment.csv',header=None,index_col=0)
df.head(5)


# In[6]:


df=df[[2,3]].reset_index(drop=True)


# In[7]:


df.head(5)


# In[8]:


df.columns=['sentiment','text']
df.head(5)


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[11]:


#686 null value pre processing of data
#show random no of data
df.sample(10)


# In[12]:


#many text data have less no of charecter , so drop the null value
df.dropna(inplace=True)


# In[13]:


# then find out length of text column
df['text'].apply(len)


# In[14]:


# check same length of charector repeated how many times
df['text'].apply(len).value_counts()


# In[15]:


df['text'].apply(len)>5


# In[16]:


# how many text len haiving more than 5 char & less than 5 char
sum(df['text'].apply(len)>5),sum(df['text'].apply(len)<=5)


# In[17]:


# df contain only data having len>5 & check shape of data before & after Pre processing
print(df.shape)
df=df[df['text'].apply(len)>5]
print(df.shape)


# In[18]:


# how many type of sentiment present ,repeated time or total no of each sentiment present
df['sentiment'].value_counts()


# In[19]:


##---===============PREPROCESSING BY  preprocess_kgptalkie================------##


# In[20]:


pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-reinstall


# In[21]:


# basic feature extraction | 
import preprocess_kgptalkie as ps


# In[22]:


df.head()


# In[23]:


df.columns


# In[24]:


# to get basic features of data
df=ps.get_basic_features(df)


# In[25]:


df.columns


# In[26]:


df.head()


# In[27]:


## for visialization import libary
import matplotlib.pyplot as plt
import seaborn as sns


# In[28]:


df.select_dtypes(include='number')


# In[29]:


# name of all columns present in the data frame
df.select_dtypes(include='number').columns


# In[30]:


plt.figure(figsize=(20,10))
num_columns=df.select_dtypes(include='number').columns

for index, col in enumerate(num_columns):
    plt.subplot(2,4,index+1)


# In[31]:


plt.figure(figsize=(20,10))
num_columns=df.select_dtypes(include='number').columns

for index, col in enumerate(num_columns):
    plt.subplot(2,4,index+1)
    sns.kdeplot(data=df,x=col)
    
plt.tight_layout()
plt.show()


# In[32]:


plt.figure(figsize=(20,10))
num_columns=df.select_dtypes(include='number').columns

for index, col in enumerate(num_columns):
    plt.subplot(2,4,index+1)
    sns.kdeplot(data=df,x=col,hue='sentiment',fill=True)
    
plt.tight_layout()
plt.show()


# In[33]:


df['sentiment'].value_counts().plot(kind='pie', autopct='%1.0f%%')


# In[34]:


## Word Cloud Visualization  ##
get_ipython().system('pip install wordcloud')


# In[35]:


from wordcloud import WordCloud,STOPWORDS


# In[36]:


stopwords=set(STOPWORDS)


# In[37]:


# most frequently use stop words
stopwords


# In[38]:


wordcloud=WordCloud(background_color='white',stopwords=stopwords,
                   max_words=300,max_font_size=40,scale=5).generate(str(df['text']))
plt.imshow(wordcloud)


# In[39]:


plt.figure(figsize=(40,20))
for index,sent in enumerate(df['sentiment'].unique()):
    plt.subplot(2,2,index+1)
    
    df1=df[df['sentiment']==sent]
    
    


# In[40]:


plt.figure(figsize=(40,20))
for index,sent in enumerate(df['sentiment'].unique()):
    plt.subplot(2,2,index+1)
    
    data=df[df['sentiment']==sent]['text']
    
    wordcloud=WordCloud(background_color='white',stopwords=stopwords,
              max_words=300,max_font_size=40,scale=5).generate(str(data))
    
    plt.imshow(wordcloud)
    
    plt.title(sent,fontsize=40)
    


# In[50]:


##=====------DATA CLEANING--------==========#


# In[41]:



df['text'] = df['text'].apply(lambda x: x.lower())
df['text'] = df['text'].apply(lambda x: ps.remove_urls(x))
df['text'] = df['text'].apply(lambda x: ps.remove_html_tags(x))
df['text'] = df['text'].apply(lambda x: ps.remove_special_chars(x))
df['text'] = df['text'].apply(lambda x: ps.remove_rt(x))


# In[42]:


## TRAIN TEST SPLIT  ##

from sklearn.model_selection import train_test_split


# In[43]:


X_train,X_test,y_train,y_test=train_test_split(df['text'],df['sentiment'],test_size=0.3,random_state=0)


# In[44]:


X_train.shape,X_test.shape


# In[56]:


##======-----MODEL BUILDING & PRE PROCESSING------======#


# In[ ]:


#Convert a collection of raw documents to a matrix of TF-IDF features.

#Equivalent to CountVectorizer followed by TfidfTransformer.


# In[45]:


# model building
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[46]:


# create a pipeline with the list of component

clf = Pipeline([('tfidf', TfidfVectorizer(stop_words=stopwords)), ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1))])
clf.fit(X_train, y_train)


# In[47]:


##====----Evaluation--------======##

from sklearn.metrics import classification_report


# In[48]:


y_pred=clf.predict(X_test)
print(classification_report(y_test,y_pred))


# In[49]:


# evaluation
from sklearn.metrics import accuracy_score

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))


# In[50]:


# save model
import pickle

pickle.dump(clf, open('twitter_sentiment.pkl', 'wb'))


# In[51]:


##------======Prediction Test===============---------##
clf.predict(['i love you'])


# In[53]:


clf.predict(['i will upset you'])


# In[ ]:




