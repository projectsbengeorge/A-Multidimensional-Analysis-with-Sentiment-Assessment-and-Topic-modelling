#!/usr/bin/env python
# coding: utf-8

# ### 1. Importing Libraries

# In[1]:



import pandas as pd
import numpy as np
import csv
import re
import nltk
import string
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import enchant
import emoji
from googletrans import Translator
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from enchant.checker import SpellChecker
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from langid.langid import LanguageIdentifier,model
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



# In[2]:


#Comments -- Read csv file and convert that to dataframes -- 

encodings_to_try = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']

for encoding in encodings_to_try:
    try:
        df = pd.read_csv('D:/amazon_review_data/Amazon_reviews.csv', encoding=encoding)
        print(f"File successfully read using {encoding} encoding.")
        break  # Break the loop if reading is successful
    except UnicodeDecodeError:
        print(f"Failed to read using {encoding} encoding.")


# ### 2. Analyse the data to find annomalies

# In[3]:


print(" \nBefore removing annomalies from DataFrame : \n\n",
      df.isnull().sum())
print("Total number of rows = ",len(df))
df.sample(n = 50)
print(df.describe())


# ###  3. Data cleaning
# 
# 
# After data exploration I found some anlomalies in  the data such as below.<br>
# 
# 1. Out of 30660 row, 21 rows are duplicates.<br>
# 2. In Review_title column: 6 null value is discovered.<br>
# 3. In Reviewer_name column: 1 null value is discovered.<br>
# 4. In Review column: 823 null value is discovered.<br>
# 5. Reviews with only special characters in both review sections and titles.<br>
# 6. Fill the empty review ratings with zeros.<br>
# 7. Need to trim the rating values to numberic only values.<br>
# 8. Transalate the foreign languages to English using Google API translator python library

# In[4]:


cleaned_df = df

#Comments -- Concat Review Title and Reviews into coloumn name "Title_and_Reviews" --
cleaned_df["Title_and_Reviews"] = cleaned_df['Review_title'].astype(str) +" "+ cleaned_df["Review"]


# #### 3.1 Remove duplicate rows

# In[5]:



cleaned_df = cleaned_df.drop_duplicates()


# #### 3.2 Remove Null values 

# In[6]:


cleaned_df = cleaned_df.dropna()
cleaned_df = cleaned_df.reset_index(drop=True)


# #### 3.3 Intial Cleaning
# 
# 1. Checking if there are any foreign languages in the text by analyzing the spelling check.<br>
# 2. Need to trim the rating values to numberic only values.<br>
# 3. Remove reviews with only special characters in review sections.<br> 
# 4. Transalate the foreign languages to English using Google API translator python library.<br>
# 

# In[7]:




#Comments -- Trimming the rating values to numberic only values-- 

for a in range (0,len(cleaned_df['Overall_product_rating'])):
    cleaned_df['Overall_product_rating'][a] = cleaned_df['Overall_product_rating'][a].replace(' out of 5', '').strip()
    
for b in range (0,len(cleaned_df['Overall_product_rating_count'])):
    cleaned_df['Overall_product_rating_count'][b] = cleaned_df['Overall_product_rating_count'][b].replace(' global ratings', '').strip() 
    if ',' in cleaned_df['Overall_product_rating_count'][b]:
        cleaned_df['Overall_product_rating_count'][b] = cleaned_df['Overall_product_rating_count'][b].replace(',', '').strip()    
    
for c in range (0,len(cleaned_df['Review_Rating'])):
    cleaned_df['Review_Rating'][c] = cleaned_df['Review_Rating'][c].replace(' out of 5 stars', '').strip() 

for d in range (0,len(cleaned_df['Review_helpful_rating'])):
    cleaned_df['Review_helpful_rating'][d] = cleaned_df['Review_helpful_rating'][d].replace(' people found this helpful', '').strip() 
    if cleaned_df['Review_helpful_rating'][d] == "One person found this helpful":   
        cleaned_df['Review_helpful_rating'][d] = cleaned_df['Review_helpful_rating'][d].replace('One person found this helpful', '1').strip()
        

        
cleaned_df['Overall_product_rating'] = pd.to_numeric(cleaned_df['Overall_product_rating'], errors='coerce')
cleaned_df['Overall_product_rating_count'] = pd.to_numeric(cleaned_df['Overall_product_rating_count'], errors='coerce') 
cleaned_df['Review_Rating'] = pd.to_numeric(cleaned_df['Review_Rating'], errors='coerce') 
cleaned_df['Review_helpful_rating'] = pd.to_numeric(cleaned_df['Review_helpful_rating'], errors='coerce')
    
#Comments -- Removing special charchters from Review title and Reviews --
#Comments -- Transalating to English --
#Comments -- Checking if there are any foreign languages 

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

def is_english(text):
    lang, _ = identifier.classify(text)
    return lang == 'en'

def translate_to_english(text):
    translator = Translator()
    translated = translator.translate(text, src='auto', dest='en')
    return translated.text

# Apply translation and language detection to the whole column
cleaned_df['Title_and_Reviews'] = cleaned_df['Title_and_Reviews'].apply(lambda x: x if is_english(x) else translate_to_english(x))
   
    

    
cleaned_df = cleaned_df.drop_duplicates()
cleaned_df = cleaned_df.dropna()
cleaned_df = cleaned_df.reset_index(drop=True)


# #### Data after cleaning 

# In[156]:


print(" \nAfter removing null values from DataFrame : \n\n",
      cleaned_df.isnull().sum())
print("Total number of rows = ",len(cleaned_df))
print(cleaned_df.describe())
cleaned_df.sample(n = 50)


# ###  Feature extraction

# In[9]:


# calculating the Character Count in the Reviews
cleaned_df['char_count'] = cleaned_df['Title_and_Reviews'].apply(len)


# calculating the Word Count
cleaned_df['word_count'] = cleaned_df['Title_and_Reviews'].apply(lambda x: len(x.split()))

# Calculating the Word Density
cleaned_df['word_density'] = cleaned_df['char_count'] / (cleaned_df['word_count']+1)


punctuation = string.punctuation
# Calculating the Punctuation Count
cleaned_df['punctuation_count'] = cleaned_df['Title_and_Reviews'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation)))


## lets summarize the Newly Created Features
cleaned_df[['char_count','word_count','word_density','punctuation_count']].describe()

cleaned_df.sample(n = 50)


# ### Sentiment analysis

# In[10]:


# init the sentiment analyzer
sia = SentimentIntensityAnalyzer()
#Comments -- calculate the Polarity of the Reviews using vader algorithm -- 


# Define a function to get sentiment score
def get_sentiment_score(text):
    return sia.polarity_scores(text)["compound"]

# Define a function to get polarity label
def get_polarity_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply the sentiment score function to the 'Review' column and store scores in a new column
cleaned_df['Sentiment_Score'] = cleaned_df['Title_and_Reviews'].apply(get_sentiment_score)

# Apply the polarity label function to the sentiment scores and store labels in another new column
cleaned_df['Polarity'] = cleaned_df['Sentiment_Score'].apply(get_polarity_label)


# In[ ]:





# ### Classification of reviews

# In[11]:


# Shorten model names
cleaned_df['Short_Model_Name'] = cleaned_df['Model'].apply(lambda x: x[:30])


# Convert a column to lowercase
cleaned_df['Title_and_Reviews'] = cleaned_df['Title_and_Reviews'].str.lower()


def classify_review(review_text):
    
       # Lists of service-related words and feature words
    service_keywords = ["service", "staff", "efficient", "delivery","delivered","amazon",
                    "customer service","replacement","service center","packing","packaging","arrived",
                    "refund","Shipping","return process","warranty","assistance","complaint",
                    "live chat","chat support","helpdesk","support team","packed","delivery boy","good condition","appario",
                    "secure package","package","customer care","cs team","delivery experience","delivery agent","exchange",
                    "damaged","shipment","box"] 

    product_keywords = ["samsung","oneplus","apple","xiaomi","camera", "microphone", "battery",
                     "display","fingerprint","sensor","charging","charge","performance","processor",
                    "ram","network","bluetooth","gps","upgrade","cam","screen","photography","speaker","android","selfie",
                       "heat","hang","internet","game","gaming","brand","finger print",
                       "face unlock","5g","4g","video"]  

    product_count = sum(1 for keyword in product_keywords if keyword.lower() in review_text.lower())
    service_count = sum(1 for keyword in service_keywords if keyword.lower() in review_text.lower())

    WE :
        return "Product Review"
    elif service_count > product_count:
        return "Service Review"
    elif len(review_text.split()) < 15 and any(word in review_text for word in service_keywords):
        
        return "Service Review"
    elif len(review_text.split()) >= 15 and sum(1 for word in service_keywords if word in review_text) >= 2:
        
        return "Service Review"
    elif any(feature in review_text for feature in product_keywords):
        return "Product Review"
    else:
        return "Product Review"


# Apply the classification function to the 'reviews' column
cleaned_df['classification'] = cleaned_df['Title_and_Reviews'].apply(classify_review)


# ###  4. Visualization

# In[12]:



def top_n_ngram(corpus,n = None,ngram = 1):
    vec = CountVectorizer(stop_words = 'english',ngram_range=(ngram,ngram)).fit(corpus)
    bag_of_words = vec.transform(corpus) #count of  all the words for each one of the review
    sum_words = bag_of_words.sum(axis =0) #count of all the word in the review
    words_freq = [(word,sum_words[0,idx]) for word,idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq,key = lambda x:x[1],reverse = True)
    return words_freq[:n]


# In[13]:




common_words = top_n_ngram(cleaned_df['Title_and_Reviews'], 10,1)
df = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
plt.figure(figsize =(10,5))
df.groupby('ReviewText').sum()['count'].sort_values(ascending=False).plot(
kind='barh', title='Top 10 unigrams in review')
print('')


# In[14]:


common_words = top_n_ngram(cleaned_df['Title_and_Reviews'], 20,2)
df = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
plt.figure(figsize =(10,5))
df.groupby('ReviewText').sum()['count'].sort_values(ascending=False).plot(
kind='barh', title='Top 20 bigrams in review')
print('')


# In[15]:


common_words = top_n_ngram(cleaned_df['Title_and_Reviews'], 20,3)
df = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
plt.figure(figsize =(10,5))
df.groupby('ReviewText').sum()['count'].sort_values(ascending=False).plot(
kind='barh', title='Top 20 Trigrams in review')
print('')


# In[16]:



px.histogram(cleaned_df, x = cleaned_df['Review_Rating'], color = cleaned_df['Review_Rating'])


# In[17]:



# Group the data by "Company" and "Short_Model_Name" and count the number of reviews for each combination
grouped_data = cleaned_df.groupby(['Company', 'Short_Model_Name']).size().reset_index(name='Review_Count')

# Create a pivot table to restructure the data for plotting
pivot_table = grouped_data.pivot(index='Short_Model_Name', columns='Company', values='Review_Count')

# Create a bar chart to visualize the distribution
plt.figure(figsize=(10, 6))
pivot_table.plot(kind='barh', stacked=True)
plt.xlabel('Model')
plt.ylabel('Number of Reviews')
plt.title('Distribution of Reviews for Each Model Across Companies')
plt.xticks(rotation=0)
plt.legend(title='Company')
plt.show()


# In[18]:




#Comments -- created a new dataframe with only unique values from model and Overall prodicy rating coloumns --

model_glb_rating_df = pd.DataFrame(None)
model_glb_rating_df['Model'] = pd.DataFrame(cleaned_df.Model.dropna().unique(), columns=['Model'])
model_glb_rating_df['Amazon_Global_ratings'] = pd.DataFrame(cleaned_df.Overall_product_rating_count.dropna().unique(), columns=['Global_ratings'])
model_glb_rating_df['Amazon_Global_ratings'] = model_glb_rating_df['Amazon_Global_ratings'].astype(int)

# Sort the DataFrame by Amazon_Global_ratings in ascending order
model_glb_rating_df = model_glb_rating_df.sort_values(by='Amazon_Global_ratings', ascending=False)


# Shorten model names
model_glb_rating_df['Short_Model'] = model_glb_rating_df['Model'].apply(lambda x: x[:30])

# Set a blue color
blue_color = '#1f77b4'

# Create a horizontal bar plot using Seaborn
plt.figure(figsize=(10, 10))
ax = sns.barplot(x='Amazon_Global_ratings', y='Short_Model', data=model_glb_rating_df, color=blue_color)

plt.title('Amazon Global Ratings by Model')
plt.xlabel('Amazon Global Ratings')
plt.ylabel('Model')

plt.show()


# In[19]:




# Calculate the averages for each column
averages = cleaned_df.groupby('Company')[['char_count', 'word_count', 'word_density', 'punctuation_count']].mean().reset_index()
averages.columns = ['Company', 'avg_char_count', 'avg_word_count', 'avg_word_density', 'avg_punctuation_count']

# Set a seaborn color palette
palette = sns.color_palette("pastel")

# Create bar graphs
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Average Metrics by Company')

# Average Character Count
axs[0, 0].bar(averages['Company'], averages['avg_char_count'], color=palette[0])
axs[0, 0].set_title('Average Character Count')
axs[0, 0].set_ylabel('Character Count')

# Average Word Count
axs[0, 1].bar(averages['Company'], averages['avg_word_count'], color=palette[1])
axs[0, 1].set_title('Average Word Count')
axs[0, 1].set_ylabel('Word Count')

# Average Word Density
axs[1, 0].bar(averages['Company'], averages['avg_word_density'], color=palette[2])
axs[1, 0].set_title('Average Word Density')
axs[1, 0].set_ylabel('Word Density')

# Average Punctuation Count
axs[1, 1].bar(averages['Company'], averages['avg_punctuation_count'], color=palette[3])
axs[1, 1].set_title('Average Punctuation Count')
axs[1, 1].set_ylabel('Punctuation Count')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# In[20]:


# Convert 'Review_helpful_rating' column to numeric
cleaned_df['Review_helpful_rating'] = pd.to_numeric(cleaned_df['Review_helpful_rating'], errors='coerce')

# Calculate average Review_helpful_rating for each model
averages = cleaned_df.groupby('Model')['Review_helpful_rating'].mean()

# Set a seaborn color palette
palette = sns.color_palette("pastel")

# Create a horizontal bar chart for average Review_helpful_rating by model
plt.figure(figsize=(10, 10))
ax = sns.barplot(x=averages.values, y=averages.index, ci=None, palette=palette)
plt.title('Average Review Helpful Rating by Model')
plt.xlabel('Average Review Helpful Rating')
plt.ylabel('Model')


plt.show()


# In[21]:



# # Define a regular expression pattern to match emoticons
# emoticon_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002700-\U000027BF]+'

# # Function to count emoticons in a text
# def count_emoticons(text):
#     emoticons = re.findall(emoticon_pattern, text)
#     return len(emoticons)



# # Apply the function to the 'Title_and_Reviews' column
# cleaned_df['Emoticon_Count'] = cleaned_df['Title_and_Reviews'].apply(count_emoticons)

# # Group by 'Model' and sum the emoticon counts
# sum_emoticons_by_model = cleaned_df.groupby('Model')['Emoticon_Count'].sum().reset_index()

# # Shorten model names
# sum_emoticons_by_model['Short_Model'] = sum_emoticons_by_model['Model'].apply(lambda x: x[:20])


# # Set a single blue color
# single_color = '#1f77b4'

# # Create a horizontal bar plot
# plt.figure(figsize=(10, 8))
# ax = sns.barplot(x='Emoticon_Count', y='Short_Model', data=sum_emoticons_by_model, ci=None, color=single_color)
# plt.title('Total Number of Emoticons in Title and Reviews by Model')
# plt.xlabel('Total Number of Emoticons')
# plt.ylabel('Model')

# plt.show()



# In[22]:


# Sort the data by Company and Review_Rating in descending order
sorted_df = cleaned_df.sort_values(by=['Company', 'Review_Rating'], ascending=[True, False])

# Create a histogram with different colors for each company's review ratings
fig = px.histogram(sorted_df, x='Review_Rating', color='Review_Rating', facet_col='Company')

# Show the plot
fig.show()



# In[23]:


#Comments -- Wordcloud Visualization --

def generate_word_cloud(text):
    # Specify a TrueType font file that is available on your system
 

    # Create a WordCloud object with the specified font path

    wordcloud = WordCloud(width=2000, height=2000, background_color='black').generate(text)
    # Display the generated word cloud using matplotlib
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.title("Words from Reviews", fontsize = 18)
    plt.show()


# Concatenate all the text rows into a single string
combined_text = ' '.join(cleaned_df['Title_and_Reviews'].tolist())

# Call the function to generate the word cloud
generate_word_cloud(combined_text)


# In[ ]:





# In[ ]:





# ## Analysis and findings

# ### product and service overall bar chart

# In[152]:


import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Group the DataFrame by classification (product or service) and calculate review counts
review_counts = cleaned_df.groupby('classification')['classification'].count()
product_reviews = review_counts.get('Product Review', 0)
service_reviews = review_counts.get('Service Review', 0)

# Define custom colors for both bars
color_product = 'skyblue'
color_service = 'lightgreen'

# Set bar width
bar_width = 0.2

# Create a slim horizontal stacked bar chart for total reviews and service reviews
plt.figure(figsize=(10, 0.7))
plt.barh(['Total Reviews'], [product_reviews], color=color_product, height=bar_width, label='Product Reviews')
plt.barh(['Total Reviews'], [service_reviews], left=[product_reviews], color=color_service, height=bar_width, label='Service Reviews')

# Create custom legend using matplotlib.patches.Patch objects
legend_labels = [
    Patch(facecolor=color_product, label=f'Product Reviews ({product_reviews})'),
    Patch(facecolor=color_service, label=f'Service Reviews ({service_reviews})')
]

plt.title(f"Total Service Reviews and Product reviews")
plt.xlabel('Count')

# Place the custom legend outside the bar chart
plt.legend(handles=legend_labels, loc='upper right', bbox_to_anchor=(1.29, 1))

plt.show()


# In[24]:


# Group the data and calculate the frequency
grouped_data = cleaned_df.groupby(['Company', 'classification']).size().reset_index(name='count')

# Define a light color palette
light_palette = sns.color_palette("pastel")

# Create the bar plot using Seaborn
plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")
bar_plot = sns.barplot(data=grouped_data, x='Company', y='count', hue='classification', palette=light_palette)
plt.xlabel('Company')
plt.ylabel('Frequency')
plt.title('Product and Service Review Classification by Company')
plt.tight_layout()
plt.legend(title='Classification')

# Annotate the bars with the exact numbers
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.0f'), 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='center', 
                      xytext=(0, 9), 
                      textcoords='offset points')

# Show the bar plot with annotations
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ### sentiment overall pie digram

# In[25]:


# Calculate the count of each polarity label
polarity_counts = cleaned_df['Polarity'].value_counts()

# Define lighter colors for each polarity label
light_colors = ['#88d498', '#e28383', '#c7c7c7']  # Light green, light red, light gray

# Create a pie chart with light colors
plt.figure(figsize=(8, 6))
plt.pie(polarity_counts, labels=polarity_counts.index, autopct='%1.1f%%', startangle=140, colors=light_colors)
plt.title('Distribution of Sentiments of all Reviews')
plt.axis('equal')
plt.show()


# ### distribution of sentiment score graph

# In[26]:


# Create a histogram with normal blue bars to visualize the distribution of sentiment scores
plt.figure(figsize=(8, 6))
plt.hist(cleaned_df['Sentiment_Score'], bins=10, color='#6F8FAF',edgecolor='black')
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()


# ### average sentimnet score by review rating individual bar chart 

# In[27]:



# Group data by Review_Rating and calculate average Sentiment_Score
average_sentiment_by_rating = cleaned_df.groupby('Review_Rating')['Sentiment_Score'].mean()


# Define a list of distinct colors for each review rating
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create a bar chart with distinct colors for each bar
plt.figure(figsize=(8, 6))
bars = plt.bar(average_sentiment_by_rating.index, average_sentiment_by_rating.values, color=color_palette)


plt.title('Average Sentiment Score by Review Rating')
plt.xlabel('Review Rating')
plt.ylabel('Average Sentiment Score')
plt.xticks(rotation=0)
plt.show()


# ### average distribution of sentiment categories by each of the models (stacked bar chart)

# In[140]:


# Shorten model names
cleaned_df['Short_Model_Name'] = cleaned_df['Model'].apply(lambda x: x[:30])

# Group data by Model and Sentiment_Category and calculate average count
average_sentiment_by_model = cleaned_df.groupby(['Short_Model_Name', 'Polarity']).size().unstack().fillna(0)
average_sentiment_by_model = average_sentiment_by_model.div(average_sentiment_by_model.sum(axis=1), axis=0)

# Define a list of custom colors
custom_colors = ['#AA786E','#1f77b4', '#2ca02c']



# Create a stacked bar chart with custom colors
plt.figure(figsize=(10, 6))
average_sentiment_by_model.plot(kind='barh', stacked=True, color=custom_colors)
plt.title('Average Distribution of Sentiment Categories by Model')
plt.xlabel('Sentiment Proportion')
plt.ylabel('Model')
plt.xticks(rotation=0)
plt.legend(title='Sentiment Category')
plt.show()


# ### word cloud for negative sentiment 

# In[150]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Filter data for positive sentiment
positive_reviews = cleaned_df[cleaned_df['Polarity'] == 'Positive']['Title_and_Reviews'].tolist()
positive_text = ' '.join(positive_reviews)

# Create word clouds for positive and negative sentiment
positive_wordcloud = WordCloud(width=400, height=400, background_color='white').generate(positive_text)

# Filter data for negative sentiment
negative_reviews = cleaned_df[cleaned_df['Polarity'] == 'Negative']['Title_and_Reviews'].tolist()
negative_text = ' '.join(negative_reviews)

# Create word clouds for positive and negative sentiment
negative_wordcloud = WordCloud(width=400, height=400, background_color='black').generate(negative_text)

# Create subplots in a 2x1 grid horizontally
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot positive sentiment word cloud
axes[0].imshow(positive_wordcloud, interpolation='bilinear')
axes[0].set_title('Word Cloud for Positive Sentiment')
axes[0].axis('off')

# Plot negative sentiment word cloud
axes[1].imshow(negative_wordcloud, interpolation='bilinear')
axes[1].set_title('Word Cloud for Negative Sentiment')
axes[1].axis('off')

plt.tight_layout()
plt.show()


# In[ ]:





# ### overall product and service sentiment bar chart 

# In[30]:


# Filter the DataFrame based on service reviews and product reviews
service_review_data = cleaned_df[cleaned_df['classification'] == 'Service Review']
product_review_data = cleaned_df[cleaned_df['classification'] == 'Product Review']

# Define the order for x-axis labels
polarity_order = ['Positive', 'Neutral', 'Negative']

# Create a 1x2 grid layout
fig, (ax2,ax1 ) = plt.subplots(1, 2, figsize=(12, 6))

# Plotting the frequency distribution of polarity for product reviews
sns.countplot(data=product_review_data, x='Polarity', order=polarity_order, palette=["#747BBA"], ax=ax2)  # Use orange color 
ax2.set_xlabel('Sentiment')
ax2.set_ylabel('Frequency')
ax2.set_title('Amazon overall Product Review Sentiment Analysis')
ax2.set_xticklabels(polarity_order, rotation=0)
ax2.tick_params(axis='x', rotation=0)

# Plotting the frequency distribution of polarity for service reviews
sns.countplot(data=service_review_data, x='Polarity', order=polarity_order, palette=["#DBAE78"], ax=ax1)  # Use blue color
ax1.set_xlabel('Sentiment')
ax1.set_ylabel('Frequency')
ax1.set_title('Amazon overall Service Review Sentiment Analysis')
ax1.set_xticklabels(polarity_order, rotation=0)
ax1.tick_params(axis='x', rotation=0)





# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()


# ### heat map on sentiment and other data

# In[31]:


# Select only numeric columns from the DataFrame
numeric_columns = cleaned_df.select_dtypes(include=[float, int])

# Calculate the correlation matrix
correlation_matrix = numeric_columns.corr()

# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt=".2f")  # 'YlGnBu' is a light color palette
plt.title("Correlation Heatmap")
plt.show()


# In[ ]:





# ### 5. individual company 

# In[171]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filtered_model2 = "Xiaomi 11 Lite NE 5G (Jazz Blue 6GB RAM 128 GB Storage) | Slimmest..."
filtered_review_type2 = "Product Review"
filtered_data2 = cleaned_df[(cleaned_df['Model'] == filtered_model2) & (cleaned_df['classification'] == filtered_review_type2)]

# Filtered model and review type
filtered_model = "Xiaomi 11 Lite NE 5G (Jazz Blue 6GB RAM 128 GB Storage) | Slimmest..."
filtered_review_type = "Service Review"
filtered_data = cleaned_df[(cleaned_df['Model'] == filtered_model) & (cleaned_df['classification'] == filtered_review_type)]


# Define the order for x-axis labels
polarity_order = ['Positive', 'Neutral', 'Negative']

# Create a 1x2 grid layout
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plotting the frequency distribution of polarity for service reviews of the specific model
sns.countplot(data=filtered_data2, x='Polarity', order=polarity_order, palette="pastel", ax=axes[0])
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Product Rating')
axes[0].set_xticklabels(polarity_order, rotation=0)
axes[0].tick_params(axis='x', rotation=0)

# Add numbers on top of the bars for the first count plot
for p in axes[0].patches:
    axes[0].annotate(format(p.get_height(), '.0f'), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', 
                     xytext = (0, 9), 
                     textcoords = 'offset points')

# Plotting the frequency distribution of polarity for product reviews of the specific model
sns.countplot(data=filtered_data, x='Polarity', order=polarity_order, palette="pastel", ax=axes[1])
axes[1].set_xlabel('Sentiment')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Service Rating')
axes[1].set_xticklabels(polarity_order, rotation=0)
axes[1].tick_params(axis='x', rotation=0)

# Add numbers on top of the bars for the second count plot
for p in axes[1].patches:
    axes[1].annotate(format(p.get_height(), '.0f'), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', 
                     xytext = (0, 9), 
                     textcoords = 'offset points')

# Adjust the layout
plt.subplots_adjust(wspace=0.7)
plt.tight_layout()

# Main title for the entire graph
plt.suptitle(f'Sentiment Analysis for {filtered_model}', fontsize=16)
plt.subplots_adjust(wspace=0.2, top=0.85)

# Show the plot
plt.show()


# In[164]:


import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Specify the model for which you want to create the bar chart
selected_model = "OnePlus 9R 5G (Carbon Black, 12GB RAM, 256 GB Storage)"

# Filter the DataFrame based on the selected model
filtered_reviews = cleaned_df[cleaned_df['Model'] == selected_model]

# Count the total number of reviews (both product and service)
total_reviews = filtered_reviews['classification'].count()

# Count the number of service reviews
service_reviews = filtered_reviews[filtered_reviews['classification'] == 'Service Review']['classification'].count()

# Define custom colors for both bars
color_product = 'skyblue'
color_service = 'lightgreen'

# Set bar width
bar_width = 0.2

# Create a slim horizontal stacked bar chart for total reviews and service reviews
plt.figure(figsize=(10, 0.7))
plt.barh(['Total Reviews'], [total_reviews - service_reviews], color=color_product, height=bar_width, label='Product Reviews')
plt.barh(['Total Reviews'], [service_reviews], left=[total_reviews - service_reviews], color=color_service, height=bar_width, label='Service Reviews')



# Create custom legend using matplotlib.patches.Patch objects
legend_labels = [
    Patch(facecolor=color_product, label=f'Product Reviews ({total_reviews - service_reviews})'),
    Patch(facecolor=color_service, label=f'Service Reviews ({service_reviews})')
]

plt.title(f"Total Service Reviews and Product reviews")
plt.xlabel('Count')

# Place the custom legend outside the bar chart
plt.legend(handles=legend_labels, loc='upper right', bbox_to_anchor=(1.29, 1))

plt.show()


# In[166]:


import matplotlib.pyplot as plt
import pandas as pd

# Specify the model for which you want to analyze the common trigrams
selected_model = "OnePlus 9R 5G (Carbon Black, 12GB RAM, 256 GB Storage)"

# Filter the DataFrame based on the selected model
filtered_reviews = cleaned_df[cleaned_df['Model'] == selected_model]

# Get the most common trigrams for positive product and service reviews separately
common_trigrams_product = top_n_ngram(filtered_reviews[filtered_reviews['classification'] == 'Product Review']['Title_and_Reviews'], 10, 3)
common_trigrams_service = top_n_ngram(filtered_reviews[filtered_reviews['classification'] == 'Service Review']['Title_and_Reviews'], 10, 3)

# Create subplots in a 2x1 grid horizontally
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Create a bar plot for product review trigram with a darker blue color
df_product = pd.DataFrame(common_trigrams_product, columns=['Trigram', 'count'])
df_product.groupby('Trigram').sum()['count'].sort_values(ascending=True).plot(
    kind='barh', title=f'Most Common Trigram in Product Reviews', ax=axes[0], color='#0096FF')  # Dark blue color
axes[0].set_xlabel('Frequency')
axes[0].set_ylabel('Product Trigram')

# Create a bar plot for service review trigram with a darker green color
df_service = pd.DataFrame(common_trigrams_service, columns=['Trigram', 'count'])
df_service.groupby('Trigram').sum()['count'].sort_values(ascending=True).plot(
    kind='barh', title=f'Most Common Trigram in Service Reviews', ax=axes[1], color='#228B22')  # Dark green color
axes[1].set_xlabel('Frequency')
axes[1].set_ylabel('Sevice Trigram')

plt.tight_layout()
plt.show()


# ## Topic modeling

# #### 3.4 Extended cleaning for topic modeling LDA

# In[34]:





# Function to remove punctuation from a text
def remove_punctuation(text):
    cleaned_text = text.translate(str.maketrans("", "", string.punctuation))
    return cleaned_text


# Apply the function to the text_column
cleaned_df['Title_and_Reviews'] = cleaned_df['Title_and_Reviews'].apply(remove_punctuation)

# Function to remove mentions from a column
def remove_mentions(text):
    cleaned_text = re.sub(r'@\w+', '', text)
    return cleaned_text

# Apply the function to the text_column
cleaned_df['Title_and_Reviews'] = cleaned_df['Title_and_Reviews'].apply(remove_mentions)


# Function to remove URLs from a text
def remove_urls(text):
    cleaned_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    return cleaned_text

# Apply the function to the text_column
cleaned_df['Title_and_Reviews'] = cleaned_df['Title_and_Reviews'].apply(remove_urls)


def remove_numbers(text):
    cleaned_text = ''.join([char for char in text if not char.isdigit()])
    return cleaned_text

# Apply the function to the text_column
cleaned_df['Title_and_Reviews'] = cleaned_df['Title_and_Reviews'].apply(remove_numbers)


# Add your custom stop words
custom_stop_words = set(["phone", "good","nt","mobile"])

# Function to remove stop words from a text
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    stop_words.update(custom_stop_words)  # Adding custom stop words
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Apply the function to the text_column
cleaned_df['Title_and_Reviews'] = cleaned_df['Title_and_Reviews'].apply(remove_stop_words)


# Create an instance of the enchant spellchecker
spellchecker = enchant.Dict("en_US")  # Use "en_US" dictionary as an example

# Function to correct misspelled words in a text

def correct_misspellings(text):
    if text is None:
        return ""  # Return an empty string for None values
    
    words = text.split()
    corrected_words = [spellchecker.suggest(word) if not spellchecker.check(word) and word is not None else word for word in words]
    corrected_text = ' '.join(corrected_word if isinstance(corrected_word, str) else '' for corrected_word in corrected_words)
    return corrected_text

# Apply the function to the 'Title_and_Reviews' column
cleaned_df['Title_and_Reviews'] = cleaned_df['Title_and_Reviews'].apply(correct_misspellings)


# Function to remove extra white spaces from a text
def remove_extra_spaces(text):
    cleaned_text = ' '.join(text.split())
    return cleaned_text

# Apply the function to the text_column
cleaned_df['Title_and_Reviews'] = cleaned_df['Title_and_Reviews'].apply(remove_extra_spaces)


# ### Check the optimal topic number

# In[44]:


import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel


# Preprocess the text data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = simple_preprocess(text, deacc=True)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


cleaned_df['processed_text'] = cleaned_df['Title_and_Reviews'].apply(preprocess_text)

# Create a dictionary and corpus
dictionary = corpora.Dictionary(cleaned_df['processed_text'])
corpus = [dictionary.doc2bow(text) for text in cleaned_df['processed_text']]


# Specify the number of topics
num_topics = 4

# Build LDA model
lda_model_build = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=42)

    
# Visualization using pyLDAvis
vis_data = gensimvis.prepare(lda_model_build, corpus, dictionary)
pyLDAvis.display(vis_data)


# In[ ]:





# ### View the topics in LDA model

# In[36]:


for num_topics, topic_keywords in lda_model_build.print_topics():
    print(f"Topic {num_topics}: {topic_keywords}\n")
doc_lda = lda_model_build[corpus]


# In[37]:


# Compute Perplexity
print('\nPerplexity: ', lda_model_build.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Calculate coherence score
coherence_model_lda = CoherenceModel(model=lda_model_build, texts=cleaned_df['processed_text'], dictionary=dictionary, coherence='c_v')
coherence_score = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_score)


# In[ ]:





# In[45]:





# List of different numbers of topics to try
num_topics_list = [1, 2, 3, 4, 5]

coherence_scores = []

# Loop through different numbers of topics
for num_topics in num_topics_list:
    # Train the LDA model
    lda_model_check = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
    
    # Calculate coherence score
    coherence_model_lda = CoherenceModel(model=lda_model_check, texts=cleaned_df['processed_text'], dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model_lda.get_coherence()
    
    coherence_scores.append(coherence_score)
    
    # Print coherence score for each number of topics
    print(f"Coherence Score for {num_topics} Topics: {coherence_score:.4f}")

# Create a line graph
plt.figure(figsize=(10, 6))
plt.plot(num_topics_list, coherence_scores, marker='o')

plt.title("Coherence Score vs. Number of Topics")
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.xticks(num_topics_list)

plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[39]:


cleaned_df.to_csv('D:/amazon_review_data/Amazon_reviews_clean.csv', mode='a', encoding='utf-8-sig',index=False, header=False)
cleaned_df.to_excel(r"D:/amazon_review_data/Amazon_reviews_clean.xlsx",index=False)


# In[ ]:





# In[ ]:





# In[ ]:




