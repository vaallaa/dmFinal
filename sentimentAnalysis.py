import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import nltk 
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


df = pd.read_csv('output.csv')

def preProcessText(df):
    stopWords = set(stopwords.words('english'))
    lemmatizeText = nltk.WordNetLemmatizer()
    processedTexts = []

    for i, row in df.iterrows():
        text = row['Text']
        spellCheck = str(TextBlob(text).correct())
        tokens = nltk.word_tokenize(spellCheck.lower())
        filteredTokens = [token for token in tokens if token not in stopWords and token.isalpha()]
        lemmatizedTokens = [lemmatizeText.lemmatize(token) for token in filteredTokens]
        textProcessed = ' '.join(lemmatizedTokens)
        processedTexts.append(textProcessed)

    df['ProcessedText'] = processedTexts
    return df
        

def sentimentAnalysis(df): 
    result = {}
    sia = SentimentIntensityAnalyzer()

    for i, row in df.iterrows():
        text = row['ProcessedText']
        rowId = row['Id']
        result[rowId] = sia.polarity_scores(text)

    result = pd.DataFrame(result).T
    return result

def mergeDataFrames(df,result): 

    result = result.reset_index().rename(columns={'index':'Id'})
    result = result.merge(df, how ='left')
    return(result)

def plotStar(result,productID):
    if productID is not None: 
        productStarResult = result[result['ProductId']==productID]
        productStarResult['Score'].value_counts().sort_index().plot(kind='bar', title='Star Count', figsize=(5, 5))
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.show()
    else:
        result['Score'].value_counts().sort_index().plot(kind='bar', title='Star Count', figsize=(5, 5))
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.show()

def sentimentScoreThreshold(result, threshold=0.5, Plot = True, product = None):
    positive_reviews = result[result['compound'] > threshold]
    negative_reviews = result[result['compound'] < -threshold]
    neutral_reviews = result[(result['compound'] >= -threshold) & (result['compound'] <= threshold)]


    print('Postive Threshold', len(positive_reviews))
    print('Negative Threshold', len(negative_reviews))
    print('Neutral Threshold', len(neutral_reviews))

    if Plot:

        if product != None: 
            productResult = result[result['ProductId']==product]
            
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            sns.barplot(data=productResult, x='Score', y='pos', ax=axs[0])
            sns.barplot(data=productResult, x='Score', y='neu', ax=axs[1])
            sns.barplot(data=productResult, x='Score', y='neg', ax=axs[2])

            axs[0].set_title('Positive')
            axs[1].set_title('Neutral')
            axs[2].set_title('Negative')

            plt.show()
        
        else:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            sns.barplot(data=result, x='Score', y='pos', ax=axs[0])
            sns.barplot(data=result, x='Score', y='neu', ax=axs[1])
            sns.barplot(data=result, x='Score', y='neg', ax=axs[2])

            axs[0].set_title('Positive')
            axs[1].set_title('Neutral')
            axs[2].set_title('Negative')

            plt.show()


def averageSentimentScore(result):
    avg_scores = result.groupby('ProductId')['compound'].mean()
    return avg_scores



result = preProcessText(df)
result = sentimentAnalysis(df)
result = mergeDataFrames(df,result)
print(result)
plotStar(result,None)
result.to_csv('output.csv', index=False)
sentimentScoreThreshold(result, Plot = True, product= None)
print(averageSentimentScore(result))
