import pandas as pd


def required_Information(df):
    columns_to_drop = ['UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Summary']
    df = df.drop(columns=columns_to_drop, axis=1)
    return df


def initalize_DF(startRow,endRow):
    df = pd.read_csv('Reviews.csv')
    scaledDownDF = required_Information(df)
    scaledDownDF = scaledDownDF.iloc[startRow:endRow+1]
    scaledDownDF.to_csv('output.csv', index = False)
    return scaledDownDF

df = initalize_DF(100,500)

print(df)

