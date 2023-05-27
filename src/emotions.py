##### IMPORT MODULES
import os
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from transformers import pipeline

##### HELPER FUNCTIONS
# Extract most likely emotions from pipeline output
def extract_emotions(score_list):
    emotions = max(score_list[0], key = lambda x:x['score'])
    emotion = emotions['label']
    likelihood = emotions['score']

    return emotion, likelihood

# Classify texts and collect outputs
def classifier_output(texts, classifier):

    # initialize empty lists
    emotions_list = list()
    likelihoods_list = list()

    # loop through all the texts
    for i in range(0, len(texts)):

        # run classifier on each text
        text = texts.iloc[i]
        score_list = classifier(text)

        # extract most likely emotion
        emotion, likelihood = extract_emotions(score_list)

        # append to prepared list
        emotions_list.append(emotion)
        likelihoods_list.append(likelihood)

    emotions_series = pd.Series(emotions_list)
    likelihoods_series = pd.Series(likelihoods_list)
    
    return emotions_series, likelihoods_series

def main():
    ##### LOAD DATA 
    filename = os.path.join(
        "in",  
        "fake_or_real_news.csv")

    data = pd.read_csv(
        filename, 
        index_col = 0) 

    # fix index column to correspond to row numbers
    data = data.reset_index(drop = True) 

    headlines = data["title"]

    ##### CLASSIFY
    classifier = pipeline(
        "text-classification", 
        model = "j-hartmann/emotion-english-distilroberta-base", 
        return_all_scores = True)

    emotions, likelihoods = classifier_output(headlines, classifier)

    data['emotions'] = emotions
    data['likelihoods'] = likelihoods

    ##### PLOT
    # POOLED PLOT
    # find how many headlines express each emotion
    together = pd.DataFrame(
        data.groupby(['emotions']).size().reset_index(name = 'counts'))

    together = together.set_index('emotions') # change index for easier plotting
    total_count = together['counts'].sum() # count total number of headlines
    together['proportion'] = together['counts'] / total_count # get proportion

    # table
    t1_path = os.path.join(
        "out", 
        "all_table.csv")
    together.to_csv(
        t1_path, 
        index = False)

    # plot
    fig = together.plot(
        figsize = (10, 7),
        y = 'proportion',
        kind = 'bar', 
        width = 0.8, 
        legend = None,
        colormap = cm.get_cmap('PiYG'))

    fig.set_xlabel('Emotion')
    fig.set_ylabel('Proportion')
    fig.set_xticklabels(
        together.index, 
        rotation = 45) # rotate x-axis labels by 45 degrees
    plt.title('Proportion of emotions for fake and real news pooled')

    plt.savefig(os.path.join(
        "out", 
        "all_plot.jpg"))

    # SEGREGATED PLOT
    # find how many headlines express each emotion, and how many of these are real/fake
    separate = data.groupby(['emotions', 'label']).size().reset_index(name = 'counts')

    # pivot to wide for easier plotting
    sep_pivoted = separate.pivot(
        columns = 'label', 
        values = 'counts', 
        index = 'emotions')

    # find proportions
    total_count_fake = sep_pivoted['FAKE'].sum()
    total_count_real = sep_pivoted['REAL'].sum()
    sep_pivoted['proportion_fake'] = sep_pivoted['FAKE'] / total_count_fake
    sep_pivoted['proportion_real'] = sep_pivoted['REAL'] / total_count_real

    # subset columns for easier plotting
    subset = sep_pivoted[['proportion_fake', 'proportion_real']]

    # table
    t2_path = os.path.join(
        "out", 
        "separate_table.csv")
    subset.to_csv(
        t2_path, 
        index = False)

    # plot
    fig2 = subset.plot(
        figsize = (10, 7),
        kind = 'bar', 
        width = 0.8, 
        colormap = cm.get_cmap('PiYG'))
    fig2.set_xlabel('Emotion')
    fig2.set_ylabel('Proportion')
    fig2.set_xticklabels(
        subset.index, 
        rotation=45) # rotate x-axis labels by 45 degrees
    fig2.legend(title = 'News type') # set legend title
    plt.title('Proportion of emotions for fake and real news separately')
    plt.savefig(os.path.join(
        "out", 
        "separate_plot.jpg"))

if __name__ == "__main__":
    main()