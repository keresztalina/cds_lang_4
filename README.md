# Assignment 4 - Using finetuned transformers via HuggingFace
This assignment is ***Part 4*** of the portfolio exam for ***Language Analytics S23***. The exam consists of 5 assignments in total (4 class assignments and 1 self-assigned project).

## Contribution
The initial assignment was created partially in collaboration with other students in the course, also making use of code provided as part of the course. The final code is my own. Several adjustments have been made since the initial hand-in.

Here is the link to the GitHub repository containing the code for this assignment: https://github.com/keresztalina/cds_lang_4

## Assignment description by Ross
*(NB! This description has been edited for brevity. Find the full instructions in ```README_rdkm.md```.)*

In previous assignments, you've done a lot of model training of various kinds of complexity, such as training document classifiers or RNN language models. This assignment is more like Assignment 1, in that it's about *feature extraction*.

For this assignment, you should use ```HuggingFace``` to extract information from the *Fake or Real News* dataset that we've worked with previously.

You should write code and documentation which addresses the following tasks:

- Initalize a ```HuggingFace``` pipeline for emotion classification
- Perform emotion classification for every *headline* in the data
- Assuming the most likely prediction is the correct label, create tables and visualisations which show the following:
  - Distribution of emotions across all of the data
  - Distribution of emotions across *only* the real news
  - Distribution of emotions across *only* the fake news
- Comparing the results, discuss if there are any key differences between the two sets of headlines

## Methods
The purpose of this script is to run an existing model on new data using a ```HuggingFace``` pipeline and then visualizing the data for easy understanding. First, the data is loaded and reformatted for processing by the model. Then, the script loops through every headline, applies the classifier to it, and extracts the emotion with the greatest likelihood. This information is then collected into a dataframe.

In order to summarize the distribution of emotion across

## Usage
### Prerequisites
This code was written and executed in the UCloud application's Coder Python interface (version 1.77.3, running Python version 3.9.2). UCloud provides virtual machines with a Linux-based operating system, therefore, the code has been optimized for Linux and may need adjustment for Windows and Mac.

### Installations
1. Clone this repository somewhere on your device. The data is already contained within the ```/cds_lang_4/in``` folder.
2. Open a terminal and navigate into the ```/cds_lang_4``` folder. Run the following lines in order to install the necessary packages:
        
        pip install --upgrade pip
        python3 -m pip install -r requirements.txt

### Run the script.
In order to run the script, make sure your current directory is still the ```/cds_lang_4``` folder. Then, from command line, run:

        python3 src/emotions.py
    











