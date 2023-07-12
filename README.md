# LearningActivities

This is a set of 2 activities built for Maydm to teach Machine Learning and Aritificial Intelligence (ML/AI) concepts through code and active usage.

## Activity 1: Sentiment Detection / Text Classification

The goal of this activity is to create a Bag of Words (BoW) language model which can tell whether a word, phrase, or sentence is `happy or sad` (or `good or evil`, `spam or not spam`, `funny or cringe`, `Bluey, Avatar: The Last Airbender, or Law and Order`, or anything else you can think of). We generally call each of these options a "sentiment", "class", or "classification".

A bag of words means we're treating each word as if it were just thrown into a bag. So we don't use the orderings of words or what context they appear in, just whether the words appear or not.

To this end, we'll implement 3 functions:

- prompt.count_words (given a list of words in a sentence, count # of times each word appears)
- learn.score_sentiments (given word counts in a sentence and in all sentiment, calculate the matching score per sentiment)
- learn.classify_sentiment (given word counts in a sentence and in all sentiments, find the highest matching score)

Once these are completed, we can run `python learn.py` to label some data for our AI to know how to classify our sentiments. Once we've labeled some data, we can run `test.py` to test out our AI and see how it performs on unseen sentences.

## Activity 2: Image Classification

Images are very different than text, so we handle them very differently in Machine Learning. We provide an existing trained model on the [Quickdraw Dataset](https://github.com/googlecreativelab/quickdraw-dataset), a fine-tuning system for drawing your own individual drawings, and a simply drawing game where you can draw some artwork and the model will tell you what it is in real-time. We can create our own drawings with `python label_data.py` and test out the trained model with `python classification_game.py`. You can toggle on and off the fine-tuning (using your own drawings) and whether to use all the possible classes, or just the ones that you've tuned on with the boolean (True or False) values at the top of `classification_game.py`.

## Resources

- [Quickdraw Dataset](https://github.com/googlecreativelab/quickdraw-dataset)
- [Other projects using Quickdraw](https://github.com/googlecreativelab/quickdraw-dataset#projects-using-the-dataset)
- [Quickdraw datasets for this project](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?prefix=&forceOnObjectsSortingFiltering=false&pli=1)