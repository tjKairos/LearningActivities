from prompt import prompt_sentence, split_prompt, count_words, print_counts
import pickle
import os.path as path

data_folder = path.join(path.dirname(path.dirname(__file__)), "data")

def gather_counts() -> dict[str, int]:
    counts = {}
    while prompt := prompt_sentence():
        words = split_prompt(prompt)
        prompt_counts = count_words(words)
        for word in prompt_counts:
            if word in counts:
                counts[word] += prompt_counts[word]
            else:
                counts[word] = prompt_counts[word]
    return counts


def gather_sentiments(sentiments: list[str]) -> dict[str, dict[str, int]]:
    sentiment_counts = {}
    for sentiment in sentiments:
        print(f"Gathering counts for Sentiment `{sentiment}`; please enter prompts with that sentiment or enter nothing to finish.")
        counts = gather_counts()
        sentiment_counts[sentiment] = counts
        print(f"Counts for {sentiment}:")
        print_counts(counts)
    return sentiment_counts


def score_sentiments(sentiment_counts: dict[str, dict[str, int]], prompt_counts: dict[str, int]) -> dict[str, int]:
    """
    Score each sentiment based on the counts of the words in the prompt
    
    For each sentiment, multiply the count of each word in the prompt by the count of that word in the sentiment.

    Example:
    Sentiment Counts:
    {
        "happy": {
            "joy": 2,
            "happy": 1
        },
        "sad": {
            "sad": 1,
            "grump": 1
        }
    }
    Sentiment: happy
    Prompt: "joy joy joy happy grump"
    Prompt Counts: {"joy": 3, "happy": 1, "grump": 1}
    Score: 3 * 2 + 1 * 1 = 7
    Final Output: {"happy": 7, "sad": 1}
    """
    scores = {}
    
    # Your code here

    return scores


def classify_sentiment(sentiment_counts: dict[str, dict[str, int]], prompt_counts: dict[str, int]) -> str:
    """
    Classify the sentiment of the prompt based on the counts of the words in the prompt

    Use your score_sentiments function to score each sentiment.
    Then look through each of the scores and pick the sentiment with the highest score.
    Return that sentiment and its score.
    """
    max_sentiment = None
    max_score = 0

    # Your code here

    return max_sentiment, max_score

def main():
    sentiments = ["happy", "sad"]
    sentiment_counts = gather_sentiments(sentiments)
    for sentiment in sentiment_counts:
        print(f"Counts for {sentiment}:")
        print_counts(sentiment_counts[sentiment])

    # Save the sentiment counts to a file for future use
    with open(f"{data_folder}/counts.pickle", "wb") as file:
        pickle.dump(sentiment_counts, file)


if __name__ == "__main__":
    main()