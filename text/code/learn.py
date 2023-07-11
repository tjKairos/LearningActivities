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
    scores = {}
    for sentiment, learned_counts in sentiment_counts.items():
        score = 0
        for word, count in prompt_counts.items():
            if word in learned_counts:
                score += learned_counts[word] * count
        scores[sentiment] = score
    return scores


def classify_sentiment(sentiment_counts: dict[str, dict[str, int]], prompt_counts: dict[str, int]) -> str:
    max_sentiment = None
    max_score = 0
    all_scores = score_sentiments(sentiment_counts, prompt_counts)
    for sentiment, score in all_scores.items():
        if score > max_score:
            max_score = score
            max_sentiment = sentiment
        print(f"Score for {sentiment}: {score}")
    return max_sentiment, max_score

def main():
    sentiments = ["happy", "sad"]
    sentiment_counts = gather_sentiments(sentiments)
    for sentiment in sentiment_counts:
        print(f"Counts for {sentiment}:")
        print_counts(sentiment_counts[sentiment])
    with open(f"{data_folder}/counts.pickle", "wb") as file:
        pickle.dump(sentiment_counts, file)


if __name__ == "__main__":
    main()