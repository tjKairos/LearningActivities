from prompt import prompt_sentence, split_prompt, count_words, print_counts
from learn import gather_counts, classify_sentiment
import pickle
import os.path as path

data_folder = path.join(path.dirname(path.dirname(__file__)), "data")


def load_counts() -> dict[str, dict[str, int]]:
    with open(f"{data_folder}/counts.pickle", "rb") as file:
        return pickle.load(file)


def main():
    print("Loading counts...")
    sentiment_counts = load_counts()
    # for sentiment, leanred_counts in sentiment_counts.items():
        # print(f"Counts for {sentiment}:")
        # print_counts(learned_counts)

    while prompt := prompt_sentence():
        words = split_prompt(prompt)
        prompt_counts = count_words(words)
        # print("Counts for prompt:")
        # print_counts(counts)
        
        sentiment, score = classify_sentiment(sentiment_counts, prompt_counts)
        print(f"Max sentiment: {sentiment}")
        print(f"Max score: {score}")

if __name__ == "__main__":
    main()
