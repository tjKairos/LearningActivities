from typing import Optional

def prompt_sentence() -> Optional[str]:
    prompt = input("Enter a prompt: ").lower()
    if prompt == "" or prompt == "exit":
        return None
    return prompt

def split_prompt(prompt: str):
    """Split the prompt into tokens"""
    # Remove punctuation
    punctuation = ".,!?;:-()[]{}'\"\\/@#$%^&*"
    for char in punctuation:
        prompt = prompt.replace(char, "")

    return prompt.split()


def count_words(words: list[str]):
    """Count the words in a list and return a dictionary of their counts"""
    counts = {}
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts


def print_counts(counts: dict[str, int]):
    """Print the counts of the words in a dictionary"""
    for word, count in counts.items():
        print(word, count)


def main():
    """Prompt the user for a prompt and print the counts of the words in it"""
    prompt = prompt_sentence()
    if prompt is None:
        print("No prompt given")
        return
    words = split_prompt(prompt)
    counts = count_words(words)
    print_counts(counts)


if __name__ == "__main__":
    main()