from typing import Optional

def prompt_sentence() -> Optional[str]:
    print("----------")
    prompt = input("Enter a prompt: ").lower()
    if prompt == "" or prompt == "exit":
        return None
    return prompt

def split_prompt(prompt: str) -> list[str]:
    """
    Split the prompt into individual words, ignoring punctuation
    
    Examples:
    "hi" -> ["hi"]
    "hello world" -> ["hello", "world"]
    "hello, world!" -> ["hello", "world"]
    """
    # Remove punctuation
    punctuation = ".,!?;:-()[]{}'\"\\/@#$%^&*"
    for char in punctuation:
        prompt = prompt.replace(char, "")

    return prompt.split()


def count_words(words: list[str]) -> dict[str, int]:
    """
    Count the words in a list and return a dictionary of their counts
    
    Examples:
    ["hi"] -> {"hi": 1}
    ["hello", "world", "hello"] -> {"hello": 2, "world": 1}
    ["hello", "world", "hello", "world"] -> {"hello": 2, "world": 2}
    """
    counts = {}

    # Your code here

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