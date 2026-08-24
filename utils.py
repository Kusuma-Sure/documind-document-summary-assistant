import re


def count_words(text):
    return len(re.findall(r"\b\w+\b", text))


def count_characters(text):
    return len(text)


def reading_time(text):
    words = count_words(text)

    minutes = max(1, round(words / 200))

    return minutes