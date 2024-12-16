import language_tool_python
import spacy
import textstat
from spellchecker import SpellChecker


def assess_spelling(text):
    """
    Returns fraction of words that are correctly-spelled.

    If -1, then not enough words to determine a score.
    """
    spell = SpellChecker()
    words = text.split()
    misspelled = spell.unknown(words)
    return 1 - len(misspelled) / len(words) if words else -1

def assess_grammar(text):
    """
    Returns fraction of words that do not incur grammatical issues.

    If -1, then not enough words to determine a score.
    """
    n_words = len(text.split())
    if n_words == 0:
        return -1
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return 1 - len(matches) / n_words

def assess_coherence(text):
    """
    Returns coherence score for text.

    If -1, then there weren't enough sentences to determine coherence.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = list(doc.sents)
    coherence_score = 0

    for i in range(len(sentences) - 1):
        similarity = sentences[i].similarity(sentences[i + 1])
        coherence_score += similarity

    return coherence_score / (len(sentences) - 1) if len(sentences) > 1 else -1

def assess_readability(text):
    """
    Returns readability score for text.

    If -1, then there weren't enough sentences to determine readability.
    """
    try:
        flesch_score = textstat.flesch_reading_ease(text)

        readability = flesch_score / 100

        if readability >= 1: # score can theoretically go over 100
            return 1

        if readability <= 0: # score can be negative, though 0 is already v bad
            return 0
        return flesch_score
    except:
        return -1

def assess_text_quality(text):
    """
    Returns text quality score for text.

    A score of 1 means perfect lexical quality and 0 is entirely
    incomprehensible. Spelling, grammar, coherence and readability are equally
    weighted.
    """
    spelling_score = assess_spelling(text)
    grammar_score = assess_grammar(text)
    coherence_score = assess_coherence(text)
    readability_score = assess_readability(text)

    scores = []

    if spelling_score != -1:
        scores.append(spelling_score)
    if grammar_score != -1:
        scores.append(grammar_score)
    if coherence_score != -1:
        scores.append(coherence_score)
    if readability_score != -1:
        scores.append(readability_score)

    return sum(scores) / len(scores)
