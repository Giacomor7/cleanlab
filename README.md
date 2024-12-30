There are no additional dependencies other than the transformers library, since
sentence-transformers could not find google/electra-small-discriminator.

Additional dependencies (other than those specified by cleanlab) NB: these are
all already added to requirements.txt:
- pyspellchecker
- language-tool-python
- spacy
- textstat

Run main.py to check the accuracy improvement from running cleanlab.

Lexical quality metrics are used to further weight cleanlab predictions based
on the spelling, grammar and coherence of the features.
