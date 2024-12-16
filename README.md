There are no additional dependencies other than the transformers library, since
sentence-transformers could not find google/electra-small-discriminator.

Additional dependencies (other than those specified by cleanlab) NB: these are
all already added to requirements.txt:
- pyspellchecker
- language-tool-python
- spacy
- textstat

Run main.py to check the accuracy improvement from running cleanlab.

I did not have time to implement unit tests because of major difficulties
loading the pretrained model suggested by cleanlab
(google/electra-small-discriminator). It probably would have been quicker to make-do
with a similar model...
