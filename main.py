import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from cleanlab.classification import CleanLearning
from cleanlab.lexical_quality_score import assess_text_quality


def main():
    data = pd.read_csv(
        "https://s.cleanlab.ai/banking-intent-classification.csv")

    raw_texts, raw_labels = data["text"].values, data["label"].values

    raw_train_texts, raw_test_texts, raw_train_labels, raw_test_labels = train_test_split(
        raw_texts, raw_labels, test_size=0.1)

    encoder = LabelEncoder()
    encoder.fit(raw_train_labels)

    train_labels = encoder.transform(raw_train_labels)
    test_labels = encoder.transform(raw_test_labels)

    # Load pretrained electra-small-discriminator model

    model_name = "google/electra-small-discriminator"
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True)

    transformer = SentenceTransformer(
        modules=[word_embedding_model, pooling_model])


    train_texts = transformer.encode(raw_train_texts)
    test_texts = transformer.encode(raw_test_texts)

    cv_n_folds = 5

    model = LogisticRegression(max_iter=400)
    cl = CleanLearning(model, cv_n_folds=cv_n_folds, verbose=False,
                       low_memory=True)

    baseline_model = LogisticRegression(max_iter=400)
    baseline_model.fit(X=train_texts, y=train_labels)

    preds = baseline_model.predict(test_texts)
    acc_og = accuracy_score(test_labels, preds)
    print(f"\n Test accuracy of original model: {acc_og}")

    cl.fit(X=train_texts, labels=train_labels,
           label_issues=cl.get_label_issues())
    pred_labels = cl.predict(test_texts)
    acc_cl = accuracy_score(test_labels, pred_labels)

    print(f"Test accuracy of cleanlab's model: {acc_cl}")

    text_quality_function = np.vectorize(assess_text_quality)
    text_quality = text_quality_function(raw_train_texts)

    model = LogisticRegression(max_iter=400)
    cl = CleanLearning(model, cv_n_folds=cv_n_folds, verbose=False,
                       low_memory=True)

    cl.fit(X=train_texts, labels=train_labels,
           label_issues=cl.get_label_issues(), sample_weight=text_quality,
           weights_affect_thresholds=True)
    pred_labels = cl.predict(test_texts)
    acc_cl_with_lexical_quality = accuracy_score(test_labels, pred_labels)

    print(
        f"Test accuracy of cleanlab's model with lexical quality scores: {acc_cl_with_lexical_quality}")

if __name__ == "__main__":
    main()
