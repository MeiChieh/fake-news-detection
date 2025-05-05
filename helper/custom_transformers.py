from sklearn.preprocessing import FunctionTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.pipeline import make_pipeline
import string
import nltk
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

nltk.download("vader_lexicon")


def add_is_weekend(df):
    """
    Adds a column to the DataFrame indicating whether the date corresponds to a weekend.

    Args:
        df (pd.DataFrame): Input DataFrame containing a "date" column with datetime objects.

    Returns:
        pd.DataFrame: Updated DataFrame with two new columns:
            - "day_of_week": Day of the week as an integer (0 for Monday, 6 for Sunday).
            - "is_weekend": Binary indicator (1 if weekend, 0 otherwise).
    """
    df.loc[:, ["day_of_week"]] = df["date"].dt.day_of_week
    df.loc[:, ["is_weekend"]] = df.day_of_week.apply(lambda i: 1 if i in [5, 6] else 0)
    return df


def detect_all_caps(text: str, threshold: float = 10) -> bool:
    """
    Detects whether the percentage of uppercase words in a text exceeds a given threshold.

    Args:
        text (str): The input text to analyze.
        threshold (float): Percentage threshold (0-100) for determining excessive uppercase usage.

    Returns:
        bool: True if the percentage of uppercase words exceeds the threshold, False otherwise.
    """
    threshold = threshold / 100
    text_ls = text.split()
    upper_count = sum(i.isupper() for i in text_ls)
    text_len = len(text_ls)
    return (upper_count / text_len) > threshold


def add_5_percent_full_caps(df):
    """
    Adds a column indicating whether the text contains more than 5% fully capitalized words.

    Args:
        df (pd.DataFrame): Input DataFrame containing a "text" column with strings.

    Returns:
        pd.DataFrame: Updated DataFrame with a new column "text_5_percent_upper" (binary indicator).
    """
    df.loc[:, ["text_5_percent_upper"]] = df.text.apply(
        lambda i: detect_all_caps(i, threshold=5)
    ).astype(int)
    return df


def add_sequence_length(df):
    """
    Adds a column representing the length of sequences (number of words) in the "sequence" column.

    Args:
        df (pd.DataFrame): Input DataFrame containing a "sequence" column with text.

    Returns:
        pd.DataFrame: Updated DataFrame with a new column "sequence_len".
    """
    df.loc[:, ["sequence_len"]] = df["sequence"].apply(lambda i: len(i.split(" ")))
    return df


def add_question_mark_count(df):
    """
    Adds a column counting the number of question marks in the "text" column.

    Args:
        df (pd.DataFrame): Input DataFrame containing a "text" column with strings.

    Returns:
        pd.DataFrame: Updated DataFrame with a new column "question_mark_count".
    """
    df.loc[:, ["question_mark_count"]] = df.text.apply(lambda i: list(i).count("?"))
    return df


def add_exclamation_mark_count(df):
    """
    Adds a column counting the number of exclamation marks in the "text" column.

    Args:
        df (pd.DataFrame): Input DataFrame containing a "text" column with strings.

    Returns:
        pd.DataFrame: Updated DataFrame with a new column "exclamation_mark_count".
    """
    df.loc[:, ["exclamation_mark_count"]] = df.text.apply(lambda i: list(i).count("!"))
    return df


def add_sentiment_scores(df):
    """
    Adds sentiment scores to the DataFrame using the VADER sentiment analyzer.

    Args:
        df (pd.DataFrame): Input DataFrame containing a "text" column with strings.

    Returns:
        pd.DataFrame: Updated DataFrame with the following new columns:
            - "senti_compound": Compound sentiment score.
            - "senti_neg": Negative sentiment score.
            - "senti_neu": Neutral sentiment score.
    """
    senti = SentimentIntensityAnalyzer()
    senti_scores = df.text.apply(lambda i: senti.polarity_scores(i)).tolist()
    df.loc[:, ["senti_compound"]] = [i["compound"] for i in senti_scores]
    df.loc[:, ["senti_neg"]] = [i["neg"] for i in senti_scores]
    df.loc[:, ["senti_neu"]] = [i["neu"] for i in senti_scores]
    
    return df


feat_cols = [
    "text_len",
    "title_len",
    "sequence_len",
    "is_weekend",
    "text_5_percent_upper",
    "question_mark_count",
    "exclamation_mark_count",
    "senti_compound",
    "senti_neg",
]


def select_feat_cols(df):
    """
    Selects specific feature columns from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with various feature columns.

    Returns:
        pd.DataFrame: DataFrame containing only the predefined feature columns.
    """
    return df[feat_cols].copy()


add_is_weekend_transformer = FunctionTransformer(add_is_weekend)
add_5_percent_full_caps_transformer = FunctionTransformer(add_5_percent_full_caps)
add_sequence_length_transformer = FunctionTransformer(add_sequence_length)
add_question_mark_count_transformer = FunctionTransformer(add_question_mark_count)
add_exclamation_mark_count_transformer = FunctionTransformer(add_exclamation_mark_count)
add_sentiment_scores_transformer = FunctionTransformer(add_sentiment_scores)
select_feat_cols_transformer = FunctionTransformer(select_feat_cols)

fe_pipeline = make_pipeline(
    add_is_weekend_transformer,
    add_5_percent_full_caps_transformer,
    add_sequence_length_transformer,
    add_question_mark_count_transformer,
    add_exclamation_mark_count_transformer,
    add_sentiment_scores_transformer,
    select_feat_cols_transformer,
)

heuristic_pipeline = Pipeline(
    [
        ("preprocessing", fe_pipeline),
        ("scaling", StandardScaler()),
        (
            "model",
            LogisticRegression(class_weight="balanced"),
        ),
    ]
)

heuristic_model_rs = RandomizedSearchCV(
    estimator=heuristic_pipeline,
    param_distributions=[
        {
            "scaling": [StandardScaler(), RobustScaler()],
            "model": [LogisticRegression(class_weight="balanced")],
            "model__C": np.logspace(-2, 2, 5),
            "model__solver": ["newton-cg", "lbfgs", "sag"],
        },
        {
            "scaling": [StandardScaler(), RobustScaler()],
            "model": [GradientBoostingClassifier(random_state=0)],
            "model__learning_rate": np.logspace(-2, 0, 3),
            "model__n_estimators": [50, 100, 250, 500],
            "model__max_depth": [3, 5, 10],
        },
    ],
    n_iter=200,
    cv=5,
    verbose=1,
    random_state=0,
    n_jobs=-1,
)


def parse_seq_clean(seq):
    """
    Cleans a sequence by removing punctuation and converting all words to lowercase.

    Args:
        seq (str): The input sequence to clean.

    Returns:
        list: A list of cleaned, lowercase words.
    """
    remove_punctuation = str.maketrans("", "", string.punctuation)
    return [word.translate(remove_punctuation).lower() for word in seq]


def clean_seq_ls(seq_ls):
    """
    Cleans a list of sequences by removing punctuation and converting words to lowercase.

    Args:
        seq_ls (list of str): List of input sequences.

    Returns:
        list of list: List of cleaned sequences, where each sequence is a list of words.
    """
    return [parse_seq_clean(seq) for seq in seq_ls]


def clean_and_join_seq_ls(df):
    """
    Cleans sequences in the DataFrame and joins words into a single string.

    Args:
        df (pd.DataFrame): Input DataFrame containing a "sequence" column with text.

    Returns:
        list: List of cleaned sequences as joined strings.
    """
    df["sequence"] = clean_seq_ls(df["sequence"])
    df.loc[:, ["sequence"]] = ["".join(i) for i in df.sequence.tolist()]
    return df["sequence"].tolist()

clean_and_join_seq_ls_transformer = FunctionTransformer(clean_and_join_seq_ls)

tfidf_pipeline = Pipeline(
    [
        ("clean_and_join_seq_ls_transformer", clean_and_join_seq_ls_transformer),
        ("tfidf", TfidfVectorizer(max_features=20000)),
        ("logits", LogisticRegression(C=1, max_iter=3000)),
    ]
)

tfidf_param_grid = {
    "tfidf__max_features": [5000, 10000, 20000],
}

tfidf_grid = GridSearchCV(tfidf_pipeline, tfidf_param_grid, cv=3, scoring="accuracy", verbose=1)
