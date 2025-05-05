import re
import string
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from IPython.display import display as dp
import pandas as pd
import numpy as np



def general_text_clean(text):
    """
    Cleans and preprocesses the input text by:
    - Stripping leading/trailing whitespace
    - Removing HTML tags
    - Preserving the domain name in URLs
    - Removing email addresses
    - Removing special characters and numbers (while preserving important ones)
    - Replacing multiple newlines with a single newline

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    
    text = text.strip()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Preserve domain name in URL and remove the rest
    text = re.sub(r"https?://([^\s/]+)[^\s]*", r"\1", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Remove special characters and numbers while preserving the important ones
    text = re.sub(r'[^a-zA-Z0-9\s.,!?"\'$%)(@#/-]', "", text)

    # Replace multiple newlines with single newline
    text = re.sub(r"\n+", "\n", text)

    return text


def extra_text_clean(text):
    """
    Cleans and preprocesses text by removing various specific phrases, URLs, image credits, and other unwanted elements.
    
    Args:
        text (str): The input text to be cleaned.
        flags (optional): The regex flags (default is re.DOTALL).
    
    Returns:
        str: The cleaned text.
    """

    text = re.sub(r"The following statements.*accuracy", "[reuters disclaimer]", text)
    text = re.sub(r"\d\d\d\d\sEDT", "", text)
    text = re.sub(r"\u200a", "", text)
    text = re.sub(r"TWITTERIMAGECONTENT", "", text)
    text = re.sub(r"IMAGE_CONTENT", "", text)
    text = re.sub(r"YOUTUBEVIDEOCONTENT", "", text)
    text = re.sub(r"Featured image via .*?Getty Images", "", text)
    text = re.sub(r"/Getty Images", "", text)
    text = re.sub(r"/ Getty Images", "", text)
    text = re.sub(r"Featured image via Twitter", "", text)
    text = re.sub(
        r"\[SOCIAL_MEDIA_HANDLE\]\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
        "{twitter timestamp} ",
        text,
    )
    text = re.sub(r"Donald J. Trump {twitter timestamp}.*", "", text)
    text = re.sub(r"{twitter timestamp}.*", "", text)
    text = re.sub(r"\[SOCIAL_MEDIA_HANDLE\]", "", text)

    # 21WIRE related
    text = re.sub(
        r"SUPPORT 21WIRE(and its work| )by (S|s)ubscribing and becoming a (MEMBER|Member) @ 21WIRE.TV",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"SUPPORT 21WIRE and its work by Subscribing and becoming a Member @ 21WIRE.TV",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"Support our work and Become a Member @ 21WIRE.TV",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"Continue this story at.*@ 21WIRE.TV", "", text, flags=re.DOTALL)
    text = re.sub(r"@ 21WIRE.TV", "", text, flags=re.DOTALL)
    text = re.sub(r"[A-Z\s\@]+21WIRE+[A-Z\s\@\.]+", "", text, flags=re.DOTALL)
    text = re.sub(r"21WIRE+[A-Z\s\@\.]+", "", text, flags=re.DOTALL)
    text = re.sub(r"[A-Z\s\@]+21WIRE", "", text, flags=re.DOTALL)

    text = re.sub(r"SEE MORE\s+[A-Z\s/.@]+", "", text, flags=re.DOTALL)

    text = re.sub(r"Shawn Helton  media", "", text, flags=re.DOTALL)
    text = re.sub(
        r"Help support us by becoming a 21WIRE Member at 21WIRE.TV",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"READ MORE ORLANDO SHOOTING NEWS AT media Orlando Files",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"READ MORE.*?Files", "", text, flags=re.DOTALL)
    text = re.sub(r"READ MORE\s+[A-Z\s]+", "", text, flags=re.DOTALL)

    text = re.sub(r"Featured image by .*?via Getty Images", "", text, flags=re.DOTALL)
    text = re.sub(r"via Getty Images", "", text, flags=re.DOTALL)
    text = re.sub(r"Featured image via Getty", "", text, flags=re.DOTALL)
    text = re.sub(r"Getty Image", "media", text, flags=re.DOTALL)
    text = re.sub(r"www/.facebook/.com", "facebook site", text, flags=re.DOTALL)

    text = re.sub(
        r"Featured image (via|from) video screen capture", "", text, flags=re.DOTALL
    )

    text = re.sub(r"Featured image is a screenshot", "", text, flags=re.DOTALL)
    text = re.sub(r"Image via screen capture", "", text, flags=re.DOTALL)
    text = re.sub(r"(f|F)eatured image.*? screen capture", "", text, flags=re.DOTALL)
    text = re.sub(
        r"Featured image (via|from) (Twitter|Facebook|YouTube|Youtube|Wikimedia|Wikimedia Commons)",
        "",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        r"Featured image (via|from|by) (Flickr|screenshots|screenshot)", "", text
    )
    text = re.sub(
        r"image (via|from) (video|image|) (screen capture|)", "", text, flags=re.DOTALL
    )
    text = re.sub(r"Featured image via screenshot", "", text, flags=re.DOTALL)
    text = re.sub(r"Featured image from video screenshot", "", text, flags=re.DOTALL)

    text = re.sub(
        r"image via (video screenshot|screengrab|screencap)", "", text, flags=re.DOTALL
    )
    text = re.sub(r"Featured image (via|by)", "", text, flags=re.DOTALL)

    text = re.sub(r"Featured image via Facebook", "", text, flags=re.DOTALL)
    text = re.sub(r"Featured image (via|) YouTube", "", text, flags=re.DOTALL)

    text = re.sub(r"Featured image courtesy of Flickr", "", text, flags=re.DOTALL)

    return text

def news_text_clean(text):
    """
    Cleans and processes news text by applying regex patterns to remove or replace unwanted content
    such as social media handles, external links, code injections, and placeholder text. It also
    removes broadcast times, spoilers, and content specific to certain media outlets.

    Args:
        text (str): The raw news text to be cleaned.

    Returns:
        str: The cleaned text after applying regex patterns and general text cleaning.
    """
    # Precompile regex patterns
    patterns = [
        (re.compile(r"pic\.twitter\.com/\S+"), "[TWITTER_IMAGE_CONTENT]"),
        (re.compile(r"[.]youtube www\.youtube\.com"), ". [YOUTUBE_VIDEO_CONTENT]"),
        (re.compile(r"www\.youtube\.com"), "[YOUTUBE_VIDEO_CONTENT]"),
        (re.compile(r"[(]bit[.]ly.*[)]"), "[IMAGE_CONTENT]"),
        (re.compile(r"\(@[A-Za-z0-9]+\)"), "[SOCIAL_MEDIA_HANDLE]"),
        (re.compile(r"\s@[A-Za-z0-9]+"), "[SOCIAL_MEDIA_HANDLE]"),
        # Remove non-ascii characters
        (re.compile(r"\u200a"), ""),
        # Remove code injections
        (re.compile(r"[\]xa0"), ""),
        (re.compile(r"CDATA.*// ]]", flags=re.DOTALL), ""),
        (re.compile(r"// [<] !\[&gt;", flags=re.DOTALL), ""),
        (re.compile(r"// [<]!\[", flags=re.DOTALL), ""),
        (re.compile(r"&gt", flags=re.DOTALL), ""),
        (
            re.compile(
                r"[(]function[(].*?'facebook-jssdk'[\]?[)][)];", flags=re.DOTALL
            ),
            "",
        ),
        (re.compile(r"[(]function[(].*}[)]", flags=re.DOTALL), ""),
        (re.compile(r"[(]function[(].*[(][)][)]", flags=re.DOTALL), ""),
        # Remove broadcast time
        (
            re.compile(r":LIVE BROADCAST TIMING: .* This week s", flags=re.DOTALL),
            " This week's",
        ),
        (
            re.compile(r"SCHEDULED SHOW TIMES:.*[)]This week s", flags=re.DOTALL),
            " This week's",
        ),
        # Remove leftovers
        (re.compile(r"[_][(] [)][_][/]", flags=re.DOTALL), ""),
        (re.compile(r"[(] [)]", flags=re.DOTALL), ""),
        (re.compile(r"[[] []]", flags=re.DOTALL), ""),
        
        # Remove clear spoilers
        (re.compile(r"(.*?) \(Reuters\) - "), ""),
        (re.compile(r"\(Reuters\) - "), ""),
        (re.compile(r"21st Century Wire", flags=re.DOTALL), "media"),
        (re.compile(r"READ MORE .* (@21WIRE.TV|2016 Files)", flags=re.DOTALL), "."),
        
        # Remove content placeholders
        (re.compile(r"TWITTERIMAGECONTENT"), ""),
        (re.compile(r"IMAGE_CONTENT"), ""),
        (re.compile(r"YOUTUBEVIDEOCONTENT"), ""),
        (re.compile(r"Featured image via .*?Getty Images"), ""),
        (re.compile(r"/Getty Images"), ""),
        (re.compile(r"/ Getty Images"), ""),
        (re.compile(r"Featured image via Twitter"), ""),
        (re.compile(r"\[SOCIAL_MEDIA_HANDLE\]\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}"), "{twitter timestamp} "),
        (re.compile(r"Donald J. Trump {twitter timestamp}.*"), ""),
        (re.compile(r"{twitter timestamp}.*"), ""),
        (re.compile(r"\[SOCIAL_MEDIA_HANDLE\]"), ""),
        (re.compile(r"Source link \[\]"), ""),
    ]

    # Apply all patterns
    for pattern, replacement in patterns:
        text = pattern.sub(replacement, text)

    # Then do the general cleaning
    text = general_text_clean(text)
    text = extra_text_clean(text)
    return text

# Function to process each row
def process_row(text):
    """
    Processes a single row of text by cleaning it using the `news_text_clean` function.

    Args:
        text (str): The raw text to be processed.

    Returns:
        str: The cleaned text after applying `news_text_clean`.
    """
    return news_text_clean(text)

def news_title_clean(title):
    
    """
    Cleans a news title by removing specific keywords, brackets, and colons.

    Args:
        title (str): The raw news title to be cleaned.

    Returns:
        str: The cleaned news title.
    """

    title = re.sub(
        r"(WATCH|REPORT|BREAKING|BREAKING NEWS|Exclusive|Factbox):", "", title
    )

    title = re.sub(r"\([A-Z/,\s]+\)", "", title)
    title = re.sub(r"\[[A-Z/,\s]+\]", "", title)
    title = re.sub(r"\[Video\]", "", title)
    title = re.sub(r":", ",", title)

    title = title.strip()

    return title


def quick_logreg(X_train=None, X_test=None, y_train=None, y_test=None):
    
    """
    Trains a logistic regression model and evaluates it using accuracy and recall.

    Args:
        X_train (array-like): Training feature data.
        X_test (array-like): Test feature data.
        y_train (array-like): Training target labels.
        y_test (array-like): Test target labels.

    Returns:
        LogisticRegression: The trained logistic regression model.
    """

    logreg = LogisticRegression(C=1, max_iter=3000)

    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    dp(
        pd.DataFrame(
            {
                "Accuracy": [accuracy_score(y_test, y_pred)],
                "Recall": [recall_score(y_test, y_pred)],
            },
            index=["scores"],
        ).round(3)
    )

    return logreg


def data_group_df(X, y):
    """
    Creates a DataFrame with text and label columns from the provided sequences and labels.

    Args:
        X (array-like): Feature data with a 'sequence' attribute.
        y (array-like): Target labels.

    Returns:
        pd.DataFrame: A DataFrame with 'text' and 'label' columns.
    """
    df = pd.DataFrame({"text": X.sequence.tolist(), "label": y.tolist()})
    return df


def moving_average(data, window_size=2):
    """
    Calculates the moving average of a given data sequence.

    Args:
        data (list): The input data sequence.
        window_size (int, optional): The size of the moving window (default is 2).

    Returns:
        list: A list of moving averages.
    """
    return [sum(data[i:i+window_size]) / window_size for i in range(len(data) - window_size + 1)]

def get_embed_avg_arr(sequence_list, w2v_model):
    """
    Computes the average word embeddings for sequences using a Word2Vec model.

    Args:
        sequence_list (list of lists): A list of tokenized sequences.
        w2v_model (gensim.models.KeyedVectors): Pre-trained Word2Vec model.

    Returns:
        np.ndarray: A NumPy array of average embeddings for each sequence.
    """
    embedded_cleaned_words = w2v_model.wv.index_to_key

    embed_avg_ls = []
    for i in sequence_list:
        seq_embed_vec = [
            w2v_model.wv.get_vector(j) for j in i if j in embedded_cleaned_words
        ]
        seq_embed_vec_mean = np.mean(
            seq_embed_vec, axis=0
        )
        embed_avg_ls.append(seq_embed_vec_mean)

    embed_avg_arr = np.array(embed_avg_ls)

    return embed_avg_arr