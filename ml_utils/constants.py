"""Constants and schema maps used across data processing utilities."""

from datetime import date


FEATURE_GROUPS = {
    "artist_level": [
        "id_author",
        "name",
        "gender",
        "birth_date",
        "description",
        "active_start",
        "active_end",
    ],
    "geographic": [
        "birth_place",
        "province",
        "region",
        "country",
        "nationality",
        "latitude",
        "longitude",
    ],
    "song_level": [
        "id_song",
        "id_artist",
        "name_artist",
        "full_title",
        "title",
        "featured_artists",
        "primary_artist",
        "language",
        "album",
        "album_name",
        "album_release_date",
        "album_type",
        "disc_number",
        "track_number",
        "duration_ms",
        "explicit",
        "popularity",
        "stats_pageviews",
        "album_image",
        "id_album",
        "lyrics",
        "modified_popularity",
        "year",
        "month",
        "day",
    ],
    "textual": [
        "swear_IT",
        "swear_EN",
        "swear_IT_words",
        "swear_EN_words",
        "n_sentences",
        "n_tokens",
        "tokens_per_sent",
        "char_per_tok",
        "lexical_density",
        "avg_token_per_clause",
    ],
    "audio": [
        "bpm",
        "centroid",
        "rolloff",
        "flux",
        "rms",
        "zcr",
        "flatness",
        "spectral_complexity",
        "pitch",
        "loudness",
    ],
}

TRACKS_DTYPE_MAP = {
    "id": "Int64",
    "id_artist": "Int64",
    "name_artist": "string",
    "full_title": "string",
    "title": "string",
    "featured_artists": "string",
    "primary_artist": "string",
    "language": "string",
    "album": "string",
    "stats_pageviews": "Int64",
    "swear_IT": "Int64",
    "swear_EN": "Int64",
    "swear_IT_words": "string",
    "swear_EN_words": "string",
    "year": "Int64",
    "month": "Int64",
    "day": "Int64",
    "n_sentences": "Int64",
    "n_tokens": "Int64",
    "tokens_per_sent": "Float64",
    "char_per_tok": "Float64",
    "lexical_density": "Float64",
    "avg_token_per_clause": "Float64",
    "bpm": "Float64",
    "centroid": "Float64",
    "rolloff": "Float64",
    "flux": "Float64",
    "rms": "Float64",
    "zcr": "Float64",
    "flatness": "Float64",
    "spectral_complexity": "Float64",
    "pitch": "Float64",
    "loudness": "Float64",
    "album_name": "string",
    "album_type": "string",
    "disc_number": "Int64",
    "track_number": "Int64",
    "duration_ms": "Int64",
    "explicit": "boolean",
    "popularity": "Float64",
    "album_image": "string",
    "id_album": "Int64",
    "lyrics": "string",
    "modified_popularity": "Float64",
}

TRACKS_DATETIME_COLUMNS = {
    "album_release_date": "datetime64[ns]",
}

ARTISTS_DTYPE_MAP = {
    "id_author": "Int64",
    "name": "string",
    "gender": "string",
    "birth_place": "string",
    "nationality": "string",
    "description": "string",
    "active_start": "Int64",
    "active_end": "Int64",
    "province": "string",
    "region": "string",
    "country": "string",
    "latitude": "Float64",
    "longitude": "Float64",
}

ARTISTS_DATETIME_COLUMNS = {
    "birth_date": "datetime64[ns]",
}

MERGED_DTYPE_MAP = {
    **TRACKS_DTYPE_MAP,
    **ARTISTS_DTYPE_MAP,
}

MERGED_DATETIME_COLUMNS = {
    **TRACKS_DATETIME_COLUMNS,
    **ARTISTS_DATETIME_COLUMNS,
}

ENGINEERED_FEATURE_DOCS = [
    {
        "feature": "swear_IT",
        "family": "textual",
        "description": "Count of Italian swear words detected in the lyrics.",
    },
    {
        "feature": "swear_EN",
        "family": "textual",
        "description": "Count of English swear words detected in the lyrics.",
    },
    {
        "feature": "swear_IT_words",
        "family": "textual",
        "description": "List of matched Italian swear words found in the lyrics.",
    },
    {
        "feature": "swear_EN_words",
        "family": "textual",
        "description": "List of matched English swear words found in the lyrics.",
    },
    {
        "feature": "n_sentences",
        "family": "textual",
        "description": "Number of detected sentences in the lyrics.",
    },
    {
        "feature": "n_tokens",
        "family": "textual",
        "description": "Number of detected tokens/words in the lyrics.",
    },
    {
        "feature": "tokens_per_sent",
        "family": "textual",
        "description": "Average number of tokens per sentence.",
    },
    {
        "feature": "char_per_tok",
        "family": "textual",
        "description": "Average number of characters per token.",
    },
    {
        "feature": "lexical_density",
        "family": "textual",
        "description": "Ratio of lexical words over all tokens.",
    },
    {
        "feature": "avg_token_per_clause",
        "family": "textual",
        "description": "Average number of tokens per clause.",
    },
    {
        "feature": "modified_popularity",
        "family": "song_level",
        "description": "Alternative popularity-like score already included in the source data.",
    },
    {
        "feature": "centroid",
        "family": "audio",
        "description": "Spectral centroid, a proxy for the brightness of the sound.",
    },
    {
        "feature": "rolloff",
        "family": "audio",
        "description": "Spectral rolloff, the frequency below which most spectral energy is concentrated.",
    },
    {
        "feature": "flux",
        "family": "audio",
        "description": "Spectral flux, measuring frame-to-frame spectral change.",
    },
    {
        "feature": "rms",
        "family": "audio",
        "description": "Root mean square energy of the audio signal.",
    },
    {
        "feature": "zcr",
        "family": "audio",
        "description": "Zero-crossing rate, often related to noisiness or percussiveness.",
    },
    {
        "feature": "flatness",
        "family": "audio",
        "description": "Spectral flatness, how tone-like versus noise-like the signal is.",
    },
    {
        "feature": "spectral_complexity",
        "family": "audio",
        "description": "Count-based spectral complexity descriptor.",
    },
    {
        "feature": "pitch",
        "family": "audio",
        "description": "Estimated pitch-related descriptor.",
    },
    {
        "feature": "loudness",
        "family": "audio",
        "description": "Perceived loudness-related descriptor.",
    },
]

BLANK_PLACEHOLDERS = [
    "N/A",
    "NA",
    "na",
    "n/a",
    "none",
    "NULL",
    "null",
    "-",
    "--",
    "---",
    "",
    " ",
    "\t",
    "?",
    "unknown",
    "Unknown",
    "missing",
    "Missing",
    "not available",
    "Not Available",
    "unavailable",
    "Unavailable",
    "no data",
    "No Data",
    "not provided",
    "Not Provided",
    "not applicable",
    "Not Applicable",
    "none of the above",
    "None of the above",
    "n.a.",
    "N.A.",
    "n.a",
    "N.A",
    "missing value",
    "Missing Value",
    "missing values",
    "Missing Values",
    "unknown value",
    "Unknown Value",
    "unknown values",
    "Unknown Values",
    "not known",
    "Not Known",
    "not sure",
    "Not Sure",
    "undisclosed",
    "Undisclosed",
    "confidential",
    "Confidential",
    "secret",
    "Secret",
    "redacted",
    "Redacted",
    "withheld",
    "Withheld",
]

REGION_TO_MACROAREA = {
    "abruzzo": "South",
    "basilicata": "South",
    "calabria": "South",
    "campania": "South",
    "emilia romagna": "North",
    "friuli venezia giulia": "North",
    "lazio": "Center",
    "liguria": "North",
    "lombardia": "North",
    "marche": "Center",
    "molise": "South",
    "piemonte": "North",
    "puglia": "South",
    "sardegna": "Sardinia",
    "sicilia": "Sicily",
    "toscana": "Center",
    "trentino alto adige": "North",
    "trentino alto adige/sudtirol": "North",
    "trentino alto adige sudtirol": "North",
    "umbria": "Center",
    "valle d aosta": "North",
    "valle d'aosta": "North",
    "veneto": "North",
}

# Domain plausibility rules: callable predicates that return True if a value is plausible.
# Used in Stage C: apply domain plausibility masks.
# Each rule is a lambda that accepts a pandas Series and returns a boolean Series.

DOMAIN_PLAUSIBILITY_RULES = {
    # Track temporal features
    "year": lambda x: (x >= 1900) & (x <= date.today().year),  # reasonable release years
    "month": lambda x: (x >= 1) & (x <= 12),
    "day": lambda x: (x >= 1) & (x <= 31),
    
    
    # Audio features: typically in normalized ranges
    "duration_ms": lambda x: (x > 0) & (x <= 1800000),  # 0ms to 30 minutes
    "bpm": lambda x: (x > 0) & (x < 300),  # typical music tempo range (beats per minute)
    "centroid": lambda x: (x >= 0) & (x <= 20000),  # frequency in Hz
    "rolloff": lambda x: (x >= 0) & (x <= 20000),  # frequency in Hz
    "flux": lambda x: (x >= 0) & (x <= 1000),  # spectral flux
    "rms": lambda x: (x >= 0) & (x <= 1.0),  # root mean square energy (typically 0-1)
    "zcr": lambda x: (x >= 0) & (x <= 1.0),  # zero crossing rate (0-1)
    "flatness": lambda x: (x >= 0) & (x <= 1.0),  # spectral flatness (0-1)
    "pitch": lambda x: (x >= 1000) & (x <= 3700),  # pitch (from separate analysis, in Hz)
    "loudness": lambda x: (x >= 0) & (x <= 200),  # loudness in -dB
    "spectral_complexity": lambda x: (x >= 0) & (x <= 100),  # complexity score
    
    # Popularity and engagement
    "popularity": lambda x: (x >= 0) & (x <= 100),  # 0-100 scale
    "modified_popularity": lambda x: (x >= 0) & (x <= 100),  # 0-100 scale
    "stats_pageviews": lambda x: (x > 0),  # positive counts
    
    # Textual features
    "n_sentences": lambda x: (x > 0),  # at least one sentence
    "n_tokens": lambda x: (x > 0),  # at least one token
    "tokens_per_sent": lambda x: (x > 0) & (x <= 1000),  # avg tokens per sentence
    "char_per_tok": lambda x: (x > 0) & (x <= 50),  # avg chars per token
    "lexical_density": lambda x: (x >= 0) & (x <= 1.0),  # ratio 0-1
    "avg_token_per_clause": lambda x: (x > 0) & (x <= 200),  # tokens per clause
    "swear_IT": lambda x: (x >= 0),  # count
    "swear_EN": lambda x: (x >= 0),  # count
    
    # Artist temporal features
    "active_start": lambda x: (x >= 1800) & (x <= date.today().year),  # reasonable artist birth year
    "active_end": lambda x: (x >= 1800) & (x <= date.today().year),  # reasonable career end year
    
    # Geographic features
    "latitude": lambda x: (x >= -90) & (x <= 90),  # valid latitude range
    "longitude": lambda x: (x >= -180) & (x <= 180),  # valid longitude range
}

# Categorical domain rules: check membership in a valid set.
# Format: {column: set_of_valid_values or callable}

DOMAIN_CATEGORICAL_RULES = {
    "gender": {"M", "F", "U", "Unknown"},  # M, F, Unknown or empty
    "language": {"en", "it", "it-IT", "en-US", "es", "fr", "de", "pt", "other"},  # common languages
    "explicit": {True, False},  # boolean
    "album_type": {"album", "single", "compilation", "EP", "ep", "album_type", "other"},  # Spotify types
}

# Cross-column consistency rules: (col1, col2) -> callable that checks logical consistency
# Used to catch data inconsistencies between related fields.

DOMAIN_CONSISTENCY_RULES = {
    # Active dates: end should be >= start (if both exist)
    ("active_start", "active_end"): lambda s1, s2: s1.isna() | s2.isna() | (s1 <= s2),
    
    # Release date should not be in the future
    ("album_release_date",): lambda dates: dates.isna() | (dates <= pd.Timestamp.now()),
    
    # Birth date should not be after today
    ("birth_date",): lambda dates: dates.isna() | (dates <= pd.Timestamp.now()),
}
