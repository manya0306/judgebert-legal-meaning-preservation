"""Bilingual (English/Hindi) legal lexicons. Hindi entries match whole whitespace-delimited tokens or
fixed phrases. These lists are a starting point, NOT an exhaustive legal ontology."""
import re

DEV = "\u0900-\u0963\u0966-\u097F"   # Devanagari WITHOUT danda (U+0964/0965) so punctuation ends a token

def en(p):  return re.compile(r"(?<![\w'])(?:%s)(?![\w])" % p, re.I)
def hi(p):  return re.compile(r"(?<![%s])(?:%s)(?![%s])" % (DEV, p, DEV))

MODALS = {
    "en": {
        "obligation": en(r"shall|must|has to|have to|is required to|are required to|required to|will|is to|are to|needs? to|obliged to"),
        "recommendation": en(r"should|ought to"),
        "permission": en(r"may|can|could|is allowed to|are allowed to|is permitted to|are permitted to|is entitled to|are entitled to|is free to"),
        "prohibited": en(r"prohibited|forbidden|not permitted|not allowed"),
    },
    "hi": {
        "obligation": hi(r"होगा|होगी|होंगे|पड़ेगा|पड़ेगी|पड़ेंगे|अनिवार्य"),
        "recommendation": hi(r"चाहिए|चाहिये"),
        "permission": hi(r"सकता|सकती|सकते|अनुमति है|स्वतंत्र है"),
        "future": re.compile(r"(?<![%s])[%s]*(?:ेगा|ेगी|ेंगे|एगा|एगी|एंगे)(?![%s])" % (DEV, DEV, DEV)),
        "prohibited": hi(r"निषिद्ध|वर्जित"),
    },
}
NEGATION = {
    "en": en(r"not|no|never|cannot|neither|nor|n't"),
    "hi": hi(r"नहीं|न|मत"),
}
WITHOUT = {"en": en(r"without"), "hi": hi(r"बिना")}

CONDITION = {
    "en": en(r"if|provided that|provided|subject to|in the event|in case of|in cases of|in case|when|where|upon|once|as long as"),
    "hi": hi(r"यदि|अगर|की स्थिति में|की दशा में|स्थिति में|जब तक|बशर्ते|जब"),
}
EXCEPTION = {
    "en": en(r"unless|except|excluding|other than|save for|apart from|save that"),
    "hi": hi(r"जब तक कि|को छोड़कर|के सिवाय|सिवाय|अन्यथा|के अलावा|बशर्ते कि"),
}
# strip subordinate clauses before judging modality / polarity of the MAIN clause
LEADING_COND = {
    "en": re.compile(r"^\s*(?:if|when|where|unless|in the event|in case|provided)\b[^,]*,\s*", re.I),
    "hi": re.compile(r"^\s*(?:यदि|अगर|जब)[^,]*,\s*(?:तो\s*)?"),
}
TRAILING_EXC = {
    "en": re.compile(r",?\s*\b(?:unless|except|provided that|save|other than|apart from)\b.*$", re.I),
    "hi": re.compile(r",?\s*(?:जब तक कि|जब तक|को छोड़कर|बशर्ते).*$"),
}

TIME_REL = {
    "en": {"within": en(r"within"), "before": en(r"before|prior to|no later than"), "after": en(r"after|following"),
           "by": en(r"by|on or before|until|till"), "during": en(r"during|throughout")},
    "hi": {"within": hi(r"के भीतर|भीतर|अंदर"), "before": hi(r"से पहले|पहले"), "after": hi(r"के बाद|बाद"),
           "by": hi(r"तक"), "during": hi(r"के दौरान|दौरान")},
}
SCOPE = {
    "en": {"restrictive": en(r"only|solely|exclusively|merely"), "universal": en(r"all|any|every|each|whole|entire"),
           "vague": en(r"some|certain|several|a few|various")},
    "hi": {"restrictive": hi(r"केवल|सिर्फ़|सिर्फ|मात्र"), "universal": hi(r"सभी|सारे|हर|प्रत्येक|किसी भी|कोई भी"),
           "vague": hi(r"कुछ|अनेक|विभिन्न")},
}
VAGUE_TIME = {"en": en(r"soon|promptly|shortly|a while|timely|later|eventually"), "hi": hi(r"जल्दी|शीघ्र|जल्द|कुछ समय|बाद में")}

# groups of legally salient notions: if the original uses a group and the simplification uses none of it -> omission
SALIENT = {
    "en": {"written form": en(r"written|in writing|writing"), "consent/permission": en(r"consent|permission|approval|authori[sz]ation|authori[sz]ed"),
           "notice": en(r"notice|notify|notification|inform|tell|told|intimat\w+")},
    "hi": {"written form": hi(r"लिखित|लिखकर"), "consent/permission": hi(r"अनुमति|सहमति|स्वीकृति"),
           "notice": re.compile("नोटिस|सूचना|सूचित|बता|जानकारी")},
}

ROLES = {
    "en": ["tenant", "landlord", "lessor", "lessee", "employee", "employer", "insurer", "policyholder", "family", "company",
           "contractor", "buyer", "seller", "borrower", "lender", "licensee", "licensor", "owner", "spouse", "children", "party", "parties"],
    "hi": ["किरायेदार", "मकान मालिक", "कर्मचारी", "नियोक्ता", "बीमाकर्ता", "पॉलिसीधारक", "परिवार", "कंपनी", "प्रतिस्पर्धी",
           "ठेकेदार", "क्रेता", "विक्रेता", "ऋणी", "ऋणदाता", "मालिक", "पति", "पत्नी", "बच्चे"],
}
MONTHS = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr", "may": "may_m", "june": "jun", "july": "jul",
    "august": "aug", "september": "sep", "october": "oct", "november": "nov", "december": "dec",
    "जनवरी": "jan", "फरवरी": "feb", "मार्च": "mar", "अप्रैल": "apr", "मई": "may_m", "जून": "jun", "जुलाई": "jul",
    "अगस्त": "aug", "सितंबर": "sep", "सितम्बर": "sep", "अक्टूबर": "oct", "नवंबर": "nov", "नवम्बर": "nov", "दिसंबर": "dec", "दिसम्बर": "dec",
}
NUMBER_WORDS = {
    "en": {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "sixty": 60, "ninety": 90, "hundred": 100},
    "hi": {"एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
           "बारह": 12, "पंद्रह": 15, "बीस": 20, "तीस": 30, "साठ": 60, "नब्बे": 90, "सौ": 100},
}
MULTIPLIERS = {
    "en": [(re.compile(r"one and a half times|1\.5 times|one and a half", re.I), "1.5x"), (re.compile(r"\btwice\b|\bdouble\b|two times", re.I), "2x"),
           (re.compile(r"\bhalf\b", re.I), "0.5x"), (re.compile(r"\bthrice\b|\btriple\b|three times", re.I), "3x")],
    "hi": [(re.compile(r"डेढ़ गुना|डेढ़"), "1.5x"), (re.compile(r"दोगुनी|दोगुना|दुगुना|दुगुनी"), "2x"), (re.compile(r"ढाई गुना|ढाई"), "2.5x"),
           (re.compile(r"तिगुना|तिगुनी"), "3x"), (re.compile(r"आधा|आधी"), "0.5x")],
}
