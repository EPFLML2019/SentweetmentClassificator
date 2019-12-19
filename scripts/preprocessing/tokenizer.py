"""
Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

import sys
import re
from nltk.tokenize import TweetTokenizer
import ntpath

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body)
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


def tokenize(text, string=False):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dDp\]]+|[(dDp\[]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"{}{}[(\[oOsS\\@/|l*]+|[\])oOsS\\@/|l*]+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"</3|<\\3","<broken_heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    #text = re_sub(r"([./<>,!?:])\1",r" \1 ")

    text = re_sub(r"([A-Z]){2,}", allcaps)
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True,  preserve_case=True)

    if string:
        return " ".join(tknzr.tokenize(text))
    else:
        return tknzr.tokenize(text)

if __name__ == '__main__':
    _, filename = sys.argv
    with open('pre_processed_' + ntpath.basename(filename), 'a+') as output:
        with open(filename) as fp:
            for cnt, line in enumerate(fp):
                tokens = tokenize(line, string=True)
                output.write(tokens + " \n")
    print ("DONE!")