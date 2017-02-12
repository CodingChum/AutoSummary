import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from collections import defaultdict
from nltk.probability import FreqDist
from heapq import nlargest
from nltk.corpus import stopwords


def getTextTechCrunch(url):
    page = requests.get(url).text
    soup = BeautifulSoup(page , "html.parser")
    article = soup.find_all("div", { "class" : "article-entry text" });
    text = ' '.join(map(lambda p: p.text, article))
    return text

def summarize(text , n):
    sents = sent_tokenize(text)
    
    assert n <= len(sents)
    words = word_tokenize(text.lower())
    _stopwords = set(stopwords.words('english') + list(punctuation))
    
    words = [word for word in words if word not in _stopwords]
    freq = FreqDist(words)
    
    ranking = defaultdict(int)
    
    for i, sent in enumerate(sents):
        for w in word_tokenize(sent.lower()):
            if w in freq:
                ranking[i] += freq[w]
                
    sents_idx = nlargest(n , ranking , key = ranking.get)
    return [sents[j] for j in sorted(sents_idx)]

articleURL = 'https://techcrunch.com/2017/01/07/using-data-science-to-beat-cancer/'
text = getTextTechCrunch(articleURL)
print(summarize(text , 2))



