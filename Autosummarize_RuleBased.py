"""
Created on Sat Apr 11 15:15:42 2020

@author: Amogh
"""


from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from string import punctuation




text = """ Apple just announced a new iPad Pro starting from $799 all the way up to $1649. Apple describes it as “Your next computer is not a computer”. The top of the line iPad Pro 12.9-inch with 1TB Storage along with its new Magic Keyboard costs a whopping $1998 + taxes.  But the big question is that can it really replace a laptop?

The new iPad Pro comes in 2 sizes, 11-inch and 12.9-inch sizes look identical to last year’s models, but there are a new A12Z processor and new camera system. Apple’s headline feature is that it has a LIDAR scanner to go along with its camera for depth sensing and AR. But perhaps more notable than the iPad Pro itself is Apple’s new Keyboard with the trackpad which costs $299 for 11-inch and $349 for the 12.9-inch model. Apple calls it Magic Keyboard, the new keyboard supports a USB‑C port for passthrough charging, Full-size keyboard with backlit key and a scissor mechanism with 1 mm travel for quiet & responsive typing. Apple also upgraded its processor to A12Z which now has 8-core and is faster than most laptops out in the market.

The most common tasks people use their laptops for are surfing the web, checking emails, Taking Notes, playing games, and watching movies are the same tasks that the iPad can perform. In fact, the iPad can perform some tasks better with the help of touch screen and Apple pencil accessory one can easily sketch, doodle, annotate, handwrite, and much more. With the help of a camera, one can easily record photos/videos and share it instantly using built-in 4G LTE so you need not have to search for a coffee shop with WiFi.

But there are many reasons that the iPad cannot totally replace a laptop. Firstly, there are specific applications which run only on Windows or Mac environment. Secondly, if you are the person who uses a laptop for manipulating large amounts of data, performing heavy tasks, playing intense games, Video Editing, and many memory-intensive tasks, the iPad is not a great choice.

Since iPads are lighter, versatile, secure, and probably a lot simple to use for most people. Couple that with your smartphone you have all the computing power you’ll need. Consider what you use your laptop, if it is just to check mail, take notes, browsing the web and some light tasks you’ll be fine with the iPad. If you still feel like you need a laptop if would suggest a Windows laptop or a MacBook.

Personally, I feel that the iPad cannot totally replace laptops. It is just an additional accessory along with a laptop and It perfectly fills the gap between a smartphone and a laptop. Apple now sells three basic iPad models—the standard version, the Air, and the Pro—as well as the cut-sized Mini The iPad range starts from $329 all the way up to $1649 toss in some accessories like apple pencil and magic keyboard and it will easily cross 2 grand. The new iPad Pro is available starting March 25 and the Magic Keyboard is coming in May. 
"""

#breaking down into sentences and words

sents = sent_tokenize(text)
sents


word_sent = word_tokenize(text.lower())
word_sent

#removing stop words and puntuation
_stopwords = set(stopwords.words('english') + list(punctuation))
_stopwords


word_sent=[word for word in word_sent if word not in _stopwords]
word_sent

#Constructing freq dist table
from nltk.probability import FreqDist
freq = FreqDist(word_sent)
freq

# Top 10 occurred words
from heapq import nlargest
nlargest(10, freq, key=freq.get)




from collections import defaultdict
ranking = defaultdict(int)

for i,sent in enumerate(sents):
    for w in word_tokenize(sent.lower()):
        if w in freq:
            ranking[i] += freq[w]
            
ranking

sents_idx = nlargest(4, ranking, key=ranking.get)
sents_idx



[sents[j] for j in sorted(sents_idx)]


#to sum it up
def summarize(text, n):
    sents = sent_tokenize(text)
    
    assert n <= len(sents)
    word_sent = word_tokenize(text.lower())
    _stopwords = set(stopwords.words('english') + list(punctuation))
    
    word_sent=[word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)
    
    
    ranking = defaultdict(int)
    
    for i,sent in enumerate(sents):
        for w in word_tokenize(sent.lower()):
            if w in freq:
                ranking[i] += freq[w]
             
        
    sents_idx = nlargest(n, ranking, key=ranking.get)
    return [sents[j] for j in sorted(sents_idx)]

summarize(text,3)



