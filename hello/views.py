from django.shortcuts import render
from django.http import HttpResponse
from gensim.summarization import keywords
from sklearn.feature_extraction.text import CountVectorizer


import requests
import urllib
import json
import numpy as np
import xml.etree.ElementTree as ET
    
    
# Create your views here.
def index(request):
    return render(request,'entrance.html')

def result(request):

    feed1 = 'https://www.mdpi.com/rss/journal/sensors'
    feed2 = 'https://rss.sciencedirect.com/publication/science/09252312'
    
    absts = list()
    titles = list()
    links = list()
    ids = list()
    
    #### Read the necessary information from RSS feeds
    response = requests.get(feed1)
    root = ET.fromstring(response.text)
    
    nodes = root.findall("./{http://purl.org/rss/1.0/}item")    
    for idx, node in enumerate(nodes):
        titles.append(node.findall("./{http://purl.org/dc/elements/1.1/}title")[0].text)
        absts.append(node.findall('./{http://purl.org/rss/1.0/}description')[0].text)
        links.append(node.findall('./{http://purl.org/rss/1.0/}link')[0].text)
        ids.append(idx)
    
    ### Feed 2
    feeds = ['https://rss.sciencedirect.com/publication/science/09252312',
             'http://rss.sciencedirect.com/publication/science/01681699',
             'http://feeds.rsc.org/rss/ay',
             'https://link.springer.com/search.rss?facet-content-type=Article&facet-journal-id=11263&channel-name=International+Journal+of+Computer+Vision']
    
    for feed in feeds:
        response = requests.get(feed)
        root = ET.fromstring(response.text)    
        nodes = root[0].findall("item")
        for idx2, node in enumerate(nodes):
            titles.append(node.findall("title")[0].text)
            absts.append(node.findall('description')[0].text)
            links.append(node.findall('link')[0].text)
            ids.append(ids[-1]+1)
    
    kws = ['convolutional','texture','nir','near-infrared',
           'spectrometer', 'spectrogram','recurrent','time series',
           'recommend engine','food security','tea','argriculture','wearable',
           'transfer learning','beer','LSTM']
    #### Count the occurence of kw in each documents by sklearn countvectorizer
    vect = CountVectorizer(vocabulary=kws)
    dtm = vect.fit_transform(absts)
    #print(vect.get_feature_names())
    #print(dtm.toarray())
    
    #### Draw the graph based on count vectorizer
    kws_counts = dtm.toarray().shape[1]
    docs_counts = dtm.toarray().shape[0]
    graph = {'nodes':list(),
             'edges':list()}
    
    for i in range(kws_counts):
        targets = np.nonzero(dtm.toarray()[:,i])[0]
        graph['nodes'].append({'name':kws[i],'group':1})
        for t in targets:
            graph['edges'].append({'source':i,'target':t + kws_counts})
    
    for i in range(docs_counts):
        graph['nodes'].append({'title':titles[i],'link':links[i],
             'group':2})
    
    return render(request, 'force2.html',
                  {'dataset':graph})
    
    

    
