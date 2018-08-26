from django.shortcuts import render
from django.http import HttpResponse
from gensim.summarization import keywords
from sklearn.feature_extraction.text import CountVectorizer


import requests
import urllib
import json
import numpy as np
import xml.etree.ElementTree as ET




def ifi_querier(q, fl="", start="0", rows="20", sort="", facet=False, facet_field=False, wt='json'):
    params = {
        "q":q,
        "indent":"true",
        "fl":fl,
        "start":start,
        "sort":sort,
        "rows":rows,
        "wt":wt}             
    payload = {'x-password': 'snK3zxC5xn4M2JkN', 'x-user': 'patcloud_prem'}
    query = "http://cdws.ificlaims.com/search/query?" + urllib.parse.urlencode(params)
    response = requests.get(query, headers=payload)
    response_in_json = json.loads(response.text)
    return response_in_json

def ifi_content(ucid):
    payload = {'x-password': 'snK3zxC5xn4M2JkN', 'x-user': 'patcloud_prem'}
    query = "http://cdws.ificlaims.com/text/fetch?ucid=" + ucid
    response = requests.get(query, headers=payload)
    root = ET.fromstring(response.text)
    node=root.findall("./bibliographic-data/ifi-integrated-content/ifi-parties/ifi-standardized-name/addressbook/name")
    node += root.findall("./bibliographic-data/ifi-integrated-content/ifi-parties/ifi-standardized-name-current/addressbook/name")
    node2=root.findall("./abstract/p")
    if len(node) > 0 and len(node2) > 0:
        return node[0].text, node2[0].text
    
    
# Create your views here.
def index(request):
    return render(request,'entrance.html')


def result(request):
    q =  request.POST['query']
    query = "tac:'" + q + "' AND pnctry:us AND ifi_publication_type:G"
    resp = ifi_querier(q=query, fl="ucid", rows="10")
    
    absts = list()
    companies = list()
    kws = list()
    ucids = list()
    for doc in resp['content']['response']['docs']:
        ucid = doc['ucid']
        if ifi_content(ucid) != None:
            company, abst = ifi_content(ucid)
        absts.append(abst)
        companies.append(company)
        ucids.append(ucid)
        #print(abst)
        #print(keywords(abst))
        kws.extend(keywords(abst).split('\n'))
    kws = list(set(kws))[1:10]
    vect = CountVectorizer(vocabulary=kws)
    
    dtm = vect.fit_transform(absts)
    #print(vect.get_feature_names())
    #print(dtm.toarray())
    
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
        graph['nodes'].append({'name':ucid,'group':2})
    
    return render(request, 'force.html',
                  {'dataset':graph})
    
