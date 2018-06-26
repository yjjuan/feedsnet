from django.shortcuts import render
from django.http import HttpResponse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing

from nltk.corpus import stopwords

import requests
import urllib
import json
import pickle
import os
from .deployExplainer2 import explainer
import numpy as np
import xml.etree.ElementTree as ET
from tinydb import TinyDB, Query

from keras.models import load_model, model_from_json
import keras
from uspto.pbd.client import UsptoPairBulkDataClient

def ifi_querier(q, fl="", start="0", rows="10", sort="", facet=False, facet_field=False, wt='json'):
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

def ifi_ipc(ucid):
    payload = {'x-password': 'snK3zxC5xn4M2JkN', 'x-user': 'patcloud_prem'}
    query = "http://cdws.ificlaims.com/text/fetch?ucid=" + ucid
    response = requests.get(query, headers=payload)
    root = ET.fromstring(response.text)
    node=root.findall("./bibliographic-data/technical-data/classifications-ipcr/classification-ipcr")
    ipc_section = node[0].text[0]
    return ipc_section
    
def ifi_fcit(ucid):
    payload = {'x-password': 'snK3zxC5xn4M2JkN', 'x-user': 'patcloud_prem'}
    query = 'http://cdws.ificlaims.com/citations/forward?ucid=' + ucid
    response = requests.get(query, headers=payload)
    response_in_json = json.loads(response.text)
    return response_in_json

def ifi_family(ucid):
    payload = {'x-password': 'snK3zxC5xn4M2JkN', 'x-user': 'patcloud_prem'}
    query = 'http://cdws.ificlaims.com/family/simple?ucid=' + ucid
    response = requests.get(query, headers=payload)
    response_in_json = json.loads(response.text)
    return response_in_json 

def status(apn):
    client = UsptoPairBulkDataClient()
    expression = "applId:" + apn
    result = client.search(expression, start=0,rows=20)
    status_desc = [result['docs'][0]['appStatus']]
    granted = ['Patented Case',
           'Publications -- Issue Fee Payment Verified',
           'Publications -- Issue Fee Payment Received',
           'Notice of Allowance Mailed -- Application Received in Office of Publications',
           'Patent Expired Due to NonPayment of Maintenance Fees Under 37 CFR 1.362',
           'Abandoned  --  Failure to Pay Issue Fee',
           'Notice of Allowance Mailed -- Application Received in Office of Publications'] 
    rejected = ["Abandoned  --  After Examiner's Answer or Board of Appeals Decision",
                'Abandoned  --  Failure to Respond to an Office Action',
                'Final Rejection Mailed',
                'Expressly Abandoned  --  During Examination']
    pending = ['Advisory Action Mailed','Application Undergoing Preexam Processing',
               'Docketed New Case - Ready for Examination',
               'Non Final Action Mailed',
               'Response to Non-Final Office Action Entered and Forwarded to Examiner']
    # -1:pending; 0:rejected; 1: granted; 2:not sure
    exam_status_label = [1 if i in granted else 0 if i in rejected else -1 if i in pending else 2 for i in status_desc]
    
    return exam_status_label[0]

# Create your views here.
def index(request):
    return render(request,'entrance.html')


def result(request):

    cur_dir = os.path.dirname(__file__)
  
    #### Select model and training data    
    keras.backend.clear_session() # kill previous model record
    #print(os.path.join(cur_dir,'bernier.h5'))
    ex_name = request.POST['examiner_name']
    model = load_model(os.path.join(cur_dir,ex_name+'.h5'))    
    
    #json_string = codecs.open(os.path.join(cur_dir,'model_json.json'), 'r', encoding='utf-8').read()
    #json_string  = json.load(open(os.path.join(cur_dir,'model_json.json')))
    #model = model_from_json(json_string)
    #model.load_weights(os.path.join(cur_dir,'model_weights.h5'))
    model_date = '2018XXXYYY'
    
    ### Test model's predictive power on random input
    print('test model...')
    x_test = np.random.rand(1,12)
    print(model.predict(x_test))
    print('test model done...')
        
    
    #### Start to vectorize this exam
    ### keywords hit rate
    ## Load the claim-text
    claims = [request.POST[i] for i in request.POST.keys() if i[-4:]=='text']
    
    ## Keyword extraction
    count = CountVectorizer()
    
    bag = count.fit_transform(claims)
    tokenIndex = count.vocabulary_ #index for each token
    
    tokenMapping = {}
    for word, num in tokenIndex.items():
        tokenMapping[num]=word
    #print (tokenMapping)
    #print (bag.toarray(),bag.toarray().shape)
    
    tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    tfidfArray = tfidf.fit_transform(bag).toarray()
    tfidfArray=np.round(tfidfArray,decimals=2)
    #print(tfidfArray)
    #print (np.nonzero(np.prod(tfidfArray,axis=0)))
    
    stop = stopwords.words('english')
    
    keywords = []
    # tf-idf > 0.2 in at least one document
    for i in np.nonzero(np.sum(tfidfArray >= 0.2,axis=0))[0]:
        if tokenMapping[i] not in stop:
            #print (tokenMapping[i])    
            keywords.append(tokenMapping[i])
    keyword_tfidf = np.max(tfidfArray,axis=0) # Use max of tf-idf as tf-idf for that keyword 
    for i in np.argsort(keyword_tfidf)[::-1]: #sort by tfidf of keywords
        if tokenMapping[i] not in stop:
            #print (tokenMapping[i])    
            keywords.append(tokenMapping[i])        
            
    keywords = keywords[:30] # Extract at most 30 keywords  
    
    ## use keywords to hit prior arts
    priorArts = [str(i) for i in request.POST['prior_arts'].split(',')]
    #print(request.POST.keys())
    
    hit_list = list()
    for pa in set(priorArts):
        #print(pa)
        hit = list()
        for keyword in keywords:
            r = ifi_querier("text:{} AND pnnum:{}".format(keyword,pa)) 
            hit.extend([1 if r['content']['response']['numFound'] > 0 else 0])
        hit_list.append(hit)
    #print(hit_list)
    
    ## Calculate the hit rate
    hit_rate = [round(sum(i)/len(keywords),2) for i in hit_list]
    
    ### Claims ratio rejected by 101, 102, 103
    total_claims = int(request.POST['total_claims'])
    claims101 = [int(i) for i in request.POST['claims101'].split(',')]
    claims102 = [int(i) for i in request.POST['claims102'].split(',')]
    claims103 = [int(i) for i in request.POST['claims103'].split(',')]
    
    claimsRatio101 = [round(i/total_claims,2) for i in claims101]
    claimsRatio102 = [round(i/total_claims,2) for i in claims102]
    claimsRatio103 = [round(i/total_claims,2) for i in claims103]
                           
    case = np.array([max(hit_rate), sum(hit_rate)/len(hit_rate), min(hit_rate),
                     max(claimsRatio101), sum(claimsRatio101)/len(claimsRatio101),min(claimsRatio101),
                     max(claimsRatio102), sum(claimsRatio102)/len(claimsRatio102),min(claimsRatio102),
                     max(claimsRatio103), sum(claimsRatio103)/len(claimsRatio103),min(claimsRatio103)])
    case = case.reshape(1,12)
    # vector must be scaled before fed into model
    case = preprocessing.scale(case)
    
    # Prediction part
    grant_prob = round(model.predict(case)[0][1], 2)
    
    
    # Explanation part
    db = TinyDB(os.path.join(cur_dir,'exam2vec_v2.json'))
    data = db.all()
    
    x = np.array([i['vec'] for i in data if i['label'] in [0,1]])
    y = [i['label'] for i in data if i['label'] in [0,1]]
    
    
    feature_names = ['Max(hit rate)','Mean(hit rate)','Min(hit rate)',
                     'Max(101)','Mean(101)','Min(101)',
                     'Max(102)','Mean(102)','Min(102)',
                     'Max(103)','Mean(103)','Min(103)']
    class_names = ['rejected','granted']
    #print(x.shape)
    exp = explainer(x,feature_names = feature_names,class_names=class_names,
                    categorical_features=[],
                    categorical_names=[],
                    kernel_width=3)
    
    explan = exp.explain(model, case.reshape(12))
    desc_list = [explan[i]['desc'] for i in explan]
    weight_list = [explan[i]['weight'] for i in explan]
    
    # Prepare format for highchart viz
    idx_sorted = np.argsort(np.abs(weight_list))
    desc_grant = [desc_list[i] if weight_list[i]>=0 else '' for i in idx_sorted]
    desc_rej = [desc_list[i] if weight_list[i]<0 else '' for i in idx_sorted]
    weight_grant = [round(weight_list[i],2) if weight_list[i]>=0 else 0 for i in idx_sorted]
    weight_rej = [round(weight_list[i],2) if weight_list[i]<0 else 0 for i in idx_sorted]
    
    # Monitor the prediction over pending applications
    q = Query()
    pend_app = [i['appNumber'] for i in db.search(q.label == -1)]
    x_pend = np.array([i['vec'] for i in db.search(q.label == -1)])
    pend_proba = [round(i[1],2) for i in model.predict(x_pend)]
    pend_now = [status(i) for i in pend_app]

    
    return render(request, 'highchart7.html',
                  {'model_date':model_date,
                   'examiner_name': ex_name,
                   'prior_arts':priorArts,
                   'hit_rate':hit_rate,
                   'claims_ratio101':claimsRatio101,
                   'claims_ratio102':claimsRatio102,
                   'claims_ratio103':claimsRatio103,
                   'claims':claims,
                   'grant_proba':grant_prob,
                   'exam_counts':x.shape[0],
                   'desc_grant':desc_grant,
                   'desc_rej':desc_rej,
                   'weight_grant':weight_grant,
                   'weight_rej':weight_rej,
                   'pend_app':pend_app,
                   'pend_proba':pend_proba,
                   'pend_now':pend_now})
    
