# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 14:18:49 2017

@author: Huang Yen Jun
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import requests
import json
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET
import urllib
import time

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

def ifi_clms(ucid):
    payload = {'x-password': 'snK3zxC5xn4M2JkN', 'x-user': 'patcloud_prem'}
    query = "http://cdws.ificlaims.com/text/fetch?ucid=" + ucid
    response = requests.get(query, headers=payload)
    root = ET.fromstring(response.text)
    node_claims=root.findall("./claims")    
    
    node=root.findall("./bibliographic-data/ifi-integrated-content/ifi-claims-summary/ifi-claims")
    attribute = node[0].attrib
    nclms = int(attribute['total'])
    return [nclms, node_claims]

def oa_citations(q):
    fields = 'applicationId,ifwNumber,Patent_PGPub,form892,form1449,citationInOA,\
    actionType,actionSubtype,artUnit,documentCd,mailDate,uspcClass,uspcSubClass,\
    headerMissing,formParagraphMissing,rejectFpMisMatch,closingMissing,\
    rejection101,rejectionDp,rejection102,rejection103,rejection112,\
    allowedClaims,objection,cite102Gt1,cite103Gt3,cite103Eq1,cite103Max,\
    signatureType'
    resp = requests.post('https://developer.uspto.gov/ds-api/oa_citations/v1/records', 
                         params={"criteria":q,'fl':fields},
                         headers={"Content-Type":"application/x-www-form-urlencoded"})
    return json.loads(resp.text)

def oa_rejections(q):
    resp = requests.post('https://developer.uspto.gov/ds-api/oa_rejections/v1/records', params={"criteria":q},headers={"Content-Type":"application/x-www-form-urlencoded"})
    return json.loads(resp.text)

'''
def ifi_nclms(ucid):
    payload = {'x-password': 'snK3zxC5xn4M2JkN', 'x-user': 'patcloud_prem'}
    query = "http://cdws.ificlaims.com/text/fetch?ucid=" + ucid
    response = requests.get(query, headers=payload)
    root = ET.fromstring(response.text)
    node=root.findall("./bibliographic-data/ifi-integrated-content/ifi-claims-summary/ifi-claims")
    attribute = node[0].attrib
    return int(attribute['total'])
'''
def ucid_gen(appNumber):
    params = {
        "q":"anorig:{} AND ifi_publication_type:A AND pnctry:us".format(appNumber),
        "fl":'ucid'}             
    payload = {'x-password': 'snK3zxC5xn4M2JkN', 'x-user': 'patcloud_prem'}
    query = "http://cdws.ificlaims.com/search/query?" + urllib.parse.urlencode(params)
    response = requests.get(query, headers=payload)
    resp = json.loads(response.text)['content']['response']
    if resp['numFound'] == 1:
        return resp['docs'][0]['ucid']
    elif resp['numFound'] == 0:
        return 'No pub. no.'
    else:
        return 'More than one pub. no.'
    
def vectorize(appNumber):
    ##### Check the existence of early publication
    params = {
        "q":"anorig:{} AND ifi_publication_type:A AND pnctry:us".format(appNumber),
        "fl":'ucid'}             
    payload = {'x-password': 'snK3zxC5xn4M2JkN', 'x-user': 'patcloud_prem'}
    query = "http://cdws.ificlaims.com/search/query?" + urllib.parse.urlencode(params)
    response = requests.get(query, headers=payload)
    resp = json.loads(response.text)['content']['response']

    if resp['numFound'] != 1:
        return 'Not able to vectorize due to invalid pub. no.'
    else:
        
        #### Assign the publication number of corresponding appnumber
        pubNumber = resp['docs'][0]['ucid']
        #print(pubNumber)
        
        #### Vectorize one examination record
        ### keyword hit_rate
        ## Loading claim text
        clms_info = ifi_clms(pubNumber)
        node = clms_info[1]
        claims = list()
        for claim in node[0].findall("./claim"):
            claim_string = ''
            for claim_text in claim[0][1:]:   #Need to eliminate the subject to claim
                #print(claim_text)
                
                if claim_text.tag in ['claim-ref', 'b', 'i']:
                    content = [claim_text.tail if claim_text.tail != None else '']
                    #print(content)
                    claim_string = claim_string + content[0] + ' '
                elif claim_text.tag == 'claim-text':
                    content = [claim_text.text if claim_text.text != None else '']
                    #print(claim_text.text)
                    claim_string = claim_string + content[0] + ' '
                 
            claims.append(claim_string)
            
        ## keyword extraction
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
        #for i in np.nonzero(tfidfArray[5,:]>=0.2)[0]:
        #    print (tokenMapping[i])
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
        #print (len(keywords))
        

        
        ## Search for prior arts
        # Make query until complete info is fetched
        query = True
        while query:
            resp = oa_citations("applicationId:{}".format(appNumber))
            if len(resp['responseHeader']['params']['fl']) > 300:
                query = False
                print('Fetched')
                
        priorArts103 = [i['Patent_PGPub'][0] for i in resp['response']['docs'] if i['citationInOA']==['1'] and i['rejection103']==['1']]
        priorArts102 = [i['Patent_PGPub'][0] for i in resp['response']['docs'] if i['citationInOA']==['1'] and i['rejection102']==['1']]
        priorArts101 = [i['Patent_PGPub'][0] for i in resp['response']['docs'] if i['citationInOA']==['1'] and i['rejection101']==['1']]
        priorArts = priorArts101 + priorArts102 + priorArts103
        
        ## Use keywords to query prior arts, and calculate keywords hit rate
        hit_list = list()
        for pa in set(priorArts):
            hit = list()
            for keyword in keywords:
                r = ifi_querier("text:{} AND pnnum:{}".format(keyword,pa)) 
                hit.extend([1 if r['content']['response']['numFound'] > 0 else 0])
            hit_list.append(hit)
        hit_rate = [sum(i)/len(keywords) for i in hit_list]  
        
        ### Claims ratio rejected by 101, 102, 103
        nclms = clms_info[0]
        #print(nclms)
        resp1 = oa_rejections("applicationId:{}".format(appNumber))
        claims_103 = [i['claimNumbers'] for i in resp1['response']['docs'] if i['actionType']==['103']]
        claims_102 = [i['claimNumbers'] for i in resp1['response']['docs'] if i['actionType']==['102']]
        claims_101 = [i['claimNumbers'] for i in resp1['response']['docs'] if i['actionType']==['101']]
        
        claimRatio_103 = [len(i[0].split(','))/nclms for i in claims_103 if len(i)!=0]
        claimRatio_102 = [len(i[0].split(','))/nclms for i in claims_102 if len(i)!=0]
        claimRatio_101 = [len(i[0].split(','))/nclms for i in claims_101 if len(i)!=0]
        
        ## Check reset the claimRatio whose value > 1.0
        claimRatio_101 = [i if i<=1 else 1 for i in claimRatio_101 ]
        claimRatio_102 = [i if i<=1 else 1 for i in claimRatio_102 ]
        claimRatio_103 = [i if i<=1 else 1 for i in claimRatio_103 ]
        
        vec = list()
        vec.extend([[max(hit_rate), sum(hit_rate)/len(hit_rate), min(hit_rate)] if len(hit_rate)!=0 else [0,0,0]][0])
        vec.extend([[max(claimRatio_101), sum(claimRatio_101)/len(claimRatio_101), min(claimRatio_101)] if len(claimRatio_101)!=0 else [0,0,0]][0])
        vec.extend([[max(claimRatio_102), sum(claimRatio_102)/len(claimRatio_102), min(claimRatio_102)] if len(claimRatio_102)!=0 else [0,0,0]][0])
        vec.extend([[max(claimRatio_103), sum(claimRatio_103)/len(claimRatio_103), min(claimRatio_103)] if len(claimRatio_103)!=0 else [0,0,0]][0])
        
        ## Need count of overlapped IPC
        
        return vec
    
if __name__ == '__main__':
    start_time = time.time()
    print(vectorize('12664088'))
    print((time.time()-start_time)/60,'mins')