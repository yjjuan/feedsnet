# -*- coding: utf-8 -*-
"""
Created on Sat May 19 11:53:01 2018

@author: kev
ref:https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch09/movieclassifier_with_update/update.py
"""

import pickle
import requests
import json
import numpy as np
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

start_time = time.time()
#### Query for new unmonetized or monetiz data
#q1 = 'ic:H AND lsconv:"ASSIGNMENT OF ASSIGNORS INTEREST" AND ifi_publication_type:g AND pd_d:[2018-04-01T00:00:00.000Z TO 2018-05-01T00:00:00.000Z]'
q1 = 'ic:H AND lsconv:license AND ifi_publication_type:g AND pd_d:[2018-04-01T00:00:00.000Z TO 2018-05-01T00:00:00.000Z]'
resp1 = ifi_querier(q1, rows=1000)
numFound = resp1['content']['response']['numFound']
monetiz_ucid = [i['ucid'] for i in resp1['content']['response']['docs']]

q2 = 'ic:H NOT lsconv:"ASSIGNMENT OF ASSIGNORS INTEREST" AND ifi_publication_type:g AND pd_d:[2018-04-01T00:00:00.000Z TO 2018-05-01T00:00:00.000Z]'
resp2 = ifi_querier(q2, rows=numFound)
unmonetiz_ucid = [i['ucid'] for i in resp2['content']['response']['docs']]

#### Vectorize the found patents.
mon_vec_list = list()
for ucid in monetiz_ucid:
    
    resp = ifi_fcit(ucid)
    fcit = len(resp['citations'][0]['ucids'])
    
    resp = ifi_family(ucid)
    family_size = len(resp['family']['members'])
    
    patent_num = ucid.split("-")[1]
    response0 = ifi_querier('pnnum:{} AND (lscode:ipr OR lscode:iprc OR lscode:pgr OR lscode:pgrc)'.format(patent_num))
    ipr = response0['content']['response']['numFound'] > 0
    
    response1 = ifi_querier('pnnum:{} AND pnctry:us AND ifi_document_category:reissue'.format(patent_num))
    reissue = response1['content']['response']['numFound'] > 0
    response2 = ifi_querier('relpnnum:{} AND ifi_document_category:reissue'.format(patent_num))
    reissued = response2['content']['response']['numFound'] > 0
                           
    mon_vec_list.append([family_size, fcit, ipr, reissue or reissued])

unmon_vec_list = list()
for ucid in unmonetiz_ucid:
    
    resp = ifi_fcit(ucid)
    fcit = len(resp['citations'][0]['ucids'])
    
    resp = ifi_family(ucid)
    family_size = len(resp['family']['members'])
    
    patent_num = ucid.split("-")[1]
    response0 = ifi_querier('pnnum:{} AND (lscode:ipr OR lscode:iprc OR lscode:pgr OR lscode:pgrc)'.format(patent_num))
    ipr = response0['content']['response']['numFound'] > 0
    
    response1 = ifi_querier('pnnum:{} AND pnctry:us AND ifi_document_category:reissue'.format(patent_num))
    reissue = response1['content']['response']['numFound'] > 0
    response2 = ifi_querier('relpnnum:{} AND ifi_document_category:reissue'.format(patent_num))
    reissued = response2['content']['response']['numFound'] > 0
                           
    unmon_vec_list.append([family_size, fcit, ipr, reissue or reissued])

X_train = np.array(mon_vec_list + unmon_vec_list)
y = np.array([1]*numFound + [0]*numFound)
print ((time.time()-start_time)/60)


clf = pickle.load(open('classifier.pickle', 'rb'))
classes = np.array([0, 1])
clf.partial_fit(X_train, y, classes=classes)
pickle.dump(open('classifier2.pickle', 'wb'))
'''
cur_dir = os.path.dirname(__file__)

clf = pickle.load(open(os.path.join(cur_dir,
                  'pkl_objects',
                  'classifier.pkl'), 'rb'))
classes = np.array([0, 1])
clf.partial_fit(X_train, y, classes=classes)
'''