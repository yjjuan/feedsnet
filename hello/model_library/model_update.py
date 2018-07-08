# -*- coding: utf-8 -*-
"""
Created on Sat May 19 11:53:01 2018

@author: kev
ref:https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch09/movieclassifier_with_update/update.py
"""

import requests
import json
import numpy as np
import urllib
import time
from tinydb import TinyDB, Query
import os
from exam_history_v2 import vectorize


from uspto.peds.client import UsptoPatentExaminationDataSystemClient
from .model import Examiner

from keras.models import load_model, model_from_json
from keras.utils import to_categorical
import keras

import sklearn.preprocessing

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

def status(apn):
    client = UsptoPatentExaminationDataSystemClient()
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

start_time = time.time()
examiner_name='JWALANT B AMIN'

#### Query for new unmonetized or monetiz data
# Load the model
model = load_model('{}.h5'.format(examiner_name))  

### Check the prediction over pending applications with actual outcome
db = TinyDB('{}.json'.format(examiner_name))
q = Query()
pend_app = [i['appNumber'] for i in db.search(q.label == -1) if i['use']==1]
x_pend = np.array([i['vec'] for i in db.search(q.label == -1) if i['use']==1])
pend_proba = [round(i[1],2) for i in model.predict(x_pend)]
pend_pred = [i > 0.5 for i in pend_proba]
pend_now = [status(i) for i in pend_app]

pend_checked = [1 if pend_pred[idx]==ans else 0 for idx, ans in enumerate(pend_now) if ans in [0, 1]]

#### Re-train the model if accuracy < 0.5 for over 10 cases
retrain = False
if len(pend_checked) >= 10 and sum(pend_checked)/len(pend_checked)<0.5:
    retrain = True

#### Update db for new data
### Loading the latest exam app numbers in exam history
client = UsptoPatentExaminationDataSystemClient()
expression = "appExamName:({})".format(examiner_name)

result = client.search(expression, start=0,rows=20)

numFound=result['numFound']
exam_status=[]
exam_app = list()

start_time = time.time()

for query_times in range(numFound//20 + 1):
    #print("{} items processed----".format(query_times*20),(time.time()-start_time)/60,'mins')
    
    result = client.search(expression, start=query_times*20,rows=20)
    
    for doc in result['docs']:
        # skip the exam record w/ missing fields
        if 'applId' in doc.keys() and 'appStatus' in doc.keys():        
            exam_status.extend([doc['appStatus']])
            exam_app.extend([doc['applId']])
### Check and update the legal status change in pending or unsure patents
for i in db.search(q.label == -1)+db.search(q.label == 2):
    if i['label'] != status(i['appNumber']):
        print(i['appNumber'])
        i['label'] = status(i['appNumber'])
        
### Check and update db for new exam records
apn_old = [i['appNumber'] for i in db.all()]
apn_new = [i for i in exam_app if i not in apn_old]
## Vectorize the new record, and insert new exam vectors
for idx, apn in enumerate(apn_new):
    vec = vectorize(apn)
    use = 0
    if type(vec)!=str and sum(vec) != 0: # no record of prior arts from uspto
        #print(num, vec)
        use = 1
    db.insert({'idx':'new '+ str(idx), 
               'appNumber': apn, 
               'use': use,
               'vec': vec,
               'label':status(apn)})
### Retrain new model
if retrain:
    data = db.all()
    
    x = np.array([i['vec'] for i in data if i['label'] in [0,1] and i['use']==1])
    y = [i['label'] for i in data if i['label'] in [0,1] and i['use']==1]
    
    #scaling
    x = sklearn.preprocessing.scale(x)
    
    # split for train and test set
    # Extract 20% in each class to be test set
    test_size = [y.count(i)//5 for i in range(2)]
    idx_test = [j for i in range(2) for j in np.random.choice(np.where(np.array(y)==i)[0],
                                          size=test_size[i], 
                                          replace=False)]
    idx_train = [i for i in range(x.shape[0]) if i not in idx_test]
    x_train = np.array([x[i,:] for i in idx_train])
    y_train = to_categorical([y[i] for i in idx_train])  ### keras onehot encoding
    x_test = np.array([x[i,:] for i in idx_test])
    y_test = to_categorical([y[i] for i in idx_test])
    model.fit(x_train,y_train,validation_split=0.2,
                              batch_size=10,epochs=200)
    model.save('{}.h5'.format(examiner_name))