from django.shortcuts import render
from django.http import HttpResponse

import requests
import urllib
import json
import pickle
import os
from .deployExplainer import explainer
import numpy as np
import xml.etree.ElementTree as ET

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

# Create your views here.
def index(request):
    return render(request,'entrance.html')


def result(request):

    cur_dir = os.path.dirname(__file__)

    ipr_list = pickle.load(open(os.path.join(cur_dir,
                      'ipr_issueNum.pickle'), 'rb')) 
    
    #### Get the corresponding ucid for input patent number
    patent_num = request.POST['patent_number']
    resp0 = ifi_querier('pnnum:{} AND pnctry:us'.format(patent_num))
    ucid = resp0['content']['response']['docs'][0]['ucid']
    
    #### Select IPC-dependent model and training data
    ipc = ifi_ipc(ucid)
    if ipc == 'C':
        train = pickle.load(open(os.path.join(cur_dir,
                          'train_C.pickle'), 'rb'))    
        model = pickle.load(open(os.path.join(cur_dir,
                          'classifier_C.pickle'), 'rb'))  
        model_name = 'Cathey'
        model_desc = 'CHEMISTRY; METALLURGY'
    else:
        train = pickle.load(open(os.path.join(cur_dir,
                          'train.pickle'), 'rb'))    
        model = pickle.load(open(os.path.join(cur_dir,
                          'classifier.pickle'), 'rb')) 
        model_name = 'Hibert'
        model_desc = 'ELECTRICITY'
        
    
    #### Start to vectorize this patent
    resp = ifi_fcit(ucid)
    fcit = len(resp['citations'][0]['ucids'])
    
    resp = ifi_family(ucid)
    family_size = len(resp['family']['members'])
    
    ipr = patent_num in ipr_list
    
    response1 = ifi_querier('pnnum:{} AND pnctry:us AND ifi_document_category:reissue'.format(patent_num))
    reissue = response1['content']['response']['numFound'] > 0
    response2 = ifi_querier('relpnnum:{} AND ifi_document_category:reissue'.format(patent_num))
    reissued = response2['content']['response']['numFound'] > 0
                           
    case = np.array([family_size, fcit, ipr, reissue or reissued])
    
    # Prediction part
    pred_value = model.predict(case.reshape(1,-1))[0]
    pred_class = ['monetized' if pred_value == 1 else 'unmonetized'][0]
    pred_proba = round(model.predict_proba(case.reshape(1,-1))[0][pred_value]*100,2)

    # Explanation part
    feature_names = ['familySize','citedby_patent_count','iprFlag','reissued_combined']
    class_names = ['unmonetized','monetized']
    categorical_features = [2,3]
    categorical_names = {2:['False','True'], 3:['False','True']}
    exp = explainer(train,feature_names = feature_names,class_names=class_names,
                                categorical_features=categorical_features,
                                categorical_names=categorical_names, kernel_width=3)
    
    explan = exp.explain(model, case)
    desc_list = [explan[i]['desc'] for i in explan]
    weight_list = [explan[i]['weight'] for i in explan]
    
    # Prepare format for highchart viz
    idx_sorted = np.argsort(np.abs(weight_list))
    desc_mon = [desc_list[i] if weight_list[i]>=0 else '' for i in idx_sorted]
    desc_unmon = [desc_list[i] if weight_list[i]<0 else '' for i in idx_sorted]
    #desc_unmon = ['', '', '', 'reissued_combined is false']
    weight_mon = [round(weight_list[i],2) if weight_list[i]>=0 else 0 for i in idx_sorted]
    weight_unmon = [round(weight_list[i],2) if weight_list[i]<0 else 0 for i in idx_sorted]

    #print(explainer.explain(model, np.array([0, 5, 1, 0])))
    #return HttpResponse(exp.explain(model, np.array([0, 5, 1, 0])))
    return render(request, 'highchart7.html',
                  {'patent_number':patent_num, 'model_name':model_name, 'model_desc':model_desc,
                   'family_size':case[0], 'fcit':case[1],'ipr':ipr, 
                   'reissue':reissue or reissued,'pred_class':pred_class,
                   'pred_proba':pred_proba, 'desc_mon':desc_mon, 
                   'desc_unmon':desc_unmon, 'weight_mon':weight_mon, 
                   'weight_unmon':weight_unmon})


def control_1(request):

    cur_dir = os.path.dirname(__file__)

    ipr_list = pickle.load(open(os.path.join(cur_dir,
                      'ipr_issueNum.pickle'), 'rb')) 
    
    #### Get the corresponding ucid for input patent number
    patent_num = request.POST['patent_number']
    resp0 = ifi_querier('pnnum:{} AND pnctry:us'.format(patent_num))
    ucid = resp0['content']['response']['docs'][0]['ucid']
    
    #### Select IPC-dependent model and training data
    ipc = ifi_ipc(ucid)
    if ipc == 'C':
        train = pickle.load(open(os.path.join(cur_dir,
                          'train_C.pickle'), 'rb'))    
        model = pickle.load(open(os.path.join(cur_dir,
                          'classifier_C.pickle'), 'rb'))  
        model_name = 'Cathey'
        model_desc = 'CHEMISTRY; METALLURGY'
    else:
        train = pickle.load(open(os.path.join(cur_dir,
                          'train.pickle'), 'rb'))    
        model = pickle.load(open(os.path.join(cur_dir,
                          'classifier.pickle'), 'rb')) 
        model_name = 'Hibert'
        model_desc = 'ELECTRICITY'
        
    
    #### Start to vectorize this patent
    resp = ifi_fcit(ucid)
    fcit = len(resp['citations'][0]['ucids'])
    
    resp = ifi_family(ucid)
    family_size = len(resp['family']['members'])
    
    ipr = patent_num in ipr_list
    
    response1 = ifi_querier('pnnum:{} AND pnctry:us AND ifi_document_category:reissue'.format(patent_num))
    reissue = response1['content']['response']['numFound'] > 0
    response2 = ifi_querier('relpnnum:{} AND ifi_document_category:reissue'.format(patent_num))
    reissued = response2['content']['response']['numFound'] > 0
                           
    case = np.array([family_size, fcit, ipr, reissue or reissued])
    
    # Prediction part
    pred_value = model.predict(case.reshape(1,-1))[0]
    pred_class = ['monetized' if pred_value == 1 else 'unmonetized'][0]
    pred_proba = round(model.predict_proba(case.reshape(1,-1))[0][pred_value]*100,2)


    return render(request, 'control1.html',
                  {'patent_number':patent_num, 'model_name':model_name, 'model_desc':model_desc,
                   'family_size':case[0], 'fcit':case[1],'ipr':ipr, 
                   'reissue':reissue or reissued,'pred_class':pred_class,
                   'pred_proba':pred_proba})

def control_2(request):

    cur_dir = os.path.dirname(__file__)

    ipr_list = pickle.load(open(os.path.join(cur_dir,
                      'ipr_issueNum.pickle'), 'rb')) 
    
    #### Get the corresponding ucid for input patent number
    patent_num = request.POST['patent_number']
    resp0 = ifi_querier('pnnum:{} AND pnctry:us'.format(patent_num))
    ucid = resp0['content']['response']['docs'][0]['ucid']
    
    #### Select IPC-independent model
    
    model = pickle.load(open(os.path.join(cur_dir,
                      'classifier_mix.pickle'), 'rb'))  
    
    #### Start to vectorize this patent
    resp = ifi_fcit(ucid)
    fcit = len(resp['citations'][0]['ucids'])
    
    resp = ifi_family(ucid)
    family_size = len(resp['family']['members'])
    
    ipr = patent_num in ipr_list
    
    response1 = ifi_querier('pnnum:{} AND pnctry:us AND ifi_document_category:reissue'.format(patent_num))
    reissue = response1['content']['response']['numFound'] > 0
    response2 = ifi_querier('relpnnum:{} AND ifi_document_category:reissue'.format(patent_num))
    reissued = response2['content']['response']['numFound'] > 0
                           
    case = np.array([family_size, fcit, ipr, reissue or reissued])
    
    # Prediction part
    pred_value = model.predict(case.reshape(1,-1))[0]
    pred_class = ['monetized' if pred_value == 1 else 'unmonetized'][0]
    pred_proba = round(model.predict_proba(case.reshape(1,-1))[0][pred_value]*100,2)

    return render(request, 'control2.html',
                  {'patent_number':patent_num,
                   'family_size':case[0], 'fcit':case[1],'ipr':ipr, 
                   'reissue':reissue or reissued,'pred_class':pred_class,
                   'pred_proba':pred_proba})