from django.shortcuts import render
from django.http import HttpResponse

import requests
import urllib
import json
import pickle
import os
from .deployExplainer import explainer
import numpy as np

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
    train = pickle.load(open(os.path.join(cur_dir,
                      'train.pickle'), 'rb'))    
    model = pickle.load(open(os.path.join(cur_dir,
                      'classifier.pickle'), 'rb')) 
    ipr_list = pickle.load(open(os.path.join(cur_dir,
                      'ipr_issueNum.pickle'), 'rb')) 
    
    ####Start to vectorize this patent
    patent_num = request.POST['patent_number']
    resp0 = ifi_querier('pnnum:{} AND pnctry:us'.format(patent_num))
    ucid = resp0['content']['response']['docs'][0]['ucid']
    
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
                  {'patent_number':patent_num, 'family_size':case[0],
                   'fcit':case[1],'ipr':ipr, 
                   'reissue':reissue or reissued,'pred_class':pred_class,
                   'pred_proba':pred_proba, 'desc_mon':desc_mon, 
                   'desc_unmon':desc_unmon, 'weight_mon':weight_mon, 
                   'weight_unmon':weight_unmon})
    '''
    cur_dir = os.path.dirname(__file__)
    clf = pickle.load(open(os.path.join(cur_dir,
                      'patent_list.pickle'), 'rb'))
    
    #query = 'http://www.patentsview.org/api/patents/query?q={"_and":[{"inventor_last_name":"Jobs"},{"assignee_lastknown_country":"US"}]}&f=["patent_number"]'
    query = 'https://api.mlab.com/api/1/databases/yjjuan01/collections/test?apiKey=ITT5lmoZVIkElhBjGP7IYrreM4Jv0OmI'
    
    result = requests.get(query)
    response = json.loads(result.text)
    
    Lambda_str = response[-2]['lamda']
    Lambda = Lambda_str[1:-1].split(', ')
    
    intensity_str = response[-1]['intensity']
    intensity = intensity_str[1:-1].split(', ')
    
    spectrum = [[float(Lambda[i]),float(intensity[i])] for i in range(len(Lambda))]

    #print(spectrum)
    #return HttpResponse(response[-1]['intensity'])
    return render(request, 'highchart5.html',
                  {'spectrum':spectrum})
    '''
    #print(spectrum)
    #return HttpResponse(clf)
    #return render(request, 'highchart4.html',
    #              {'spectrum':spectrum})

    #return "hello"




