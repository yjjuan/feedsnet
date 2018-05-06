from django.shortcuts import render
from django.http import HttpResponse

import requests
import json
import pickle
import os
from .deployExplainer import explainer
import numpy as np

# Create your views here.
def index(request):

    cur_dir = os.path.dirname(__file__)
    train = pickle.load(open(os.path.join(cur_dir,
                      'train.pickle'), 'rb'))    
    model = pickle.load(open(os.path.join(cur_dir,
                      'classifier.pickle'), 'rb'))    

    feature_names = ['familySize','citedby_patent_count','iprFlag','reissued_combined']
    class_names = ['unmonetized','monetized']
    categorical_features = [2,3]
    categorical_names = {2:['False','True'], 3:['False','True']}
    
    exp = explainer(train,feature_names = feature_names,class_names=class_names,
                                categorical_features=categorical_features,
                                categorical_names=categorical_names, kernel_width=3)
    
    explan = exp.explain(model, np.array([0, 5, 1, 0]))
    desc_list = [explan[i]['desc'] for i in explan]
    weight_list = [explan[i]['weight'] for i in explan]
    
    #print(explainer.explain(model, np.array([0, 5, 1, 0])))
    #return HttpResponse(exp.explain(model, np.array([0, 5, 1, 0])))
    return render(request, 'highchart6.html',
                  {'desc_list':desc_list, 'weight_list':weight_list})
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




