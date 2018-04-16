from django.shortcuts import render
from django.http import HttpResponse

import requests
import json
import pickle
import os

# Create your views here.
def index(request):
    
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
<<<<<<< HEAD
    #print(spectrum)
    #return HttpResponse(response[-1]['intensity'])
    return render(request, 'highchart5.html',
                  {'spectrum':spectrum})
=======
    print(spectrum)
    return HttpResponse(clf)
    #return render(request, 'highchart4.html',
    #              {'spectrum':spectrum})
>>>>>>> a562d75d2f2424b59f3405ac21a79375da5a97e8
    #return "hello"




