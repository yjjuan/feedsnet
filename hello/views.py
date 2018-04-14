from django.shortcuts import render
from django.http import HttpResponse

import requests
import json

# Create your views here.
def index(request):
    
    #query = 'http://www.patentsview.org/api/patents/query?q={"_and":[{"inventor_last_name":"Jobs"},{"assignee_lastknown_country":"US"}]}&f=["patent_number"]'
    query = 'https://api.mlab.com/api/1/databases/yjjuan01/collections/test?apiKey=ITT5lmoZVIkElhBjGP7IYrreM4Jv0OmI'
    
    result = requests.get(query)
    response = json.loads(result.text)
    
    Lambda_str = response[-2]['lamda']
    Lambda = Lambda_str[1:-1].split(', ')
    
    intensity_str = response[-1]['intensity']
    intensity = intensity_str[1:-1].split(', ')
    
    spectrum = [[float(Lambda[i]),float(intensity[i])] for i in range(len(Lambda))]
    print(spectrum)
    #return HttpResponse(response[-1]['intensity'])
    return render(request, 'highchart3.html',
                  {'spectrum':spectrum})
    #return "hello"




