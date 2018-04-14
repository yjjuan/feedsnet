from django.shortcuts import render
from django.http import HttpResponse

import requests
import json

# Create your views here.
def index(request):
    
    #query = 'http://www.patentsview.org/api/patents/query?q={"_and":[{"inventor_last_name":"Jobs"},{"assignee_lastknown_country":"US"}]}&f=["patent_number"]'
    query = 'https://api.mlab.com/api/1/databases/yjjuan01/collections/test?apiKey=ITT5lmoZVIkElhBjGP7IYrreM4Jv0OmI'
    
    result = requests.get(query)
    response = json.loads(result)
    return HttpResponse(response[-1])
    #return render(request, 'index.html')
    #return "hello"




