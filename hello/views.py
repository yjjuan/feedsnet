from django.shortcuts import render
from django.http import HttpResponse

import requests

# Create your views here.
def index(request):
    
    query = 'http://www.patentsview.org/api/patents/query?q={"_and":[{"inventor_last_name":"Jobs"},{"assignee_lastknown_country":"US"}]}&f=["patent_number"]'
    
    result = requests.get(query)
    return HttpResponse(result)
    #return render(request, 'index.html')
    #return "hello"




