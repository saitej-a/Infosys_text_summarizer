from django.shortcuts import render
from . import generate
from django.http import JsonResponse
# Create your views here.
def index(request):
    
    if request.method=='POST':
        text=request.POST.get('article')
        range_inp=request.POST.get('range')
        if request.POST.get('method')=='1':
            summary=generate.Extract(text,int(range_inp)*6)
            return JsonResponse({'summary':summary})
        elif request.POST.get('method')=='0':
            
            summary=generate.Abstract(text,int(range_inp)*6)
            return JsonResponse({'summary':summary})
    return render(request,'index.html')