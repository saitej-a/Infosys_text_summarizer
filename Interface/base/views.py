from django.shortcuts import render
# from . import generate
from django.http import JsonResponse
# Create your views here.
def index(request):
    
    # if request.method=='POST':
        
    #     text=request.POST.get('article')
    #     if request.POST.get('method')=='1':
    #         summary=generate.ex_summarize(text)[0]['summary_text']
    #         return JsonResponse({'summary':summary})
    #     elif request.POST.get('method')=='0':
            
    #         summary=generate.ab_summarize(text)[0]['summary_text']
            # return JsonResponse({'summary':summary})
    return render(request,'index.html')