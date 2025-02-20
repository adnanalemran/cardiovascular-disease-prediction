import os
import joblib
from django.shortcuts import render

def homes(request):
    return render(request, 'homes.html')

def result(request):
    model_path = os.path.join(os.path.dirname(__file__), '../output/trained_model.pkl')
  
    model = joblib.load(model_path)  # Load model with correct path
    print(model)
    lis = [
        int(request.GET['age']),
        int(request.GET['sex']),
        int(request.GET['cp']),
        int(request.GET['trestbps']),
        int(request.GET['chol']),
        int(request.GET['thalach']),
        float(request.GET['oldpeak']),  # ✅ Fix: Convert to float
        int(request.GET['ca']),
        int(request.GET['thal'])
    ]
    ans = model.predict([lis])

    return render(request, 'result.html', {'ans': ans})
