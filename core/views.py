from django.shortcuts import render, redirect
from .models import Dataset

def home(request):
    """Redirects to the upload page."""
    return redirect('upload')

def upload_file(request):
    """Handles file uploads."""
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        dataset = Dataset.objects.create(file=uploaded_file, name=uploaded_file.name)
        
        # Store dataset ID in session for downstream pages
        request.session['dataset_id'] = dataset.id
        
        return redirect('report')

    recent_uploads = Dataset.objects.order_by('-uploaded_at')[:5]
    return render(request, 'upload.html', {'recent_uploads': recent_uploads})

import pandas as pd
import os

def report(request):
    """Displays EDA report."""
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        return redirect('upload')
    
    dataset = Dataset.objects.get(id=dataset_id)
    file_path = dataset.file.path
    
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            return render(request, 'report.html', {'error': 'Unsupported file format'})
            
        # Basic Stats
        context = {
            'dataset': dataset,
            'rows': df.shape[0],
            'columns': df.shape[1],
            'missing_values': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'columns_list': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'head': df.head(10).to_html(classes='min-w-full text-left text-sm whitespace-nowrap', index=False, border=0),
            'describe': df.describe().to_html(classes='min-w-full text-left text-sm whitespace-nowrap', border=0),
        }
        
    except Exception as e:
        context = {'error': str(e)}

    return render(request, 'report.html', context)

def clean_data(request):
    """Data cleaning interface."""
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        return redirect('upload')
    
    dataset = Dataset.objects.get(id=dataset_id)
    file_path = dataset.file.path
    
    # helper for loading
    def load_df(path):
        if path.endswith('.csv'): return pd.read_csv(path)
        if path.endswith('.xlsx') or path.endswith('.xls'): return pd.read_excel(path)
        if path.endswith('.json'): return pd.read_json(path)
        return None

    # helper for saving
    def save_df(df, path):
        if path.endswith('.csv'): df.to_csv(path, index=False)
        elif path.endswith('.xlsx') or path.endswith('.xls'): df.to_excel(path, index=False)
        elif path.endswith('.json'): df.to_json(path)

    df = load_df(file_path)
    if df is None: return render(request, 'clean.html', {'error': 'Failed to load data'})

    if request.method == 'POST':
        action = request.POST.get('action')
        
        try:
            if action == 'drop_col':
                col = request.POST.get('column')
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            
            elif action == 'rename_col':
                old_name = request.POST.get('old_name')
                new_name = request.POST.get('new_name')
                if old_name in df.columns and new_name:
                    df.rename(columns={old_name: new_name}, inplace=True)

            elif action == 'fill_na':
                col = request.POST.get('column')
                method = request.POST.get('method')
                if col in df.columns:
                    if method == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif method == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].median(), inplace=True)
                    elif method == 'mode':
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif method == 'drop':
                         df.dropna(subset=[col], inplace=True)
            
            elif action == 'drop_duplicates':
                df.drop_duplicates(inplace=True)
            
            save_df(df, file_path)
            return redirect('clean')
            
        except Exception as e:
            error = str(e) # Pass error to context if needed

    # Prepare context
    context = {
        'dataset': dataset,
        'columns': df.columns.tolist(),
        'table': df.head(50).to_html(classes='min-w-full text-left text-sm whitespace-nowrap', index=False, border=0),
        'total_rows': len(df)
    }

    return render(request, 'clean.html', context)

from django.http import JsonResponse
import numpy as np

def visualize(request):
    """Visualization interface."""
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        return redirect('upload')
    
    dataset = Dataset.objects.get(id=dataset_id)
    file_path = dataset.file.path
    
    # Simple load helper (duplicate logic, could be refactored)
    try:
        if file_path.endswith('.csv'): df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'): df = pd.read_excel(file_path)
        elif file_path.endswith('.json'): df = pd.read_json(file_path)
        else: return redirect('report')
    except:
        return redirect('report')

    # AJAX Data Request
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        chart_type = request.GET.get('type')
        x_col = request.GET.get('x')
        y_col = request.GET.get('y')
        
        try:
            data = {}
            if chart_type in ['bar', 'line', 'scatter']:
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    # Limit to top 100 to prevent browser crash on huge datasets
                    sample = df.head(100) if len(df) > 100 else df
                    data = {
                        'labels': sample[x_col].tolist(),
                        'values': sample[y_col].tolist()
                    }
            elif chart_type == 'pie':
                if x_col and x_col in df.columns:
                    vc = df[x_col].value_counts().head(10) # Top 10 categories
                    data = {
                        'labels': vc.index.tolist(),
                        'values': vc.values.tolist()
                    }
            elif chart_type == 'histogram':
                if x_col and x_col in df.columns:
                     # Calculate histogram bins using numpy
                    vals = df[x_col].dropna()
                    hist, bins = np.histogram(vals, bins='auto')
                    data = {
                        'labels': [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(hist))],
                        'values': hist.tolist()
                    }

            return JsonResponse(data)
        except Exception as e:
             return JsonResponse({'error': str(e)}, status=400)

    # Context for Page Load
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    context = {
        'dataset': dataset,
        'columns': df.columns.tolist(),
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
    }
    return render(request, 'visualize.html', context)

import requests
from django.conf import settings

def ai_insights(request):
    """AI Chat interface."""
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        return redirect('upload')
    
    dataset = Dataset.objects.get(id=dataset_id)
    
    if request.method == 'POST':
        user_message = request.POST.get('message')
        
        # Prepare context from dataset (head + info)
        file_path = dataset.file.path
        try:
            if file_path.endswith('.csv'): df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'): df = pd.read_excel(file_path)
            elif file_path.endswith('.json'): df = pd.read_json(file_path)
            else: df = None
        except: df = None

        data_context = ""
        if df is not None:
            data_context = f"""
            Dataset Context:
            Columns: {', '.join(df.columns)}
            Shape: {df.shape}
            Sample Data:
            {df.head(5).to_string()}
            """

        system_prompt = f"""
        You are Wanda Maximoff (Scarlet Witch), a powerful reality-warping entity helping a data scientist.
        Your tone should be mystical, powerful, yet helpful and precise.
        Use metaphors about chaos magic, reality, and hexes, but keep the data analysis technical and correct.
        
        Analyze the user's question based on this dataset snippet:
        {data_context}
        
        Keep responses concise (under 200 words) and format them with bullet points or clear sections.
        """

        try:
            headers = {
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.7
            }
            
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
            response_data = response.json()
            
            if 'choices' in response_data:
                ai_reply = response_data['choices'][0]['message']['content']
                return JsonResponse({'reply': ai_reply})
            else:
                return JsonResponse({'error': 'The chaotic energies interfered (API Error).'}, status=500)
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return render(request, 'ai_insights.html', {'dataset': dataset})

