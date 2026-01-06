from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Dataset

def home(request):
    """Redirects to the upload page."""
    return redirect('upload')

def clear_session(request):
    """Clears the session - useful for testing navigation restrictions."""
    request.session.flush()
    messages.info(request, "Session cleared. Upload a dataset to begin.")
    return redirect('upload')

def upload_file(request):
    """Handles file uploads."""
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        alias = request.POST.get('alias', '').strip()
        dataset = Dataset.objects.create(
            file=uploaded_file, 
            name=alias or uploaded_file.name
        )
        
        # Store dataset ID in session for downstream pages
        request.session['dataset_id'] = dataset.id
        
        return redirect('report')
    
    # Handle direct load from recent interactions
    dataset_id = request.GET.get('id')
    if dataset_id:
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            request.session['dataset_id'] = dataset.id
            return redirect('report')
        except Dataset.DoesNotExist:
            pass

    recent_uploads = Dataset.objects.order_by('-uploaded_at')[:5]
    return render(request, 'upload.html', {'recent_uploads': recent_uploads})

import pandas as pd
import os

def report(request):
    """Displays EDA report."""
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        messages.warning(request, "Manifest a reality fragment first to access the Report.")
        return redirect('upload')
    
    try:
        dataset = Dataset.objects.get(id=dataset_id)
    except Dataset.DoesNotExist:
        request.session.pop('dataset_id', None)
        messages.warning(request, "Dataset no longer exists. Please upload a new one.")
        return redirect('upload')
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
        
        # Per-Column Missing Value Analysis
        missing_per_column = []
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            missing_per_column.append({
                'column': col,
                'missing_count': missing_count,
                'missing_pct': round(missing_pct, 2)
            })
        
        # Per-Column Unique Value Count
        unique_per_column = []
        for col in df.columns:
            unique_count = df[col].nunique()
            unique_pct = (unique_count / len(df)) * 100
            unique_per_column.append({
                'column': col,
                'unique_count': unique_count,
                'unique_pct': round(unique_pct, 2)
            })
        
        # Categorical Value Frequency Summary (top 3-5 values)
        categorical_summary = []
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                value_counts = df[col].value_counts(normalize=True).head(5)
                top_values = [
                    {'value': str(val), 'percentage': round(pct * 100, 2)}
                    for val, pct in value_counts.items()
                ]
                categorical_summary.append({
                    'column': col,
                    'top_values': top_values
                })
        
        # Numeric Distribution Shape Indicators
        numeric_distribution = []
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            try:
                skewness = df[col].skew()
                # Determine skew direction
                if skewness > 1:
                    skew_label = 'Right Skewed'
                elif skewness < -1:
                    skew_label = 'Left Skewed'
                else:
                    skew_label = 'Symmetric'
                
                # Detect outliers using IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_pct = (outliers / len(df)) * 100
                
                numeric_distribution.append({
                    'column': col,
                    'skewness': round(skewness, 2),
                    'skew_label': skew_label,
                    'outliers': outliers,
                    'outlier_percentage': round(outlier_pct, 2),
                    'outlier_pct_val': round(outlier_pct, 2)
                })
            except:
                pass
        
        print("DEBUG DISTRIBUTION:", numeric_distribution)
            
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
            # New analytics
            'missing_per_column': missing_per_column,
            'unique_per_column': unique_per_column,
            'categorical_summary': categorical_summary,
            'numeric_distribution': numeric_distribution,
        }
        
    except Exception as e:
        context = {'error': str(e)}

    return render(request, 'report.html', context)

def clean_data(request):
    """Data cleaning interface."""
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        messages.warning(request, "Manifest a reality fragment first to access the Cleaning ritual.")
        return redirect('upload')
    
    try:
        dataset = Dataset.objects.get(id=dataset_id)
    except Dataset.DoesNotExist:
        request.session.pop('dataset_id', None)
        messages.warning(request, "Dataset no longer exists. Please upload a new one.")
        return redirect('upload')
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
            
            # NEW: Data Type Conversion
            elif action == 'convert_dtype':
                col = request.POST.get('column')
                dtype = request.POST.get('dtype')
                if col in df.columns:
                    try:
                        if dtype == 'numeric':
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        elif dtype == 'categorical':
                            df[col] = df[col].astype('category')
                        elif dtype == 'datetime':
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        elif dtype == 'string':
                            df[col] = df[col].astype(str)
                    except:
                        pass
            
            # NEW: Text Cleaning
            elif action == 'clean_text':
                col = request.POST.get('column')
                operation = request.POST.get('operation')
                if col in df.columns and df[col].dtype == 'object':
                    if operation == 'trim':
                        df[col] = df[col].str.strip()
                    elif operation == 'lowercase':
                        df[col] = df[col].str.lower()
                    elif operation == 'uppercase':
                        df[col] = df[col].str.upper()
                    elif operation == 'empty_to_nan':
                        df[col] = df[col].replace('', pd.NA)
            
            # NEW: Value Replacement
            elif action == 'replace_value':
                col = request.POST.get('column')
                old_value = request.POST.get('old_value')
                new_value = request.POST.get('new_value')
                if col in df.columns:
                    df[col] = df[col].replace(old_value, new_value)
            
            # NEW: Row Filtering
            elif action == 'filter_rows':
                col = request.POST.get('column')
                operator = request.POST.get('operator')
                value = request.POST.get('value')
                if col in df.columns:
                    try:
                        if operator == 'equals':
                            df = df[df[col] == value]
                        elif operator == 'not_equals':
                            df = df[df[col] != value]
                        elif operator == 'greater':
                            df = df[df[col] > float(value)]
                        elif operator == 'less':
                            df = df[df[col] < float(value)]
                        elif operator == 'contains':
                            df = df[df[col].astype(str).str.contains(value, na=False)]
                    except:
                        pass
            
            # NEW: Outlier Handling
            elif action == 'handle_outliers':
                col = request.POST.get('column')
                method = request.POST.get('method')
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if method == 'iqr':
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
                    elif method == 'zscore':
                        from scipy import stats
                        z_scores = np.abs(stats.zscore(df[col].dropna()))
                        df = df[(z_scores < 3) | df[col].isna()]
            
            # NEW: Rare Category Grouping
            elif action == 'group_rare':
                col = request.POST.get('column')
                threshold = float(request.POST.get('threshold', 0.01))
                if col in df.columns:
                    value_counts = df[col].value_counts(normalize=True)
                    rare_categories = value_counts[value_counts < threshold].index
                    df[col] = df[col].replace(rare_categories, 'Other')
            
            # NEW: Duplicate Removal by Columns
            elif action == 'drop_duplicates_cols':
                cols = request.POST.getlist('columns')
                if cols:
                    df.drop_duplicates(subset=cols, inplace=True)
            
            # NEW: Datetime Feature Extraction
            elif action == 'extract_datetime':
                col = request.POST.get('column')
                features = request.POST.getlist('features')
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        if 'year' in features:
                            df[f'{col}_year'] = df[col].dt.year
                        if 'month' in features:
                            df[f'{col}_month'] = df[col].dt.month
                        if 'day' in features:
                            df[f'{col}_day'] = df[col].dt.day
                        if 'dayofweek' in features:
                            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                        if 'hour' in features:
                            df[f'{col}_hour'] = df[col].dt.hour
                    except:
                        pass
            
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
        messages.warning(request, "Manifest a reality fragment first to access Visualizations.")
        return redirect('upload')
    
    try:
        dataset = Dataset.objects.get(id=dataset_id)
    except Dataset.DoesNotExist:
        request.session.pop('dataset_id', None)
        messages.warning(request, "Dataset no longer exists. Please upload a new one.")
        return redirect('upload')
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
            
            # Existing chart types
            if chart_type in ['bar', 'line', 'scatter']:
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    sample = df.head(100) if len(df) > 100 else df
                    data = {
                        'labels': sample[x_col].tolist(),
                        'values': sample[y_col].tolist()
                    }
            
            elif chart_type == 'pie':
                if x_col and x_col in df.columns:
                    vc = df[x_col].value_counts().head(10)
                    data = {
                        'labels': vc.index.tolist(),
                        'values': vc.values.tolist()
                    }
            
            elif chart_type == 'histogram':
                if x_col and x_col in df.columns:
                    vals = df[x_col].dropna()
                    hist, bins = np.histogram(vals, bins='auto')
                    data = {
                        'labels': [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(hist))],
                        'values': hist.tolist()
                    }
            
            # NEW: Box Plot
            elif chart_type == 'boxplot':
                if x_col and x_col in df.columns and pd.api.types.is_numeric_dtype(df[x_col]):
                    vals = df[x_col].dropna()
                    q1, median, q3 = vals.quantile([0.25, 0.5, 0.75])
                    iqr = q3 - q1
                    lower_whisker = max(vals.min(), q1 - 1.5 * iqr)
                    upper_whisker = min(vals.max(), q3 + 1.5 * iqr)
                    outliers = vals[(vals < lower_whisker) | (vals > upper_whisker)].tolist()
                    data = {
                        'min': lower_whisker,
                        'q1': q1,
                        'median': median,
                        'q3': q3,
                        'max': upper_whisker,
                        'outliers': outliers[:50],  # Limit outliers
                        'label': x_col
                    }
            
            # NEW: Count Plot (for categorical data)
            elif chart_type == 'countplot':
                if x_col and x_col in df.columns:
                    vc = df[x_col].value_counts().head(20)
                    data = {
                        'labels': vc.index.tolist(),
                        'values': vc.values.tolist()
                    }
            
            # NEW: Correlation Heatmap
            elif chart_type == 'correlation':
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    data = {
                        'labels': corr_matrix.columns.tolist(),
                        'data': corr_matrix.values.tolist()
                    }
            
            # NEW: Missing Value Bar Plot
            elif chart_type == 'missing':
                missing_counts = df.isnull().sum()
                missing_data = missing_counts[missing_counts > 0].sort_values(ascending=False)
                data = {
                    'labels': missing_data.index.tolist(),
                    'values': missing_data.values.tolist()
                }
            
            # NEW: Grouped/Stacked Bar Chart
            elif chart_type == 'grouped_bar':
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    grouped = df.groupby(x_col)[y_col].value_counts().unstack(fill_value=0)
                    data = {
                        'labels': grouped.index.tolist(),
                        'datasets': [
                            {'label': str(col), 'data': grouped[col].tolist()}
                            for col in grouped.columns[:5]  # Limit to 5 groups
                        ]
                    }
            
            # NEW: Violin Plot (approximated with box plot data + distribution)
            elif chart_type == 'violin':
                if x_col and x_col in df.columns and pd.api.types.is_numeric_dtype(df[x_col]):
                    vals = df[x_col].dropna()
                    # Create histogram for distribution shape
                    hist, bins = np.histogram(vals, bins=20)
                    data = {
                        'distribution': hist.tolist(),
                        'bins': bins.tolist(),
                        'q1': vals.quantile(0.25),
                        'median': vals.quantile(0.5),
                        'q3': vals.quantile(0.75),
                        'label': x_col
                    }
            
            # NEW: Pairwise Scatter Plot
            elif chart_type == 'pairplot':
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                        sample = df[[x_col, y_col]].dropna().head(200)
                        data = {
                            'x': sample[x_col].tolist(),
                            'y': sample[y_col].tolist(),
                            'labels': [x_col, y_col]
                        }
            
            # NEW: Time Series Plot
            elif chart_type == 'timeseries':
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    try:
                        df_copy = df[[x_col, y_col]].copy()
                        df_copy[x_col] = pd.to_datetime(df_copy[x_col])
                        df_sorted = df_copy.sort_values(x_col).head(200)
                        data = {
                            'labels': df_sorted[x_col].dt.strftime('%Y-%m-%d').tolist(),
                            'values': df_sorted[y_col].tolist()
                        }
                    except:
                        data = {'error': 'Invalid datetime column'}

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
        messages.warning(request, "Manifest a reality fragment first to access AI Insights.")
        return redirect('upload')
    
    try:
        dataset = Dataset.objects.get(id=dataset_id)
    except Dataset.DoesNotExist:
        request.session.pop('dataset_id', None)
        messages.warning(request, "Dataset no longer exists. Please upload a new one.")
        return redirect('upload')
    
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

