from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.conf import settings
from .models import Dataset
import pandas as pd
import os
import requests
import json

def home(request):
    """Redirects to the upload page."""
    return redirect('upload')

def calculate_chaos_score(df):
    """Calculates a chaos score from 0-100 (Higher = Messier)."""
    if df.empty: return 0
    
    # 1. Missing Values (Up to 40 points)
    missing_pct = df.isnull().sum().sum() / (df.size or 1)
    missing_score = min(missing_pct * 100 * 2, 40)
    
    # 2. Duplicate Rows (Up to 20 points)
    dup_pct = df.duplicated().sum() / len(df)
    dup_score = min(dup_pct * 100 * 2, 20)
    
    # 3. High Cardinality / Messy Columns (Up to 20 points)
    # Check for columns that look like IDs but are 'object'
    messy_cols = 0
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.9:
            messy_cols += 1
    cardinality_score = min((messy_cols / len(df.columns)) * 100, 20)
    
    # 4. Outliers (Placeholder - Up to 20 points)
    # Just a rough check on numeric skewness for now
    numeric_df = df.select_dtypes(include=['number'])
    outlier_score = 0
    if not numeric_df.empty:
        skew = numeric_df.skew().abs().mean()
        outlier_score = min(skew * 5, 20)
        
    return int(min(missing_score + dup_score + cardinality_score + outlier_score, 100))

def get_groq_completion(prompt):
    """Calls the Groq API using requests."""
    api_key = getattr(settings, 'GROQ_API_KEY', '')
    if not api_key:
        return None
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are Wanda Maximoff, the Scarlet Witch. You are an expert data scientist who uses chaos magic to clean data. Your tone is mystical, elegant, and slightly dark. Provide actionable data cleaning advice in HTML format (use <div>, <b>, <p> tags). Keep it concise."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Groq API Error: {e}")
        return None

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
        
        # Save temporary for analysis
        dataset = Dataset.objects.create(
            file=uploaded_file, 
            name=alias or uploaded_file.name
        )
        
        try:
            # Analyze metadata
            df = pd.read_csv(dataset.file.path) if dataset.file.path.endswith('.csv') else pd.read_excel(dataset.file.path)
            dataset.total_rows = len(df)
            dataset.total_cols = len(df.columns)
            dataset.chaos_score = calculate_chaos_score(df)
            dataset.save()
        except:
            pass # Fallback to defaults
            
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

    # Hall of Records: Load all history
    history = Dataset.objects.order_by('-uploaded_at')
    return render(request, 'upload.html', {'history': history})

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

    # --- UNDO/REDO SYSTEM ---
    history_dir = os.path.join(settings.MEDIA_ROOT, 'history', str(dataset_id))
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    def get_history_stacks():
        undo_stack = request.session.get(f'undo_{dataset_id}', [])
        redo_stack = request.session.get(f'redo_{dataset_id}', [])
        return undo_stack, redo_stack

    def save_history_stacks(undo, redo):
        request.session[f'undo_{dataset_id}'] = undo
        request.session[f'redo_{dataset_id}'] = redo
        request.session.modified = True

    def save_checkpoint(df):
        undo, redo = get_history_stacks()
        # Save current state to history before changing it
        import time
        timestamp = int(time.time() * 1000)
        checkpoint_path = os.path.join(history_dir, f'cp_{timestamp}.csv')
        df.to_csv(checkpoint_path, index=False)
        
        undo.append(checkpoint_path)
        # Limit history to 10
        if len(undo) > 10:
            old = undo.pop(0)
            if os.path.exists(old): os.remove(old)
            
        # Clear redo stack on new action
        for r in redo:
            if os.path.exists(r): os.remove(r)
        
        save_history_stacks(undo, [])

    df = load_df(file_path)
    if df is None: return render(request, 'clean.html', {'error': 'Failed to load data'})

    if request.method == 'POST':
        action = request.POST.get('action')
        
        # AJAX Handler for Guidance
        if action == 'get_guidance':
            cols_info = df.dtypes.to_dict()
            missing_info = df.isnull().sum().to_dict()
            prompt = f"Dataset sample:\n{df.head(5).to_csv()}\nColumns: {cols_info}\nMissing Values: {missing_info}\nSuggest 3 specific rituals (cleaning steps) to restore balance to this reality. Use scarlet-themed metaphors."
            guidance_html = get_groq_completion(prompt)
            if guidance_html:
                return JsonResponse({'status': 'success', 'html': guidance_html})
            return JsonResponse({'status': 'error', 'message': 'The void is silent.'})

        try:
            if action == 'auto_ritual':
                save_checkpoint(df)
                # 1. Automagical Cleaning
                # Strip strings
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].astype(str).str.strip()
                
                # Fill numeric with median, categorical with mode
                for col in df.columns:
                    if df[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col].fillna(df[col].median(), inplace=True)
                        else:
                            df[col].fillna(df[col].mode()[0], inplace=True)
                
                # Drop duplicates
                df.drop_duplicates(inplace=True)
                
                # Recalculate chaos and save
                dataset.chaos_score = calculate_chaos_score(df)
                dataset.save()
                save_df(df, file_path)
                messages.success(request, "The reality has been stabilized by the Auto-Ritual.")
                return redirect('clean')
            if action == 'undo':
                undo, redo = get_history_stacks()
                if undo:
                    # Save current live state to redo stack
                    import time
                    timestamp = int(time.time() * 1000)
                    redo_cp = os.path.join(history_dir, f'redo_{timestamp}.csv')
                    df.to_csv(redo_cp, index=False)
                    redo.append(redo_cp)
                    
                    # Target state
                    target_path = undo.pop()
                    df = pd.read_csv(target_path)
                    save_df(df, file_path)
                    os.remove(target_path) # Clean up history file after use
                    
                    save_history_stacks(undo, redo)
                    return redirect('clean')

            elif action == 'redo':
                undo, redo = get_history_stacks()
                if redo:
                    # Save current live state back to undo stack
                    import time
                    timestamp = int(time.time() * 1000)
                    undo_cp = os.path.join(history_dir, f'undo_{timestamp}.csv')
                    df.to_csv(undo_cp, index=False)
                    undo.append(undo_cp)
                    
                    # Target state
                    target_path = redo.pop()
                    df = pd.read_csv(target_path)
                    save_df(df, file_path)
                    os.remove(target_path)
                    
                    save_history_stacks(undo, redo)
                    return redirect('clean')

            elif action == 'drop_col':
                save_checkpoint(df)
                col = request.POST.get('column')
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            
            elif action == 'rename_col':
                save_checkpoint(df)
                old_name = request.POST.get('old_name')
                new_name = request.POST.get('new_name')
                if old_name in df.columns and new_name:
                    df.rename(columns={old_name: new_name}, inplace=True)

            elif action == 'fill_na':
                save_checkpoint(df)
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
                save_checkpoint(df)
                df.drop_duplicates(inplace=True)
            
            # ... and so on for other actions. 
            # To be thorough I should add save_checkpoint(df) to all other actions.
            
            elif action == 'convert_dtype':
                save_checkpoint(df)
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
            
            elif action == 'clean_text':
                save_checkpoint(df)
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
            
            elif action == 'replace_value':
                save_checkpoint(df)
                col = request.POST.get('column')
                old_value = request.POST.get('old_value')
                new_value = request.POST.get('new_value')
                if col in df.columns:
                    df[col] = df[col].replace(old_value, new_value)
            
            elif action == 'filter_rows':
                save_checkpoint(df)
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
            
            elif action == 'handle_outliers':
                save_checkpoint(df)
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
            
            elif action == 'group_rare':
                save_checkpoint(df)
                col = request.POST.get('column')
                threshold = float(request.POST.get('threshold', 0.01))
                if col in df.columns:
                    value_counts = df[col].value_counts(normalize=True)
                    rare_categories = value_counts[value_counts < threshold].index
                    df[col] = df[col].replace(rare_categories, 'Other')
            
            elif action == 'drop_duplicates_cols':
                save_checkpoint(df)
                cols = request.POST.getlist('columns')
                if cols:
                    df.drop_duplicates(subset=cols, inplace=True)
            
            elif action == 'extract_datetime':
                save_checkpoint(df)
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
            
            # Recalculate Chaos Score after any ritual
            if action not in ['undo', 'redo', 'get_guidance']:
                save_df(df, file_path)
                dataset.chaos_score = calculate_chaos_score(df)
                dataset.total_rows = len(df)
                dataset.total_cols = len(df.columns)
                dataset.save()
            
            return redirect('clean')
            
        except Exception as e:
            messages.error(request, f"Reality distortion: {str(e)}")
            return redirect('clean')

    # GET request - check if guidance is requested
    if request.GET.get('action') == 'get_guidance':
        cols_info = df.dtypes.astype(str).to_dict()
        missing_info = df.isnull().sum().to_dict()
        
        system_instruction = """
        You are Wanda Maximoff, the Scarlet Witch. You guide the user in cleaning their data using the provided app tools.
        DO NOT provide Python code. Provide 3 specific, actionable suggestions based on the data analysis.
        For each suggestion, include a button that opens the relevant tool in the app.
        
        Available Tools & Actions (use exactly this HTML for buttons):
        - To clean text: <button onclick="openModal('clean-text-modal')" class="btn-energy text-[10px] px-2 py-1 mt-2">Clean Text</button>
        - To fill missing: <button onclick="openModal('fill-modal')" class="btn-energy text-[10px] px-2 py-1 mt-2">Fill Missing</button>
        - To drop columns: <button onclick="openModal('drop-modal')" class="btn-energy text-[10px] px-2 py-1 mt-2">Drop Columns</button>
        - To remove outliers: <button onclick="openModal('outliers-modal')" class="btn-energy text-[10px] px-2 py-1 mt-2">Handle Outliers</button>
        - To extract dates: <button onclick="openModal('datetime-extract-modal')" class="btn-energy text-[10px] px-2 py-1 mt-2">Extract Dates</button>
        - To filter rows: <button onclick="openModal('filter-rows-modal')" class="btn-energy text-[10px] px-2 py-1 mt-2">Filter Data</button>
        
        Format each suggestion as a div block:
        <div class="p-4 bg-white/5 border border-white/10 rounded-lg mb-4 hover:border-scarlet/30 transition-all">
            <h4 class="text-scarlet font-cinzel text-sm font-bold mb-1">Title</h4>
            <p class="text-gray-400 text-xs mb-2">Mystical explanation...</p>
            [Button Code Here]
        </div>
        """
        
        prompt = f"System: {system_instruction}\n\nUser Data Context:\nColumns: {cols_info}\nMissing Values: {missing_info}\nSample: {df.head(3).to_dict()}"
        
        guidance_html = get_groq_completion(prompt)
        if guidance_html:
            return JsonResponse({'status': 'success', 'html': guidance_html})
        return JsonResponse({'status': 'error', 'message': 'The void is silent.'})

    # Prepare context
    context = {
        'dataset': dataset,
        'columns': df.columns.tolist(),
        'table': df.to_html(classes='min-w-full text-left text-sm whitespace-nowrap', index=False, border=0),
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

