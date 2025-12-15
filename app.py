from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import json
from datetime import datetime, timedelta

app = Flask(__name__)

# Load data sekali saat server start dari pickle
data_segmentasi = pd.read_pickle("data/data_segmentasi.pkl")
data_segmentasi['month'] = data_segmentasi['trans_datetime'].dt.month

# Load data harian untuk tampilan historis
try:
    data_harian = pd.read_csv("data/data_harian.csv")
    # Rename kolom sesuai dengan nama di CSV
    if 'trans_date_trans_time' in data_harian.columns:
        data_harian.rename(columns={
            'trans_date_trans_time': 'date',
            'transaction_count': 'transactions'
        }, inplace=True)
    # Pastikan kolom tanggal dalam format datetime
    if 'date' in data_harian.columns:
        data_harian['date'] = pd.to_datetime(data_harian['date'])
    elif 'tanggal' in data_harian.columns:
        data_harian['date'] = pd.to_datetime(data_harian['tanggal'])
    print(f"Data harian loaded: {len(data_harian)} records")
except Exception as e:
    data_harian = None
    print(f"Warning: Could not load data_harian.csv: {e}")

# Load model SARIMAX
try:
    import os
    model_path = os.path.join('data', 'model_sarimax_transaksi.pkl')
    with open(model_path, 'rb') as f:
        model_sarimax = pickle.load(f)
    print(f"Model SARIMAX loaded successfully from {model_path}!")
    print(f"Model type: {type(model_sarimax)}")
    
    # Hitung data transaksi per hari untuk evaluasi
    daily_transactions = data_segmentasi.groupby(data_segmentasi['trans_datetime'].dt.date).size()
    print(f"Daily transactions data: {len(daily_transactions)} days")
except Exception as e:
    model_sarimax = None
    daily_transactions = None
    print(f"Warning: Could not load SARIMAX model: {e}")

# Fungsi untuk calculate error metrics
def calculate_error_metrics(actual, predicted):
    """Calculate RMSE, MAE, MAPE for model evaluation"""
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Convert to numpy arrays and handle NaN
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    
    # Remove any NaN or infinite values
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not np.any(mask):
        return None
    
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) < 2:
        return None
    
    try:
        # RMSE
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # MAE
        mae = mean_absolute_error(actual, predicted)
        
        # MAPE - avoid division by zero
        mape_mask = actual != 0
        if np.any(mape_mask):
            mape = np.mean(np.abs((actual[mape_mask] - predicted[mape_mask]) / actual[mape_mask])) * 100
        else:
            mape = 0.0
        
        # R²
        r2 = r2_score(actual, predicted)
        
        # Replace NaN with None for JSON serialization
        return {
            'rmse': float(rmse) if np.isfinite(rmse) else None,
            'mae': float(mae) if np.isfinite(mae) else None,
            'mape': float(mape) if np.isfinite(mape) else None,
            'r2': float(r2) if np.isfinite(r2) else None
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

# Fungsi generate data untuk map
def get_map_data(month=None, cluster_spend=None, label_spend=None,
                 cluster_geo=None, label_geo=None, sample_size=5000):
    df = data_segmentasi.copy()

    # Filter
    if month: df = df[df['month'] == month]
    if cluster_spend is not None: df = df[df['cluster_spend'] == cluster_spend]
    if label_spend: df = df[df['label'] == label_spend]
    if cluster_geo is not None: df = df[df['cluster_geo_dim2'] == cluster_geo]
    if label_geo: df = df[df['label_geo_dim2'] == label_geo]

    # Summary per zone
    summary = df.groupby(['cluster_spend','label','cluster_geo_dim2','label_geo_dim2']).agg(
        count=('cc_num','count'),
        avg_distance=('distance_km','mean'),
        center_lat=('lat','mean'),
        center_long=('long','mean')
    ).reset_index().to_dict(orient='records')

    # Sample marker supaya browser nggak lemot
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    records_sample = df_sample.to_dict(orient='records')

    return {'summary': summary, 'records_sample': records_sample, 'total': len(df)}

# ================= ROUTES =================

@app.route("/")
def home():
    # Home page kosong atau bisa diisi konten statis
    return render_template("home.html")

@app.route("/prediksi")
def prediksi():
    return render_template("prediksi.html")

@app.route("/prediksi/run", methods=['POST'])
def run_prediksi():
    if model_sarimax is None:
        return jsonify({'error': 'Model belum dimuat. Pastikan file model_sarimax_transaksi.pkl ada di folder data/'}), 500
    
    try:
        # Ambil parameter dari request
        data = request.json
        steps = int(data.get('steps', 7))  # Default 7 hari ke depan
        user_data = data.get('user_data', None)  # Data transaksi dari user
        
        # Jika user input data sendiri, gunakan itu sebagai basis
        if user_data and len(user_data) > 0:
            # User memberikan data historis sendiri
            import numpy as np
            user_values = [float(d['value']) for d in user_data]
            
            # Lakukan prediksi dari data user
            forecast = model_sarimax.forecast(steps=steps)
            
            # Adjust forecast berdasarkan rata-rata data user vs data training
            user_avg = np.mean(user_values)
            training_avg = daily_transactions.mean() if daily_transactions is not None else user_avg
            adjustment_factor = user_avg / training_avg if training_avg > 0 else 1
            
            adjusted_forecast = forecast * adjustment_factor
            
            # Buat tanggal untuk hasil prediksi
            last_date_str = user_data[-1]['date']
            last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
            future_dates = [last_date + timedelta(days=i+1) for i in range(steps)]
            
            # Calculate error metrics on user data (backtesting)
            split_idx = int(len(user_values) * 0.7)
            if split_idx > 0 and len(user_values) > split_idx:
                test_actual = user_values[split_idx:]
                test_steps = len(test_actual)
                test_forecast = model_sarimax.forecast(steps=test_steps)
                test_forecast_adjusted = test_forecast * adjustment_factor
                
                error_metrics = calculate_error_metrics(test_actual, test_forecast_adjusted[:len(test_actual)])
            else:
                error_metrics = None
            
            # Format hasil
            results = []
            for date, value in zip(future_dates, adjusted_forecast):
                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_transactions': float(value)
                })
            
            # Get model goodness of fit
            goodness_of_fit = {
                'aic': float(model_sarimax.aic) if hasattr(model_sarimax, 'aic') else None,
                'bic': float(model_sarimax.bic) if hasattr(model_sarimax, 'bic') else None,
                'hqic': float(model_sarimax.hqic) if hasattr(model_sarimax, 'hqic') else None,
            }
            
            return jsonify({
                'success': True,
                'predictions': results,
                'model_type': 'SARIMAX (Adjusted)',
                'last_actual_date': last_date.strftime('%Y-%m-%d'),
                'adjustment_info': f'Adjusted by factor {adjustment_factor:.2f} (avg: {user_avg:.0f} vs training: {training_avg:.0f})',
                'error_metrics': error_metrics,
                'goodness_of_fit': goodness_of_fit
            })
        else:
            # Gunakan data default dari dataset
            import numpy as np
            
            # Lakukan prediksi
            forecast = model_sarimax.forecast(steps=steps)
            
            # Buat tanggal untuk hasil prediksi
            last_date = data_segmentasi['trans_datetime'].max()
            future_dates = [last_date + timedelta(days=i+1) for i in range(steps)]
            
            # Calculate error metrics on test set (last 30% of data)
            if daily_transactions is not None and len(daily_transactions) > 10:
                split_idx = int(len(daily_transactions) * 0.7)
                test_actual = daily_transactions.values[split_idx:]
                test_steps = min(len(test_actual), 30)
                test_forecast = model_sarimax.forecast(steps=test_steps)
                error_metrics = calculate_error_metrics(test_actual[:test_steps], test_forecast[:test_steps])
            else:
                error_metrics = None
            
            # Format hasil prediksi
            results = []
            for date, value in zip(future_dates, forecast):
                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_transactions': float(value)
                })
            
            # Get historical data dari data_harian.csv
            historical = []
            if data_harian is not None:
                # Filter data dari 01-01-2019 sampai 21-06-2020
                start_date = pd.to_datetime('2019-01-01')
                end_date = pd.to_datetime('2020-06-21')
                
                df_filtered = data_harian[(data_harian['date'] >= start_date) & 
                                          (data_harian['date'] <= end_date)].copy()
                df_filtered = df_filtered.sort_values('date')
                
                for _, row in df_filtered.iterrows():
                    historical.append({
                        'date': row['date'].strftime('%Y-%m-%d'),
                        'transactions': int(row.get('transactions', row.get('jumlah_transaksi', 0)))
                    })
                
            
            # Get model goodness of fit
            goodness_of_fit = {
                'aic': float(model_sarimax.aic) if hasattr(model_sarimax, 'aic') else None,
                'bic': float(model_sarimax.bic) if hasattr(model_sarimax, 'bic') else None,
                'hqic': float(model_sarimax.hqic) if hasattr(model_sarimax, 'hqic') else None,
            }
            
            return jsonify({
                'success': True,
                'predictions': results,
                'historical': historical,
                'model_type': 'SARIMAX',
                'last_actual_date': last_date.strftime('%Y-%m-%d'),
                'error_metrics': error_metrics,
                'goodness_of_fit': goodness_of_fit
            })
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route("/segmentasi")
def segmentasi():
    # Halaman segmentasi dengan data dari data_segmentasi.pkl
    return render_template("segmentasi.html")

@app.route("/segmentasi/data")
def segmentasi_data():
    # API untuk mendapatkan data segmentasi
    try:
        # Ambil parameter filter
        cluster_spend = request.args.get('cluster_spend', type=int)
        cluster_geo = request.args.get('cluster_geo', type=int)
        limit = request.args.get('limit', default=100, type=int)
        
        df = data_segmentasi.copy()
        
        # Apply filters
        if cluster_spend is not None:
            df = df[df['cluster_spend'] == cluster_spend]
        if cluster_geo is not None:
            df = df[df['cluster_geo_dim2'] == cluster_geo]
        
        # Get summary statistics
        summary = {
            'total_records': len(df),
            'cluster_spend_counts': df['cluster_spend'].value_counts().to_dict(),
            'cluster_geo_counts': df['cluster_geo_dim2'].value_counts().to_dict(),
            'label_spend_counts': df['label'].value_counts().to_dict(),
            'label_geo_counts': df['label_geo_dim2'].value_counts().to_dict(),
            'avg_distance': float(df['distance_km'].mean()) if 'distance_km' in df.columns else 0
        }
        
        # Sample data untuk ditampilkan
        sample_data = df.head(limit).to_dict(orient='records')
        
        # Convert timestamps to string
        for record in sample_data:
            if 'trans_datetime' in record and record['trans_datetime']:
                record['trans_datetime'] = record['trans_datetime'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'success': True,
            'summary': summary,
            'data': sample_data
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route("/map")
def map_view():
    return render_template("map.html")

@app.route("/get_data")
def get_data():
    month = request.args.get('month', type=int)
    cluster_spend = request.args.get('cluster_spend', type=int)
    label_spend = request.args.get('label_spend', type=str)
    cluster_geo = request.args.get('cluster_geo', type=int)
    label_geo = request.args.get('label_geo', type=str)

    data = get_map_data(month, cluster_spend, label_spend, cluster_geo, label_geo)
    return jsonify(data)

# =========================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
