from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load data sekali saat server start dari pickle
data_segmentasi = pd.read_pickle("data/data_segmentasi.pkl")
data_segmentasi['month'] = data_segmentasi['trans_datetime'].dt.month

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
    # Halaman prediksi kosong atau placeholder
    return render_template("prediksi.html")

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
    app.run(debug=True)
