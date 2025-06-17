from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
import numpy as np
from math import sqrt, ceil

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# === Gemini API Key ===
genai.configure(api_key="AIzaSyC5Zk54SNRj64kj7MsGidogOkkwEsE_sH0")  # Replace with your real key

# === Routes ===

@app.route('/')
def landing():
    return render_template('index.html')  # Landing page (front/index.html)

@app.route('/frontend/<path:filename>')
def serve_frontend(filename):
    return send_from_directory('static/frontend', filename)

@app.route('/api/calculate', methods=['POST'])
def calculate():
    data = request.json
    scenario = data.get('scenario')
    params = data.get('params')

    if not scenario or not params:
        return jsonify({"error": "Missing scenario or parameters"}), 400

    # === Scenario Handling ===
    if scenario == "wireless":
        result = wireless_system_calc(params)
        result_text = "Wireless system outputs:\n" + "\n".join([f"{k}: {v:.2f} bps" for k, v in result.items()])
    elif scenario == "ofdm":
        result = ofdm_calc(params)
        result_text = f"OFDM outputs:\nData rate: {result['data_rate']:.2f} bps\nSpectral efficiency: {result['spectral_efficiency']:.2f} bps/Hz"
    elif scenario == "link_budget":
        result = link_budget_calc(params)
        result_text = f"Link budget outputs:\nFSPL: {result['fspl']:.2f} dB\nReceived power: {result['rx_power_dbm']:.2f} dBm"
    elif scenario == "cellular":
        result = cellular_calc(params)
        result_text = f"Cellular system outputs:\nNumber of cells: {result['n_cells']}\nTotal users: {result['total_users']}"
    else:
        return jsonify({"error": "Invalid scenario"}), 400

    # === Gemini Explanation ===
    try:
        prompt = result_text + "\nExplain how these results were calculated and what they mean."
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        explanation = response.text
    except Exception as e:
        explanation = f"Failed to generate explanation: {str(e)}"

    return jsonify({"result": result, "explanation": explanation})


# === Calculation Functions ===

def wireless_system_calc(params):
    sampler_rate = float(params['bandwidth']) * 2
    quantizer_rate = sampler_rate * int(params['bits_per_sample'])
    source_encoder_rate = quantizer_rate * 0.7
    channel_encoder_rate = source_encoder_rate / 0.8
    interleaver_rate = channel_encoder_rate
    burst_formatter_rate = interleaver_rate
    return {
        "sampler": sampler_rate,
        "quantizer": quantizer_rate,
        "source_encoder": source_encoder_rate,
        "channel_encoder": channel_encoder_rate,
        "interleaver": interleaver_rate,
        "burst_formatter": burst_formatter_rate
    }

def ofdm_calc(params):
    n_subcarriers = int(params['n_subcarriers'])
    bits_per_symbol = int(params['bits_per_symbol'])
    symbols_per_sec = float(params['symbols_per_sec'])
    bandwidth = float(params['bandwidth'])
    data_rate = n_subcarriers * bits_per_symbol * symbols_per_sec
    spectral_efficiency = data_rate / bandwidth
    return {
        "data_rate": data_rate,
        "spectral_efficiency": spectral_efficiency
    }

def link_budget_calc(params):
    tx_power_dbm = float(params['tx_power_dbm'])
    tx_gain_db = float(params['tx_gain_db'])
    rx_gain_db = float(params['rx_gain_db'])
    distance_km = float(params['distance_km'])
    freq_mhz = float(params['freq_mhz'])
    fspl = 20 * np.log10(distance_km * 1000) + 20 * np.log10(freq_mhz) - 27.55
    rx_power_dbm = tx_power_dbm + tx_gain_db + rx_gain_db - fspl
    return {
        "fspl": fspl,
        "rx_power_dbm": rx_power_dbm
    }

def cellular_calc(params):
    area_km2 = float(params['area_km2'])
    cell_radius_km = float(params['cell_radius_km'])
    users_per_cell = int(params['users_per_cell'])
    cell_area = 3 * sqrt(3) / 2 * (cell_radius_km ** 2)
    n_cells = ceil(area_km2 / cell_area)
    total_users = n_cells * users_per_cell
    return {
        "n_cells": n_cells,
        "total_users": total_users
    }


# === Run Server ===

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
