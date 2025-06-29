from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
import numpy as np
import math
from math import sqrt, ceil

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/api/*": {"origins": "*"}})


# === Gemini API Key ===
genai.configure(api_key="AIzaSyC5Zk54SNRj64kj7MsGidogOkkwEsE_sH0")  # Replace with your real key

# === Routes ===

@app.route('/')
def landing():
    return render_template('index.html')  # Landing page (front/index.html)

@app.route('/frontend/<path:filename>')
def serve_frontend(filename):
    return send_from_directory('static/frontend', filename)

def wireless_system_calc(params):
    results = {}
    bw = float(params.get('bandwidth_khz', 0)) or None
    bits = int(params.get('bits_per_sample', 0)) or None
    src_cr = float(params.get('src_compression_rate', 0)) or None
    chan_cr = float(params.get('channel_code_rate', 0)) or None
    burst_ov = float(params.get('burst_formatting_overhead', 0)) or 0.0  # e.g. 0.1 for 10%

    if bw is not None and bw > 0:
        f_s = 2 * bw * 1000  
        results['1-sampling_frequency_hz'] = f_s
    else:
        f_s = None

    if bits is not None and bits > 0:
        q_levels = 2 ** bits
        results['2-quantization_levels'] = q_levels
    else:
        q_levels = None

    if f_s and bits:
        q_bps = f_s * bits
        results['3-quantizer_bitrate_bps'] = q_bps
    else:
        q_bps = None

    if q_bps and src_cr is not None and src_cr > 0:
        se_bps = q_bps * src_cr
        results['4-source_encoder_bitrate_bps'] = se_bps
    else:
        se_bps = None

    if se_bps and chan_cr is not None and chan_cr > 0:
        ce_bps = se_bps / chan_cr
        results['5-channel_encoder_bitrate_bps'] = ce_bps
        results['6-interleaver_bitrate_bps'] = ce_bps
    else:
        ce_bps = None

    if ce_bps is not None and burst_ov > 0:
        burst_bps = ce_bps * (1 + burst_ov)
        results['7-burst_formatted_bitrate_bps'] = burst_bps

    return results

def ofdm_lte_problem_calc(params):
    results = {}
    rb_bw = float(params.get('rb_bandwidth_khz', 0)) or None
    subc_bw = float(params.get('subcarrier_spacing_khz', 0)) or None
    n_syms = int(params.get('n_ofdm_symbols', 0)) or None
    mod_order = int(params.get('mod_order', 0)) or None
    n_rbs = int(params.get('n_parallel_rbs', 0)) or None
    rb_dur = float(params.get('rb_duration_ms', 0)) or None

    if rb_bw and subc_bw and subc_bw > 0:
        n_subc = int(rb_bw / subc_bw)
        results['1-n_subcarriers'] = n_subc
    else:
        n_subc = None

    if mod_order and mod_order > 0:
        bits_per_sym = int(np.log2(mod_order))
        results['2-bits_per_symbol'] = bits_per_sym
    else:
        bits_per_sym = None

    if n_subc and bits_per_sym:
        bits_per_ofdm = n_subc * bits_per_sym
        results['3-bits_per_ofdm_symbol'] = bits_per_ofdm
        results['4-bits_per_resource_element'] = bits_per_sym
    else:
        bits_per_ofdm = None

    if bits_per_ofdm and n_syms:
        bits_per_rb = bits_per_ofdm * n_syms
        results['5-bits_per_resource_block'] = bits_per_rb
    else:
        bits_per_rb = None

    if bits_per_rb and n_rbs:
        bits_all = bits_per_rb * n_rbs
        results['6-bits_in_all_rbs'] = bits_all
    else:
        bits_all = None

    if bits_all and rb_dur and rb_dur > 0:
        max_rate = bits_all / (rb_dur / 1000)
        results['7-max_rate_bps'] = max_rate

        if rb_bw and n_rbs:
            total_bw_hz = rb_bw * n_rbs * 1e3
            spectral_eff = max_rate / total_bw_hz if total_bw_hz > 0 else None
            results['8-spectral_efficiency_bps_per_hz'] = spectral_eff

    return results


def link_budget_calc(params):

    def get_float(val, default=0.0):
        try:
            if val is None or val == '':
                return default
            return float(val)
        except:
            return default

    k = 1.38e-23  
    linear_to_db = lambda x: 10 * math.log10(x)

    tx_power_db = get_float(params.get("tx_power_db"))
    rx_power_db = get_float(params.get("rx_power_db"))
    tx_gain_db = get_float(params.get("tx_gain_db"))
    tx_amplifier_gain_db = get_float(params.get("tx_amplifier_gain_db"))
    tx_loss_db = get_float(params.get("tx_loss_db"))
    rx_gain_db = get_float(params.get("rx_gain_db"))
    rx_amplifier_gain_db = get_float(params.get("rx_amplifier_gain_db"))
    rx_loss_db = get_float(params.get("rx_loss_db"))
    path_loss_db = get_float(params.get("path_loss_db"))
    other_loss_db = get_float(params.get("other_loss_db"))
    fade_margin_db = get_float(params.get("fade_margin_db"))
    data_rate_bps = get_float(params.get("data_rate_bps"))
    noise_figure_db = get_float(params.get("noise_figure_db"))
    noise_temp_K = get_float(params.get("noise_temp_K"), 290)
    required_ebn0_db = get_float(params.get("required_ebn0_db"))
    freq_mhz = get_float(params.get("freq_mhz"))
    distance_m = get_float(params.get("distance_m"))

    results = {}

    kt_dbw = linear_to_db(k * noise_temp_K)       
    r_db = linear_to_db(data_rate_bps) if data_rate_bps else 0  

    if rx_power_db == 0 and tx_power_db != 0:
        rx_power_actual_db = (
            tx_power_db
            + tx_gain_db
            + tx_amplifier_gain_db
            + rx_gain_db
            + rx_amplifier_gain_db
            - tx_loss_db
            - rx_loss_db
            - path_loss_db
            - other_loss_db
            - fade_margin_db
        )
        results['rx_power_db'] = rx_power_actual_db
    else:
        rx_power_actual_db = rx_power_db if rx_power_db != 0 else None

    if required_ebn0_db != 0 and data_rate_bps != 0 and fade_margin_db != 0 and noise_figure_db != 0:
        rx_power_min_db = (
            required_ebn0_db
            + kt_dbw
            + fade_margin_db
            + noise_figure_db
            + r_db
        )
        results['required_rx_power_db_for_ebn0'] = rx_power_min_db
    else:
        rx_power_min_db = None

    if rx_power_min_db is not None:
        required_tx_power_db = (
            rx_power_min_db
            + path_loss_db
            + other_loss_db
            + fade_margin_db
            + tx_loss_db
            - tx_gain_db
            - rx_gain_db
            - rx_amplifier_gain_db
        )
        results['required_tx_power_db_for_ebn0'] = required_tx_power_db

    if rx_power_actual_db is not None and required_ebn0_db == 0 and data_rate_bps != 0 and noise_figure_db !=0  and fade_margin_db !=0:
        ebn0_at_detector_db = rx_power_actual_db - (kt_dbw + fade_margin_db + noise_figure_db + r_db)
        results['ebn0_at_detector_db'] = ebn0_at_detector_db

    if rx_power_actual_db is not None and  noise_figure_db ==0 and required_ebn0_db != 0 and data_rate_bps != 0   and fade_margin_db !=0:
        noise_figure_calc = rx_power_actual_db - required_ebn0_db - kt_dbw - r_db - fade_margin_db
        results['calculated_noise_figure_db'] = noise_figure_calc

    if rx_power_actual_db is not None and required_ebn0_db != 0 and data_rate_bps != 0 and noise_figure_db !=0  and fade_margin_db ==0:
        fade_margin_db_calc = rx_power_actual_db - required_ebn0_db - kt_dbw - r_db - noise_figure_db
        results['calculated_fade_margin_db'] = fade_margin_db_calc

    if rx_power_actual_db is not None and required_ebn0_db != 0 and data_rate_bps == 0 and noise_figure_db !=0  and fade_margin_db !=0 :
        r_db_calc = rx_power_actual_db - required_ebn0_db - kt_dbw - noise_figure_db - fade_margin_db
        results['calculated_r_db'] = r_db_calc
        results['calculated_r_linear'] = 10 ** (r_db_calc / 10)


    if freq_mhz != 0 and distance_m != 0:
        fspl_db = 20 * math.log10(distance_m) + 20 * math.log10(freq_mhz) + 32.44
        results['fspl_db'] = fspl_db

    if tx_power_db != 0 and rx_power_actual_db is not None:
        allowable_path_loss = tx_power_db - rx_power_actual_db
        results['allowable_path_loss_db'] = allowable_path_loss

    return results

def cellular_system_calc(params):
    results = {}

    timeslots_per_carrier = int(params.get('timeslots_per_carrier', 0)) or None
    area_m2 = float(params.get('area_km2', 0)) or None
    n_users = int(params.get('n_users', 0)) or None
    call_rate = float(params.get('call_rate', 0)) or None
    avg_call_duration_min = float(params.get('avg_call_duration_min', 0)) or None
    min_sir_db = int(params.get('min_sir_db', 0)) or None
    ref_power_db = float(params.get('ref_power_db', 0)) or None
    ref_distance_m = float(params.get('ref_distance_m', 0)) or None
    path_loss_exp = float(params.get('path_loss_exp', 0)) or None
    receiver_sensitivity_uw = float(params.get('receiver_sensitivity_uw', 0)) * 10 ** -6 or None
    cochannel_interferers = int(params.get('cochannel_interferers', 0)) or None
    n_channels = int(params.get('n_channels', 0)) or None
    
    if ref_power_db is not None and ref_distance_m and path_loss_exp and receiver_sensitivity_uw:
        ref_poewr =10 ** (ref_power_db/ 10)
        r= (receiver_sensitivity_uw /ref_poewr )** (1/3)
        d_max = ref_distance_m  /r

        results['1-Max_distance_m'] = d_max
    else:
        d_max = None

    if d_max is not None:
        cell_area_km2 = (3 * sqrt(3) / 2) * (d_max ) ** 2
        results['2-Max_cell_area_m2'] = cell_area_km2
    else:
        cell_area_km2 = None

    if area_m2 is not None and cell_area_km2 and cell_area_km2 > 0:
        n_cells = int(ceil(area_m2 / cell_area_km2))
        results['3-N_cells'] = n_cells
    else:
        n_cells = None

    if n_users and call_rate and avg_call_duration_min:
        avg_call_duration_hr = avg_call_duration_min / 60
        user_traffic = call_rate * avg_call_duration_hr / 24
        total_traffic = n_users * user_traffic
        results['4-Total_traffic_erlangs'] = total_traffic
    else:
        total_traffic = None

    if total_traffic and n_cells:
        traffic_per_cell = total_traffic / n_cells
        results['5-Traffic_per_cell_erlangs'] = traffic_per_cell
    else:
        traffic_per_cell = None

    if cochannel_interferers:
        min_sir =10 ** (min_sir_db/ 10)
        N_cluster = math.ceil((1/3) * ((cochannel_interferers * min_sir) ** (2 / path_loss_exp)))

        results['6-Cluster_size_N'] = N_cluster
    else:
        N_cluster = None

    if n_channels and timeslots_per_carrier:
        total_carriers = int(ceil(n_channels / timeslots_per_carrier))
        results['7-Total_carriers'] = total_carriers
        results['8-Number of carrier of whole system'] = total_carriers * N_cluster 
        results['9-Total of  cluster'] = n_cells /N_cluster

    else:
        results['total_carriers'] = None

    return results




def ask_ai_for_explanation(scenario, params, result):
    prompt = (
        f"You are an expert wireless/mobile networks assistant.\n"
        f"Scenario: {scenario}\n"
        f"User Inputs: {params}\n"
        f"Calculated Outputs: {result}\n\n"
        "Please:\n"
        "1. Validate the user inputs. If anything is missing, odd, or inconsistent, mention it clearly.\n"
        "2. Then, step by step, explain how the outputs were computed from the inputs (formulas, logic, assumptions).\n"
        "3. For each output value, explain what it means and why it's important for a wireless or mobile network designer.\n"
        "Use clear language, with short paragraphs or lists. Assume the reader is a student or junior engineer."
    )
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

@app.route('/api/calculate', methods=['POST'])
def calculate():
    data = request.json
    scenario = data['scenario']
    params = data['params']

    # Call correct calculator
    if scenario == "wireless":
        result = wireless_system_calc(params)
    elif scenario == "ofdm":
        result = ofdm_lte_problem_calc(params)
    elif scenario == "link_budget":
        result = link_budget_calc(params)
    elif scenario == "cellular":
        result = cellular_system_calc(params)
    else:
        return jsonify({"error": "Invalid scenario"}), 400


    explanation = ask_ai_for_explanation(scenario, params, result)
    return jsonify({"result": result, "explanation": explanation})

# === Run Server ===

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
