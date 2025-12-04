"""
NILM Dashboard with Full FFT Analysis
v4.3 - Fixed frequency detection during noise calibration
"""

import streamlit as st
import serial
import serial.tools.list_ports
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from collections import deque
import json

# ===== CONFIG =====
st.set_page_config(
    page_title="NILM Dashboard - Full FFT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== CSS =====
st.markdown("""
    <style>
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    .stMetric label {
        color: #495057 !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #212529 !important;
        font-size: 1.8rem;
    }
    .calibration-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
    .status-idle { background-color: #e9ecef; color: #495057; }
    .status-calibrating { background-color: #fff3cd; color: #856404; }
    .status-ready { background-color: #d1e7dd; color: #0f5132; }
    .status-running { background-color: #cfe2ff; color: #084298; }
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    </style>
""", unsafe_allow_html=True)

# ===== UTILITIES =====
def init_serial(port, baudrate=115200):
    try:
        ser = serial.Serial(port, baudrate, timeout=0.5)
        time.sleep(1)
        return ser
    except Exception as e:
        st.error(f"Serial error: {e}")
        return None

def find_serial_ports():
    return [port.device for port in serial.tools.list_ports.comports()]

def send_command(ser, command):
    if ser and ser.is_open:
        try:
            ser.write(f"{command}\n".encode())
            time.sleep(0.05)
        except:
            pass

def parse_json_data(line):
    try:
        return json.loads(line)
    except:
        return None

def detect_grid_frequency(fft_data, freq_resolution, expected_freq=50.0, threshold=10.0):
    """
    Detect actual grid frequency from FFT peak
    Returns detected frequency only if peak is strong enough (not noise)
    """
    # Search around expected frequency (±5 Hz)
    search_start = max(0, int((expected_freq - 5) / freq_resolution))
    search_end = min(len(fft_data), int((expected_freq + 5) / freq_resolution))
    
    if search_end <= search_start:
        return expected_freq
    
    search_region = fft_data[search_start:search_end]
    
    # Check if there's a significant peak (not just noise)
    peak_magnitude = np.max(search_region)
    noise_floor = np.median(fft_data)  # Estimate noise floor
    
    # Only detect frequency if peak is significantly above noise
    if peak_magnitude < threshold or peak_magnitude < noise_floor * 3:
        return expected_freq  # Return expected if no strong signal
    
    peak_idx = np.argmax(search_region)
    detected_freq = (search_start + peak_idx) * freq_resolution
    
    # Sanity check: frequency must be reasonable
    if 45.0 <= detected_freq <= 65.0:
        return detected_freq
    else:
        return expected_freq

# ===== SESSION STATE INIT =====
def reset_calibration_state():
    """Reset ALL calibration-related state"""
    st.session_state.calibration_state = 'idle'
    st.session_state.noise_fft = None
    st.session_state.calibration_samples = []
    st.session_state.gain_calibrated = 1.0
    st.session_state.calibration_timeout = 0

if 'serial_connected' not in st.session_state:
    st.session_state.serial_connected = False
if 'ser' not in st.session_state:
    st.session_state.ser = None

# Data storage - 3 minutes rolling window @ 2Hz = 360 samples
ROLLING_WINDOW_SIZE = 360
if 'history_fft' not in st.session_state:
    st.session_state.history_fft = deque(maxlen=100)
if 'history_metrics' not in st.session_state:
    st.session_state.history_metrics = deque(maxlen=ROLLING_WINDOW_SIZE)
if 'history_timestamps' not in st.session_state:
    st.session_state.history_timestamps = deque(maxlen=ROLLING_WINDOW_SIZE)

if 'latest_fft' not in st.session_state:
    st.session_state.latest_fft = None
if 'latest_metrics' not in st.session_state:
    st.session_state.latest_metrics = {}

# Calibration
if 'calibration_state' not in st.session_state:
    st.session_state.calibration_state = 'idle'
if 'noise_fft' not in st.session_state:
    st.session_state.noise_fft = None
if 'calibration_samples' not in st.session_state:
    st.session_state.calibration_samples = []
if 'calibration_timeout' not in st.session_state:
    st.session_state.calibration_timeout = 0

# Parameters
if 'reference_power' not in st.session_state:
    st.session_state.reference_power = 1250.0
if 'grid_frequency' not in st.session_state:
    st.session_state.grid_frequency = 50.0
if 'grid_voltage' not in st.session_state:
    st.session_state.grid_voltage = 100.0
if 'gain_calibrated' not in st.session_state:
    st.session_state.gain_calibrated = 1.0
if 'detected_frequency' not in st.session_state:
    st.session_state.detected_frequency = 50.0

# ===== SIDEBAR =====
with st.sidebar:
    st.title("⚡ NILM Monitor")
    st.markdown("---")
    
    st.subheader("Configuration")
    available_ports = find_serial_ports()
    
    if available_ports:
        selected_port = st.selectbox("Serial port", available_ports, index=0)
        baudrate = st.selectbox("Baud rate", [9600, 115200, 230400], index=1)
    else:
        st.warning("No serial port")
        selected_port = None
        baudrate = 115200
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔌 Connect", width='stretch'):
            if selected_port:
                if st.session_state.ser:
                    try:
                        st.session_state.ser.close()
                    except:
                        pass
                st.session_state.ser = init_serial(selected_port, baudrate)
                if st.session_state.ser:
                    st.session_state.serial_connected = True
                    st.success("Connected!")
                else:
                    st.session_state.serial_connected = False
                    st.error("Failed")
    
    with col2:
        if st.button("🔌 Disconnect", width='stretch'):
            if st.session_state.ser:
                try:
                    st.session_state.ser.close()
                except:
                    pass
                st.session_state.serial_connected = False
                st.session_state.ser = None
                reset_calibration_state()
                st.info("Disconnected")
    
    if st.session_state.serial_connected:
        st.success("✅ ESP32 connected")
    else:
        st.error("❌ ESP32 disconnected")
    
    st.markdown("---")
    
    # Grid parameters
    st.subheader("Grid Parameters")
    
    # Manual frequency input (used as default/fallback)
    grid_freq_manual = st.number_input("Expected Frequency (Hz)", 45.0, 65.0, 50.0, 0.1,
                                       help="Used as fallback when no signal detected")
    
    if grid_freq_manual != st.session_state.grid_frequency:
        st.session_state.grid_frequency = grid_freq_manual
        st.session_state.detected_frequency = grid_freq_manual
        send_command(st.session_state.ser, f"GRID_FREQ:{grid_freq_manual}")
    
    grid_volt = st.number_input("Voltage (V)", 80.0, 250.0, 100.0, 1.0)
    
    if grid_volt != st.session_state.grid_voltage:
        st.session_state.grid_voltage = grid_volt
        send_command(st.session_state.ser, f"GRID_VOLT:{grid_volt}")
    
    # Display detected frequency (read-only) - only show if different from expected
    if abs(st.session_state.detected_frequency - st.session_state.grid_frequency) > 0.1:
        st.metric("Detected Frequency", f"{st.session_state.detected_frequency:.2f} Hz",
                 delta=f"{st.session_state.detected_frequency - st.session_state.grid_frequency:+.2f} Hz")
    else:
        st.metric("Detected Frequency", f"{st.session_state.detected_frequency:.2f} Hz")
    
    st.markdown("---")
    
    # Calibration
    st.subheader("🎯 Calibration")
    
    status_map = {
        'idle': ('⚪ Not calibrated', 'status-idle'),
        'noise': ('🟡 Measuring noise...', 'status-calibrating'),
        'reference': ('🟡 Measuring reference...', 'status-calibrating'),
        'ready': ('🟢 Calibrated', 'status-ready'),
        'running': ('🔵 Running', 'status-running')
    }
    status_text, status_class = status_map[st.session_state.calibration_state]
    st.markdown(f'<div class="calibration-status {status_class}">{status_text}</div>', 
                unsafe_allow_html=True)
    
    # Step 1: Noise
    st.write("**Step 1:** Noise baseline (50 samples)")
    st.caption("⚠️ Unplug all devices")
    
    col_n1, col_n2 = st.columns([3, 1])
    with col_n1:
        if st.button("📊 Measure noise", width='stretch',
                     disabled=not (st.session_state.serial_connected and 
                                  st.session_state.calibration_state in ['idle', 'ready'])):
            st.session_state.calibration_state = 'noise'
            st.session_state.calibration_samples = []
            st.session_state.noise_fft = None
            st.session_state.calibration_timeout = time.time()
            st.rerun()
    with col_n2:
        if st.button("❌", width='stretch',
                     disabled=st.session_state.calibration_state != 'noise'):
            st.session_state.calibration_state = 'idle'
            st.session_state.calibration_samples = []
            st.rerun()
    
    # Step 2: Reference
    st.write("**Step 2:** Reference device (50 samples)")
    reference_power = st.number_input("Reference power (W)", 100.0, 3000.0, 1250.0, 50.0)
    st.session_state.reference_power = reference_power
    st.caption("⚡ Plug reference device")
    
    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        if st.button("⚡ Measure reference", width='stretch',
                     disabled=not (st.session_state.serial_connected and
                                  st.session_state.noise_fft is not None and
                                  st.session_state.calibration_state == 'idle')):
            st.session_state.calibration_state = 'reference'
            st.session_state.calibration_samples = []
            st.session_state.calibration_timeout = time.time()
            st.rerun()
    with col_r2:
        if st.button("✖️", width='stretch',
                     disabled=st.session_state.calibration_state != 'reference'):
            st.session_state.calibration_state = 'idle'
            st.session_state.calibration_samples = []
            st.rerun()
    
    # Step 3: Start
    st.write("**Step 3:** Start monitoring")
    if st.button("🚀 Start System", width='stretch',
                 disabled=st.session_state.calibration_state != 'ready'):
        # Clear history when starting fresh
        st.session_state.history_metrics.clear()
        st.session_state.history_timestamps.clear()
        st.session_state.calibration_state = 'running'
        st.rerun()
    
    # Controls
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🔄 Reset", width='stretch'):
            reset_calibration_state()
            st.session_state.history_metrics.clear()
            st.session_state.history_timestamps.clear()
            st.session_state.detected_frequency = st.session_state.grid_frequency
            st.rerun()
    
    with col_s2:
        if st.button("⏸️ Stop", width='stretch',
                     disabled=st.session_state.calibration_state != 'running'):
            st.session_state.calibration_state = 'ready'
            st.rerun()
    
    # Display calibration info
    if st.session_state.noise_fft is not None:
        with st.expander("📋 Noise baseline"):
            noise_rms = np.sqrt(np.mean(np.array(st.session_state.noise_fft)**2))
            st.write(f"RMS noise: {noise_rms:.3f} mV")
    
    if st.session_state.calibration_state in ['ready', 'running']:
        with st.expander("📋 Calibration"):
            st.write(f"**Gain:** {st.session_state.gain_calibrated:.6f}")
            st.write(f"**Reference:** {st.session_state.reference_power:.0f} W")
    
    st.markdown("---")
    st.caption("NILM Dashboard v4.3")

# ===== MAIN CONTENT =====
st.title("⚡ NILM Dashboard - Full FFT Analysis")

# ===== DATA ACQUISITION =====
if st.session_state.serial_connected and st.session_state.ser and st.session_state.ser.is_open:
    # Timeout check
    if st.session_state.calibration_state in ['noise', 'reference']:
        if time.time() - st.session_state.calibration_timeout > 60:
            st.error("⏱️ Timeout! No data received in 60s")
            st.session_state.calibration_state = 'idle'
            st.session_state.calibration_samples = []
            st.rerun()
            
    try:
        lines_read = 0
        max_lines = 10
        
        while st.session_state.ser.in_waiting > 0 and lines_read < max_lines:
            try:
                line = st.session_state.ser.readline().decode('utf-8', errors='ignore').strip()
            except:
                continue
                
            if not line or len(line) < 10:
                continue
                
            lines_read += 1
            data = parse_json_data(line)
            
            if not data or 'fft' not in data:
                continue
                
            fft_raw = data.get('fft')
            if not isinstance(fft_raw, list) or len(fft_raw) == 0:
                continue
                
            fft_data = np.array(fft_raw)
            freq_resolution = data.get('freq_resolution', 7.8125)
            
            # ===== FREQUENCY DETECTION =====
            # Only detect frequency when NOT measuring noise (no signal during noise cal)
            if st.session_state.calibration_state != 'noise':
                detected_freq = detect_grid_frequency(
                    fft_data, 
                    freq_resolution, 
                    st.session_state.grid_frequency,
                    threshold=10.0  # Minimum peak magnitude
                )
                
                # Update detected frequency
                st.session_state.detected_frequency = detected_freq
                
                # Update ESP32 if frequency changed significantly
                if abs(detected_freq - st.session_state.grid_frequency) > 0.5:
                    st.session_state.grid_frequency = detected_freq
                    send_command(st.session_state.ser, f"GRID_FREQ:{detected_freq}")
            # During noise calibration, keep the expected frequency
            else:
                st.session_state.detected_frequency = st.session_state.grid_frequency
            
            # ===== CALIBRATION PROCESSING =====
            if st.session_state.calibration_state == 'noise':
                st.session_state.calibration_samples.append(fft_data)
                
                if len(st.session_state.calibration_samples) >= 50:
                    st.session_state.noise_fft = np.mean(
                        st.session_state.calibration_samples, axis=0
                    ).tolist()
                    st.session_state.calibration_state = 'idle'
                    st.session_state.calibration_samples = []
                    st.success("✅ Noise baseline captured!")
                    time.sleep(0.5)
                    st.rerun()
            
            elif st.session_state.calibration_state == 'reference':
                st.session_state.calibration_samples.append({
                    'fft': fft_data,
                    'v_diff_mv': data.get('v_diff_mv', 0),
                    'v_rms': data.get('v_rms', 0)
                })
                
                if len(st.session_state.calibration_samples) >= 50:
                    avg_v_rms = np.mean([s['v_rms'] for s in st.session_state.calibration_samples])
                    avg_v_diff_mv = np.mean([s['v_diff_mv'] for s in st.session_state.calibration_samples])
                    
                    expected_current = st.session_state.reference_power / st.session_state.grid_voltage
                    
                    if avg_v_rms > 0:
                        st.session_state.gain_calibrated = expected_current / avg_v_rms
                        send_command(st.session_state.ser, f"CAL_GAIN:{st.session_state.gain_calibrated}")
                    
                    st.session_state.calibration_state = 'ready'
                    st.session_state.calibration_samples = []
                    st.success(f"✅ Calibrated! Gain: {st.session_state.gain_calibrated:.6f}")
                    st.success(f"Avg V_diff: {avg_v_diff_mv:.2f} mV")
                    time.sleep(0.5)
                    st.rerun()
            
            # ===== NORMAL OPERATION =====
            elif st.session_state.calibration_state in ['ready', 'running']:
                st.session_state.latest_fft = fft_data
                st.session_state.latest_metrics = data
                st.session_state.history_fft.append(fft_data)
                st.session_state.history_metrics.append(data)
                st.session_state.history_timestamps.append(datetime.now())
    
    except Exception as e:
        st.error(f"Error: {e}")

# ===== CALIBRATION PROGRESS =====
if st.session_state.calibration_state in ['noise', 'reference']:
    progress = len(st.session_state.calibration_samples) / 50
    elapsed = int(time.time() - st.session_state.calibration_timeout)
    st.progress(progress, text=f"Samples: {len(st.session_state.calibration_samples)}/50 ({elapsed}s)")

# ===== METRICS DISPLAY =====
if st.session_state.calibration_state in ['ready', 'running'] and st.session_state.latest_metrics:
    metrics = st.session_state.latest_metrics
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("V_diff (mV)", f"{metrics.get('v_diff_mv', 0):.2f}")
    
    with col2:
        st.metric("I_RMS (A)", f"{metrics.get('i_rms', 0):.3f}")
    
    with col3:
        st.metric("Power (W)", f"{metrics.get('power', 0):.0f}")
    
    with col4:
        st.metric("THD (%)", f"{metrics.get('thd', 0)*100:.2f}")
    
    with col5:
        st.metric("Frequency (Hz)", f"{st.session_state.detected_frequency:.2f}")

# ===== TABS =====
if st.session_state.calibration_state in ['ready', 'running']:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Full FFT Spectrum", 
        "⚡ Power & Current", 
        "📈 Time Series",
        "🔬 Advanced"
    ])
    
    # TAB 1: FULL FFT
    with tab1:
        if st.session_state.latest_fft is not None:
            fft_data = st.session_state.latest_fft
            fft_bins = len(fft_data)
            freq_resolution = st.session_state.latest_metrics.get('freq_resolution', 7.8125)
            
            frequencies = np.arange(fft_bins) * freq_resolution
            
            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(
                x=frequencies, y=fft_data, mode='lines', name='FFT Magnitude',
                line=dict(color='#1f77b4', width=1), fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.3)'
            ))
            
            # Mark harmonics using DETECTED frequency
            grid_freq = st.session_state.detected_frequency
            for h in range(1, 17):
                harm_freq = grid_freq * h
                if harm_freq < frequencies[-1]:
                    fig_fft.add_vline(
                        x=harm_freq, line_dash="dash",
                        line_color="red" if h == 1 else "orange",
                        annotation_text=f"H{h}", annotation_position="top"
                    )
            
            fig_fft.update_layout(
                title=f"Complete FFT Spectrum ({fft_bins} bins, Δf={freq_resolution:.2f} Hz, f={grid_freq:.2f} Hz)",
                xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (mV)",
                height=500, template="plotly_white", hovermode='x'
            )
            
            st.plotly_chart(fig_fft, width='stretch')
            
            # Harmonics details
            st.subheader("Harmonics Detail")
            col_z1, col_z2 = st.columns(2)
            
            with col_z1:
                fig_low = go.Figure()
                harm_mags = []
                for h in range(1, 9):
                    freq = grid_freq * h
                    bin_idx = int(freq / freq_resolution)
                    if bin_idx < len(fft_data):
                        harm_mags.append(fft_data[bin_idx])
                
                fig_low.add_trace(go.Bar(
                    x=[f"H{i+1}" for i in range(len(harm_mags))], y=harm_mags,
                    marker=dict(color=harm_mags, colorscale='Viridis', showscale=True),
                    text=[f"{m:.2f}" for m in harm_mags], textposition='outside'
                ))
                fig_low.update_layout(
                    title="Harmonics H1-H8", xaxis_title="Harmonic",
                    yaxis_title="Magnitude (mV)", height=350, template="plotly_white"
                )
                st.plotly_chart(fig_low, width='stretch')
            
            with col_z2:
                fig_high = go.Figure()
                harm_mags_high = []
                for h in range(9, 17):
                    freq = grid_freq * h
                    bin_idx = int(freq / freq_resolution)
                    if bin_idx < len(fft_data):
                        harm_mags_high.append(fft_data[bin_idx])
                
                if harm_mags_high:
                    fig_high.add_trace(go.Bar(
                        x=[f"H{i+9}" for i in range(len(harm_mags_high))], y=harm_mags_high,
                        marker=dict(color=harm_mags_high, colorscale='Plasma', showscale=True),
                        text=[f"{m:.2f}" for m in harm_mags_high], textposition='outside'
                    ))
                    fig_high.update_layout(
                        title="Harmonics H9-H16", xaxis_title="Harmonic",
                        yaxis_title="Magnitude (mV)", height=350, template="plotly_white"
                    )
                    st.plotly_chart(fig_high, width='stretch')
        else:
            st.info("Waiting for FFT data...")

    # TAB 2: POWER & CURRENT (3-minute rolling window)
    with tab2:
        if len(st.session_state.history_timestamps) > 1:
            df = pd.DataFrame(list(st.session_state.history_metrics))
            df['timestamp'] = list(st.session_state.history_timestamps)
            
            # Filter last 3 minutes
            now = datetime.now()
            three_min_ago = now - timedelta(minutes=3)
            df = df[df['timestamp'] >= three_min_ago]
            
            if len(df) > 0:
                df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
                
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    fig_power = go.Figure()
                    fig_power.add_trace(go.Scatter(
                        x=df['time_sec'], y=df['power'] * 1.05, fill=None, mode='lines',
                        line=dict(width=0), showlegend=False, hoverinfo='skip'
                    ))
                    fig_power.add_trace(go.Scatter(
                        x=df['time_sec'], y=df['power'] * 0.95, fill='tonexty', mode='lines',
                        line=dict(width=0), fillcolor='rgba(31, 119, 180, 0.2)',
                        name='±5%', hoverinfo='skip'
                    ))
                    fig_power.add_trace(go.Scatter(
                        x=df['time_sec'], y=df['power'], mode='lines', name='Power',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    fig_power.update_layout(
                        title="Power (W) - Last 3 Minutes",
                        xaxis_title="Time (s)", yaxis_title="Power (W)",
                        height=400, template="plotly_white",
                        xaxis=dict(range=[0, 180])  # Fixed 3-minute window
                    )
                    st.plotly_chart(fig_power, width='stretch')
                
                with col_p2:
                    fig_current = px.line(df, x='time_sec', y='i_rms', 
                                         title='Current (A) - Last 3 Minutes',
                                         labels={'time_sec': 'Time (s)', 'i_rms': 'Current (A)'})
                    fig_current.update_traces(line_color='#ff7f0e', line_width=2)
                    fig_current.update_layout(
                        height=400, template="plotly_white",
                        xaxis=dict(range=[0, 180])  # Fixed 3-minute window
                    )
                    st.plotly_chart(fig_current, width='stretch')

    # TAB 3: TIME SERIES
    with tab3:
        if len(st.session_state.history_timestamps) > 1:
            df = pd.DataFrame(list(st.session_state.history_metrics))
            df['timestamp'] = list(st.session_state.history_timestamps)
            df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                fig_vrms = px.line(df, x='time_sec', y='v_rms', title='V_RMS (V)',
                                  labels={'time_sec': 'Time (s)', 'v_rms': 'V_RMS (V)'})
                fig_vrms.update_traces(line_color='#2ca02c', line_width=2)
                fig_vrms.update_layout(height=350, template="plotly_white")
                st.plotly_chart(fig_vrms, width='stretch')

    # TAB 4: ADVANCED
    with tab4:
        st.subheader("📋 Raw Data")
        if st.session_state.latest_metrics:
            st.json(st.session_state.latest_metrics)
        
        st.subheader("💾 Export Data")
        if len(st.session_state.history_metrics) > 0:
            df_export = pd.DataFrame(list(st.session_state.history_metrics))
            csv = df_export.to_csv(index=False)
            st.download_button("⬇️ Download CSV", csv, "nilm_data.csv", "text/csv",
                             width='stretch')
else:
    st.info("👈 Complete calibration to start monitoring")

# ===== AUTO REFRESH =====
if st.session_state.serial_connected and st.session_state.ser and st.session_state.ser.is_open:
    if st.session_state.calibration_state in ['noise', 'reference']:
        time.sleep(0.1)
    elif st.session_state.calibration_state == 'running':
        time.sleep(0.3)
    st.rerun()