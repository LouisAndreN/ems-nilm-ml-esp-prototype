import streamlit as st
import serial
import serial.tools.list_ports
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime
import time
from collections import deque
import json

# ===== CONFIG =====
st.set_page_config(
    page_title="EMS Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CSS =====
st.markdown("""
    <style>
    /* Fix metrics visibility */
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
    .stMetric [data-testid="stMetricDelta"] {
        color: #6c757d !important;
    }
    
    /* Calibration status */
    .calibration-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
    .status-idle {
        background-color: #e9ecef;
        color: #495057;
    }
    .status-calibrating {
        background-color: #fff3cd;
        color: #856404;
    }
    .status-ready {
        background-color: #d1e7dd;
        color: #0f5132;
    }
    </style>
""", unsafe_allow_html=True)

# ===== UTILITIES =====
def init_serial_connection(port, baudrate=115200):
    """Initialize serial connection"""
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)
        return ser
    except Exception as e:
        st.error(f"Serialconexion error: {e}")
        return None

def find_serial_ports():
    """List available ports"""
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

def parse_harmonics_line(line):
    """
    Parse line harmonics from ESP32
    """
    try:
        harmonics = {}
        parts = line.strip().split(',')
        for part in parts:
            if ':' in part:
                h_name, h_val = part.split(':')
                h_num = int(h_name.replace('H', ''))
                harmonics[h_num] = float(h_val)
        return harmonics
    except Exception as e:
        return None
    
def parse_metrics_line(line):
    """
    Parse metric lines from ESP32
    """
    try:
        metrics = {}
        parts = line.strip().split(',')
        for part in parts:
            if ':' in part:
                key, val = part.split(':')
                metrics[key] = float(val)
        return metrics
    except:
        return None
    
def calculate_thd(harmonics):
    """Calculate Total Harmonic Distortion"""
    if len(harmonics) < 2:
        return 0.0
    H1 = harmonics.get(1, 0)
    if H1 == 0:
        return 0.0
    sum_squares = sum(harmonics[i]**2 for i in range(2, 9) if i in harmonics)
    return np.sqrt(sum_squares) / H1

def calculate_rms(harmonics):
    """Calculate RMS froms harmonics"""
    return harmonics.get(1, 0)

def apply_calibration(harmonics, offset, gain):
    """Apply calibration to harmonics"""
    if not harmonics or not offset:
        return harmonics
    
    calibrated = {}
    for h_num, h_val in harmonics.items():
        # Remove offset and apply gain
        offset_val = offset.get(h_num, 0)
        calibrated[h_num] = max(0, (h_val - offset_val) * gain)
    
    return calibrated

# ===== SESSION STATE INIT =====
if 'serial_connected' not in st.session_state:
    st.session_state.serial_connected = False
if 'ser' not in st.session_state:
    st.session_state.ser = None
if 'history_harmonics' not in st.session_state:
    st.session_state.history_harmonics = deque(maxlen=100)
if 'history_rms' not in st.session_state:
    st.session_state.history_rms = deque(maxlen=100)
if 'history_thd' not in st.session_state:
    st.session_state.history_thd = deque(maxlen=100)
if 'history_timestamps' not in st.session_state:
    st.session_state.history_timestamps = deque(maxlen=100)
if 'latest_harmonics' not in st.session_state:
    st.session_state.latest_harmonics = {}
if 'latest_rms' not in st.session_state:
    st.session_state.latest_rms = 0
if 'latest_thd' not in st.session_state:
    st.session_state.latest_thd = 0

# ===== CALIBRATION STATE =====
if 'calibration_state' not in st.session_state:
    st.session_state.calibration_state = 'idle'  # idle, noise, reference, ready
if 'noise_offset' not in st.session_state:
    st.session_state.noise_offset = {}
if 'reference_harmonics' not in st.session_state:
    st.session_state.reference_harmonics = {}
if 'calibration_gain' not in st.session_state:
    st.session_state.calibration_gain = 1.0
if 'calibration_samples' not in st.session_state:
    st.session_state.calibration_samples = []
if 'reference_power' not in st.session_state:
    st.session_state.reference_power = 1250.0  # Kettle by default

# ===== SIDEBAR =====
with st.sidebar:
    st.title("⚡ EMS Monitor")
    st.markdown("---")

    # Port serial selection
    st.subheader("Configuration")
    available_ports = find_serial_ports()
    
    if available_ports:
        selected_port = st.selectbox(
            "Serial port",
            available_ports,
            index=0
        )
        baudrate = st.selectbox(
            "Baud rate",
            [9600, 115200, 230400],
            index=1
        )
    else:
        st.warning("No serial port detected")
        selected_port = None
        baudrate = 115200

    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔌 Connect", use_container_width=True):
            if selected_port:
                if st.session_state.ser:
                    try:
                        st.session_state.ser.close()
                    except:
                        pass
                st.session_state.ser = init_serial_connection(selected_port, baudrate)
                if st.session_state.ser:
                    st.session_state.serial_connected = True
                    st.success("Connected !")
                else:
                    st.session_state.serial_connected = False
                    st.error("Connexion fail")
            else:
                st.error("Select port")
                
    with col2:
        if st.button("🔌 Disconect", use_container_width=True):
            if st.session_state.ser:
                try:
                    st.session_state.ser.close()
                except:
                    pass
                st.session_state.serial_connected = False
                st.session_state.ser = None
                st.info("Disconected")

    # Status
    if st.session_state.serial_connected:
        st.success("ESP32 connected")
    else:
        st.error("ESP32 disconected")
    
    st.markdown("---")

    # ===== CALIBRATION SECTION =====
    st.subheader("Calibration")
    
    # Status calibration
    status_map = {
        'idle': ('Non calibrated', 'status-idle'),
        'noise': ('Noise measure...', 'status-calibrating'),
        'reference': ('Reference measure...', 'status-calibrating'),
        'ready': ('Calibrated', 'status-ready')
    }
    status_text, status_class = status_map[st.session_state.calibration_state]
    st.markdown(f'<div class="calibration-status {status_class}">{status_text}</div>', 
                unsafe_allow_html=True)
    
    # Measure noise without load
    st.write("**Step 1:** Noise measure")
    st.caption("Unplug all devices")
    
    # Condition: connected AND (idle OR ready to recalibrate)
    can_measure_noise = (st.session_state.serial_connected and 
                        st.session_state.calibration_state in ['idle', 'ready'])
    
    if st.button("Noise measure", use_container_width=True, 
                 disabled=not can_measure_noise):
        st.session_state.calibration_state = 'noise'
        st.session_state.calibration_samples = []
        st.session_state.noise_offset = {}
        st.rerun()
    
    if not st.session_state.serial_connected:
        st.caption("Connect ESP32 first")
    
    # Reference measure
    st.write("Reference measure")
    reference_power = st.number_input(
        "Reference power (W)",
        min_value=100.0,
        max_value=3000.0,
        value=1250.0,
        step=50.0,
        help="Reference device power"
    )
    st.session_state.reference_power = reference_power
    
    st.caption("Plug reference device")
    
    # Condition: connected AND offset measured AND no calibration in progress
    can_measure_reference = (st.session_state.serial_connected and 
                            len(st.session_state.noise_offset) > 0 and
                            st.session_state.calibration_state == 'idle')
    
    if st.button("Mesure reference", use_container_width=True,
                 disabled=not can_measure_reference):
        st.session_state.calibration_state = 'reference'
        st.session_state.calibration_samples = []
        st.session_state.reference_harmonics = {}
        st.rerun()
    
    if not st.session_state.serial_connected:
        st.caption("Connect ESP32 first")
    elif len(st.session_state.noise_offset) == 0:
        st.caption("Measure noise first")
    
    # Reset calibration
    if st.button("Reset calibration", use_container_width=True):
        st.session_state.calibration_state = 'idle'
        st.session_state.noise_offset = {}
        st.session_state.reference_harmonics = {}
        st.session_state.calibration_gain = 1.0
        st.session_state.calibration_samples = []
        st.rerun()
    
    # Plot calibation values
    if st.session_state.noise_offset:
        with st.expander("Offset noise"):
            st.json({f"H{k}": f"{v:.6f}" for k, v in st.session_state.noise_offset.items()})
    
    if st.session_state.reference_harmonics:
        with st.expander("Reference"):
            st.json({f"H{k}": f"{v:.6f}" for k, v in st.session_state.reference_harmonics.items()})
            st.write(f"Gain: {st.session_state.calibration_gain:.4f}")
    
    st.markdown("---")

    # Plot options
    st.subheader("Plot")
    show_normalized = st.checkbox("Normalized harmonics", value=True)
    show_history = st.checkbox("Temporal history", value=True)
    show_raw = st.checkbox("Plot raw signal", value=False, 
                          help="Plot before calibration")
    refresh_rate = st.slider("Refresh rate (ms)", 100, 2000, 500, 100)
    
    st.markdown("---")
    st.caption("EMS Dashboard v2.0 - with calibration")

# ===== MAIN CONTENT =====
st.title("EMS Dashboard - Realtime Analysis")

# ===== READ DATA AND CALIBRATION =====
if st.session_state.serial_connected and st.session_state.ser:
    try:
        if st.session_state.ser.in_waiting > 0:
            line = st.session_state.ser.readline().decode('utf-8', errors='ignore').strip()

            if line:
                current_harmonics = None
                current_metrics = None

                # Parse data
                if line.startswith('H1:'):
                    current_harmonics = parse_harmonics_line(line)
                elif line.startswith('RMS:'):
                    current_metrics = parse_metrics_line(line)
                elif line.startswith('{'):
                    try:
                        data = json.loads(line)
                        current_harmonics = data.get('harmonics')
                        current_metrics = data.get('metrics')
                    except:
                        pass

                if current_harmonics:
                    # ===== PROCESS CALIBRATION =====
                    if st.session_state.calibration_state == 'noise':
                        # Collect noise samples
                        st.session_state.calibration_samples.append(current_harmonics)
                        
                        if len(st.session_state.calibration_samples) >= 20:  # ~10s à 2Hz
                            # Calculate mean of harmonics (offset)
                            noise_offset = {}
                            for h_num in range(1, 9):
                                values = [s.get(h_num, 0) for s in st.session_state.calibration_samples]
                                noise_offset[h_num] = np.mean(values)
                            
                            st.session_state.noise_offset = noise_offset
                            st.session_state.calibration_state = 'idle'
                            st.session_state.calibration_samples = []
                            st.success("Noise offset measured !")
                            time.sleep(1)
                            st.rerun()
                    
                    elif st.session_state.calibration_state == 'reference':
                        # Collect reference samples
                        st.session_state.calibration_samples.append(current_harmonics)
                        
                        if len(st.session_state.calibration_samples) >= 20:
                            # Calculate harmonics reference mean
                            ref_harmonics = {}
                            for h_num in range(1, 9):
                                values = [s.get(h_num, 0) for s in st.session_state.calibration_samples]
                                ref_harmonics[h_num] = np.mean(values)
                            
                            # Substract offset
                            for h_num in ref_harmonics:
                                ref_harmonics[h_num] -= st.session_state.noise_offset.get(h_num, 0)
                            
                            st.session_state.reference_harmonics = ref_harmonics
                            
                            # Calculate gain (suppose H1 corresponds to power)
                            ref_rms = calculate_rms(ref_harmonics)
                            if ref_rms > 0:
                                # Gain to normalize reference power
                                # Power ≈ V * I, and I ∝ H1
                                # If reference = 1250W at 100V → I ≈ 12.5A
                                expected_current = st.session_state.reference_power / 100.0
                                st.session_state.calibration_gain = expected_current / ref_rms
                            
                            st.session_state.calibration_state = 'ready'
                            st.session_state.calibration_samples = []
                            st.success("Calibration completed !")
                            time.sleep(1)
                            st.rerun()
                    
                    # ===== NORMAL OPERATION =====
                    else:
                        # Apply calibration if ready
                        if st.session_state.calibration_state == 'ready':
                            calibrated_harmonics = apply_calibration(
                                current_harmonics,
                                st.session_state.noise_offset,
                                st.session_state.calibration_gain
                            )
                        else:
                            calibrated_harmonics = current_harmonics
                        
                        # Stor raw ND calibrate
                        if show_raw:
                            st.session_state.latest_harmonics = current_harmonics
                        else:
                            st.session_state.latest_harmonics = calibrated_harmonics
                        
                        st.session_state.history_harmonics.append(calibrated_harmonics)
                        st.session_state.history_timestamps.append(datetime.now())

                        # Metrics current
                        if not current_metrics:
                            rms = calculate_rms(calibrated_harmonics)
                            thd = calculate_thd(calibrated_harmonics)
                        else:
                            rms = current_metrics.get('RMS', 0)
                            thd = current_metrics.get('THD', 0)
                        
                        st.session_state.latest_rms = rms
                        st.session_state.latest_thd = thd
                        st.session_state.history_rms.append(rms)
                        st.session_state.history_thd.append(thd)
                    
    except Exception as e:
        st.error(f"Reading error: {e}")

# ===== PROGRESS BAR CALIBRATION =====
if st.session_state.calibration_state in ['noise', 'reference']:
    progress = len(st.session_state.calibration_samples) / 20
    st.progress(progress, text=f"Acquisition: {len(st.session_state.calibration_samples)}/20 samples")

# ===== PLOT METRICS =====
st.markdown("### Realtime metrics")
metrics_container = st.container()
with metrics_container:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.calibration_state == 'ready':
            power_estimate = st.session_state.latest_rms * 100.0  # V * I
            st.metric(
                "Estimated power",
                f"{power_estimate:.0f} W",
                delta=None
            )
        else:
            st.metric(
                "RMS Current",
                f"{st.session_state.latest_rms:.3f}",
                delta=None
            )
    
    with col2:
        st.metric(
            "THD",
            f"{st.session_state.latest_thd*100:.2f}%",
            delta=None,
            help="Total Harmonic Distortion"
        )
    
    with col3:
        h1_val = st.session_state.latest_harmonics.get(1, 0) if st.session_state.latest_harmonics else 0
        st.metric(
            "Fundamental (H1)",
            f"{h1_val:.4f}",
            delta=None
        )
    
    with col4:
        if st.session_state.latest_harmonics:
            harm_no_h1 = {k: v for k, v in st.session_state.latest_harmonics.items() if k > 1}
            if harm_no_h1:
                dominant = max(harm_no_h1.items(), key=lambda x: x[1])
                st.metric(
                    "Main harmonic",
                    f"H{dominant[0]}",
                    delta=f"{dominant[1]:.4f}"
                )
            else:
                st.metric("Main harmonic", "N/A")
        else:
            st.metric("Main harmonic", "N/A")

# ===== MAIN GRAPHICS =====
st.markdown("---")
col_left, col_right = st.columns(2)

# GRAPH 1: Bar chart harmonics
with col_left:
    st.subheader("Harmonics H1-H8")
    
    if st.session_state.latest_harmonics:
        h_labels = [f'H{i}' for i in range(1, 9)]
        h_values = [st.session_state.latest_harmonics.get(i, 0) for i in range(1, 9)]

        if show_normalized and h_values[0] > 0:
            h_values_norm = [v / h_values[0] for v in h_values]
            y_label = "Normalized amplitude"
        else:
            h_values_norm = h_values
            y_label = "Amplitude"

        fig_harmonics = go.Figure()
        fig_harmonics.add_trace(go.Bar(
            x=h_labels,
            y=h_values_norm,
            marker=dict(
                color=h_values_norm,
                colorscale='Viridis',
                showscale=True
            ),
            text=[f'{v:.4f}' for v in h_values_norm],
            textposition='outside'
        ))
        fig_harmonics.update_layout(
            xaxis_title="Harmonic",
            yaxis_title=y_label,
            height=400,
            template="plotly_white",
            hovermode='x unified'
        )
        st.plotly_chart(fig_harmonics, use_container_width=True)
    else:
        st.info("Waiting for data...")

# GRAPH 2: Radar chart
with col_right:
    st.subheader("Spectral signature")
    
    if st.session_state.latest_harmonics:
        h_labels = [f'H{i}' for i in range(1, 9)]
        h_values = [st.session_state.latest_harmonics.get(i, 0) for i in range(1, 9)]

        if h_values[0] > 0:
            h_values_norm = [v / h_values[0] for v in h_values]
        else:
            h_values_norm = h_values

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=h_values_norm,
            theta=h_labels,
            fill='toself',
            name='Harmonics',
            line=dict(color='rgb(99, 110, 250)')
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(h_values_norm) * 1.1 if max(h_values_norm) > 0 else 1]
                )
            ),
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Waiting for data...")

# ===== HISTORY =====
if show_history:
    st.markdown("---")
    st.subheader("History")
    
    if len(st.session_state.history_timestamps) > 1:
        df_history = pd.DataFrame({
            'timestamp': list(st.session_state.history_timestamps),
            'RMS': list(st.session_state.history_rms),
            'THD': list(st.session_state.history_thd)
        })

        df_history['time_sec'] = (df_history['timestamp'] - 
                                   df_history['timestamp'].iloc[0]).dt.total_seconds()

        col_h1, col_h2 = st.columns(2)
        
        with col_h1:
            if st.session_state.calibration_state == 'ready':
                df_history['Power'] = df_history['RMS'] * 100
                fig_power = px.line(
                    df_history,
                    x='time_sec',
                    y='Power',
                    title='Estimated power (W)',
                    labels={'time_sec': 'Temps (s)', 'Power': 'Puissance (W)'}
                )
                fig_power.update_traces(line_color='#1f77b4', line_width=2)
                fig_power.update_layout(height=300, template="plotly_white")
                st.plotly_chart(fig_power, use_container_width=True)
            else:
                fig_rms = px.line(
                    df_history,
                    x='time_sec',
                    y='RMS',
                    title='RMS Signal',
                    labels={'time_sec': 'Time (s)', 'RMS': 'RMS'}
                )
                fig_rms.update_traces(line_color='#1f77b4', line_width=2)
                fig_rms.update_layout(height=300, template="plotly_white")
                st.plotly_chart(fig_rms, use_container_width=True)

        with col_h2:
            fig_thd = px.line(
                df_history,
                x='time_sec',
                y='THD',
                title='Total Harmonic Distortion',
                labels={'time_sec': 'Time (s)', 'THD': 'THD'}
            )
            fig_thd.update_traces(line_color='#ff7f0e', line_width=2)
            fig_thd.update_layout(height=300, template="plotly_white")
            st.plotly_chart(fig_thd, use_container_width=True)

        with st.expander("Raw Data (last 10)"):
            df_display = df_history[['timestamp', 'RMS', 'THD']].tail(10).copy()
            df_display['timestamp'] = df_display['timestamp'].dt.strftime('%H:%M:%S.%f').str[:-3]
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("Collect data in progress...")

# ===== AUTO REFRESH =====
if st.session_state.serial_connected:
    time.sleep(refresh_rate / 1000)
    st.rerun()
else:

    st.info("Connect to ESP32 from sidebar to begin")
