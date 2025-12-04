# 🏡🔌 Home Energy Management System prototype with NILM (Non-Intrusive Load Monitoring) Device

This project provides a full end-to-end Energy Management System, using  Non-Intrusive Load Monitoring (NILM) system designed for the Japanese electrical environment (100V, 50/60Hz).
It integrates custom low-cost hardware, real household data, and lightweight machine learning models to estimate appliance-level energy usage in real time.

The goal is to provide an accessible and practical system for monitoring household energy usage, detecting anomalies, and predicting future consumption patterns — with a strong focus on real-world deployment rather than academic demonstration.

**Key functionnalities :**
- Real-time appliance-level energy disaggregation (fridge, AC, IH cooking heater, computer, smartphone charging, lighting, etc.)
- Forecasting of total household consumption and prediction of device usage patterns
- From Custom analog conditioning circuit design to ML deployment
- Dashboard for live monitoring and historical analysis
- Edge preprocessing (ESP32) + Central ML (Raspberry Pi 5)
- Support for Japanese standards (seasonal patterns, 30A breaker, 100V, 50/60Hz)

## Used material

- Current transformer SCT-013-030 (30A/1V)
- 16-bit ADC ADS1115 (860 SPS)
- Adafruit 3.5mm TRRS jack breakout
- ESP32-WROOM-32E
- 2× 10kΩ resistors (divider)
- 1× 1kΩ resistor (anti-aliasing filter)
- 1× 100Ω resistor (protection)
- 1× 10µF capacitor (coupling)
- 1× 470nF capacitor (anti-aliasing filter)
- Male plug and female plug JP type
- AWG14 cables (can use AWG16 cables)
- Proster multimeter with clamper

See `docs/schematic-circuit.png` for complete circuit.

The SCT-013 current sensor connects via a 3.5mm TRRS jack.
The current components allows to get until the 8th harmonics, which can be limited for monitoring more advanced devices such as induction heater, solderer, ...

# Status : 🚧 **In progress**
- Dashboard for visualization
- Individual device signatures recording for classification library
- Consideration of PCM1808 to get more harmonics for advanced devices and Deep Learning Model evaluation for prediction and anomaly detection

# How to run

- Install the project on your computer
- Run `streamlit run <path_to_project_folder>/EMS-dashboard.py`
- The interactive dashboard should open in your browser
- Follow instructions to setup the system (calibration without load, with load reference, add new device to library, ...)

## Future features

Some devices can be tracked but because of the relative low precision of the ADS1115, the system needs to be improved with an audio card 48kHz (PCM1808 ?) to get more harmonics until 24kHz (Nyquist theorem) to detect specific devices such as induction heater.

# 日本語

## 概要



