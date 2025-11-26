# Non-Intrusive Load Monitoring System & Energy Monitoring

The goal of this project is to create a monitoring system for home to track energy consumption of home devices such as frdge, induction heater, air conditionner, computers, charging smartphones, lights, ..., to detect anomalies if possible to anticipate maintenance, predict consumption of the household, anticipate devices usage time to predict device usage patterns to avoid tripping the circuit breaker.

This system is deployed in a japanese residence but can also be adapted for industrial fields, in manufacture or other urban buildings.

## Used material

- SCT-013-005 (5A/1V) and SCT-013-030 (30A/1V)
- ADS1115
- Adafruit 3.5mm TRRS jack breakout
- ESP32 S3 R16N8 (or ESP32-WROOM-32E)
- 2× 10kΩ resistors (bias)
- 1× 1kΩ resistor (series)
- 1× 100Ω resistor (protection)
- 1× 10µF capacitor (coupling)
- 2× 100nF capacitors (filtering + decoupling)
- Male plug and female plug JP type
- AWG14 cables
- Proster multimeter

See `docs/hardware_schematic.pdf` for complete circuit.


The SCT-013 current sensor connects via a 3.5mm TRRS jack.


## Future features

Some devices can be tracked but because of the relative low precision of the ADS1115, the system needs to be improved with an audio card 48kHz (PCM2704 ?) to get more harmonics to detect specific devices such as induction heater.

# 日本語

## 概要



