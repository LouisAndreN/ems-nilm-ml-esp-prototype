// Impedances at 50Hz
Z_C1 = 1/(2π × 50 × 10µF) = 318 Ω
Z_R1 = 10 kΩ
Z_R2 = 10 kΩ
Z_C_VREF = 1/(2π × 50 × 100nF) = 31.8 kΩ
Z_Rpull = 1 kΩ
Z_C2 = 1/(2π × 50 × 470nF) = 6.77 kΩ
Z_Rseries = 100 Ω
Z_ADC_A0 ≈ 1-10 MΩ
Z_ADC_A1 ≈ 1-10 MΩ

// VREF to GND impedance
Z_VREF_to_GND = Z_R2 + Z_C_VREF = 41.8 kΩ

// Thevenin impedance at VREF
Z_thevenin = Z_R1 || Z_VREF_to_GND
Z_thevenin = 10k || 41.8k
Z_thevenin = (10k × 41.8k) / (10k + 41.8k)
Z_thevenin = 418k / 51.8k ≈ 8.07 kΩ

// Path A0 from VREF impedance
Z_to_A0 = Z_Rpull + (Z_Rseries || Z_C2 || Z_ADC_A0)

// Z_C2 >> Rseries
// Z_ADC_A0 >> Rseries
=> Z_to_A0 ≈ Z_Rpull + Z_Rseries = 1000 + 100 = 1100 Ω

// Total impedance seen from SCT
// 1. To 3.3V through R1 : 10 kΩ
// 2. To GND through R2+C_VREF : 41.8 kΩ
// 3. To A0 through Rpull+Rseries : 1.1 kΩ

Z_parallel_VREF = 10k || 41.8k || 1.1k

- 10k || 41.8k = 8.07 kΩ 
=> 8.07k || 1.1k = (8070 × 1100) / (8070 + 1100)
              = 8877000 / 9170 ≈ 968 Ω

Z_total = Z_C1 + Z_parallel_VREF
        = 318 + 968 = 1286 Ω

// Voltage at VREF (AC) :
// Divider Z_C1 / Z_parallel
V_VREF_AC / V_SCT = Z_parallel_VREF / (Z_C1 + Z_parallel_VREF)
V_VREF_AC / V_SCT = 968 / 1286 = 0.753

V_SCT = 417 mV // with 12.5A on SCT-013-030
V_VREF_AC = 417 × 0.753 = 314 mV RMS

// Voltage at A0 (via Rpull+Rseries)
// Divider Rpull/(Rpull+Rseries) from VREF

// Impedance between VREF and Junction B
Z_at_B = Z_Rseries || Z_C2 || Z_ADC_A0 ≈ Z_Rseries = 100 Ω

// Divider :
V_B / V_VREF = Z_at_B / (Z_Rpull + Z_at_B)
V_B / V_VREF = 100 / (1000 + 100) = 0.091

V_B = 314 mV × 0.091 = 28.6 mV

// Impedance from B to A0
Z_ADC_A0 >> Z_Rseries => V_A0 ≈ V_B = 28.6 mV RMS

// Voltage at A1 (VREF)
Z_ADC_A1 >> Z_thevenin => V_A1 ≈ V_VREF_AC = 314 mV RMS

// Differential measure A1 - A0
V_diff = V_A1 - V_A0
V_diff = 314 mV - 28.6 mV = 285.4 mV RMS

// Capaciter C_VREF in serial with R3 => high-pass filter
// 50 Hz << fc => low frequency blocked
Z_C_VREF = 31.8 kΩ // 50 Hz
Z_R1 = 10 kΩ

// Signal attenuated by capacitive divider
V_after_filter / V_before = Z_R2 / (Z_R2 + Z_C_VREF)
V_after_filter / V_before = 10k / (10k + 31.8k)
V_after_filter / V_before = 10k / 41.8k = 0.239

=> Only 24% of the signal is received at 50 Hz

// Signal at VREF with R2+C_VREF filter
V_VREF_AC_crude = 314 mV // before

// High-pass filter
V_VREF_AC_real = 314 mV × 0.239 = 75 mV

// Signal at A1
V_A1 ≈ 75 mV
// Signal at A0 through Rpull/Rseries
V_A0 = 75 mV × 0.091 = 6.8 mV
// Differential measure A1 - A0
V_diff = 75 - 6.8 = 68 mV // vs 62 mV = real measure of 1250W electric kettle
=> Ratio = 1.1
