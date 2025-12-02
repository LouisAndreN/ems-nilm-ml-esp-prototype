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
Z_VREF_to_GND = sqrt(Z_R2^2 + Z_C_VREF^2) = 33.3 kΩ

// Thevenin impedance at VREF
Z_th = Z_R1 || Z_VREF_to_GND
Z_th = 10k || 33.3k
Z_th = (10k × 33.3k) / (10k + 33.3k)
Z_th = 333k / 43.3k ≈ 7.7 kΩ

// Path A0 from VREF impedance
Z_to_A0 = Z_Rpull + (Z_Rseries || Z_C2 || Z_ADC_A0)

// Z_C2 >> Z_Rseries
// Z_ADC_A0 >> Z_Rseries
=> Z_to_A0 ≈ Z_Rpull + Z_Rseries = 1000 + 100 = 1100 Ω

// Total impedance seen from SCT
// 1. To 3.3V through R1 : 10 kΩ
// 2. To GND through R2+C_VREF : 33.3 kΩ
// 3. To A0 through Rpull+Rseries : 1.1 kΩ

Z_parallel_VREF = 10k || 33.3k || 1.1k // = Z_th || Z_to_A0
                = 7.7k || 1.1k 
                = (7690 × 1100) / (7690 + 1100)
                = 8459000 / 8790 
                ≈ 962 Ω

Z_total = Z_C1 + Z_parallel_VREF
        = 318 + 962 = 1280 Ω

// Voltage at VREF (AC) :
// Divider Z_C1 / Z_parallel
V_VREF_AC / V_SCT = Z_parallel_VREF / (Z_C1 + Z_parallel_VREF)
V_VREF_AC / V_SCT = 962 / 1280 = 0.752

V_SCT = 417 mV // electric kettle 1250W with AC 100V => 12.5A on SCT-013-030 (12.5/30 = 0.417)
V_VREF_AC = 417 × 0.752 = 314 mV RMS

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
Z_ADC_A1 >> Z_th => V_A1 ≈ V_VREF_AC = 314 mV RMS

// Differential measure A1 - A0
V_diff = V_A1 - V_A0
V_diff = 314 mV - 28.6 mV = 285.4 mV RMS

// Capaciter C_VREF in serial with R2 => high-pass filter
// 50 Hz < fc = 159Hz => low frequency blocked
Z_C_VREF = 31.8 kΩ // 50 Hz
Z_R2 = 10 kΩ

// Signal attenuated by capacitive divider
V_after_filter / V_before = Z_R2 / sqrt(Z_R2^2 + Z_C_VREF^2)
V_after_filter / V_before = 10k / sqrt(10k^2 + 31.8k^2)
V_after_filter / V_before ≈ 10k / 33.3k = 0.3

=> Only 30% of the signal is received at 50 Hz

// Signal at VREF with R2+C_VREF filter
V_VREF_AC_crude = 314 mV // before

// High-pass filter
V_VREF_AC_real = 314 mV × 0.3 = 94 mV

// Signal at A1
V_A1 ≈ 94 mV
// Signal at A0 through divider (Rseries / (Rpull + Rseries))
V_A0 = 94 mV × 0.091 = 8.6 mV
  
// Differential measure A1 - A0
V_diff = 94 - 8.6 = 85 mV // for 1250W electric kettle load
