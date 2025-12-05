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
Z_Rburden = 60 Ω // SCT-013 30A/1V => burden resistor typically ~60Ω (Current ratio 1:1800)

////////////////////////////////////////////////////////////////////////
//////////////////// Voltage divider bridge from VREF //////////////////
////////////////////////////////////////////////////////////////////////
// VREF to GND impedance
Z_VREF_to_GND = sqrt(Z_R2^2 + Z_C_VREF^2) = 33.3 kΩ

// Thevenin impedance at VREF
Z_th_VREF = Z_R1 || Z_VREF_to_GND
Z_th_VREF = 10k || 33.3k
Z_th_VREF = (10k × 33.3k) / (10k + 33.3k)
Z_th_VREF = 333k / 43.3k ≈ 7.7 kΩ
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

  
////////////////////////////////////////////////////////////////////////
////////////////// Impedance at junction B, path to A0 /////////////////
//////////////////////////////////////////////////////////////////////// 
// Path A0 from VREF impedance
Z_to_A0 = Z_Rpull + (Z_Rseries || Z_C2 || Z_ADC_A0)

// Z_C2 >> Z_Rseries
// Z_ADC_A0 >> Z_Rseries
=> Z_to_A0 ≈ Z_Rpull + Z_Rseries = 1000 + 100 = 1100 Ω
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
///////////////////// Thevenin equivalent from SCT /////////////////////
////////////////////////////////////////////////////////////////////////  
// Total impedance seen from SCT
// 1. To 3.3V through R1 : 10 kΩ
// 2. To GND through R2+C_VREF : 33.3 kΩ
// 3. To A0 through Rpull+Rseries : 1.1 kΩ

Z_parallel_VREF = 10k || 33.3k || 1.1k // = Z_th || Z_to_A0
                = 7.7k || 1.1k 
                = (7690 × 1100) / (7690 + 1100)
                = 8459000 / 8790 
                ≈ 962 Ω

Z_total = Z_Rburden + Z_C1 + Z_parallel_VREF
        = 60 + 318 + 962 = 1340 Ω
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
///////////// Capacitive divider - signal before filtering /////////////
////////////////////////////////////////////////////////////////////////
// Voltage at VREF (AC) :
// Divider Z_C1 / Z_parallel
VREF_before_filter / V_SCT = Z_parallel_VREF / Z_total
VREF_before_filter / V_SCT = 962 / 1340 = 0.718

V_SCT = 417 mV // electric kettle 1250W with AC 100V => 12.5A on SCT-013-030 (12.5/30 = 0.417V)

// Voltage drop in R_SCT_internal
V_drop_burden = V_SCT × (Z_Rburden / Z_total)
V_drop_burden = 417 × (60 / 1340) = 18.7 mV

// Available voltage after burden
V_after_burden = V_SCT - V_drop_burden
V_after_burden = 417 - 18.7 = 398.3 mV
  
VREF_before_filter = 398.3 × 0.718 = 286.0 mV RMS
// Part of the signal is "eaten" by the capacitor C1
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
  

////////////////////////////////////////////////////////////////////////
//////// Capacitor C_VREF in serial with R2 => high-pass filter ////////
////////////////////////////////////////////////////////////////////////  
// 50 Hz < fc = 159Hz => low frequency blocked
Z_C_VREF = 31.8 kΩ // 50 Hz
Z_R2 = 10 kΩ

// Signal attenuated by capacitive divider
VREF_after_filter / VREF_before_filter = Z_R2 / sqrt(Z_R2^2 + Z_C_VREF^2)
VREF_after_filter / VREF_before_filter = 10k / sqrt(10k^2 + 31.8k^2)
VREF_after_filter / VREF_before_filter ≈ 10k / 33.3k = 0.3

// => Only 30% of the signal is received at 50 Hz

// Apply high-pass filter to signal
VREF = VREF_before_filter × 0.30
VREF = 286.0 mV × 0.30
VREF = 85.8 mV RMS = V_A1 // Z_ADC_A1 >> Z_th_VREF
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
///////////////////////// Resistive divider to A0 //////////////////////
////////////////////////////////////////////////////////////////////////  
// Voltage at A0 (via Rpull+Rseries)
// Divider Rpull/(Rpull+Rseries) from VREF

// Impedance between VREF and Junction B
Z_B = Z_Rseries || Z_C2 || Z_ADC_A0 ≈ Z_Rseries = 100 Ω

// Divider :
V_B / VREF = Z_B / (Z_Rpull + Z_B)
V_B / VREF = 100 / (1000 + 100) = 0.091
=> V_B = 85.8 mV × 0.091 = 7.8 mV RMS = V_A0 // Z_ADC_A0 >> Z_Rseries


///////////////////////// Differential measure A1 - A0 ////////////////////// 
V_diff = V_A1 - V_A0
V_diff = 85.8 - 7.8 = 78.0 mV RMS // for 1250W electric kettle load
// vs 62 mV RMS measured => Ratio = 78.0 / 62 = 1.26 
  
// Theorical system sensibility :
// S = 78.0 mV / 1250 W = 62.4 µV/W
