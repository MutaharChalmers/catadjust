# CatAdjust - Tools for adjusting catastrophe models
Current tools:
- ELT Rate Adjustment Tool - Given an aggregated or location-level ELT with optional location-level hazard, and target loss or hazard EEF curve(s), this tool adjusts the rates of all events in the ELT such that the loss or hazard EEF curves from the rate-adjusted ELT match the target loss or hazard EEF curve(s) as closely as possible.
- ELT Loss Adjustment Tool - Given an aggregated or location-level ELT, and target loss EEF curve(s), this tool adjusts the losses of all events in the ELT such that the loss EEF curves from the loss-adjusted ELT match the target loss EEF curve(s) as closely as possible. Note that this only adjusts mean losses, so doesn't take account of secondary uncertainty.

Note that although the adjustments are made to an ELT, any N-year YELT can be interpreted as an ELT by attributing to each event a rate of 1/N, assuming each event occurs exactly once.
