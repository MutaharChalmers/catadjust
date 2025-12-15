# CatAdjust - Tools for adjusting catastrophe models
Current tools:
- Rate Adjustment Tool - Given an aggregated or location-level Event Loss Table (ELT) or Event Hazard Table (EHT), and target loss or hazard EEF curve(s), this tool adjusts the rates of all events in the ELT/EHT such that the loss or hazard EEF curves from the rate-adjusted ELT/EHT match the target loss or hazard EEF curve(s) as closely as possible.
- Loss Adjustment Tool - Given an aggregated or location-level ELT, and target loss EEF curve(s), this tool adjusts the losses of all events in the ELT such that the loss EEF curves from the loss-adjusted ELT match the target loss EEF curve(s) as closely as possible. Note that this only adjusts mean losses, so does not take account of secondary uncertainty.

Note that although the adjustments are made to an ELT or EHT, any N-year YELT/YEHT can be interpreted as an ELT/EHT by attributing to each event a rate of 1/N, assuming each event occurs exactly once.
