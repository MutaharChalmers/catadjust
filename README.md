# CatAdjust - Tools for adjusting catastrophe models
Current tools:
- Hazard ELT Adjustment Tool - Given a location-level ELT with hazard values, and a collection of *target* hazard EEF curves at all locations, this tool adjusts the rates of all events in the ELT, such that the hazard EEF curves from the rate-adjusted ELT match the target hazard EEF curves as closely as possible.

Future tools:
- Loss Adjustment Tool - Given a stochastic ELT or YLT from a catastrophe model, and some historic loss experience, this tool adjusts the losses in the ELT or YLT such that the adjusted OEP curve matches the empirical historic loss OEP curve over some user-defined range.

- Industry Loss Conversion Tool - Given two stochastic industry ELTs or YLTs with losses at sub-national (e.g. CRESTA) level, scale the losses from one ELT/YLT such that the EP curves match the EP curves derived from the other ELT/YLT both at aggregate and sub-national level.
 

