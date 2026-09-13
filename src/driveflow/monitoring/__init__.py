"""Monitoring layer for the cross-domain AI layer (docs/design_ai_layer_transversal.md). Rule
schemas/YAML live in monitoring.rules; the rule-evaluating agents (Sec. 4.3, Sec. 8 step 9) live
in monitoring.agents -- GatewayAgent (Raspberry Pi 5 tier) and ServerAgent (PC tier). No ESP32
agent module: that tier's watchdog is pure hard thresholds with no ML/state (Sec. 4.3), already
covered by the rule YAML + schema alone (Sec. 8 step 1/2) -- there is no separate agent script for
it to build.
"""
