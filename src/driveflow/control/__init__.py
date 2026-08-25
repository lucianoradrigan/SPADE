"""Controllers acting on a simulated plant. Per docs/propuesta_consolidacion.pdf Sec. 4.1
(Principle #3), the controller is swappable -- classical PI/PID (this Macro-fase A) and MPC are
GEM-derived baselines; DPC (Macro-fase B) is a controller of the same kind, not a special case.
"""
