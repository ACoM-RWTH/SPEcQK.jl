"""
BGK (Bhatnagar-Gross-Krook) collision operator utilities.
"""

"""
    tau_BGK(n, T, omega)

Compute the BGK collision time scale.

# Positional arguments:
* `n`: density
* `T`: temperature
* `omega`: collision frequency parameter

# Returns:
* `tau`: collision time scale (currently returns 1.0 as a dummy value)
"""
function tau_BGK(n, T, omega)
    return 1.0
end
