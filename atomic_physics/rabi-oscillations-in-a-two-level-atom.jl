# Libraries
using LinearAlgebra
using Plots, ColorSchemes, LaTeXStrings

# Plot parameters
default(fontfamily = "Computer modern", 
            legend = true, 
          tickfont = font(12), 
         guidefont = font(16), 
         titlefont = font(18),
         linewidth = 2, 
            grid = true,
            size = (800, 400),
          margin = 10Plots.mm)

# Physical parameters
Ω = 1  # Rabi frequency 
Δ = 1  # Detuning

# Time parameters
t₀ = 0                         # initial time (s)
t₁ = 10                        # final time   (s)
Δt = 0.01                      # step size    (s)
n = Int64((t₁ - t₀) / Δt)      # iterations
t = t₀:Δt:(t₁ - Δt)            # time vector  (s)

# Fourth-Order Runge-Kutta function
function RK4(f, x₀, y₀, z₀, h)
    k1y = h .* f(x₀, y₀, z₀)[1]
    k1z = h .* f(x₀, y₀, z₀)[2]
    #
    k2y = h .* f(x₀ + h / 2, y₀ + k1y / 2, z₀ + k1z / 2)[1]
    k2z = h .* f(x₀ + h / 2, y₀ + k1y / 2, z₀ + k1z / 2)[2]
    #
    k3y = h .* f(x₀ + h / 2, y₀ + k2y / 2, z₀ + k2z / 2)[1]
    k3z = h .* f(x₀ + h / 2, y₀ + k2y / 2, z₀ + k2z / 2)[2]
    #
    k4y = h .* f(x₀ + h, y₀ + k3y, z₀ + k3z)[1]
    k4z = h .* f(x₀ + h, y₀ + k3y, z₀ + k3z)[2]

    # Approximation
    y1 = ComplexF64(y₀ + (k1y + 2 * k2y + 2 * k3y + k4y) / 6)
    z1 = ComplexF64(z₀ + (k1z + 2 * k2z + 2 * k3z + k4z) / 6)
    return y1, z1
end

# Differential equations
function f(t, c1, c2)
    return ComplexF64(-1im * Ω * exp(1im * Δ * t) * c2), ComplexF64(-1im * Ω * exp(-1im * Δ * t) * c1)
end

# Initial arrays
c₁, c₁[1] = zeros(ComplexF64, n), ComplexF64(1 + 0im)
c₂, c₂[1] = zeros(ComplexF64, n), ComplexF64(0 + 0im)

# RK4 method evaluation
for i in 1:(n - 1)
    c₁[i + 1], c₂[i + 1] = RK4(f, t[i], c₁[i], c₂[i], Δt)
end

# Plot
plot(t, abs2.(c₁), label = L"$|\tilde{c}_{1}(t)|^2$", c=:blue, xlabel = L"$t (s)$", ylabel = L"\textrm{Population}")
plot!(t, abs2.(c₂), label = L"$|\tilde{c}_{2}(t)|^2$", c=:green)
plot!(t, abs2.(c₁) + abs2.(c₂), label = L"|\tilde{c}_{1}(t)|^2 + |\tilde{c}_{2}(t)|^2", c=:red)
title!("Rabi oscillations in an optically-driven two-level atom")
savefig("juliaplot2.pdf")