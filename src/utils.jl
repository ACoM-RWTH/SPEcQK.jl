@muladd begin
"""
    unroll(A)

Unroll a matrix or tensor `A` into a vector.

# Returns
* `vector`: flattened version of `A`
"""
function unroll(A)
    # unroll matrix or tensor into vector
    return vcat(A...)
end

"""
    unroll!(A_flat, A)

Unroll a matrix or tensor `A` into a pre-allocated vector `A_flat`.
"""
function unroll!(A_flat, A)
    for i in eachindex(A)
        A_flat[i] = A[i]
    end
end

"""
    rollup(A_flat, n_vx, n_vy)

Roll a flat vector `A_flat` into an `n_vx` by `n_vy` matrix.

# Returns
* `matrix`: reshaped version of `A_flat`
"""
function rollup(A_flat, n_vx, n_vy)
    # roll flat vector into n_vx * n_vy matrix
    return reshape(A_flat, (n_vx, n_vy))
end

"""
    rollup(A_flat, n_vx, n_vy, n_vz)

Roll a flat vector `A_flat` into an `n_vx` by `n_vy` by `n_vz` tensor.

# Returns
* `tensor`: reshaped version of `A_flat`
"""
function rollup(A_flat, n_vx, n_vy, n_vz)
    # roll flat vector into n_vx * n_vy matrix
    return reshape(A_flat, (n_vx, n_vy, n_vz))
end

function convert_to_central_3D!(moments_central, moments, moment_powers, n_mom)
    
    #
    # first moment is density!
    moments_central[1] = moments[1]

    vx = 0.0
    vy = 0.0
    vz = 0.0


    for nm in 2:n_mom
        if moment_powers[nm] == [1,0,0]
            vx = moments[nm]
        end
        if moment_powers[nm] == [0,1,0]
            vy = moments[nm]
        end
        if moment_powers[nm] == [0,0,1]
            vz = moments[nm]
        end
    end

    # println("$vx $vy $vz")

    for nm in 2:n_mom
        mp = moment_powers[nm]

        mc = 0.0

        for i in 0:mp[1]
            for j in 0:mp[2]
                for k in 0:mp[3]
                    tmp = (-1.0)^(i+j+k) * binomial(mp[1], i) * binomial(mp[2], j) * binomial(mp[3], k)

                    for nm2 in 1:n_mom
                        if moment_powers[nm2] == [mp[1] - i, mp[2] - j, mp[3] - k]
                            mc += tmp * moments[nm2] * vx^i * vy^j * vz^k
                            break
                        end
                    end

                    
                    # moments_central
                end
            end
        end
        moments_central[nm] = mc
    end
end

"""
    find_index(moment_powers, index_to_search)

Find the index of a moment power in the `moment_powers` array.

# Returns
* `index`: index of the moment power in `moment_powers`
"""
function find_index(moment_powers, index_to_search)
    for (i,mom) in enumerate(moment_powers)
        if (mom[1] == index_to_search[1]) && (mom[2] == index_to_search[2]) && (mom[3] == index_to_search[3])
            return i
        end
    end
end

"""
    build_next_moment_index_direction!(index, moment_powers, n_moment_constraints, direction_binary_vec)

Construct index for the next moment in a given direction, i.e. if `moment_powers[i] = (a,b,c)`,
then `index[i] = find_index(moment_powers, (a+direction_binary_vec[1], b+direction_binary_vec[2], c+direction_binary_vec[3]))`.
So one can quickly access the required next-order moment.
"""
function build_next_moment_index_direction!(index, moment_powers, n_moment_constraints, direction_binary_vec)
    for i in 1:n_moment_constraints
        for j in eachindex(moment_powers)
            a, b, c = moment_powers[j]
            index[i] = find_index(moment_powers, (a+direction_binary_vec[1], b+direction_binary_vec[2], c+direction_binary_vec[3]))
        end
    end
end
end