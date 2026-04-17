@muladd begin
function unroll(A)
    # unroll matrix or tensor into vector
    return vcat(A...)
end

function unroll!(A_flat, A)
    for i in eachindex(A)
        A_flat[i] = A[i]
    end
end

function rollup(A_flat, n_vx, n_vy)
    # roll flat vector into n_vx * n_vy matrix
    return reshape(A_flat, (n_vx, n_vy))
end

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
end

function find_index(moment_powers, index_to_search)
    for (i,mom) in enumerate(moment_powers)
        if (mom[1] == index_to_search[1]) && (mom[2] == index_to_search[2]) && (mom[3] == index_to_search[3])
            return i
        end
    end
end