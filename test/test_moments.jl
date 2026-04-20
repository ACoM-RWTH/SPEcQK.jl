@testset "test moments" begin
    @testset "test moment indices" begin
        mom_pow_set = all_powers_up_to_M_3D(3)

        @test length(mom_pow_set) == 20

        for i in 0:3
            for j in 0:3
                for k in 0:3
                    if (i+j+k) <= 3
                        @test (i,j,k) in mom_pow_set
                    else
                        @test (i,j,k) ∉ mom_pow_set
                    end
                end
            end
        end

        mom_pow_set = all_powers_up_to_M_3D(5)

        @test length(mom_pow_set) == 56

        for i in 0:5
            for j in 0:5
                for k in 0:5
                    if (i+j+k) <= 5
                        @test (i,j,k) in mom_pow_set
                    else
                        @test (i,j,k) ∉ mom_pow_set
                    end
                end
            end
        end

        for i in 0:5
            for j in 0:5
                for k in 0:5
                    fi = find_index(mom_pow_set, (i,j,k))
                    if (i+j+k) <= 5
                        @test fi != -1
                        @test mom_pow_set[fi] == (i,j,k)
                    else
                        @test fi == -1
                    end
                end
            end
        end

        next_dir_index_x = build_next_moment_index_direction(mom_pow_set, 4, [1,0,0])
        for i in 0:5
            for j in 0:5
                for k in 0:5
                    if (i+j+k) <= 4
                        fi = find_index(mom_pow_set, (i,j,k))
                        @test mom_pow_set[next_dir_index_x[fi]] == (i+1,j,k)
                    end
                end
            end
        end
    end
end