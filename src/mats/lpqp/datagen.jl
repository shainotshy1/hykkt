using Test
using Random
using SparseArrays
using LinearAlgebra
using MatrixMarket


function writeVector(filename::String, vector)
    stream = open(filename, "w")
    write(stream, "%%MatrixMarket matrix\n")
    s = size(vector, 1)
    write(stream, "$(s) 1\n")
    for i in 1:s
        write(stream, "$(vector[i])\n")
    end
    close(stream)
end


m = 50 # num constraints
n= 7 #num variables
rng = MersenneTwister(1235) # this is the seed
# These lines are from Boyd's example on how to generate an LP with a known solution 
# I'm using absolute values here to make sure all zeros is feasible (this is also realistic in bid optimization)
z = randn(rng,m)
zervec = zeros(m)
sst = max.(z, zervec)
y = sst-z
A = abs.(randn(rng,m,n))
foreach(normalize!,eachrow(A))
xst = abs.(randn(rng,n))
b = A*xst+sst
c = -A'*y
# this generates the quadratic part of the problem
Q1 = sprand(rng, n, n, 1.0/n)
Q = Q1'*Q1 + I
A = sparse(A)
vals = 1.0 ./ b .^ 2

H = Diagonal(vals)
v_in = 1.0 * ones(n)

t = 1.7
expected = t * Q * v_in + (A' * (H * (A * v_in)))

H = sparse(vals)
v_in = sparse(v_in)
expected = sparse(expected)

curr = pwd() * "/src/mats/lpqp"
q_filename = curr * "/q3.mtx"
a_filename = curr * "/a3.mtx"
vin_filename = curr * "/vin3.mtx"
h_filename = curr * "/h3.mtx"
expected_filename = curr * "/exp3.mtx"

# Save the sparse matrix as an .mtx file
MatrixMarket.mmwrite(q_filename, Q);
MatrixMarket.mmwrite(a_filename, A);
writeVector(h_filename, H)
writeVector(vin_filename, v_in)
writeVector(expected_filename, expected)