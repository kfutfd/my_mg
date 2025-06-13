import lattice
import mg
import cupy as cp
import bicgstab
nx = 4
ny = nx
nc = 2

#生成随即组态
U = lattice.load_or_generate_U(nx, ny, nc, "./U_data/")
print("生成的矩阵形状：", U.shape)

#确定格点
fine_op = lattice.operator_para(U, nx, ny, nc)


V = cp.random.rand(nx,ny,nc*2, dtype=cp.float64).view(cp.complex128)

my_mg = mg.mg(fine_op, 1)


Vout = lattice.apply_mat(V,fine_op)
V0 = bicgstab.bicgstab(Vout, op=fine_op)
print(V[0,:,0] - V0[0,:,0])