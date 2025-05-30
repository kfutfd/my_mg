import cupy as cp
import numpy as np
import lattice
import bicgstab
from cupyx.scipy.sparse.linalg import eigsh
from cupyx.scipy.sparse.linalg import LinearOperator

      

class mg:

    blocksize = [4, 2, 2, 2, 2, 2] #每一层的单个方向的压缩程度
    coarse_dof = [8, 12, 12, 12, 12, 12] #新一层的内禀维度
    R_null_vec = [] #
    coarse_info = []
    coarse_map = []

    #生成近零空间向量
    def near_null_vec(self, P_null_vec_coarse, coarse_dof, coarse_op):
        for i in range(0, coarse_dof):
            #施密特正交化
            for k in range(0,i):
                P_null_vec_coarse[i,:,:,:] -= cp.vdot(P_null_vec_coarse[i,:,:,:],P_null_vec_coarse[k,:,:,:])/cp.vdot(P_null_vec_coarse[k,:,:,:],P_null_vec_coarse[k,:,:,:])*P_null_vec_coarse[k,:,:,:]

            #Ar
            Ar = lattice.apply_mat(P_null_vec_coarse[i,:,:,:], coarse_op)
            #-Ar
            Ar = -Ar
            #x = (A^-1)*(-Ar)
            x = bicgstab.bicgstab(Ar, op=coarse_op, tol=5e-5)
            #V = x+r
            P_null_vec_coarse[i,:,:,:] += x
            print(P_null_vec_coarse[i,:,0,0])


            #施密特正交化
            for k in range(0,i):
                P_null_vec_coarse[i,:,:,:] -= cp.vdot(P_null_vec_coarse[i,:,:,:],P_null_vec_coarse[k,:,:,:])/cp.vdot(P_null_vec_coarse[k,:,:,:],P_null_vec_coarse[k,:,:,:])*P_null_vec_coarse[k,:,:,:]

            print("after",P_null_vec_coarse[i,:,0,0])
            print(lattice.apply_mat(P_null_vec_coarse[i,:,:,:], op = coarse_op)[:,0,0])

        return P_null_vec_coarse
    
    def fermi_f2c(self, fermi, fine_op, coarse_op):
        return 0

    def index_to_coord(self, ptr, coarse_op):
        c_coarse = ptr % coarse_op.c_coarse
        xy = ptr // coarse_op.c_coarse
        y_coarse = xy % coarse_op.y_coarse
        x_coarse = xy // coarse_op.y_coarse
        return x_coarse, y_coarse, c_coarse

    def build_mapping(self, fine_op, coarse_op):
        for i in range(0,fine_op.volume):
            x_coarse, y_coarse, c_coarse = self.index_to_coord(i, coarse_op)
            x_fine = 0
            y_fine = 0
            count = 0
            self.recursive_site_build(map, x_coarse, y_coarse, x_fine, y_fine, 0, count)

    def recursive_site_build(self, map, x_coarse, y_coarse, x_fine, y+fine, step, count):




    def __init__(self, fine_size, n_refine, ifeigen=0):
        self.fine_size = fine_size
        self.n_refine = n_refine
        U = self.fine_size.U
        nx = self.fine_size.nx
        ny = self.fine_size.ny
        nc = self.fine_size.nc
        if(ifeigen == 0):
            for i in range(0,n_refine):


                coarse_op = lattice.operator_para(U, nx, ny, nc)
                self.coarse_info.append(coarse_op)######################################

                P_null_vec_coarse = cp.random.rand(self.coarse_dof[i], nx, ny, nc*2, dtype=cp.float64).view(cp.complex128)
                # print(P_null_vec_coarse[7,:,:,:])
                P_null_vec_coarse = self.near_null_vec(P_null_vec_coarse, self.coarse_dof[i], coarse_op)
                self.R_null_vec.append(P_null_vec_coarse)######################################
                print(P_null_vec_coarse.shape)
                print(cp.sum(cp.vdot(P_null_vec_coarse[0,:,:,:], P_null_vec_coarse[1,:,:,:])))
                
                
                # print(P_null_vec_coarse[1,:,:,:])

                nx = int(nx/self.blocksize[i])
                ny = int(ny/self.blocksize[i])
                nc_c = nc
                nc = int(self.coarse_dof[i])

                map_f2c = [[int(0)]*int(self.blocksize[i]*self.blocksize[i]*nc_c)] * int(nx*ny*nc)
                map = np.array(map_f2c)
                self.coarse_map.append(map)######################################



        else:
            a=0

    
        


