import cupy as cp
import numpy as np
import lattice
import bicgstab
from cupyx.scipy.sparse.linalg import eigsh
from cupyx.scipy.sparse.linalg import LinearOperator

      

class mg:

    blocksize = [2, 2, 2, 2, 2, 2] #每一层的单个方向的压缩程度
    coarse_dof = [4, 12, 12, 12, 12, 12] #新一层的内禀维度
    R_null_vec = [] #
    mg_ops = []
    coarse_map = []

    #生成近零空间向量
    def near_null_vec(self, P_null_vec_coarse, coarse_dof, coarse_op):
        for i in range(0, coarse_dof):
            #施密特正交化
            for k in range(0,i):
                P_null_vec_coarse[i,:,:,:] -= cp.vdot(cp.conj(P_null_vec_coarse[i,:,:,:]),P_null_vec_coarse[k,:,:,:])/cp.vdot(cp.conj(P_null_vec_coarse[k,:,:,:]),P_null_vec_coarse[k,:,:,:])*P_null_vec_coarse[k,:,:,:]

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
                P_null_vec_coarse[i,:,:,:] -= cp.vdot(cp.conj(P_null_vec_coarse[i,:,:,:]),P_null_vec_coarse[k,:,:,:])/cp.vdot(cp.conj(P_null_vec_coarse[k,:,:,:]),P_null_vec_coarse[k,:,:,:])*P_null_vec_coarse[k,:,:,:]

            # P_null_vec_coarse[i,:] = cp.zeros_like(P_null_vec_coarse[i,:])
            P_null_vec_coarse[i,:] = P_null_vec_coarse[i,:]/cp.sqrt(cp.vdot(cp.conj(P_null_vec_coarse[i,:]), P_null_vec_coarse[i,:]))
            if i==0:
                print("after",P_null_vec_coarse[i,:])
            # print(lattice.apply_mat(P_null_vec_coarse[i,:,:,:], op = coarse_op)[:,0,0])

        return P_null_vec_coarse
    
    def vol_index_dof_to_cv_index(self, i, i_dof, coarse_op):
        return coarse_op.nc*i + i_dof

    def index_to_coord(self, ptr, coarse_op):
        y_coarse = ptr % coarse_op.ny
        x_coarse = ptr // coarse_op.ny
        return x_coarse, y_coarse
    
    #网格物理地址与内存地址转化
    def coord_to_index(self, coords, coarse_op, i):
        ptr = (coarse_op.ny*coords[0] + coords[1])*coarse_op.nc + i
        return ptr


    '''
    建立粗网格与细网格之间的对应关系
    '''
    def build_mapping(self, map_id, fine_op, coarse_op):
        for i in range(0,coarse_op.volume):
            x_coarse, y_coarse = self.index_to_coord(i, coarse_op)
            coarse_coords = [x_coarse, y_coarse]
            # print("coarse_coords = ",coarse_coords)
            coords = [0, 0]
            blocksizes = int(fine_op.nx/coarse_op.nx)
            count = [0]
            self.recursive_site_build(map_id, coarse_coords, coords, 0, count, blocksizes, fine_op, i)

    '''
    build_mapping用到的递归函数
    '''
    def recursive_site_build(self, map_id, coarse_coords, coords, step, count, blocksizes, fine_op, fine_ptr):
        if(step < 2):
            for i in range(coarse_coords[step]*blocksizes, (coarse_coords[step]+1)*blocksizes):
                coords[step] = i
                self.recursive_site_build(map_id, coarse_coords, coords, step+1, count, blocksizes, fine_op, fine_ptr)
        else:
            for i in range(0, fine_op.nc):
                self.coarse_map[map_id][fine_ptr][count[0]] = self.coord_to_index(coords, fine_op, i)
                count[0] = count[0] + 1

    def restrict_f2c(self, fine_leve, fermi_in, fermi_out, nevc, fine_sites_per_coarse):
        fermi_in = fermi_in.reshape(-1)
        fermi_out = fermi_out.reshape(-1)
        for i in range(0,self.mg_ops[fine_leve+1].volume):
            for i_dof in range(0,nevc):
                cv_index = self.vol_index_dof_to_cv_index(i, i_dof, self.mg_ops[fine_leve+1])
                for j in range(0, fine_sites_per_coarse):
                    fermi_out[cv_index] += np.conj(self.R_null_vec[fine_leve][i_dof][self.coarse_map[fine_leve][i][j]])*fermi_in[self.coarse_map[fine_leve][i][j]]



    def prolong_c2f(self, fine_leve, fermi_in, fermi_out, nevc, fine_sites_per_coarse):
        fermi_in = fermi_in.reshape(-1)
        fermi_out = fermi_out.reshape(-1)
        for i in range(0,self.mg_ops[fine_leve+1].volume):
            for i_dof in range(0,nevc):
                cv_index = self.vol_index_dof_to_cv_index(i, i_dof, self.mg_ops[fine_leve+1])
                for j in range(0, fine_sites_per_coarse):
                    fermi_out[self.coarse_map[fine_leve][i][j]] += self.R_null_vec[fine_leve][i_dof][self.coarse_map[fine_leve][i][j]]*fermi_in[cv_index]


    def local_orthogonalization(self, fine_leve, nevc, fine_sites_per_coarse):
        for i in range(0,self.mg_ops[fine_leve+1].volume):
            for i_dof in range(0,nevc):
                cv_index = self.vol_index_dof_to_cv_index(i, i_dof, self.mg_ops[fine_leve+1])
                for k in range(0,i_dof):

                    k_dot = 0
                    k_i_dof_dot = 0
                    for j in range(0, fine_sites_per_coarse):
                        k_dot += np.conj(self.R_null_vec[fine_leve][k][self.coarse_map[fine_leve][i][j]])*self.R_null_vec[fine_leve][k][self.coarse_map[fine_leve][i][j]]
                        k_i_dof_dot += np.conj(self.R_null_vec[fine_leve][k][self.coarse_map[fine_leve][i][j]])*self.R_null_vec[fine_leve][i_dof][self.coarse_map[fine_leve][i][j]]
                    for j in range(0, fine_sites_per_coarse):
                        self.R_null_vec[fine_leve][i_dof][self.coarse_map[fine_leve][i][j]] -= self.R_null_vec[fine_leve][k][self.coarse_map[fine_leve][i][j]] * k_i_dof_dot / k_dot 
                    
                i_dof_dot = 0
                for j in range(0, fine_sites_per_coarse):
                    i_dof_dot += np.conj(self.R_null_vec[fine_leve][i_dof][self.coarse_map[fine_leve][i][j]])*self.R_null_vec[fine_leve][i_dof][self.coarse_map[fine_leve][i][j]]

                for j in range(0, fine_sites_per_coarse):
                    self.R_null_vec[fine_leve][i_dof][self.coarse_map[fine_leve][i][j]] /= cp.sqrt(i_dof_dot)
                a=0




    def fermi_f2c(self, fermi, fine_op, coarse_op):
        return 0
    

    def __init__(self, fine_op, n_refine, ifeigen=0):
        self.n_refine = n_refine
        self.fine_op = fine_op
        U = self.fine_op.U
        nx = self.fine_op.nx
        ny = self.fine_op.ny
        nc = self.fine_op.nc
        if(ifeigen == 0):
            self.mg_ops.append(fine_op)######################################
            for i in range(0,n_refine):

                P_null_vec_coarse = cp.random.rand(self.coarse_dof[i], nx, ny, nc*2, dtype=cp.float64).view(cp.complex128)
                # print(P_null_vec_coarse[7,:,:,:])
                P_null_vec_coarse = self.near_null_vec(P_null_vec_coarse, self.coarse_dof[i], fine_op)
                P_null_vec_coarse = P_null_vec_coarse.reshape(P_null_vec_coarse.shape[0],-1)
                self.R_null_vec.append(P_null_vec_coarse)######################################

                
                rand_fermi =  cp.random.rand(nx, ny, nc*2, dtype=cp.float64).view(cp.complex128)
                rand_fermi =  cp.zeros_like(rand_fermi)
                rand_fermi =  cp.ones_like(rand_fermi)
                # rand_fermi[0,0,0] = 1000
                # print(P_null_vec_coarse[1,:,:,:])

                nx = int(nx/self.blocksize[i])
                ny = int(ny/self.blocksize[i])
                nc_c = nc
                nc = int(self.coarse_dof[i])

                map = [[int(0)]*int(self.blocksize[i]*self.blocksize[i]*nc_c)] * int(nx*ny)
                fine_sites_per_coarse = int(self.blocksize[i]*self.blocksize[i]*nc_c)

                map = np.array(map)
                self.coarse_map.append(map)######################################         



                coarse_op = lattice.operator_para(nx, ny, nc)
                self.mg_ops.append(coarse_op)
                self.build_mapping( i, fine_op, coarse_op)
                print(self.coarse_map[i].shape)
                print(self.coarse_map[i])
                print(len(np.unique(self.coarse_map[i])) < len(self.coarse_map[i]))

                self.local_orthogonalization(i, self.coarse_dof[i], fine_sites_per_coarse)
                
                
                fermi_out = cp.random.rand(nx, ny, nc*2, dtype=cp.float64).view(cp.complex128).reshape((nx,ny,nc))
                fermi_out = cp.zeros_like(fermi_out)
                # fermi_out[0,0,0] = 1
                # fermi_out[0,1,0] = 1
                # fermi_out[1,0,0] = 1
                # fermi_out[1,1,0] = 1
                fermi_out = fermi_out
                fermi_out_pr = cp.zeros_like(fermi_out)
                fermi_out_r = cp.zeros_like(rand_fermi)
                # self.restrict_f2c( i, rand_fermi, fermi_out, self.coarse_dof[i], fine_sites_per_coarse)
                # self.prolong_c2f( i, fermi_out, fermi_out_r, self.coarse_dof[i], fine_sites_per_coarse)
                # self.restrict_f2c( i, fermi_out_r, fermi_out_pr, self.coarse_dof[i], fine_sites_per_coarse)
                # print(rand_fermi)
                # print(fermi_out)
                # print(fermi_out_r)
                # print(fermi_out_pr)

                ################################# transfer ######################################
                self.mg_ops[i+1].clover = cp.zeros_like(self.mg_ops[i+1].clover)
                # clover
                for color in range(0,self.mg_ops[i+1].nc):
                    
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[:,:,color] = 1

                    # print("fermi_tmp_coarse = ",fermi_tmp_coarse)

                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine, self.coarse_dof[i], fine_sites_per_coarse)
                    fermi_tmp_Afine = lattice.apply_clover(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse, self.coarse_dof[i], fine_sites_per_coarse)

                    self.mg_ops[i+1].clover[:,:,:,color] = fermi_tmp_coarse[:,:,:]

                    # print("fermi_tmp_coarse = ",fermi_tmp_coarse)
                    # print("clover = ", self.mg_ops[i+1].clover)

                # wilson
                # self.mg_ops[i+1].clover = cp.zeros_like(self.mg_ops[i+1].clover)
                self.mg_ops[i+1].hopping = cp.zeros_like(self.mg_ops[i+1].hopping)
                for color in range(0,self.mg_ops[i+1].nc):
                    # xp=even
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[0:-1:2,:,color] = 1
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine, self.coarse_dof[i], fine_sites_per_coarse)
                    fermi_tmp_Afine = lattice.apply_hopping_x_p(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse, self.coarse_dof[i], fine_sites_per_coarse)
                    self.mg_ops[i+1].hopping[0,1::2,:,:,color] = fermi_tmp_coarse[1::2,:,:]
                    self.mg_ops[i+1].clover[0:-1:2,:,:,color] += fermi_tmp_coarse[0:-1:2,:,:]
                    print("fermi_tmp_coarse = ", fermi_tmp_coarse)

                    # xp=odd
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[1::2,:,color] = 1
                    print("fermi_tmp_coarse", fermi_tmp_coarse)
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine, self.coarse_dof[i], fine_sites_per_coarse)
                    fermi_tmp_Afine = lattice.apply_hopping_x_p(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse, self.coarse_dof[i], fine_sites_per_coarse)
                    self.mg_ops[i+1].hopping[0,0:-1:2,:,:,color] = fermi_tmp_coarse[0:-1:2,:,:]
                    self.mg_ops[i+1].clover[1::2,:,:,color] += fermi_tmp_coarse[1::2,:,:]

                    # xm=even
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[0:-1:2,:,color] = 1
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine, self.coarse_dof[i], fine_sites_per_coarse)
                    fermi_tmp_Afine = lattice.apply_hopping_x_m(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse, self.coarse_dof[i], fine_sites_per_coarse)
                    self.mg_ops[i+1].hopping[1,1::2,:,:,color] = fermi_tmp_coarse[1::2,:,:]
                    self.mg_ops[i+1].clover[0:-1:2,:,:,color] += fermi_tmp_coarse[0:-1:2,:,:]

                    # xm=odd
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[1::2,:,color] = 1
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine, self.coarse_dof[i], fine_sites_per_coarse)
                    fermi_tmp_Afine = lattice.apply_hopping_x_m(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse, self.coarse_dof[i], fine_sites_per_coarse)
                    self.mg_ops[i+1].hopping[1,0:-1:2,:,:,color] = fermi_tmp_coarse[0:-1:2,:,:]
                    self.mg_ops[i+1].clover[1::2,:,:,color] += fermi_tmp_coarse[1::2,:,:]

                    # yp=even
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[:,0:-1:2,color] = 1
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine, self.coarse_dof[i], fine_sites_per_coarse)
                    fermi_tmp_Afine = lattice.apply_hopping_y_p(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse, self.coarse_dof[i], fine_sites_per_coarse)
                    self.mg_ops[i+1].hopping[2,:,1::2,:,color] = fermi_tmp_coarse[:,1::2,:]
                    self.mg_ops[i+1].clover[:,0:-1:2,:,color] += fermi_tmp_coarse[:,0:-1:2,:]

                    # yp=odd
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[:,1::2,color] = 1
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine, self.coarse_dof[i], fine_sites_per_coarse)
                    fermi_tmp_Afine = lattice.apply_hopping_y_p(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse, self.coarse_dof[i], fine_sites_per_coarse)
                    self.mg_ops[i+1].hopping[2,:,0:-1:2,:,color] = fermi_tmp_coarse[:,0:-1:2,:]
                    self.mg_ops[i+1].clover[:,1::2,:,color] += fermi_tmp_coarse[:,1::2,:]

                    # ym=even
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[:,0:-1:2,color] = 1
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine, self.coarse_dof[i], fine_sites_per_coarse)
                    fermi_tmp_Afine = lattice.apply_hopping_y_m(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse, self.coarse_dof[i], fine_sites_per_coarse)
                    self.mg_ops[i+1].hopping[3,:,1::2,:,color] = fermi_tmp_coarse[:,1::2,:]
                    self.mg_ops[i+1].clover[:,0:-1:2,:,color] += fermi_tmp_coarse[:,0:-1:2,:]

                    # ym=odd
                    fermi_tmp_coarse = cp.zeros_like(fermi_out)
                    fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                    fermi_tmp_Afine = cp.zeros_like(fermi_out_r)

                    fermi_tmp_coarse[:,1::2,color] = 1
                    self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine, self.coarse_dof[i], fine_sites_per_coarse)
                    fermi_tmp_Afine = lattice.apply_hopping_y_m(fermi_tmp_fine, self.mg_ops[i])
                    fermi_tmp_coarse = cp.zeros_like(fermi_tmp_coarse)
                    self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse, self.coarse_dof[i], fine_sites_per_coarse)
                    self.mg_ops[i+1].hopping[3,:,0:-1:2,:,color] = fermi_tmp_coarse[:,0:-1:2,:]
                    self.mg_ops[i+1].clover[:,1::2,:,color] += fermi_tmp_coarse[:,1::2,:]

                fermi_tmp_coarse = cp.zeros_like(fermi_out)
                fermi_tmp_fine = cp.zeros_like(fermi_out_r)
                fermi_tmp_Afine = cp.zeros_like(fermi_out_r)
                fermi_tmp_coarse_n = cp.zeros_like(fermi_out)
                fermi_tmp_coarse_t = cp.zeros_like(fermi_out)
                fermi_tmp_coarse[:,:,0] = 1

                print("self.mg_ops[i+1].hopping", self.mg_ops[i+1].hopping[0,:,:,:,:])
                fermi_tmp_coarse_n = lattice.apply_mat(fermi_tmp_coarse,self.mg_ops[i+1])
                self.prolong_c2f( i, fermi_tmp_coarse, fermi_tmp_fine, self.coarse_dof[i], fine_sites_per_coarse)
                fermi_tmp_Afine = lattice.apply_mat(fermi_tmp_fine, self.mg_ops[i])
                self.restrict_f2c( i, fermi_tmp_Afine, fermi_tmp_coarse_t, self.coarse_dof[i], fine_sites_per_coarse)


                # print("clover = ", self.mg_ops[i+1].clover)
                print("self.mg_ops[i+1].hopping", self.mg_ops[i+1].hopping[0,:,:,:,:])

                print("fermi_tmp_coarse_n = ",fermi_tmp_coarse_n)
                print("fermi_tmp_coarse_t = ",fermi_tmp_coarse_t)
                print("dif = ",cp.linalg.norm(fermi_tmp_coarse_n-fermi_tmp_coarse_t))

                



        else:
            a=0

    
        


