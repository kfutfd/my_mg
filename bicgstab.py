import cupy as cp
import mg
import lattice

def bicgstab(b, x0=None, op=None, max_iter=30000, tol=1e-18):
    """
    使用BiCGSTAB方法求解 Ax = b，其中矩阵A的乘法操作被替换为 lattice.apply_mat(V, op)。

    参数：
    - lattice: 具有 apply_mat 方法的 lattice 对象。
    - b: 右侧向量。
    - x0: 初始解（默认为零向量）。
    - op: apply_mat 的操作参数。
    - max_iter: 最大迭代次数。
    - tol: 收敛容差。

    返回：
    - x: 解向量。
    """
    # 初始化解向量 x
    if x0 is None:
        x = cp.zeros_like(b)
    else:
        x = x0.copy()

    # 计算初始残差 r = b - Ax
    r = b - lattice.apply_mat(x, op)
    # print(r)
    r0 = r.copy()  # 保存初始残差 r0
    p = r.copy()   # 初始化搜索方向 p
    rho = 1
    rho1 = 1
    w = 1
    alpha = 1
    # 主迭代循环
    for k in range(max_iter):
        # 计算 Ap = A * p
        Ap = lattice.apply_mat(p, op)
        
        # 计算步长 alpha
        alpha = cp.vdot(r0, r) / cp.vdot(r0, Ap)
        # print("alpha = ", alpha)
                
        x += alpha * p

        # 更新中间残差 r_1 = r - alpha * Ap
        r_1 = r - alpha * Ap

        # 检查是否收敛
        print(cp.linalg.norm(r_1))
        if cp.linalg.norm(r_1) < tol:
            return x
        
        # 计算 t = A * r
        t = lattice.apply_mat(r, op)
        
        # 计算 omega
        omega = cp.vdot(t, r) / cp.vdot(t, t)
        
        # 更新解 x
        x += omega * r_1
        
        # 更新残差 r = r_1 - omega * t
        r_1 = r_1 - omega * lattice.apply_mat(r_1, op)
        
        # 检查是否收敛
        if cp.linalg.norm(r) < tol:
            return x
        
        # 计算 beta
        beta = (cp.vdot(r_1, r_1) / cp.vdot(r, r)) 
        
        # 更新搜索方向 p
        p = r_1 + alpha*beta/omega*p - alpha*beta*Ap

        r = r_1

    # 如果未收敛，抛出错误
    raise ValueError("BiCGSTAB 未能在最大迭代次数内收敛。")
