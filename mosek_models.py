from mosek.fusion import *
import sys


def obtain_mosek_model(M,N):

    # with Model() as model:
    model = Model()
    # Create variable 'x' of length 4
    theta = model.variable("theta",M, Domain.unbounded())
    lbd = model.variable("lbd", [M,N], Domain.unbounded())
    ka = model.variable("ka",1,Domain.greaterThan(0))

    xd = model.parameter("xd",N)
    xr = model.parameter("xr",N)
    xp = model.parameter("xp",[N,M])
    # ka = model.parameter("ka")
    c = model.parameter("c")

    v = {}
    for m in range(M):
        v[m] = model.variable([N,N+1],Domain.unbounded())
        model.constraint(Expr.sub(theta.index(m),Expr.sum(lbd.pick([[m,i] for i in range(N)]))), Domain.greaterThan(0.0))

        for k in range(N):
            for j in range(k,N+1,1):
                if j < N:
                    model.constraint(Expr.sub(Expr.sum(lbd.pick([[m,ind] for ind in range(k,j+1)])),\
                                                Expr.sum(v[m].pick([[ind,j] for ind in range(k,j+1)])),),\
                                            Domain.greaterThan(0) )
                else:
                    model.constraint(Expr.sub(Expr.sum(lbd.pick([[m,ind] for ind in range(k,j)])),\
                                                Expr.sum(v[m].pick([[ind,j] for ind in range(k,j)])),),\
                                            Domain.greaterThan(0) )

        for i in range(N):
            for j in range(i,N+1):
            # for j in [0,N-1,N]:
                if i == 0:
                    # rhs = np.maximum(-(j-i)*xr[0] + xp[N-1,m],\
                    # -(j-i)*xr[0] + xd[N-1] - ka * (xd[N-1] - xp[N-1,m]))
                    model.constraint(Expr.sub(v[m].pick([[i,j]]),Expr.add(Expr.mul(i-j,xr.index(0)),xp.slice([N-1,m],[N,m+1]))),Domain.greaterThan(0.0))
                    model.constraint(Expr.sub(v[m].pick([[i,j]]),Expr.sub(Expr.add(Expr.mul(i-j,xr.index(0)), xp.slice([N-1,m],[N,m+1])), \
                                                                          Expr.sub(Expr.mul(ka,xd.index(N-1)),Expr.mul(ka,xp.slice([N-1,m],[N,m+1]))))),Domain.greaterThan(0))
                                                                            # Expr.mul(ka,Expr.sub(xd.index(N-1), xp.slice([N-1,m],[N,m+1]) )))),Domain.greaterThan(0))

                else:
                    # rhs = np.maximum(-(j-i)*(xr[i] - xr[i-1] ) + (j-i+1)*xp[i-1,m],
                    # -(j-i)*(xr[i] - xr[i-1] ) + (j-i+1)*xd[i-1] - ka * (xd[i-1] - xp[i-1,m]) )
                    model.constraint(Expr.sub(v[m].pick([[i,j]]),\
                                                Expr.add(Expr.mul(i-j,Expr.sub(xr.index(i),xr.index(i-1))), \
                                                        Expr.mul(j-i+1,xp.slice([i-1,m],[i,m+1]))) ),Domain.greaterThan(0))
                    model.constraint(Expr.sub(v[m].pick([[i,j]]),\
                                                Expr.sub(Expr.add(Expr.mul(i-j,Expr.sub(xr.index(i),xr.index(i-1))), \
                                                                Expr.mul(j-i+1,xd.index([i-1]))), \
                                                                Expr.sub(Expr.mul(ka,xd.index(i-1)),Expr.mul(ka,xp.slice([i-1,m],[i,m+1]))))),Domain.greaterThan(0))

                                                                    # Expr.mul(ka,Expr.sub(xd.index(i-1), xp.slice([i-1,m],[i,m+1]))))),Domain.greaterThan(0))
                # cons_name = 'v_'+str(i)+'_'+str(j)
                # model.constraint(cons_name,v[m].pick([[i,j]]),Domain.greaterThan(rhs))

    model.objective("obj", ObjectiveSense.Minimize, Expr.add(Expr.mul(c,ka),Expr.mul(1/M,Expr.sum(theta))))


    # xr_tem = x_tem @ r
    # xd_tem = x_tem @ d_bar
    # xp_tem = x_tem @ p_hat

    # xr.setValue(xr_tem)
    # xd.setValue(xd_tem)
    # xp.setValue(xp_tem)
    # ka.setValue(ka_tem)

    # model.solve()
    # sol = theta.level()
    # obj_val = c*ka_tem + (1/M)*np.sum(sol)


    return model


# random release time model
def obtain_mosek_random_model(M,N):

    # with Model() as model:
    model = Model()
    # Create variable 'x' of length 4
    theta = model.variable("theta",M, Domain.unbounded())
    ka = model.variable("ka",1,Domain.greaterThan(0))


    # t0 = {}
    # t1 = {}
    # t2 = {}
    t0 = model.variable("t0", [N], Domain.unbounded())
    t1 = model.variable("t1", [N,N], Domain.unbounded())
    t2 = model.variable("t2", [N,N], Domain.unbounded())
    x = model.parameter("x",[N,N])


    p_bar = model.parameter("p_bar",N)
    p_low = model.parameter("p_low",N)
    p_hat = model.parameter("p_hat",[N,M])

    r_bar = model.parameter("r_bar",N)
    r_low = model.parameter("r_low",N)
    r_hat = model.parameter("r_hat",[N,M])
    # ka = model.parameter("ka")
    c = model.parameter("c")

    phi = {}
    bet = {}
    up = {}
    vp = {}
    ur = {}
    vr = {}

    wp = {}
    sp = {}
    wr = {}
    sr = {}
    phi_p = {}
    phi_r = {}
    pi_p = {}
    pi_r = {}


    for m in range(M):

        # t0[m] = model.variable([N], Domain.unbounded())
        # t1[m] = model.variable([N,N], Domain.unbounded())
        # t2[m] = model.variable([N,N], Domain.unbounded())

        phi[m] = model.variable(N,Domain.unbounded())
        bet[m] = model.variable(N,Domain.unbounded())
        up[m] = model.variable(N,Domain.lessThan(0))
        ur[m] = model.variable(N,Domain.lessThan(0))

        vp[m] = model.variable(N,Domain.greaterThan(0))
        vr[m] = model.variable(N,Domain.greaterThan(0))


        # model.addConstr(theta[m] >= quicksum(t0) + quicksum([-bet[m][j]*p_hat[j,m] - phi[m][j]*r_hat[j,m] for j in range(N)])\
        #     + quicksum([up[m][j]*p_low[j] + vp[m][j]*p_bar[j] + ur[m][j]*r_low[j] + vr[m][j]*r_bar[j] for j in range(N) ]))

        model.constraint(Expr.sub(theta.index(m),\
                                  Expr.add(
                                        Expr.add(
                                            Expr.add(
                                                Expr.add(
                                                    Expr.sub(
                                                            Expr.sub(Expr.sum(t0),
                                                                    Expr.dot(p_hat.slice([0,m],[N,m+1]).reshape([N]),bet[m])
                                                                    ),
                                                            Expr.dot(r_hat.slice([0,m],[N,m+1]).reshape([N]),phi[m]) 
                                                            ),
                                                        Expr.dot(up[m],p_low)
                                                        ),
                                                    Expr.dot(vp[m],p_bar)
                                                    ),
                                                Expr.dot(ur[m],r_low)
                                                ),
                                            Expr.dot(vr[m],r_bar)
                                            ),
                                        ),
                            Domain.greaterThan(0)
                        )
        for j in range(N):
            # model.addConstr(bet[m][j]==up[m][j] + vp[m][j] - quicksum([t1[i,j] for i in range(N)]) - 1)
            model.constraint(Expr.sub(bet[m].index(j),
                                      Expr.sub(Expr.add(up[m].index(j),
                                                        vp[m].index(j)),
                                                Expr.sum(t1.pick([[i,j] for i in range(N)]))
                                                )
                                    ),Domain.equalsTo(-1)  
                            )
            
            
            # model.addConstr(phi[m][j]==ur[m][j] + vr[m][j] - quicksum([t2[i,j] for i in range(N)]))

            model.constraint(Expr.sub(phi[m].index(j),
                                      Expr.sub(Expr.add(ur[m].index(j),
                                                        vr[m].index(j)),
                                                Expr.sum(t2.pick([[i,j] for i in range(N)]))
                                                )
                                    ),Domain.equalsTo(0)  
                            )

            # model.addConstr(ka >= bet[m][j] )
            model.constraint(Expr.sub(ka,bet[m].index(j)),Domain.greaterThan(0))
            # model.addConstr(ka >= -bet[m][j])
            model.constraint(Expr.add(ka,bet[m].index(j)),Domain.greaterThan(0))
            # model.addConstr(ka >= phi[m][j] )
            model.constraint(Expr.sub(ka,phi[m].index(j)),Domain.greaterThan(0))
            # model.addConstr(ka >= -phi[m][j])
            model.constraint(Expr.add(ka,phi[m].index(j)),Domain.greaterThan(0))



        wp[m] = model.variable([N,N],Domain.lessThan(0))
        wr[m] = model.variable([N,N],Domain.lessThan(0))
        sp[m] = model.variable([N,N],Domain.greaterThan(0))
        sr[m] = model.variable([N,N],Domain.greaterThan(0))

        phi_p[m] = model.variable([N,N],Domain.lessThan(0))
        phi_r[m] = model.variable([N,N],Domain.lessThan(0))
        pi_p[m] = model.variable([N,N],Domain.greaterThan(0))
        pi_r[m] = model.variable([N,N],Domain.greaterThan(0))


        for i in range(N):
            # model.addConstr(t0[i] >= quicksum([wr[m][i,j]*r_low[j] + sr[m][i,j]*r_bar[j] + wp[m][i,j]*p_low[j] + sp[m][i,j]*p_bar[j] for j in range(N)]))

            model.constraint(Expr.sub(t0.index(i),
                                      Expr.add(Expr.add(Expr.add(Expr.dot(wp[m].slice([i,0],[i+1,N]).reshape([N]),p_low),\
                                               Expr.dot(sp[m].slice([i,0],[i+1,N]).reshape([N]),p_bar)),\
                                               Expr.dot(sr[m].slice([i,0],[i+1,N]).reshape([N]),r_bar)),\
                                               Expr.dot(wr[m].slice([i,0],[i+1,N]).reshape([N]),r_low))
                                      ),Domain.greaterThan(0))
            for j in range(N):
                # model.addConstr(wr[m][i,j] + sr[m][i,j] == x[i,j] - t2[i,j])
                model.constraint(Expr.add(Expr.sub(Expr.add(wr[m].index([i,j]),sr[m].index([i,j])),x.index([i,j])),t2.index([i,j])),
                                 Domain.equalsTo(0))
                # model.addConstr(wp[m][i,j] + sp[m][i,j] == -t1[i,j])
                model.constraint(Expr.add(Expr.add(wp[m].index([i,j]),sp[m].index([i,j])),t1.index([i,j])),
                                 Domain.equalsTo(0))


            if i == 0:
                # model.addConstr(t0[i] >= quicksum([phi_r[m][i,j]*r_low[j] + pi_r[m][i,j]*r_bar[j] + phi_p[m][i,j]*p_low[j] + pi_p[m][i,j]*p_bar[j] for j in range(N)]))
                model.constraint(Expr.sub(t0.index(i),
                                      Expr.add(Expr.add(Expr.add(Expr.dot(phi_r[m].slice([i,0],[i+1,N]).reshape([N]),r_low),\
                                               Expr.dot(pi_r[m].slice([i,0],[i+1,N]).reshape([N]),r_bar)),\
                                               Expr.dot(phi_p[m].slice([i,0],[i+1,N]).reshape([N]),p_low)),\
                                               Expr.dot(pi_p[m].slice([i,0],[i+1,N]).reshape([N]),p_bar))
                                      ),Domain.greaterThan(0))
                
                for j in range(N):
                    # model.addConstr(phi_r[m][i,j] + pi_r[m][i,j] == -t2[i,j])
                    model.constraint(Expr.add(Expr.add(phi_r[m].index([i,j]),pi_r[m].index([i,j])),t2.index([i,j])),Domain.equalsTo(0))
                    # model.addConstr(phi_p[m][i,j] + pi_p[m][i,j] == -t1[i,j])   
                    model.constraint(Expr.add(Expr.add(phi_p[m].index([i,j]),pi_p[m].index([i,j])),t1.index([i,j])),Domain.equalsTo(0))

            if i > 0:
                # model.addConstr(t0[i] - t0[i-1] >= quicksum([phi_r[m][i,j]*r_low[j] + pi_r[m][i,j]*r_bar[j] + phi_p[m][i,j]*p_low[j] + pi_p[m][i,j]*p_bar[j] for j in range(N)]))
                model.constraint(Expr.sub(Expr.sub(t0.index(i),t0.index(i-1)),
                                      Expr.add(Expr.add(Expr.add(Expr.dot(phi_r[m].slice([i,0],[i+1,N]).reshape([N]),r_low),\
                                               Expr.dot(pi_r[m].slice([i,0],[i+1,N]).reshape([N]),r_bar)),\
                                               Expr.dot(phi_p[m].slice([i,0],[i+1,N]).reshape([N]),p_low)),\
                                               Expr.dot(pi_p[m].slice([i,0],[i+1,N]).reshape([N]),p_bar))
                                      ),Domain.greaterThan(0))
                
                for j in range(N):
                    # model.addConstr(phi_r[m][i,j] + pi_r[m][i,j] == t2[i-1,j]-t2[i,j])
                    model.constraint(Expr.sub(Expr.add(Expr.add(phi_r[m].index([i,j]),pi_r[m].index([i,j])),t2.index([i,j])),t2.index([i-1,j])),Domain.equalsTo(0))

                    # model.addConstr(phi_p[m][i,j] + pi_p[m][i,j] == x[i-1,j] + t1[i-1,j]-t1[i,j])  
                    model.constraint(Expr.sub(Expr.add(Expr.add(phi_p[m].index([i,j]),pi_p[m].index([i,j])),t1.index([i,j])),Expr.add(x.index([i-1,j]),t1.index([i-1,j]))),Domain.equalsTo(0))

    model.objective("obj", ObjectiveSense.Minimize, Expr.add(Expr.mul(c,ka),Expr.mul(1/M,Expr.sum(theta))))

    return model







# -------------- history -----------------------------------
def obtain_mosek_RS_model(M,N):

    # with Model() as model:
    model = Model()
    # Create variable 'x' of length 4
    theta = model.variable("theta",M, Domain.unbounded())
    lbd = model.variable("lbd", [M,N], Domain.unbounded())
    # ka = model.variable("ka",1,Domain.greaterThan(0))

    xd = model.parameter("xd",N)
    xr = model.parameter("xr",N)
    xp = model.parameter("xp",[N,M])
    ka = model.parameter("ka")
    # tau = model.parameter("tau")

    v = {}
    for m in range(M):
        v[m] = model.variable([N,N+1],Domain.unbounded())
        model.constraint(Expr.sub(theta.index(m),Expr.sum(lbd.pick([[m,i] for i in range(N)]))), Domain.greaterThan(0.0))

        for k in range(N):
            for j in range(k,N+1,1):
                if j < N:
                    model.constraint(Expr.sub(Expr.sum(lbd.pick([[m,ind] for ind in range(k,j+1)])),\
                                                Expr.sum(v[m].pick([[ind,j] for ind in range(k,j+1)])),),\
                                            Domain.greaterThan(0) )
                else:
                    model.constraint(Expr.sub(Expr.sum(lbd.pick([[m,ind] for ind in range(k,j)])),\
                                                Expr.sum(v[m].pick([[ind,j] for ind in range(k,j)])),),\
                                            Domain.greaterThan(0) )

        for i in range(N):
            for j in range(i,N+1):
                if i == 0:
                    model.constraint(Expr.sub(v[m].pick([[i,j]]),Expr.add(Expr.mul(i-j,xr.index(0)),xp.slice([N-1,m],[N,m+1]))),Domain.greaterThan(0.0))
                    model.constraint(Expr.sub(v[m].pick([[i,j]]),Expr.sub(Expr.add(Expr.mul(i-j,xr.index(0)), xp.slice([N-1,m],[N,m+1])), \
                                                                          Expr.sub(Expr.mul(ka,xd.index(N-1)),Expr.mul(ka,xp.slice([N-1,m],[N,m+1]))))),Domain.greaterThan(0))

                else:
                    model.constraint(Expr.sub(v[m].pick([[i,j]]),\
                                                Expr.add(Expr.mul(i-j,Expr.sub(xr.index(i),xr.index(i-1))), \
                                                        Expr.mul(j-i+1,xp.slice([i-1,m],[i,m+1]))) ),Domain.greaterThan(0))
                    model.constraint(Expr.sub(v[m].pick([[i,j]]),\
                                                Expr.sub(Expr.add(Expr.mul(i-j,Expr.sub(xr.index(i),xr.index(i-1))), \
                                                                Expr.mul(j-i+1,xd.index([i-1]))), \
                                                                Expr.sub(Expr.mul(ka,xd.index(i-1)),Expr.mul(ka,xp.slice([i-1,m],[i,m+1]))))),Domain.greaterThan(0))

    # model.constraint(Expr.sub(Expr.mul(1/M,Expr.sum(theta)),tau),Domain.lessThan(0))
    model.objective("obj", ObjectiveSense.Minimize, Expr.mul(1/M,Expr.sum(theta)))

    return model


def obtain_mosek_full_model(M,N):

    # with Model() as model:
    model = Model()
    # Create variable 'x' of length 4
    theta = model.variable("theta",M, Domain.unbounded())
    lbd = model.variable("lbd", [M,N], Domain.unbounded())
    # ka = model.variable("ka",1,Domain.greaterThan(0))
    x = model.variable("x",[N,N],Domain.binary())


    d = model.parameter("d",N)
    r = model.parameter("r",N)
    p = model.parameter("p",[N,M])
    ka = model.parameter("ka")


    xd = Expr.mul(x,d)
    xr = Expr.mul(x,r)
    xp = Expr.mul(x,p)
    v = {}
    for m in range(M):
        v[m] = model.variable([N,N+1],Domain.unbounded())
        model.constraint(Expr.sub(theta.index(m),Expr.sum(lbd.pick([[m,i] for i in range(N)]))), Domain.greaterThan(0.0))

        for k in range(N):
            for j in range(k,N+1,1):
                if j < N:
                    model.constraint(Expr.sub(Expr.sum(lbd.pick([[m,ind] for ind in range(k,j+1)])),\
                                                Expr.sum(v[m].pick([[ind,j] for ind in range(k,j+1)])),),\
                                            Domain.greaterThan(0) )
                else:
                    model.constraint(Expr.sub(Expr.sum(lbd.pick([[m,ind] for ind in range(k,j)])),\
                                                Expr.sum(v[m].pick([[ind,j] for ind in range(k,j)])),),\
                                            Domain.greaterThan(0) )

        for i in range(N):
            for j in range(i,N+1):
                if i == 0: 
                    model.constraint(Expr.sub(v[m].pick([[i,j]]),Expr.add(Expr.mul(i-j,xr.slice([0],[1])),xp.slice([N-1,m],[N,m+1]))),Domain.greaterThan(0.0))
                    model.constraint(Expr.sub(v[m].pick([[i,j]]),Expr.sub(Expr.add(Expr.mul(i-j,xr.slice([0],[1])), xp.slice([N-1,m],[N,m+1])), \
                                                                          Expr.sub(Expr.mul(ka,xd.slice([N-1],[N])),Expr.mul(ka,xp.slice([N-1,m],[N,m+1]))))),Domain.greaterThan(0))

                else:
                    model.constraint(Expr.sub(v[m].pick([[i,j]]),\
                                                Expr.add(Expr.mul(i-j,Expr.sub(xr.slice([i],[i+1]),xr.slice([i-1],[i]))), \
                                                        Expr.mul(j-i+1,xp.slice([i-1,m],[i,m+1]))) ),Domain.greaterThan(0))
                    model.constraint(Expr.sub(v[m].pick([[i,j]]),\
                                                Expr.sub(Expr.add(Expr.mul(i-j,Expr.sub(xr.slice([i],[i+1]),xr.slice([i-1],[i]))), \
                                                                Expr.mul(j-i+1,xd.slice([i-1],[i]))), \
                                                                Expr.sub(Expr.mul(ka,xd.slice([i-1],[i])),Expr.mul(ka,xp.slice([i-1,m],[i,m+1]))))),Domain.greaterThan(0))


    for i in range(N):
        model.constraint(Expr.sum(x.slice([i,0],[i+1,N])),Domain.equalsTo(1))
        model.constraint(Expr.sum(x.slice([0,i],[N,i+1])),Domain.equalsTo(1))

    model.objective("obj", ObjectiveSense.Minimize, Expr.mul(1/M,Expr.sum(theta)))


    # xr_tem = x_tem @ r
    # xd_tem = x_tem @ d_bar
    # xp_tem = x_tem @ p_hat

    return model