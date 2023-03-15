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
    xi = model.variable("xi", [M,N], Domain.unbounded())
    phi = model.variable("phi", [M,N], Domain.unbounded())
    ka = model.variable("ka",1,Domain.greaterThan(0))


    t0 = model.variable("t0", [N], Domain.unbounded())
    t1 = model.variable("t1", [N,N], Domain.unbounded())
    t2 = model.variable("t2", [N,N], Domain.unbounded())



    x = model.parameter("x",[N,N])
    p_bar = model.parameter("p_bar",N)
    p_low = model.parameter("p_low",N)
    p_hat = model.parameter("p_hat",[M,N])

    r_bar = model.parameter("r_bar",N)
    r_low = model.parameter("r_low",N)
    r_hat = model.parameter("r_hat",[M,N])
    # ka = model.parameter("ka")
    c = model.parameter("c")

    wr = {}
    wp = {}
    vr = {}
    vp = {}
    for m in range(M):
        wr[m] = model.variable([N,N],Domain.unbounded())
        wp[m] = model.variable([N,N],Domain.unbounded())
        vr[m] = model.variable([N,N],Domain.unbounded())
        vp[m] = model.variable([N,N],Domain.unbounded())
        model.constraint(Expr.sub(theta.index(m),\
                                  Expr.add(Expr.sum(t0),\
                                   Expr.add(Expr.sum(xi.pick([[m,i] for i in range(N)])),\
                                            Expr.sum(phi.pick([[m,i] for i in range(N)])) ))), \
                                                Domain.greaterThan(0.0))
        for j in range(N):
            # --- processing time 
            model.constraint(Expr.sub(xi.pick([[m,j]]), \
                                      Expr.add(
                                        Expr.sub(
                                                Expr.mul(p_bar.index(j),Expr.sum(t1.pick([[i,j] for i in range(N)]))),\
                                                Expr.sub(Expr.mul(ka,p_bar.index(j)),Expr.mul(ka,p_hat.index([m,j])))),\
                                                    p_bar.index(j)
                                                )),
                             Domain.greaterThan(0))

            model.constraint(Expr.sub(xi.pick([[m,j]]), \
                                                Expr.add(
                                                Expr.mul(p_hat.index([m,j]),Expr.sum(t1.pick([[i,j] for i in range(N)]))),\
                                                    p_hat.index([m,j]))),
                                        Domain.greaterThan(0))


            model.constraint(Expr.sub(xi.pick([[m,j]]), \
                                      Expr.add(
                                        Expr.sub(
                                                Expr.mul(p_low.index(j),Expr.sum(t1.pick([[i,j] for i in range(N)]))),\
                                                Expr.sub(Expr.mul(ka,p_hat.index([m,j])),Expr.mul(ka,p_low.index(j)))),\
                                                    p_low.index(j)
                                                )),
                             Domain.greaterThan(0))

            # release time 
            model.constraint(Expr.sub(phi.pick([[m,j]]), \
                                        Expr.sub(
                                                Expr.mul(r_bar.index(j),Expr.sum(t2.pick([[i,j] for i in range(N)]))),\
                                                Expr.sub(Expr.mul(ka,r_bar.index(j)),Expr.mul(ka,r_hat.index([m,j])))),\
                                                ),
                             Domain.greaterThan(0))
            
            model.constraint(Expr.sub(phi.pick([[m,j]]), \
                                        Expr.mul(r_hat.index([m,j]),Expr.sum(t2.pick([[i,j] for i in range(N)]))) ),
                                        Domain.greaterThan(0))


            model.constraint(Expr.sub(phi.pick([[m,j]]), \
                                        Expr.sub(
                                                Expr.mul(r_low.index(j),Expr.sum(t2.pick([[i,j] for i in range(N)]))),\
                                                Expr.sub(Expr.mul(ka,r_hat.index([m,j])),Expr.mul(ka,r_low.index(j)))
                                                ),\
                                                ),
                             Domain.greaterThan(0))

        for i in range(N):
            model.constraint(Expr.sub(t0.index(i),Expr.add(Expr.sum(wr[m].pick([[i,j] for j in range(N)])),\
                                                          Expr.sum(wp[m].pick([[i,j] for j in range(N)]))) ),\
                                                          Domain.greaterThan(0))
            if i == 0:
                model.constraint(Expr.sub(t0.index(i),Expr.add(Expr.sum(vr[m].pick([[i,j] for j in range(N)])),\
                                                            Expr.sum(vp[m].pick([[i,j] for j in range(N)]))) ),\
                                                            Domain.greaterThan(0))
            else:
                model.constraint(Expr.sub(Expr.sub(t0.index(i),t0.index(i-1)),Expr.add(Expr.sum(vr[m].pick([[i,j] for j in range(N)])),\
                                                            Expr.sum(vp[m].pick([[i,j] for j in range(N)]))) ),\
                                                            Domain.greaterThan(0))
            

            for j in range(N):
                model.constraint(Expr.sub(wr[m].index([i,j]), Expr.sub(Expr.mul(r_bar.index(j),x.index([i,j])),Expr.mul(r_bar.index(j),t2.index([i,j])))),\
                                        Domain.greaterThan(0) )
                model.constraint(Expr.sub(wr[m].index([i,j]), Expr.sub(Expr.mul(r_low.index(j),x.index([i,j])),Expr.mul(r_low.index(j),t2.index([i,j])))),\
                                        Domain.greaterThan(0) )
                model.constraint(Expr.add(wp[m].index([i,j]), Expr.mul(p_bar.index(j),t1.index([i,j]))),\
                                        Domain.greaterThan(0) )
                model.constraint(Expr.add(wp[m].index([i,j]), Expr.mul(p_low.index(j),t1.index([i,j]))),\
                                        Domain.greaterThan(0) )
                
                if i == 0:
                    model.constraint(Expr.add(vr[m].index([i,j]), Expr.mul(r_bar.index(j),t2.index([i,j]))),\
                                        Domain.greaterThan(0))
                    model.constraint(Expr.add(vr[m].index([i,j]), Expr.mul(r_low.index(j),t2.index([i,j]))),\
                                        Domain.greaterThan(0))
                    
                    model.constraint(Expr.add(vp[m].index([i,j]), Expr.mul(p_bar.index(j),t1.index([i,j])
                                                                   )),\
                                            Domain.greaterThan(0) )
                    model.constraint(Expr.add(vp[m].index([i,j]), Expr.mul(p_low.index(j),t1.index([i,j])
                                                                   )),\
                                            Domain.greaterThan(0) )
                else:
                    model.constraint(Expr.sub(vr[m].index([i,j]), 
                                                Expr.sub(Expr.mul(t2.index([i-1,j]),r_bar.index(j)),\
                                                        Expr.mul(t2.index([i,j]),r_bar.index(j)))
                                                        ),
                                        Domain.greaterThan(0))
                    model.constraint(Expr.sub(vr[m].index([i,j]), 
                                                Expr.sub(Expr.mul(t2.index([i-1,j]),r_low.index(j)),\
                                                        Expr.mul(t2.index([i,j]),r_low.index(j)))
                                                        ),
                                        Domain.greaterThan(0))
                    
                    model.constraint(Expr.sub(vp[m].index([i,j]), 
                                              Expr.add(
                                                Expr.sub(Expr.mul(x.index([i-1,j]),p_bar.index(j)),\
                                                        Expr.mul(t1.index([i,j]),p_bar.index(j))),\
                                                        Expr.mul(t1.index([i-1,j]),p_bar.index(j))
                                                        )),
                                        Domain.greaterThan(0))
                    model.constraint(Expr.sub(vp[m].index([i,j]), 
                                              Expr.add(
                                                Expr.sub(Expr.mul(x.index([i-1,j]),p_low.index(j)),\
                                                        Expr.mul(t1.index([i,j]),p_low.index(j))),\
                                                        Expr.mul(t1.index([i-1,j]),p_low.index(j))
                                                        )),
                                        Domain.greaterThan(0))


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