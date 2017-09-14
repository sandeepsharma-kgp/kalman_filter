def kalman_filter(lat, lon):
    x = np.matrix([[0.0, 0.0, 0.0, 0.0]]).T
    dt = .001

    A = np.matrix([[1.0, 0.0, dt, 0.0],
                   [0.0, 1.0, 0.0, dt],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])

    H = np.matrix([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0]])

    rax = 0.032816**2
    ray = 0.054553**2

    R = np.matrix([[rax, 0.0],
                   [0.0, ray]])

    sv = 100

    G = np.matrix([[0.5 * dt**2],
                   [0.5 * dt**2],
                   [dt],
                   [dt]])

    Q = G * G.T * sv**2

    I = np.eye(4)
    P = 1.0 * np.eye(4)
    xt = []
    yt = []
    dxt = []
    dyt = []
    Zx = []
    Zy = []
    Px = []
    Py = []
    Pdx = []
    Pdy = []
    Rdx = []
    Rdy = []
    Kx = []
    Ky = []
    Kdx = []
    Kdy = []

    measurements = np.vstack((lat, lon))

    for n in range(len(measurements[0])):

        # Adaptive Measurement Covariance R from last i Measurements
        # as an Maximum Likelihood Estimation
        i = 5
        if n > i:
            R = np.matrix([[np.std(measurements[0, (n - i):n])**2, 0.0],
                           [0.0, np.std(measurements[1, (n - i):n])**2]])

        # Time Update (Prediction)
        # ========================
        # Project the state ahead
        x = A * x

        # Project the error covariance ahead
        P = A * P * A.T + Q

        # Measurement Update (Correction)
        # ===============================
        # Compute the Kalman Gain
        S = H * P * H.T + R
        K = (P * H.T) * np.linalg.pinv(S)

        # Update the estimate via z
        Z = measurements[:, n].reshape(2, 1)
        y = Z - (H * x)                            # Innovation or Residual
        x = x + (K * y)

        # Update the error covariance
        P = (I - (K * H)) * P

        # Save states for Plotting
        xt.append(float(x[0]))
        yt.append(float(x[1]))
        dxt.append(float(x[2]))
        dyt.append(float(x[3]))
        Zx.append(float(Z[0]))
        Zy.append(float(Z[1]))
        Px.append(float(P[0, 0]))
        Py.append(float(P[1, 1]))
        Pdx.append(float(P[2, 2]))
        Pdy.append(float(P[3, 3]))
        Rdx.append(float(R[0, 0]))
        Rdy.append(float(R[1, 1]))
        Kx.append(float(K[0, 0]))
        Ky.append(float(K[1, 0]))
        Kdx.append(float(K[2, 0]))
        Kdy.append(float(K[3, 0]))

    return xt[6:], yt[6:]


# a = AutoMileage.objects.get(pk='HPTRIP201762281036769225579HP')
a = AutoMileage.objects.get(pk='HPTRIP20176271825306280808395HP')

f = a.coordinate_map
np.set_printoptions(formatter={'float_kind': '{:f}'.format})
c = []
x = []
y = []
lat = []
lon = []
time = []
kx = []
ky = []
for k, v in sorted(f.iteritems()):
    if type(v) is dict:
        print k
        c = v['gps_point']
        x = []
        y = []
        for i in c[1:]:
            x.append(i.split(",")[0])
            y.append(i.split(",")[1])
            lat.append(i.split(",")[0])
            lon.append(i.split(",")[1])
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        # x = map(radians,x)
        # y = map(radians,y)
        xt, yt = kalman_filter(x, y)
        kx.extend(xt)
        ky.extend(yt)
        