import numpy as np

def sortControlPoints(control_points):
    control_points.sort(key = lambda x: x["x"])
    
    x = []
    y = []
    xvis = []
    yvis = []
    xmin = np.inf
    xmax = -np.inf
    for i in range(len(control_points)):
        if (control_points[i]["type"] != 'spawn'):
            x.append(control_points[i]["x"])
            y.append(control_points[i]["y"])
            xmin = control_points[i]["x"] if control_points[i]["x"] < xmin else xmin
            xmax = control_points[i]["x"] if control_points[i]["x"] > xmax else xmax
        if (control_points[i]["type"] != 'hidden'):
            xvis.append(control_points[i]["x"])
            yvis.append(control_points[i]["y"])

    return {"x": x, "y": y, "xvis": xvis, "yvis": yvis, "xmin": xmin, "xmax": xmax}

def calcTangents(x, y, method, tension):
    method = 'fritschbutland' if method == 'undefined' else method.lower()
    
    n = len(x)
    delta = np.zeros((n-1,))
    m = np.zeros((n,))
    for k in range(n-1):
        deltak = (y[k+1] - y[k]) / (x[k+1] - x[k] + np.finfo('float').eps)
        delta[k] = deltak
        if (k == 0):  # left endpoint, same for all methods
            m[k] = deltak
        elif (method == 'cardinal'):
            m[k] = (1 - tension) * (y[k+1] - y[k-1]) / (x[k+1] - x[k-1] + np.finfo('float').eps);
        elif (method == 'fritschbutland'):
            alpha = (1 + (x[k+1] - x[k]) / (x[k+1] - x[k-1] + np.finfo('float').eps)) / 3;  # Not the same alpha as below.
            m[k] = 0 if delta[k-1] * deltak <= 0 else delta[k-1] * deltak / (alpha*deltak + (1-alpha)*delta[k-1])
        elif (method == 'fritschcarlson'):
            # If any consecutive secant lines change sign (i.e. curve changes direction), initialize the tangent to zero.
            # This is needed to make the interpolation monotonic. Otherwise set tangent to the average of the secants.
            m[k] = 0 if delta[k-1] * deltak < 0 else (delta[k-1] + deltak + np.finfo('float').eps) / 2
        elif (method == 'steffen'):
            p = ((x[k+1] - x[k]) * delta[k-1] + (x[k] - x[k-1]) * deltak) / (x[k+1] - x[k-1] + np.finfo('float').eps)
            m[k] = (np.sign(delta[k-1]) + np.sign(deltak)) * np.min(np.abs(delta[k-1]), np.abs(deltak), 0.5*np.abs(p))
        else:    # FiniteDifference
            m[k] = (delta[k-1] + deltak) / 2

    m[n-1] = delta[n-2]
    if (method != 'fritschcarlson'):
        return {"m": m, "delta": delta}

    """
    Fritsch & Carlson derived necessary and sufficient conditions for monotonicity in their 1980 paper. Splines will be
    monotonic if all tangents are in a certain region of the alpha-beta plane, with alpha and beta as defined below.
    A robust choice is to put alpha & beta within a circle around origo with radius 3. The FritschCarlson algorithm
    makes simple initial estimates of tangents and then does another pass over data points to move any outlier tangents
    into the monotonic region. FritschButland & Steffen algorithms make more elaborate first estimates of tangents that
    are guaranteed to lie in the monotonic region, so no second pass is necessary. """
    
    # Second pass of FritschCarlson: adjust any non-monotonic tangents.
    for k in range(n-1):
        deltak = delta[k]
        if (deltak == 0):
            m[k] = 0
            m[k+1] = 0
            continue

        alpha = m[k] / deltak
        beta = m[k+1] / deltak
        tau = 3 / np.sqrt(np.pow(alpha,2) + np.pow(beta,2))
        if (tau < 1): # if we're outside the circle with radius 3 then move onto the circle
            m[k] = tau * alpha * deltak
            m[k+1] = tau * beta * deltak

    return {"m": m, "delta": delta}


def interpolateCubicHermite(xeval, xbp, ybp, method, tension):
    # first we need to determine tangents (m)
    n = len(xbp)
    obj = calcTangents(xbp, ybp, method, tension)
    xbp = np.array(xbp)
    m = np.array(obj["m"])          # length n
    delta = np.array(obj["delta"])  # length n-1
    c = np.zeros((n-1,))
    d = np.zeros((n-1,))

    if method.lower() == 'linear':
        m[:-1] = delta.copy()
    else:
        xdiff = xbp[1:]-xbp[:-1]
        c = (3*delta - 2*m[:-1] - m[1:]) / (xdiff + np.finfo(m.dtype).eps)
        d = (m[:-1] + m[1:] - 2*delta) / (xdiff**2 + np.finfo(m.dtype).eps)

    f = np.empty((xeval.size,))
    k = 0
    for i in range(f.size):
        x = xeval[i]
        if (x < xbp[0]) or (x > xbp[n-1]):
            raise ValueError("interpolateCubicHermite: x value {} outside breakpoint range [{}, {}]".format(x,xbp[0],xbp[n-1]))

        while (k < n-1) and (x > xbp[k+1]):
            k += 1

        xdiff = x - xbp[k];
        f[i] = ybp[k] + m[k]*xdiff + c[k]*(xdiff**2) + d[k]*(xdiff**3); 
    
    return f


def getSpline(points,type):
    spline = [];

    if (type == "cardinal"):
        pass
        #spline = getSplineCardinal(points);
    elif (type == "monotone"):
        sortedControlPoints = sortControlPoints(points)
        interpolationmethod = 'FritschButland' # Linear, FiniteDifference, Cardinal, FritschCarlson, FritschButland, Steffen
        interpolationtension = 0.5
        xx = np.linspace(sortedControlPoints["xmin"], sortedControlPoints["xmax"], sortedControlPoints["xmax"]-sortedControlPoints["xmin"]+1);
        yy = interpolateCubicHermite(xx, sortedControlPoints["x"], sortedControlPoints["y"], interpolationmethod, interpolationtension);
        
        for i in range(xx.size):
            point = {};
            point["x"]=xx[i];
            point["y"]=yy[i];
            spline.append(point);

    return spline