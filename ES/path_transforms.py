def lead(X):
    """
    Returns the lead component of the lead lag transform of a path X (or single dimension of a path value)

    :param X: X is a path consists of tuples (time, value)
    :return: A list of points which represent the lead component of the lead-lag transformation of the path X
    """
    lead_path = []
    for j in range(2*len(X)):
        i = j // 2
        if j % 2 != 0:
            i += 1
        lead_path.append(X[i][1])
    return lead_path


def lag(X):
    """
    Returns the lag component of the lead lag transform of a path X (or single dimension of a path value)

    :param X: X is a path consists of tuples (time, value)
    :return: A list of points which represent the lag component of the lead-lag transformation of the path X
    """
    lag_path = []
    for j in range(2*len(X)):
        i = j // 2
        lag_path.append(X[i][1])
    return lag_path

