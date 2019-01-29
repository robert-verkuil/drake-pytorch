# Drake Cost of Constraint Pseudocode
def make_custom_constraint(net_constructor):
    # Constraint interface necessitates taking
    # a flat vector of decision variables.
    def constraint(decision_variables):
        x, u, net_params = unpack(decision_variables)
        
        # Construct a fresh network here with net_params
        net = net_contructor(net_params)

        # Use NNSystem logic to forward pass, and possibly propagate gradients
        if isinstance(decision_variables[0], float):
            out = NNSystemForward_[float](x, net)
        else:
            out = NNSystemForward_[AutoDiffXd](x, net)
        return 0.5*(out - u).dot(out - u)

    return constraint


# Gradient Test Case Pseudocode
def test_constraint(constraint):
    DELTA = 1e-3
    ATOL  = 1e-3
    random_floats = np.random.rand(n_inputs)
    random_ads    = np.array([AutoDiffXd(val, []) for val in random_floats])
    for i in range(n_inputs):
        # Use AutoDiff
        ad = constraint(random_ads)

        # Use Finite Differencing
        before = constraint(random_inputs)
        random_inputs[i] += DELTA
        after  = constraint(random_inputs)
        random_inputs[i] -= DELTA # Undo
        fd = (after - before) / (2 * delta)
        assert np.sum(np.abs(ad - fd)) < ATOL



