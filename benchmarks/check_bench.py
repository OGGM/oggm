"""Utility script to check that the benchmark run.

#FIXME: this should somehow be added to CI
"""

if __name__ == '__main__':

    import hef_dynamics
    import massbalance
    import numerics
    import track_model_results

    for func in dir(hef_dynamics):
        if 'time_' not in func:
            continue
        print('% -- run -- hef_dynamics.{}'.format(func))
        func = getattr(hef_dynamics, func)
        hef_dynamics.setup()
        func()
        hef_dynamics.teardown()

    for func in dir(massbalance):
        if 'time_' not in func:
            continue
        print('% -- run -- massbalance.{}'.format(func))
        func = getattr(massbalance, func)
        massbalance.setup()
        func()
        massbalance.teardown()

    for func in dir(numerics):
        if 'time_' not in func:
            continue
        print('% -- run -- numerics.{}'.format(func))
        func = getattr(numerics, func)
        func()

    c = track_model_results.hef_prepro()
    gdir = c.setup_cache()
    for func in dir(c):
        if 'track_' not in func:
            continue
        func = getattr(c, func)
        print('% -- run -- hef_prepro.{} -- out: {}'.format(func.__name__,
                                                            func(gdir)))

    c = track_model_results.full_workflow()
    gdir = c.setup_cache()
    for func in dir(c):
        if 'track_' not in func:
            continue
        func = getattr(c, func)
        print('% -- run -- full_workflow.{} -- out: {}'.format(func.__name__,
                                                               func(gdir)))

    c = track_model_results.columbia_calving()
    gdir = c.setup_cache()
    for func in dir(c):
        if 'track_' not in func:
            continue
        func = getattr(c, func)
        print('% -- run -- columbia_calving.{} -- out: '
              '{}'.format(func.__name__, func(gdir)))
