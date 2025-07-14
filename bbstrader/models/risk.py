

class HMMRiskManager():

    def __init__(
        self,
        data,
        states,
        iterations,
        end_date,
        csv_filepath,
        **kwargs,
    ):
        import warnings

        warnings.warn(
            "The class 'HMMRiskManager' is deprecated. ",
            DeprecationWarning,
        )


def build_hmm_models(symbol_list=None, **kwargs):
    import warnings

    warnings.warn(
        "The function 'build_hmm_models' is deprecated.",
        DeprecationWarning,
    )
