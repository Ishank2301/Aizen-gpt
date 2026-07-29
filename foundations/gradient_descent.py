class Solution:
    def get_minimizer(
        self,
        iterations: int,
        learning_rate: float,
        init: int
    ) -> float:
        return init if not iterations else round(
            init * (1 - 2 * learning_rate) ** iterations,
            5
        )