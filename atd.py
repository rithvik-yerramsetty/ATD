import sys
import warnings
from math import sqrt

if sys.version_info < (3, 9):
    warnings.warn("You're suggested to upgrade your Python interpreter.", category=ImportWarning)

try:
    from typing import Any, Iterable, Optional, Tuple, Union, Callable, Final, final
    from abc import abstractmethod
    from functools import wraps
except ImportError:
    warnings.warn("Unable to import type hinting library, "
                  "possibly because you have to upgrade your Python interpreter.", category=ImportWarning)


    def original_decorator(obj: Callable) -> Callable:
        return obj


    abstractmethod = final = wraps = original_decorator
    Any = Iterable = Optional = Tuple = Union = Callable = Final = None

try:
    if sys.version_info < (3, 10):
        # Support for old version
        from backend_manager_39 import Backend, Matrix, Decimal, isinstance, extend_with_000, extend_with_010
    else:
        from backend_manager import Backend, Matrix, Decimal, extend_with_000, extend_with_010
except ImportError:
    raise ImportError("Unable to import the specified backend!")
    exit(-1)

meta_data: dict = {"trace_update_mode": {},
                   "w_update_emphasizes": ["complexity", "accuracy"],
                   "rcond": 1e-5}  # Meta data
TraceUpdateFunction: Final = Callable[[Any, Matrix, Decimal, Optional[Matrix],
                                       Optional[Decimal], Optional[Decimal],
                                       Optional[Decimal]], Matrix]


def learn_func_wrapper(
        func: Callable[[Any, Matrix, Matrix, float, float, int], Any]
) -> Callable[[Any, Matrix, Matrix, float, float, int], Any]:
    """
    The decorator for the learn function. Helpful for checking input.
    """
    if not callable(func):
        raise ValueError("Unexpected decorator usage.")

    @wraps(func)
    def _learn_func(
            self: AbstractAgent,
            observation: Matrix,
            next_observation: Matrix,
            reward: float,
            discount: float,
            t: int
    ) -> Any:
        assert observation.shape == (
            self.observation_space_n,), f"Bad observation shape. Expected ({self.observation_space_n},), not {observation.shape}"
        assert next_observation.shape == (
            self.observation_space_n,), f"Bad next observation shape. Expected ({self.observation_space_n},), not {next_observation.shape}"
        if not (isinstance(reward, Decimal) and isinstance(discount, Decimal)
                and isinstance(t, int) and isinstance(self, AbstractAgent)):
            raise TypeError("Invalid input type!")
        if not (t >= 0 and 0 <= discount <= 1):
            raise ValueError("Invalid hyperparameter!")

        self.lr = self.lr_func(t)  # Calculate the new learning rate

        return func(self, observation, next_observation, reward, discount, t)

    return _learn_func


def register_trace_update_func(
        mode_name: str
) -> Callable[[TraceUpdateFunction], TraceUpdateFunction]:
    """
    Decorator for registering trace update functions.
    """

    def _trace_update_func_wrapper(
            func: TraceUpdateFunction
    ) -> TraceUpdateFunction:
        """
        Decorator for trace update functions. Helpful for checking input.
        """

        if not callable(func):
            raise ValueError("Unexpected decorator usage.")
        if not isinstance(mode_name, str):
            raise TypeError("Invalid trace update mode type.")

        @wraps(func)
        def _trace_update_func(self: Any, observation: Matrix,
                               discount: Decimal, e: Optional[Matrix] = None,
                               lambd: Optional[Decimal] = None, rho: Optional[Decimal] = 1.,
                               i: Optional[Decimal] = 1.) -> Matrix:
            assert observation.shape == (
                self.observation_space_n,), f"Bad observation shape. Expected ({self.observation_space_n},), not {observation.shape}"
            if not (isinstance(discount, Decimal) and isinstance(lambd, Decimal)
                    and isinstance(self, AbstractAgent)):
                raise TypeError("Invalid input type!")
            if not 0 <= discount <= 1:
                raise ValueError("Invalid discount parameter!")
            if e is None:
                e = self.e
            if lambd is None:
                lambd = self.lambd

            return func(self=self, observation=observation, discount=discount, e=e, lambd=lambd, rho=rho, i=i)

        meta_data["trace_update_mode"][mode_name] = _trace_update_func
        return _trace_update_func

    return _trace_update_func_wrapper


class AbstractAgent:
    """
    AbstractAgent
    ======

    The abstract agent class, offering some fundamental functions.

    Parameters
    ------
    observation_space_n :
        The shape(1-D) of observation space
    action_space_n :
        The shape(1-D) of action space
    lr :
        learning rate, could be a function with time step as input and learning rate as output, or a float representing
        constant learning rate
    lambd :
        λ for trace updating
    trace_update_mode :
        Trace update mode, should be in ``conventional | emphatic`` . Default is ``conventional``.

    Raises
    ------
    TypeError
        Invalid input type
    AssertionError
        Unable to deal with the learning rate input
    """

    def __init__(self, observation_space_n: int, action_space_n: int,
                 lr: Union[Callable[[int], Decimal], Decimal], lambd: Optional[Decimal] = 0,
                 trace_update_mode: Optional[str] = "conventional") -> None:
        if not (isinstance(observation_space_n, int)
                and isinstance(action_space_n, int)
                and isinstance(lambd, Decimal)
                and isinstance(meta_data["rcond"], Decimal)
                and isinstance(trace_update_mode, str)):
            raise TypeError("Invalid input type!")
        if trace_update_mode not in meta_data["trace_update_mode"].keys():
            warnings.warn(
                f"Not supported trace update mode: {trace_update_mode}! Will be set to conventional。")
            trace_update_mode = "conventional"
        if isinstance(lr, Decimal):
            self.lr_func = lambda t: lr
        else:
            assert callable(lr), "Unable to deal with the learning rate input."
            self.lr_func = lr

        self.observation_space_n = observation_space_n
        self.action_space_n = action_space_n
        self.lambd = lambd
        self.trace_update = meta_data["trace_update_mode"][trace_update_mode]  # type: TraceUpdateFunction

        self.reinit()
        self.reset()

    def reinit(self) -> None:
        """
        Make the agent forget what it learned.
        """
        self.w = Backend.empty(self.observation_space_n)  # Initialize the weight arbitrarily

    def reset(self) -> None:
        """
        Reset everything of the agent. Should be invoked when a game begins.
        """
        self.F = 0
        self.M = 0
        self.e = Backend.zeros(self.observation_space_n)

    @abstractmethod
    def learn(
            self,
            observation: Matrix,
            next_observation: Matrix,
            reward: Decimal,
            discount: Decimal,
            t: int
    ) -> Any:
        """
        Train the agent. Should be decorated with ``@learn_func_wrapper`` .

        Parameters
        ------
        observation :
            Current observation
        next_observation :
            Next observation
        reward :
            Reward
        discount :
            γ discount. 0 for the terminal step and 0.99 for the rest for example
        t :
            Time step. Starts from 0

        Returns
        ------
        Any :
            The loss

        Raises
        ------
        NotImplementedError
            This learn function has not been implemented yet
        AssertionError
            Invalid input shape
        TypeError
            Invalid input type
        ValueError
            Invalid hyperparameter
        """
        raise NotImplementedError("The agent is not trainable!")

    def decide(self, next_observations: Iterable[Matrix]) -> int:
        """
        Ask the agent to pick an action.

        Parameters
        ------
        next_observations :
            A list consisted of all the next observations

        Returns
        ------
        action : int
            The action index picked by the agent

        Raises
        ------
        ValueError
            Unexpected error
        """
        warnings.simplefilter("default", DeprecationWarning)
        warnings.warn("This function has not been tested yet!", category=DeprecationWarning)

        try:
            next_v = [self.w @ next_observation
                      for next_observation in next_observations]
        except ValueError:
            print("Unexcepted error, maybe the input shape is invalid?")
            return -1

        return Backend.argmax(next_v)

    @staticmethod
    @final
    def trace_update(self, observation: Matrix, discount: Decimal, e: Optional[Matrix] = None,
                     lambd: Optional[Decimal] = None, rho: Optional[Decimal] = 1.,
                     i: Optional[Decimal] = 1.) -> Matrix:
        """
        Parameters
        ------
        self :
            The agent object for trace update
        observation :
            Current observation
        discount :
            γ discount. 0 for the terminal step and 0.99 for the rest for example
        e :
            Previous trace. Omit it to use the one stored in the agent
        lambd :
            λ for trace updating. Omit it to use the one stored in the agent
        rho :
            Only needed when emphatic trace update is required.
            In the off-policy context, it is the quotient of the probability to choose the action if applied the target
            policy π and the probability if applied the behaviour policy b, namely :math:`\\frac{π(a)}{b(a)}` .
            In the on-policy context, it should be 1.
        i :
            Only needed when emphatic trace update is required.
            How much is the agent interested in the current observation. If averagely interested, then it is 1.

        Returns
        ------
        Matrix
            New trace

        Raises
        ------
        AssertionError
            Invalid input shape
        TypeError
            Invalid input type
        ValueError
            Invalid γ discount
        """
        ...

    @staticmethod
    @register_trace_update_func("conventional")
    def __trace_update(*, self, observation: Matrix, discount: Decimal, e: Optional[Matrix] = None,
                       lambd: Optional[Decimal] = None, **kwargs) -> Matrix:
        """
        Internal function.
        The implementation of concrete conventional trace update algorithm.
        """
        return discount * lambd * e + observation

    @staticmethod
    @register_trace_update_func("emphatic")
    def __emphatic_trace_update(*, self, observation: Matrix, discount: Decimal, e: Optional[Matrix] = None,
                                lambd: Optional[Decimal] = None, rho: Optional[Decimal] = 1.,
                                i: Optional[Decimal] = 1., **kwargs) -> Matrix:
        """
        Internal function.
        The implementation of concrete emphatic trace update algorithm.
        """
        if not (isinstance(rho, Decimal) and isinstance(i, Decimal)):
            raise TypeError("Invalid input type!")

        self.F = rho * discount * self.F + i
        self.M = lambd * i + (1 - lambd) * self.F

        return rho * (discount * lambd * e + self.M * observation)


class TDAgent(AbstractAgent):
    """
    TDAgent
    ======

    Conventional temporal difference learning algorithm.

    See Also
    ------
    ``TDAgent``
    """

    @learn_func_wrapper
    def learn(
            self,
            observation: Matrix,
            next_observation: Matrix,
            reward: Decimal,
            discount: Decimal,
            t: int
    ) -> Any:
        self.e = self.trace_update(self, observation, discount, self.e, self.lambd)  # Updates the trace
        delta = reward + discount * self.w @ next_observation - self.w @ observation  # Calculate the TD error
        self.w += self.lr * delta * self.e  # Updates the weight

        return delta


class PlainATDAgent(AbstractAgent):
    """
    PlainATDAgent
    ======

    Plain accelerated temporal difference learning algorithm(ATD).

    Parameters
    ------
    eta :
        Learning rate for semi-gradient TD.
    lr :
        Learning rate for semi-gradient mean squared projected Bellman error(MSPBE).
    """

    def __init__(self,
                 eta: Decimal,
                 lr: Optional[Union[Callable[[int], Decimal], Decimal]] = lambda t: 1 / (t + 1),
                 **kwargs) -> None:
        super().__init__(lr=lr, **kwargs)
        if not (isinstance(eta, Decimal)):
            raise TypeError("Invalid input type!")

        self.eta = eta

    def reinit(self) -> None:
        super(PlainATDAgent, self).reinit()
        self.A = Backend.zeros((self.observation_space_n, self.observation_space_n))

    @learn_func_wrapper
    def learn(
            self,
            observation: Matrix,
            next_observation: Matrix,
            reward: Decimal,
            discount: Decimal,
            t: int
    ) -> Any:
        beta = 1 / (t + 1)  # As this value is frequently used, assign it to a variable β
        delta = reward + discount * self.w @ next_observation - self.w @ observation  # Calculates the TD error
        self.e = self.trace_update(self, observation, discount, self.e, self.lambd)  # Updates the trace

        # Calculates the matrix A. A should be the expectation, so use incremental update method to reduce complexity
        self.A = (1 - beta) * self.A + beta * self.e.reshape((self.observation_space_n, 1)) \
                 @ (observation - discount * next_observation).reshape((1, self.observation_space_n))



        # Am, t = em,i(xi − γxi + 1)

        self.w += (self.lr * Backend.linalg.pinv(self.A, rcond=meta_data["rcond"]) + self.eta *
                   Backend.eye(self.observation_space_n)) @ (delta * self.e)  # Updates the weight accordingly
        # Originally 1/(1+t) is used, replacing it with beta

        return delta


class SVDATDAgent(AbstractAgent):
    """
    SVDATDAgent
    ======

    The ATD algorithm based on SVD decomposition.

    Parameters
    ------
    eta :
        Learning rate for semi-gradient TD.
    lr :
        Learning rate for semi-gradient mean squared projected Bellman error(MSPBE).

    See Also
    ------
    ``PlainATDAgent``
    """

    def __init__(self,
                 eta: Decimal,
                 lr: Optional[Union[Callable[[int], Decimal], Decimal]] = lambda t: 1 / (t + 1),
                 **kwargs) -> None:
        super().__init__(lr=lr, **kwargs)
        if not (isinstance(eta, Decimal)):
            raise TypeError("Invalid input type!")

        self.eta = eta

    def reinit(self) -> None:
        super(SVDATDAgent, self).reinit()
        self.U, self.V, self.Sigma = Backend.empty(
            (self.observation_space_n, 0)), Backend.empty((self.observation_space_n, 0)), Backend.empty((0, 0))

    def svd_update(
            self,
            U: Matrix,
            Sigma: Matrix,
            V: Matrix,
            z: Matrix,
            d: Matrix
    ) -> Tuple[Matrix, Matrix, Matrix]:
        """
        SVD update. It is the same as
        :math:`\\mathbf{U}' \\mathbf{\\Sigma} '\\mathbf{V'}^\\top =
        \\mathbf{U}\\mathbf{\\Sigma}\\mathbf{V}^\\top + \\mathbf{z}\\mathbf{d}^\\top`

        Parameters
        ------
        U :
            The matrix U
        Sigma :
            The matrix ∑
        V :
            The matrix V
        z :
            The vector z
        d :
            The vector d

        Returns
        ------
        Tuple[Matrix, Matrix, Matrix]
            The new updated U'、∑'、V'

        Raises
        ------
        TypeError
            Wrong input type
        ValueError
            Cannot multiply the matrices.
        """
        try:
            U, Sigma, V, z, d = Backend.convert_to_matrix_func(U), Backend.convert_to_matrix_func(
                Sigma), Backend.convert_to_matrix_func(V), Backend.convert_to_matrix_func(
                z), Backend.convert_to_matrix_func(d)
        except TypeError:
            warnings.warn("Wrong input type!")
            return U, Sigma, V
        if U.ndim != 2 \
                or Sigma.ndim != 2 \
                or V.ndim != 2 \
                or U.shape[1] != Sigma.shape[0] \
                or V.shape[1] != Sigma.shape[1] \
                or U.shape[0] != z.shape[0] \
                or V.shape[0] != d.shape[0]:
            raise ValueError("Unable to handle the input!")

        m = U.T @ z
        p = z - U @ m
        n = V.T @ d
        q = d - V @ n

        p_l2 = Backend.linalg.norm(p)
        q_l2 = Backend.linalg.norm(q)

        K = extend_with_000(Sigma) + Backend.vstack((m, p_l2)
                                                    ) @ Backend.vstack((n, q_l2)).T

        p = p / p_l2 if p_l2 > 0 else Backend.zeros_like(p)
        q = q / q_l2 if q_l2 > 0 else Backend.zeros_like(q)
        U = Backend.hstack((U, p))
        V = Backend.hstack((V, q))

        return U, K, V

    @learn_func_wrapper
    def learn(
            self,
            observation: Matrix,
            next_observation: Matrix,
            reward: Decimal,
            discount: Decimal,
            t: int
    ) -> Any:
        beta = 1 / (t + 1)
        delta = reward + discount * self.w @ next_observation - self.w @ observation
        self.e = self.trace_update(self, observation, discount, self.e, self.lambd)

        self.U, self.Sigma, self.V = \
            self.svd_update(
                self.U,
                (1 - beta) * self.Sigma,
                self.V,
                sqrt(beta) * self.e.reshape((self.observation_space_n, 1)),
                sqrt(beta) * (observation - discount *
                              next_observation).reshape((self.observation_space_n, 1))
            )  # Uses SVD update to reduce the complexity, enhancing the performance

        self.w += (self.lr *
                   Backend.linalg.pinv(self.U @ self.Sigma @ self.V.T, rcond=meta_data["rcond"]) +
                   self.eta *
                   Backend.eye(self.observation_space_n)) @ (delta * self.e)

        return delta


class DiagonalizedSVDATDAgent(SVDATDAgent):
    """
    DiagonalizedSVDATDAgent
    ======

    Diagonalizing :math:`\\mathbf{\\Sigma}` and SVD decomposition based ATD。

    Parameters
    ------
    k :
        The largest allowed size of matrices(k*k)
    svd_diagonalizing :
        Decides whether to use svd decomposition to diagonalize the matrix with orthogonality. Default is `False`
    w_update_emphasizes :
        Decides which one comes first when updating the weight. Should be one of ``accuracy | complexity``
    """

    def __init__(self, k: int,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if not (isinstance(k, int)):
            raise TypeError("Invalid input type!")

        self.k = k

    def reinit(self) -> None:
        super(DiagonalizedSVDATDAgent, self).reinit()
        self.L, self.R = Backend.empty((0, 0)), Backend.empty((0, 0))

    def svd_update(
            self,
            U: Matrix,
            Sigma: Matrix,
            V: Matrix,
            z: Matrix,
            d: Matrix
    ) -> Tuple[Matrix, Matrix, Matrix]:
        try:
            U, Sigma, V, z, d = Backend.convert_to_matrix_func(U), Backend.convert_to_matrix_func(
                Sigma), Backend.convert_to_matrix_func(V), Backend.convert_to_matrix_func(
                z), Backend.convert_to_matrix_func(d)
        except TypeError:
            warnings.warn("Wrong input type!")
            return U, Sigma, V
        if U.ndim != 2 \
                or Sigma.ndim != 2 \
                or V.ndim != 2 \
                or self.L.shape[1] != Sigma.shape[0] \
                or self.R.shape[1] != Sigma.shape[1] \
                or self.L.shape[0] != U.shape[1] \
                or self.R.shape[0] != V.shape[1] \
                or U.shape[0] != z.shape[0] \
                or V.shape[0] != d.shape[0]:
            raise ValueError("Unable to handle the input!")

        m = self.L.T @ (U.T @ z)
        p = z - U @ (self.L @ m)
        n = self.R.T @ (V.T @ d)
        q = d - V @ (self.R @ n)

        p_l2 = Backend.linalg.norm(p)
        q_l2 = Backend.linalg.norm(q)

        K = extend_with_000(Sigma) + Backend.vstack((m, p_l2)
                                                    ) @ Backend.vstack((n, q_l2)).T

        L_, Sigma, R_ = Backend.linalg.svd(K)
        Sigma = Backend.diagflat(Sigma)
        R_ = R_.T


        self.L = extend_with_010(self.L) @ L_
        self.R = extend_with_010(self.R) @ R_
        # Takes zero vector if the vector is infinitesimal, as it doesn't affects the Moore-Penrose inverse
        p = p / p_l2 if p_l2 > meta_data["rcond"] else Backend.zeros_like(p)
        q = q / q_l2 if q_l2 > meta_data["rcond"] else Backend.zeros_like(q)
        U = Backend.hstack((U, p))
        V = Backend.hstack((V, q))

        if self.L.shape[1] >= 2 * self.k:
            Sigma = Sigma[:self.k, :self.k]
            U = U @ self.L
            U = U[:, :self.k]
            V = V @ self.R
            V = V[:, :self.k]
            self.L, self.R = Backend.eye(self.k), Backend.eye(self.k)

        return U, Sigma, V

    @learn_func_wrapper
    def learn(
            self,
            observation: Matrix,
            next_observation: Matrix,
            reward: Decimal,
            discount: Decimal,
            t: int
    ) -> Any:
        beta = 1 / (t + 1)
        delta = reward + discount * self.w @ next_observation - self.w @ observation
        self.e = self.trace_update(self, observation, discount, self.e, self.lambd)

        self.U, self.Sigma, self.V = \
            self.svd_update(
                self.U,
                (1 - beta) * self.Sigma,
                self.V,
                sqrt(beta) * self.e.reshape((self.observation_space_n, 1)), #z
                sqrt(beta) * (observation - discount *
                              next_observation).reshape((self.observation_space_n, 1)) #d
            )  # Uses SVD update to reduce the complexity, enhancing the performance

        self.w += (self.lr *
                   Backend.linalg.pinv(self.U @ self.L @ self.Sigma @ (self.V @ self.R).T,
                                       rcond=meta_data["rcond"]) +
                   self.eta *
                   Backend.eye(self.observation_space_n)) @ (delta * self.e)

        return delta


class tLSTDAgent(SVDATDAgent):
    """
    tLSTDAgent
    ======

    tLSTD
    Parameters
    ------
    k :
        rank
    svd_diagonalizing :
        Decides whether to use svd decomposition to diagonalize the matrix with orthogonality. Default is `False`
    w_update_emphasizes :
        Decides which one comes first when updating the weight. Should be one of ``accuracy | complexity``
    """

    def __init__(self, k: int,
                  **kwargs) -> None:
        super().__init__(**kwargs)
        if not (isinstance(k, int) ):
            raise TypeError("Invalid input type!")

        self.k = k

    def reinit(self) -> None:
        super(tLSTDAgent, self).reinit()
        self.L, self.R = Backend.empty((0, 0)), Backend.empty((0, 0))
        self.b = Backend.empty(self.observation_space_n)

    def svd_update(
            self,
            U: Matrix,
            Sigma: Matrix,
            V: Matrix,
            z: Matrix,
            d: Matrix
    ) -> Tuple[Matrix, Matrix, Matrix]:
        try:
            U, Sigma, V, z, d = Backend.convert_to_matrix_func(U), Backend.convert_to_matrix_func(
                Sigma), Backend.convert_to_matrix_func(V), Backend.convert_to_matrix_func(
                z), Backend.convert_to_matrix_func(d)
        except TypeError:
            warnings.warn("Wrong input type!")
            return U, Sigma, V
        if U.ndim != 2 \
                or Sigma.ndim != 2 \
                or V.ndim != 2 \
                or self.L.shape[1] != Sigma.shape[0] \
                or self.R.shape[1] != Sigma.shape[1] \
                or self.L.shape[0] != U.shape[1] \
                or self.R.shape[0] != V.shape[1] \
                or U.shape[0] != z.shape[0] \
                or V.shape[0] != d.shape[0]:
            raise ValueError("Unable to handle the input!")

        m = self.L.T @ (U.T @ z)
        p = z - U @ (self.L @ m)
        n = self.R.T @ (V.T @ d)
        q = d - V @ (self.R @ n)

        p_l2 = Backend.linalg.norm(p)
        q_l2 = Backend.linalg.norm(q)

        K = extend_with_000(Sigma) + Backend.vstack((m, p_l2)
                                                    ) @ Backend.vstack((n, q_l2)).T

        L_, Sigma, R_ = Backend.linalg.svd(K)
        Sigma = Backend.diagflat(Sigma)
        R_ = R_.T

        self.L = extend_with_010(self.L) @ L_
        self.R = extend_with_010(self.R) @ R_
        # Takes zero vector if the vector is infinitesimal, as it doesn't affects the Moore-Penrose inverse
        p = p / p_l2 if p_l2 > meta_data["rcond"] else Backend.zeros_like(p)
        q = q / q_l2 if q_l2 > meta_data["rcond"] else Backend.zeros_like(q)
        U = Backend.hstack((U, p))
        V = Backend.hstack((V, q))

        if self.L.shape[1] >= 2 * self.k:
            Sigma = Sigma[:self.k, :self.k]
            U = U @ self.L
            U = U[:, :self.k]
            V = V @ self.R
            V = V[:, :self.k]
            self.L, self.R = Backend.eye(self.k), Backend.eye(self.k)

        return U, Sigma, V

    import numpy as np

    def compute_weights(self, U, Sigma, V, L, R, b):
        # Solve Ax = b where A = ULΣR>V> and A^-1 = VRΣ^-1L>U>

        # Step 1: Compute b~ = L>Ub
        b_tilde = L.T @ U.T @ b

        # Step 2: Determine which singular values to invert
        sigma_hat = Sigma[0, 0]  # Estimated largest singular value
        sigma_inv = Backend.zeros_like(Sigma)  # Initialize inverted singular values matrix
        for i in range(Sigma.shape[0]):
            if Sigma[i, i] > 0.01 * sigma_hat:
                sigma_inv[i, i] = 1 / Sigma[i, i]
            else:
                break

        # Step 3: Compute weights w = VRΣ^-1b~
        w = V @ sigma_inv @ (R.T @ b_tilde)

        return w

    @learn_func_wrapper
    def learn(
            self,
            observation: Matrix,
            next_observation: Matrix,
            reward: Decimal,
            discount: Decimal,
            t: int
    ) -> Any:

        beta = 1 / (t + 1)
        delta = reward + discount * self.w @ next_observation - self.w @ observation
        self.e = self.trace_update(self, observation, discount, self.e, self.lambd)

        self.U, self.Sigma, self.V = \
            self.svd_update(
                self.U,
                (1 - beta) * self.Sigma,
                self.V,
                sqrt(beta) * self.e.reshape((self.observation_space_n, 1)),
                sqrt(beta) * (observation - discount *
                              next_observation).reshape((self.observation_space_n, 1))
            )  # Uses SVD update to reduce the complexity, enhancing the performance
        self.b = (1-beta)*self.b + beta*delta*self.k

        self.w = self.compute_weights(self.U, self.Sigma, self.V, self.L, self.R, self.b)


        return delta




print(
    """
    ATD algorithm module has been ready.
    """.strip()
)

if __name__ == "__main__":
    from atd import TDAgent, SVDATDAgent, DiagonalizedSVDATDAgent, PlainATDAgent
