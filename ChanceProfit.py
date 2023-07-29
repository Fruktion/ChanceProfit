"""
Exercise for Sylwia, chance-profit.
"""

from __future__ import annotations
from typing import NoReturn, final, Final, Self

import numpy as np

import numba
import random
import concurrent.futures
import multiprocessing
import multiprocessing.sharedctypes
import plotly
import os
import numbers
import enum
import time
import warnings
import tqdm
import types


class ProgressBar:

    """
    Progress bar class showing the progress bar in the console.
    """

    def __init__(self, total: int | float = 100, description: str | None = None) -> None:

        """
        Constructor for the ProgressBar class.

        Args:
            total (int | float): The total value to be achieved by the progress bar. 100 by default.
            description (str): Description to the progress bar shown on the left. None by default.

        Raises:
            AssertionError: If the type of any argument does not match the correct one.
        """

        assert isinstance(description, str) or description is None, f"description type should match either " \
                                                                    f"{str.__name__} or {types.NoneType.__name__}. " \
                                                                    f"{type(description).__name__} given instead."

        assert isinstance(total, (int, float)), f"total type should match either {int.__name__} or {float.__name__}. " \
                                                f"{type(total).__name__} given instead."

        self.__progress_bar: tqdm.tqdm = tqdm.tqdm(total=total, desc=description)

    @property
    def progress_bar(self) -> tqdm.tqdm:

        """
        Getter for the self.__progress_bar attribute.

        Returns:
            tqdm.tqdm: The value of the self.__progress_bar attribute.
        """

        return self.__progress_bar

    def increase(self, value: int | float = 1) -> None:

        """
        Method used for increasing the value of the progress bar.

        Args:
            value (int | float): The value to be added to the current progress value. 1 by default.

        Raises:
            AssertionError: If the value argument is not int or float.
        """

        assert isinstance(value, (int, float)), f"value type should match either {int.__name__} or {float.__name__}. " \
                                                f"{type(value).__name__} given instead."

        self.progress_bar.update(value)

    def __add__(self, other: int | float) -> Self:

        """
        Overriden special method that makes possible increasing the progress bar value using the "+" operator.

        Args:
            other (int | float): A value to be added to current progress on the progress bar. Must be positive.

        Examples:
            "bar + 5" = "bar.increase(5)"
        """

        self.increase(value=other)

        return self

    def __del__(self) -> None:

        """
        Overriden __del__ special method used for object deletion. Automatically called by the garbage collector.
        In here closes the progress bar by calling the self.progress_bar.close() method.
        """

        self.progress_bar.close()


class OrdinalNumbers(enum.Enum):

    """
    Class for ordinal numbers.
    """

    ST: Final[int] = 1
    ND: Final[int] = 2
    RD: Final[int] = 3
    TH: Final[set[int]] = {0, 4, 5, 6, 7, 8, 9}

    def __str__(self) -> str:

        """
        Overriden special method to provide lower-case members' names.

        Returns:
            str: The lower-case member name.

        Examples:
            # Proper use:
            >>> print(OrdinalNumbers(7))
            th

            # The following syntax returns upper-case member name:
            >>> print(OrdinalNumbers(7).name)
            TH
        """

        return self.name.lower()

    @classmethod
    def _missing_(cls, key: int) -> Self:

        """
        Overriden special class method to find the unspecified key.

        Args:
            key (int): The key of the enum.Enum class element.

        Returns:
            Self: The OrdinalNumbers class object.
        """

        assert isinstance(key, int), f"key should be of type {int.__name__}. {type(key).__name__} given instead."

        for member in cls.__members__.values():
            match isinstance(member.value, (set, tuple)):
                case True:
                    if key in member.value:
                        return member
                    else:
                        continue
                case False:
                    if key == member.value:
                        return member
                    else:
                        continue
        else:
            raise ValueError(f"Such element does not exist in the {cls.__name__} class.")


class NumericalTools:

    """
    Class implementing methods for numerical functions and tools.
    """

    numeric_types: set[type] = {numbers.Number, np.number}

    @classmethod
    def is_numeric(cls, obj: object) -> bool:

        """
        Class Method for determining if the given object is of the numerical type (numbers.Number, np.number).

        Args:
            obj (object): Object that is going to be checked for its numerical properties.

        Returns:
            bool: True - if the given object is either int, np.int16, float, np.float64, (...).
        """

        return isinstance(obj, tuple(cls.numeric_types))

    @staticmethod
    def rescale_values(values: list[int | float] | tuple[int | float], lower_bound: float,
                       higher_bound: float) -> list[int | float]:
        """
        Static method prescaling the iterable of values to the given range.

        Args:
            values (list[int | float] | tuple[int | float]): The iterable of values to be rescaled.
            lower_bound (int | float): The minimum value of the rescaled sequence.
            higher_bound (int | float): The maximum value of the rescaled sequence.

        Returns:
            list[int | float]: The list of rescaled values from the input iterable.
        """

        min_val: float = min(values)
        amplitudes_ratio: float = (higher_bound - lower_bound) / (max(values) - min_val)

        return [(value - min_val) * amplitudes_ratio + lower_bound for value in values]

    @classmethod
    def seq_n_diff(cls, sequence: list[int | float] | tuple[int | float], n: int = 1) -> tuple[int | float]:

        """
        Class method for calculation of the sequence n-difference.

        Args:
            sequence (list[int | float] | tuple[int | float]): The sequence for for calculation.
            n (int): The order of the difference. 1 by default.

        Returns:
            tuple[int | float]: The tuple of the n-difference sequence.

        Raises:
            AssertionError: If any of the following occurs:
                - sequence is not a list nor a tuple,
                - sequence length is not higher than 1,
                - at least one element in the sequence is not int nor a float,
                - order of the sequence "n" is not an int,
                - order of the sequence "n" is higher or equal to the number of elements in the sequence,
                - order of the sequence "n" is lower or equal to 0.
        """

        assert isinstance(sequence, (list, tuple)), f"sequence should be of type {list.__name__} or {tuple.__name__}." \
                                                    f" {type(sequence).__name__} given instead."

        assert len(sequence) > 1, f"Number of elements in the sequence should be greater or equal to 2. " \
                                  f"Current cardinality: {len(sequence)}."

        assert all(
            cls.is_numeric(element) for element in sequence
        ), f"Each element in the sequence should be of type either {numbers.Number.__name__} or {np.number.__name__}." \
           f" At least one element is of type " \
           f"{type(list(filter(lambda element: type(element) not in cls.numeric_types, sequence))[0]).__name__}."

        assert isinstance(n, int), f"n should be of type {int.__name__}. {type(n).__name__} given instead."

        assert n < len(sequence), f"n should be lower than number of elements in the sequence. n: {n}, number of " \
                                  f"elements in the sequence: {len(sequence)}."

        assert n > 0, f"n should be higher than 0. {n} given instead."

        difference_sequence: list[int | float] = [
            element - prev_element for prev_element, element in zip(sequence, sequence[1:])
        ]

        for _ in range(n - 1):
            difference_sequence: list[int | float] = [
                element - prev_element for prev_element, element in zip(difference_sequence, difference_sequence[1:])
            ]

        return tuple(difference_sequence)

    @staticmethod
    def scientific_notation(number: int | float) -> str:

        """
        Method for conversion of a number to a string with its scientific notation.

        Args:
            number (int | float): The number to be converted.

        Returns:
            str: The number represented in the scientific notation.

        Raises:
            AssertionError: If the number type is not int nor float.
        """

        assert isinstance(number, (int, float)), f"number should be of type {int.__name__} or {float.__name__}. " \
                                                 f"{type(number).__name__} given instead."

        power: int = 0
        while True:
            try:
                number: float = float(number / 10 ** power)
            except OverflowError:
                power += 1
                continue

            number: str = str(number)
            e_plus_pos: int = number.find('e+')
            if e_plus_pos == -1:
                return number
            else:
                return number[:e_plus_pos] + 'e+' + str(int(number[e_plus_pos + 1:]) + power)

    @staticmethod
    def integer_multiply(first_element: int | float, float_element: float) -> int | float:

        """
        Method multiplying two numbers together. Useful for calculating of big numbers that need to be expressed as
        integers due to floating-point numbers 64-bit limits, where 11 bits are used for exponent storing.

        Args:
            first_element (int | float): The integer or float number to multiply with the floating-point number.
            float_element (float): The floating-point number to multiply with the integer number.

        Returns:
            int: The integer being a result of the multiplication with the limited precision (the decimal points are cut
            off).

        Raises:
            AssertionError: If the first_element is not of type int nor float or float_element is not of type float.
        """

        assert isinstance(first_element, (int, float)), f"first_element should be of type {int.__name__} or " \
                                                        f"{float.__name__}. {type(first_element).__name__} " \
                                                        f"given instead."

        assert isinstance(float_element, float), f"float_element should be of type {float.__name__}. " \
                                                 f"{type(float_element).__name__} given instead."

        try:
            default_multiply: float = first_element * float_element
            if default_multiply == np.inf:
                pass
            else:
                return default_multiply
        except OverflowError:
            pass

        float_element: str = str(float_element)
        dot_position: int = float_element.find('.')

        fractional_part_str: str = float_element[dot_position + 1:]
        return (
            int(first_element) * int(float_element[:dot_position])
        ) + (
            int(first_element) * int(fractional_part_str) // 10 ** len(fractional_part_str)
        )


class ValuesPlotter:

    """
    Class used to create a plot using given lists of real values.

    This class defines a constructor to create a plot using a list or tuple of real numbers passed as arguments.
    Plotly is used to plot these values, and an HTML file is created for the plot.

    Attributes:
        css_colors (tuple[str]): The constant tuple of the available colors.

    Constructor Args:
        values (list[list[int | float]] | tuple[list[int | float], ...]): A list or tuple of lists of real numbers for
         plotting.
        title (str): Title for the plot. Default is 'Plot'.
        additional_lines (list[int | float] | tuple[int | float] | None): Optional list or tuple of real numbers to
        include as additional lines in the plot. Default is None.
        background_color (str): Background color for the plot. Default is dark gray ('rgb(145, 145, 145)').
    """

    css_colors: Final[tuple[str]] = ('aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
                                     'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood',
                                     'chartreuse', 'chocolate', 'coral', 'cornsilk', 'crimson', 'cyan', 'darkblue',
                                     'darkgray', 'darkgrey', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen',
                                     'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
                                     'darkslateblue', 'darkslategray', 'darkslategrey', 'darkviolet', 'deeppink',
                                     'dimgray', 'dimgrey', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia',
                                     'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'green', 'greenyellow', 'honeydew',
                                     'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush',
                                     'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
                                     'lightgoldenrodyellow', 'lightgray', 'lightgrey', 'lightgreen', 'lightpink',
                                     'lightsalmon', 'lightskyblue', 'lightsteelblue', 'lightyellow', 'lime', 'linen',
                                     'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid',
                                     'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
                                     'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'orange',
                                     'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'papayawhip',
                                     'peachpuff', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown',
                                     'royalblue', 'rebeccapurple', 'saddlebrown', 'salmon', 'sandybrown', 'seashell',
                                     'sienna', 'silver', 'skyblue', 'slateblue', 'snow', 'springgreen', 'tan',
                                     'thistle', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow',
                                     'yellowgreen')

    def __init__(self, values: list[list | tuple] | tuple[list, tuple, ...], title: str = 'Plot',
                 additional_lines: list[int | float] | tuple[int | float] | None = None,
                 background_color: str = 'rgb(145, 145, 145)') -> None:

        """
        The constructor creates a plot of the values given as arguments.

        Args:
            values: A list or a tuple of lists of real values that are going to be plotted.
            background_color (str): Background color for the plot. Default is gray ('rgb(145, 145, 145)').

        Raises:
            AssertionError: If values is not a list or tuple, or if any element it contains is not a real number.
        """

        assert isinstance(values, (list, tuple)), f"values is expected to be of type {list.__name__} or " \
                                                  f"{tuple.__name__}. {type(values).__name__} given instead."

        for sublist in values:
            assert isinstance(sublist, (list, tuple)), f"Each element in values should be of type {list.__name__} or " \
                                                       f"{tuple.__name__}. {type(sublist).__name__} detected."

            for element in sublist:
                assert isinstance(element, (int, float)), f"Every element in the given values should be of type " \
                                                          f"{int.__name__} or {float.__name__}. " \
                                                          f"{type(element).__name__} detected."

        if additional_lines is not None:
            assert isinstance(additional_lines, (tuple, list)), f"additional_lines should be of type {list.__name__} " \
                                                                f"or {tuple.__name__}. " \
                                                                f"{type(additional_lines).__name__} given instead."

            for const_func_value in additional_lines:
                values.append(list())
                for _ in range(len(values[0])):
                    values[-1].append(const_func_value)

        assert isinstance(background_color, str), f"background color should be of type {str.__name__}. " \
                                                  f"{type(background_color).__name__} given instead."

        self.__fig: plotly.graph_objects.Figure = plotly.graph_objects.Figure(
            data=[
                plotly.graph_objects.Scatter(
                    y=sublist,
                    line=plotly.graph_objects.scatter.Line(
                        color=random.choice(self.css_colors)
                    )
                )
                for sublist in values
            ]
        ).update_layout(
            title=title,
            plot_bgcolor=background_color
        )

        self.__title: str = title
        self.__background_color: str = background_color

    @property
    def fig(self) -> plotly.graph_objects.Figure:

        """
        Getter for the self.__fig attribute

        Returns:
            plotly.graph_objects.Figure: the value of the self.__fig attribute
        """

        return self.__fig

    @property
    def title(self) -> str:

        """
        Getter for the self.__title attribute

        Returns:
            str: the value of the self.__title attribute
        """

        return self.__title

    @property
    def background_color(self) -> str:

        """
        Getter for the self.__background_color attribute

        Returns:
            str: the value of the self.__background_color attribute
        """

        return self.__background_color

    @background_color.setter
    def background_color(self, color: str) -> None:

        """
        Setter for the self.__background_color attribute.

        Args:
            color (str): New background color to be set.
        """

        assert isinstance(color, str), f"background_color should be of type {str.__name__}. " \
                                       f"{type(color).__name__} given instead."
        self.__background_color: str = color
        self.__fig.update_layout(plot_bgcolor=color)

    def create(self, show: bool = False, log_scale: bool = False, background_color: str | None = None) -> None:
        """
        Method used for showing the graph and saving it to the file.

        Args:
            show (bool): Specifies if the created chart should be instantly open in the default browser immediately
            after creation. False by default.
            log_scale (bool): Specifies if the y-axis should have a logarithmic scale. False by default.
            background_color (str): Background color for the plot. If provided, it overrides the color set in the
            constructor.

        Raises:
            AssertionError: If the type of the show or log_scale arguments is not bool or the type of the
            background_color is not str.
        """

        assert isinstance(show, bool), f"show argument should be of type {bool.__name__}. " \
                                       f"{type(show).__name__} given instead."

        assert isinstance(log_scale, bool), f"log_scale argument should be of type {bool.__name__}. " \
                                            f"{type(log_scale).__name__} given instead."

        assert isinstance(background_color, str) or background_color is None, f"background_color should be of type " \
                                                                              f"{str.__name__} or be {None}. " \
                                                                              f"{type(background_color).__name__} " \
                                                                              f"with value {background_color} given " \
                                                                              f"instead."

        if background_color is not None:
            self.background_color: str = background_color

        if log_scale:
            self.__fig.update_yaxes(type="log")

        self.__fig.write_html(f"{self.title}.html", auto_open=show)


class FileManager:

    """
    Class for managing files.
    """

    @staticmethod
    def delete_file_endswith(endswith: str) -> None:

        """
        Method for file deletion within the base project directory.

        Args:
            endswith (str): The extension of the files to delete. E.g. ".html".
        """

        current_directory = os.getcwd()
        for filename in os.listdir(current_directory):
            if filename.endswith(endswith):
                os.remove(os.path.join(current_directory, filename))


class ChanceProfit:

    """
    Class for testing the chance-profit.
    """

    @staticmethod
    @numba.jit(forceobj=True)
    def __test(prev_max: float, invest_part: float) -> tuple[int, list[int | float]]:

        """
        Class method for testing the strategy.

        Args:
            prev_max (float): Previously calculated max_value.
            invest_part (float): Part of the whole wallet to invest.

        Returns:
            tuple[int, list[float]]: The number of trials and the list of the wallet balance if function of time.

        Notes:
            Method is compiled with Numba in order to provide higher performance.
        """

        assert isinstance(prev_max, float), f"prev_max should be of type {float.__name__}. " \
                                            f"{type(prev_max).__name__} given instead."

        assert isinstance(invest_part, float), f"invest_part should be of type {float.__name__}. " \
                                               f"{type(invest_part).__name__} given instead."

        """
        Integer_multiply:
        try:
            default_multiply: float = first_element * float_element
            if default_multiply == np.inf:
                pass
            else:
                return default_multiply
        except OverflowError:
            pass

        float_element: str = str(float_element)
        dot_position: int = float_element.find('.')

        fractional_part_str: str = float_element[dot_position + 1:]
        return (
            int(first_element) * int(float_element[:dot_position])
        ) + (
            int(first_element) * int(fractional_part_str) // 10 ** len(fractional_part_str)
        )
        """

        success_chance: Final[float] = 0.5
        profit_after_success: Final[float] = 2
        kill_border: Final[float] = 1 - invest_part

        trials_counter: int = 0
        wallet_balances: list[int | float] = [0.0]

        while max(wallet_balances) <= prev_max:

            wallet_balances: list[int | float] = [1.0]
            trials_counter += 1
            while wallet_balances[-1] > kill_border and len(wallet_balances) <= 10 ** 6:

                if np.random.random() <= success_chance:
                    wallet_balances.append(
                        NumericalTools.integer_multiply(
                            wallet_balances[-1],
                            1 + invest_part * profit_after_success
                        )
                    )
                else:
                    wallet_balances.append(
                        NumericalTools.integer_multiply(
                            wallet_balances[-1],
                            1 - invest_part
                        )
                    )

        return trials_counter, wallet_balances

    def _constant_run(self, max_value: multiprocessing.sharedctypes.Synchronized[float],
                      trials_counter: multiprocessing.sharedctypes.Synchronized[int],
                      lock: multiprocessing.Lock, invest_part: float, max_value_limit: float) -> None | NoReturn:

        """
        Method for the constant run of the ChanceProfit.test() method.

        Args:
            max_value (multiprocessing.sharedctypes.Synchronized[float]): Shared between processes variable that stores
            the information about the maximal achieved value in the simulation.
            trials_counter (multiprocessing.sharedctypes.Synchronized[int]): Shared between processes variable that
            stores the information about the total number of taken trials.
            lock (multiprocessing.Lock): Lock not to lead to race conditions.
            invest_part (float): The constant describing what part of the wallet is supposed to be used in the
            simulation.
            max_value_limit (float): The upper limit of the max_value to be reached until the method returns. If 'inf'
            then the _constant_run is going to run indefinitely.
        """

        # Ignore the warning "RuntimeWarning: overflow encountered in scalar multiply
        #   default_multiply: float = first_element * float_element" so the output is clear of it.
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in scalar multiply")

        # Apply the same as above to the following warning "RuntimeWarning: overflow encountered in multiply
        #   default_multiply: float = first_element * float_element".
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in multiply")

        trials: int = ...
        simulation_wallet_balances: list[int | float] = ...
        infinite_run_condition: bool = ...

        match max_value_limit == np.inf:
            case True:
                infinite_run_condition: bool = True
            case False:
                infinite_run_condition: bool = False

        while infinite_run_condition or max_value.value <= max_value_limit:
            while True:
                try:
                    trials, simulation_wallet_balances = self.__test(max_value.value, invest_part)
                except MemoryError:
                    continue
                else:
                    break

            with lock:
                trials_counter.value += trials

            if (max_balance := max(simulation_wallet_balances)) > max_value.value:

                with lock:
                    max_value.value = max_balance

    def run_test(self) -> None:

        """
        Method for running the ChanceProfit.test() method concurrently.
        """

        max_value_limit: Final[int] = 10 ** 100
        max_timeout: Final[int] = 60  # 60 seconds

        manager: multiprocessing.Manager = multiprocessing.Manager()
        lock: multiprocessing.Lock = manager.Lock()

        invest_part_step: float = 0.01
        invest_part_vals: list[float] = [
            round(
                number,
                len(str(invest_part_step)[str(invest_part_step).find('.'):])
            )
            for number in np.arange(0, 1, invest_part_step)
        ]
        repeat_to_minimize_errors_range: range = range(10)
        time_invest_part_dict: dict[float, float] = {invest_part: 0 for invest_part in invest_part_vals}
        progress_bar: ProgressBar = ProgressBar(
            total=len(invest_part_vals) * len(repeat_to_minimize_errors_range),
            description="Invest_part progress"
        )

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for invest_part in invest_part_vals:
                for _ in repeat_to_minimize_errors_range:
                    max_value: multiprocessing.sharedctypes.Synchronized[float] = manager.Value('d', 0.0)
                    trials_counter: multiprocessing.sharedctypes.Synchronized[int] = manager.Value('i', 0)
                    start_time: float = time.perf_counter()
                    futures: list[concurrent.futures.Future[None | NoReturn]] = [
                        executor.submit(
                            self._constant_run,
                            max_value=max_value,
                            trials_counter=trials_counter,
                            lock=lock,
                            invest_part=invest_part,
                            max_value_limit=max_value_limit
                        )
                        for _ in range(os.cpu_count())
                    ]

                    try:

                        done_and_not_done_futures = concurrent.futures.wait(
                            fs=futures,
                            timeout=max_timeout
                        )
                        for completed_future, not_completed_future in zip(
                                done_and_not_done_futures.done,
                                done_and_not_done_futures.not_done
                        ):
                            completed_future.result()
                            not_completed_future.cancel()

                    except concurrent.futures.TimeoutError:
                        pass

                    time_invest_part_dict[invest_part] += time.perf_counter() - start_time
                    progress_bar.increase()

        print(f"\nBest time "
              f"{(t_i_p_d_val := list(time_invest_part_dict.values()))[min_time_arg := np.argmin(t_i_p_d_val)]} "
              f"for invest_part: {list(time_invest_part_dict.keys())[min_time_arg]}.")

        ValuesPlotter(
            values=[
                list(time_invest_part_dict.values())
            ],
            title='t_in_f_of_invest_part',
            additional_lines=[60]
        ).create(True)


@final
class Main:

    """
    The main class for the whole program execution. Everything except the program entry point should be run within this
    class.

    Notes:
        This class shouldn't be inherited.
    """

    @classmethod
    def main(cls) -> None:

        """
        The main method for the whole program execution. Everything that is done in the program shall be run within
        this method. This rule applies to all other method calls.
        """

        # Delete all existing .html files in the directory.
        FileManager.delete_file_endswith('.html')

        # Run the simulation.
        ChanceProfit().run_test()


if __name__ == '__main__':

    """
    The entry point of the whole program.
    If any uncaught exception is raised during runtime, then it is going to be caught and re-raised here. Such behavior
    is recommended when it comes to exception handling of exceptions raised during execution on many threads.
    """

    try:
        Main.main()
    except Exception as main_exception:
        raise main_exception
