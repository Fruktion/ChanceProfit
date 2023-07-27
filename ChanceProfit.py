"""
Exercise for Sylwia, chance-profit.
"""

from __future__ import annotations
from typing import NoReturn, final, Final

import numpy as np

import numba
import random
import concurrent.futures
import multiprocessing
import multiprocessing.sharedctypes
import plotly
import os


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
    @numba.njit()
    def __test(prev_max: float) -> tuple[int, list[float]]:

        """
        Class method for testing the strategy.

        Args:
            prev_max (float): Previously calculated max_value.

        Returns:
            tuple[int, list[float]]: The number of trials and the list of the wallet balance if function of time.
        """

        success_chance: Final[float] = 0.5
        profit_after_success: Final[float] = 2
        invest_part: Final[float] = 0.5
        kill_border: Final[float] = 0.5

        trials_counter: int = 0
        wallet_balances: list[float] = [0.0]
        while max(wallet_balances) <= prev_max:

            wallet_balances: list[float] = [1.0]
            trials_counter += 1
            while wallet_balances[-1] > kill_border:

                if np.random.random() <= success_chance:
                    wallet_balances.append(wallet_balances[-1] * (1 + invest_part * profit_after_success))
                else:
                    wallet_balances.append(wallet_balances[-1] * (1 - invest_part))

        return trials_counter, wallet_balances

    def _constant_run(self, max_value: multiprocessing.sharedctypes.Synchronized[float],
                      trials_counter: multiprocessing.sharedctypes.Synchronized[int],
                      lock: multiprocessing.Lock) -> NoReturn:

        """
        Method for the constant run of the ChanceProfit.test() method.

        Args:
            max_value (multiprocessing.sharedctypes.Synchronized[float]): Shared between processes variable that stores
            the information about the maximal achieved value in the simulation.
            trials_counter (multiprocessing.sharedctypes.Synchronized[int]): Shared between processes variable that
            stores the information about the total number of taken trials.
            lock (multiprocessing.Lock): Lock not to lead to race conditions.
        """

        create_plot_and_show_value: bool = False

        trials: int = ...
        simulation_wallet_balances: list[float] = ...

        while True:

            while True:
                try:
                    trials, simulation_wallet_balances = self.__test(max_value.value)
                except MemoryError:
                    continue
                else:
                    break

            with lock:
                trials_counter.value += trials

            if (max_balance := max(simulation_wallet_balances)) > max_value.value:

                with lock:
                    max_value.value = max_balance

                create_plot_and_show_value: bool = True

            if create_plot_and_show_value:

                match int((trades_to_max := str(np.argmax(simulation_wallet_balances)))[-1]):
                    case 1:
                        th_trade: str = trades_to_max + 'st'
                    case 2:
                        th_trade: str = trades_to_max + 'nd'
                    case 3:
                        th_trade: str = trades_to_max + 'rd'
                    case _:
                        th_trade: str = trades_to_max + 'th'

                match (trials_counter_value := trials_counter.value):
                    case 1:
                        trial_s: str = 'trial'
                    case _:
                        trial_s: str = 'trials'

                print(f"Max balance out of 1: {max_balance} ({th_trade} trade) after "
                      f"{trials_counter_value} {trial_s}. Max balance / trials: {max_balance / trials_counter_value}.")

                ValuesPlotter(
                    values=[
                        simulation_wallet_balances
                    ],
                    title=f"{max_balance}"
                ).create(log_scale=True)
                create_plot_and_show_value: bool = False

    def run_test(self) -> NoReturn:

        """
        Method for running the ChanceProfit.test() method concurrently.
        """

        manager: multiprocessing.Manager = multiprocessing.Manager()
        max_value: multiprocessing.sharedctypes.Synchronized[float] = manager.Value('d', 0.0)
        trials_counter: multiprocessing.sharedctypes.Synchronized[int] = manager.Value('i', 0)
        lock: multiprocessing.Lock = manager.Lock()

        with concurrent.futures.ProcessPoolExecutor() as executor:

            futures: list[concurrent.futures.Future[NoReturn]] = [
                executor.submit(
                    self._constant_run,
                    max_value=max_value,
                    trials_counter=trials_counter,
                    lock=lock
                )
                for _ in range(os.cpu_count())
            ]

            for future in concurrent.futures.as_completed(futures):
                future.result()


@final
class Main:

    """
    The main class for the whole program execution.
    """

    @classmethod
    def main(cls) -> None:

        """
        The main method for the whole program execution. Everything that is done within the program shall be run within
        this method.
        """

        # Delete all existing .html files in the directory.
        FileManager.delete_file_endswith('.html')

        # Run the simulation.
        ChanceProfit().run_test()


if __name__ == '__main__':

    try:
        Main.main()
    except Exception as main_exception:
        raise main_exception
