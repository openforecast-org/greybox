"""Formula parser for greybox models.

This module provides formula parsing functionality similar to R's formula system.
Users can call the formula function to get X (design matrix) and y, then pass
them to the model fit method.
"""

import inspect
import re
from typing import Any

import numpy as np
import pandas as pd

from .xreg import B  # re-exported for convenience

TRANSFORMATIONS = {
    "log": np.log,
    "log10": np.log10,
    "log2": np.log2,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
}


def _get_caller_globals():
    """Get globals from the caller's caller (the user's code).

    Returns
    -------
    dict
        Global namespace from the user's code.
    """
    frame = inspect.currentframe()
    if frame is None:
        return {}
    try:
        caller_frame = frame.f_back.f_back
        if caller_frame is not None:
            return caller_frame.f_globals
        return {}
    finally:
        del frame


def _apply_transformation(var_name, data_dict, n_obs, caller_globals=None):
    """Evaluate a variable of the formula, as R's model.frame() does.

    Parameters
    ----------
    var_name : str
        Variable in R syntax, e.g. "log(x)", "x^2", "I(x * z)" or "B(x,1)".
    data_dict : dict
        Data dictionary.
    n_obs : int
        Number of observations.
    caller_globals : dict, optional
        Global namespace from user's code to resolve custom functions.

    Returns
    -------
    np.ndarray
        Transformed variable values.
    """
    namespace = {**(caller_globals or {}), **TRANSFORMATIONS, "I": np.asarray, "B": B}
    namespace["trend"] = np.arange(1, n_obs + 1, dtype=float)

    # The columns of the data, under names that Python can evaluate (e.g. "x.1")
    expression = var_name
    for i, name in enumerate(data_dict):
        pattern = rf"(?<![\w.]){re.escape(name)}(?![\w.]|\s*\()"
        if re.search(pattern, expression):
            expression = re.sub(pattern, f"__data{i}__", expression)
            namespace[f"__data{i}__"] = np.asarray(data_dict[name], dtype=float)

    try:
        return np.asarray(eval(expression.replace("^", "**"), namespace), dtype=float)
    except NameError as error:
        if re.search(rf"\b{error.name}\s*\(", var_name):
            raise ValueError(
                f"Unknown function '{error.name}' in transformation '{var_name}'. "
                f"Either use a built-in transformation (log, sqrt, exp, etc.) or "
                f"make sure '{error.name}' is defined or imported in your global "
                f"scope."
            ) from None
        raise ValueError(f"Variable '{error.name}' not found in data") from None


def formula(
    formula_str: str,
    data,
    return_type: str = "both",
    as_dataframe: bool = True,
) -> Any:
    """Parse formula string and return design matrix and/or response.

    The response and the design matrix are those of R's model.frame() and
    model.matrix() with the treatment contrasts.

    Parameters
    ----------
    formula_str : str
        Formula string in R-style, e.g., "y ~ x1 + x2" or "~ x1 + x2" (no y).
        Supports "." (all the other columns of data), "- 1" and "0 +" (no
        intercept), interactions "x:z" and "x * z", transformations log(x),
        sqrt(x), x^2, etc. Use I() to protect expressions: I(x^2), I(x * z).
        Categorical, object and string columns are factors, coded with the
        treatment contrasts, as R does (the ordered ones as the unordered).
    data : dict or DataFrame
        Data containing variables. Can be dict, DataFrame, or dict of arrays.
    return_type : str, optional
        What to return: "both" (default), "X", "y", "variables" or "terms".
    as_dataframe : bool, optional
        If True, return pandas DataFrames instead of numpy arrays.
        This preserves variable names in column names. Default is True.

    Returns
    -------
    Depends on return_type and as_dataframe:
        - "both": tuple (y, X) where y is 1D array (or Series)
          and X is 2D array (or DataFrame), with intercept unless removed
        - "X": design matrix only (or DataFrame if as_dataframe=True)
        - "y": response only
        - "variables": names of the columns of X, except the intercept
        - "terms": the labels of the terms, as R's attr(terms, "term.labels")

    Examples
    --------
    >>> data = {'y': [1, 2, 3], 'x1': [4, 5, 6], 'x2': [7, 8, 9]}
    >>> y, X = formula("y ~ x1 + x2", data)
    >>> X_only = formula("~ x1 + x2", data, return_type="X")

    >>> # With transformations: y is log-transformed, X has x and x^2
    >>> data_t = {'y': [1, 2, 3], 'x': [1, 2, 3]}
    >>> y, X = formula("log(y) ~ x + x^2", data_t)

    >>> # Return as DataFrames with column names
    >>> y, X = formula("y ~ x1 + x2", data, as_dataframe=True)
    >>> list(X.columns)
    ['(Intercept)', 'x1', 'x2']

    >>> # All the other variables, without the intercept
    >>> list(formula("y ~ 0 + .", data)[1].columns)
    ['x1', 'x2']

    >>> # Factors and interactions
    >>> data_f = {'y': [1, 2, 3, 4], 'x': [1, 2, 3, 5], 'g': ['a', 'b', 'c', 'a']}
    >>> list(formula("y ~ x * g", data_f)[1].columns)
    ['(Intercept)', 'x', 'gb', 'gc', 'x:gb', 'x:gc']

    >>> # With custom functions (defined or imported in your global scope)
    >>> def my_transform(x):
    ...     return x * 2
    >>> y, X = formula("y ~ my_transform(x)", data_t)

    >>> # With imported functions (e.g., from scipy)
    >>> from scipy.special import erfc
    >>> data_e = {'y': [1, 2, 3], 'x': [0.5, 1.0, 1.5]}
    >>> y, X = formula("y ~ erfc(x)", data_e)

    >>> # Custom function on LHS (response variable)
    >>> y, X = formula("my_transform(y) ~ x", data_t)
    """
    formula_str = formula_str.strip()

    lhs = ""
    if "~" in formula_str:
        lhs, formula_str = (part.strip() for part in formula_str.split("~", 1))
    if not formula_str:
        formula_str = "1"

    if isinstance(data, dict):
        data_dict = {str(name): values for name, values in data.items()}
    else:
        try:
            data_dict = {str(name): data[name] for name in data.columns}
        except AttributeError:
            raise ValueError("data must be dict or DataFrame") from None

    caller_globals = _get_caller_globals()
    n_obs = len(next(iter(data_dict.values())))

    # The variables as R deparses them: spaces around +, -, * and comparisons
    chunks = _parse_formula_terms(formula_str)
    canonical: dict[str, str] = {}
    for variable in [lhs] + [v for _, groups in chunks for g in groups for v in g]:
        name = variable
        for pattern, replacement in (
            (r"\s+", ""),
            (r"(?<=[\w.)])([+*-])", r" \1 "),
            (r"(==|!=|>=|<=|<|>|&|\|)", r" \1 "),
            (r",", ", "),
            (r"\bB\(([\w.]+), ", r"B(\1,"),
        ):
            name = re.sub(pattern, replacement, name)
        canonical[variable] = name
    lhs = canonical[lhs]

    # The terms, as R's terms(): left to right, then ordered by their degree
    intercept = True
    terms: list[frozenset[str]] = []
    order = [lhs] if lhs else []
    for negative, groups in chunks:
        names = [[canonical[v] for v in group] for group in groups]
        if names in ([["0"]], [["1"]]):
            intercept = (names[0][0] == "1") != negative
            continue
        if names == [["."]]:
            appearing = [
                name
                for name in data_dict
                if not re.search(rf"(?<![\w.]){re.escape(name)}(?![\w.(])", lhs)
            ]
            new_terms = [frozenset([name]) for name in appearing]
        else:
            appearing = [v for group in names for v in group]
            new_terms = [frozenset(names[0])]
            for group in names[1:]:
                new_terms += [frozenset(group)]
                new_terms += [term | frozenset(group) for term in new_terms[:-1]]
        order += [v for v in dict.fromkeys(appearing) if v not in order]
        if negative:
            terms = [term for term in terms if term not in new_terms]
        else:
            terms += [term for term in dict.fromkeys(new_terms) if term not in terms]
    terms = sorted(terms, key=len)

    if return_type == "terms":
        return [":".join(sorted(term, key=order.index)) for term in terms]

    y = None
    if lhs:
        if lhs in data_dict:
            y = np.asarray(data_dict[lhs], dtype=float)
        else:
            y = _apply_transformation(lhs, data_dict, n_obs, caller_globals)

    if return_type == "y":
        return y

    # The columns of each variable: the dummies of all levels for the factors
    blocks: dict[str, list[tuple[str, np.ndarray]]] = {}
    factors: set[str] = set()
    for name in [v for v in order if any(v in term for term in terms)]:
        values = pd.Series(data_dict.get(name, []))
        if name in data_dict and (
            isinstance(values.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(values)
        ):
            categories = values.astype("category").cat
            codes = categories.codes.to_numpy()
            dummies = (codes[:, None] == np.arange(len(categories.categories))) * 1.0
            dummies[codes < 0] = np.nan
            blocks[name] = [
                (
                    name
                    + (
                        str(int(level))
                        if isinstance(level, float) and level.is_integer()
                        else str(level)
                    ),
                    dummies[:, i],
                )
                for i, level in enumerate(categories.categories)
            ]
            factors.add(name)
        elif name in data_dict:
            blocks[name] = [(name, np.asarray(data_dict[name], dtype=float))]
        else:
            blocks[name] = [
                (name, _apply_transformation(name, data_dict, n_obs, caller_globals))
            ]

    # R's model.matrix(): a factor is coded by contrasts if the term without it
    # is in a preceding term; without intercept, the first factor by dummies
    first_factor: tuple[int, str] | None = None
    if not intercept:
        first_factor = next(
            (
                (j, v)
                for j, term in enumerate(terms)
                for v in sorted(term, key=order.index)
                if v in factors
            ),
            None,
        )
    X_columns = [("(Intercept)", np.ones(n_obs))] if intercept else []
    for j, term in enumerate(terms):
        columns = [("", np.ones(n_obs))]
        for name in sorted(term, key=order.index):
            contrasts = (
                name in factors
                and (j, name) != first_factor
                and (len(term) == 1 or any(term - {name} <= t for t in terms[:j]))
            )
            columns = [
                (f"{prefix}:{column}" if prefix else column, values * block)
                for column, block in blocks[name][int(contrasts) :]
                for prefix, values in columns
            ]
        X_columns += columns

    var_names = [name for name, _ in X_columns]
    X = np.column_stack([np.empty((n_obs, 0))] + [column for _, column in X_columns])
    coef_names = [n for n in var_names if n != "(Intercept)"]

    if as_dataframe:
        X_df = pd.DataFrame(X, columns=var_names)

        if return_type == "X":
            return X_df
        elif return_type == "variables":
            return coef_names

        y_df = pd.Series(y, name=lhs) if y is not None else None

        return y_df, X_df

    if return_type == "X":
        return X
    elif return_type == "variables":
        return coef_names

    return y, X


def _parse_formula_terms(formula_str: str) -> list[tuple[bool, list[list[str]]]]:
    """Split the right-hand side of a formula into its summands.

    Returns, for each summand in its order, whether it is subtracted and its
    variables, grouped by "*" and, inside the groups, by ":". The operators in
    parentheses belong to the variables, e.g. log(x + 1), as does the minus of
    a power, e.g. x^-1.
    """
    chunks: list[tuple[bool, list[list[str]]]] = []
    negative = False
    groups: list[list[str]] = [[]]
    variable = ""
    depth = 0
    for char in formula_str + "+":
        depth += (char == "(") - (char == ")")
        if depth or char not in "+-*:" or variable.rstrip().endswith("^"):
            variable += char
            continue
        groups[-1].append(variable.strip())
        variable = ""
        if char in "*:":
            groups += [[]] if char == "*" else []
            continue
        if groups != [[""]]:
            chunks.append((negative, groups))
        negative, groups = char == "-", [[]]

    return chunks


def expand_formula(formula_str):
    """Expand formula with interaction terms.

    Parameters
    ----------
    formula_str : str
        Formula string, e.g., "y ~ x1 * x2"

    Returns
    -------
    str
        Expanded formula with explicit interaction terms.
    """
    if "~" not in formula_str:
        return formula_str

    lhs, rhs = formula_str.split("~", 1)

    rhs = rhs.strip()

    rhs = re.sub(r"(\w+)\s*\*\s*(\w+)", r"\1 + \2 + \1:\2", rhs)

    return f"{lhs.strip()} ~ {rhs}"
