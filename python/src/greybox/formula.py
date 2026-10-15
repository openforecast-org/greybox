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

# The logarithm of R, with its base: log10() and log2() for 10 and 2, as R
TRANSFORMATIONS = {
    "log": lambda x, base=np.e: (
        np.log10(x)
        if base == 10
        else np.log2(x)
        if base == 2
        else np.log(x) / np.log(base)
    ),
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
        # A comparison stays logical, a factor of R's model.matrix()
        result = np.asarray(eval(expression.replace("^", "**"), namespace))
        return result if result.dtype == bool else result.astype(float)
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
        intercept), the operators of R's formulas ("+", "-", ":", "*", "/",
        "%in%" and "^", e.g. "(x + z)^2" for x + z + x:z, so that "x^2" is x),
        with brackets, offset() (not a column, as in R's model.matrix()) and
        transformations log(x), log(x, 2), sqrt(x), etc. Use I() to protect
        expressions: I(x^2), I(x * z). Categorical, object and string columns
        are factors, coded with the treatment contrasts, as R does (the ordered
        ones as the unordered); the logical ones are factors with the levels
        FALSE and TRUE.
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
    >>> y, X = formula("log(y) ~ x + I(x^2)", data_t)

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
    # The formula as R deparses it: spaces around +, -, *, = and comparisons, except
    # in the backquoted names
    parts = formula_str.strip().split("`")
    for pattern, replacement in (
        (r"\s+", ""),
        (r"(?<=[\w.)])([+*-])", r" \1 "),
        (r"(==|!=|>=|<=|<|>|&|\|)", r" \1 "),
        (r"(?<![=!<>])=(?!=)", " = "),
        (r",", ", "),
        (r"\bB\(([\w.]+), ", r"B(\1,"),
    ):
        parts[::2] = [re.sub(pattern, replacement, part) for part in parts[::2]]
    formula_str = "`".join(parts)

    lhs = ""
    if "~" in formula_str:
        lhs, formula_str = (part.strip() for part in formula_str.split("~", 1))

    if isinstance(data, dict):
        data_dict = {str(name): values for name, values in data.items()}
    else:
        try:
            data_dict = {str(name): data[name] for name in data.columns}
        except AttributeError:
            raise ValueError("data must be dict or DataFrame") from None

    caller_globals = _get_caller_globals()
    n_obs = len(next(iter(data_dict.values())))

    # "." is all the other variables of the data, as a group
    others = [
        f"`{name}`"
        for name in data_dict
        if not re.search(rf"(?<![\w.]){re.escape(name)}(?![\w.(])", lhs)
    ]
    formula_str = re.sub(
        r"(?<![\w.`])\.(?![\w.`])", f"({' + '.join(others)})", formula_str
    )

    # The terms, as R's terms(): left to right, then ordered by their degree, without
    # the offsets, which are not the columns of R's model.matrix()
    parsed_terms, parsed_intercept, order = _parse_formula_terms(formula_str)
    intercept = parsed_intercept is not False
    terms = sorted(
        (term for term in parsed_terms if not any(v[:7] == "offset(" for v in term)),
        key=len,
    )

    if return_type == "terms":
        return [":".join(sorted(term, key=order.index)) for term in terms]

    # The response is not needed for the design matrix of the new data
    y = None
    if lhs and return_type in ("both", "y"):
        if lhs in data_dict:
            y = np.asarray(data_dict[lhs], dtype=float)
        else:
            y = _apply_transformation(lhs, data_dict, n_obs, caller_globals).astype(
                float
            )

    if return_type == "y":
        return y

    # The columns of each variable: the dummies of all levels for the factors (the
    # logical variables with the levels FALSE and TRUE), with the levels of the
    # training data at the prediction, as R's xlevels
    xlevels = getattr(data, "attrs", {}).get("_xlevels", {})
    blocks: dict[str, list[tuple[str, np.ndarray]]] = {}
    factors: dict[str, list] = {}
    for name in [v for v in order if any(v in term for term in terms)]:
        evaluated = name not in data_dict
        values = pd.Series(
            _apply_transformation(name, data_dict, n_obs, caller_globals)
            if evaluated
            else data_dict[name]
        )
        if pd.api.types.infer_dtype(values, skipna=True) == "boolean":
            values = values.astype(pd.CategoricalDtype([False, True]))
        if name in xlevels:
            unseen = values[values.notna()]
            unseen = unseen[~unseen.isin(xlevels[name])].unique()
            if len(unseen):
                raise ValueError(
                    f"factor {name} has new levels {', '.join(map(str, unseen))}"
                )
            values = values.astype(pd.CategoricalDtype(xlevels[name]))
        if isinstance(values.dtype, pd.CategoricalDtype) or (
            not evaluated and pd.api.types.is_string_dtype(values)
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
                        else str(level).upper()
                        if isinstance(level, (bool, np.bool_))
                        else str(level)
                    ),
                    dummies[:, i],
                )
                for i, level in enumerate(categories.categories)
            ]
            factors[name] = list(categories.categories)
        else:
            blocks[name] = [(name, values.to_numpy(dtype=float))]

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
        X_df.attrs["_xlevels"] = factors

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


def _parse_formula_terms(
    formula_str: str,
) -> tuple[list[frozenset[str]], bool | None, list[str]]:
    """Expand the right-hand side of a formula into its terms, as R's terms.formula.

    Returns the terms (the sets of their variables) in R's order before the
    sorting by degree, the intercept set by "1" or "0" (None if neither) and
    the variables in their order of appearance. The operators are those of R:
    +, -, *, /, %in%, : and ^ (the power of a term is the term), with R's
    precedence. The operators in calls belong to the variables, e.g. log(x + 1),
    as do those in backquoted names.
    """
    text = formula_str.strip()
    if not text:
        return [], None, []

    # The binary operators outside calls, by precedence: + and -, * and /, %in%, :, ^
    found: list[list[int]] = [[], [], [], [], []]
    depth, quoted, operand = 0, False, False
    for position, char in enumerate(text):
        if quoted or depth or char in "()`":
            quoted ^= char == "`"
            depth += 0 if quoted else (char == "(") - (char == ")")
            operand = True
        elif char in "+-*/%:^":
            if (operand or char not in "+-") and (
                char != "%" or text.startswith("%in%", position)
            ):
                found[[0, 0, 1, 1, 2, 3, 4]["+-*/%:^".index(char)]].append(position)
            operand = False
        elif not char.isspace():
            operand = True

    kind = next((k for k, positions in enumerate(found) if positions), None)
    if text[0] in "+-" and (kind is None or kind == 4):
        terms, intercept, variables = _parse_formula_terms(text[1:])
        if text[0] == "-":
            terms, intercept = [], None if intercept is None else not intercept
        return terms, intercept, variables
    if kind is None and text[0] == "(":
        return _parse_formula_terms(text[1:-1])
    if kind is None:
        if text in ("0", "1"):
            return [], text == "1", []
        name = text[1:-1] if text[0] == "`" else text
        return [frozenset([name])], None, [name]

    # Split at the last operator of the lowest precedence, at the first power
    position = found[kind][0 if kind == 4 else -1]
    operator = "%in%" if kind == 2 else text[position]
    left, intercept, variables = _parse_formula_terms(text[:position])
    right, right_intercept, right_variables = _parse_formula_terms(
        "" if kind == 4 else text[position + len(operator) :]
    )
    variables += [v for v in right_variables if v not in variables]
    if right_intercept is not None:
        intercept = right_intercept if operator != "-" else not right_intercept

    if operator == "-":
        terms = [term for term in left if term not in right]
    elif operator == "+":
        terms = left + right
    elif operator == "*":
        terms = left + right + [a | b for a in left for b in right]
    elif operator == ":":
        terms = [a | b for a in left for b in right]
    elif operator == "/":
        terms = left + [frozenset().union(*left) | b for b in right]
    elif operator == "%in%":
        terms = [a | frozenset().union(*right) for a in left]
    else:
        power = text[position + 1 :].strip()
        if not re.fullmatch(r"\d+(\.\d*)?", power) or float(power) < 2:
            raise ValueError(f"Invalid power in formula: '{text}'")
        terms = left
        for _ in range(int(float(power)) - 1):
            terms = list(dict.fromkeys(a | b for a in left for b in terms))
    return list(dict.fromkeys(terms)), intercept, variables


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
