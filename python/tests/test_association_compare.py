"""Compare association(), pcor(), mcor(), cramer(), determination() and determ()
between R and Python.

The R functions with the fixes of this branch (association() with ``y`` and a
fixed method, the incomplete observations in pcor(), determination() and mcor(),
the ``use`` options of mcor() and cramer()) are sourced from the R/ folder of the
repository, so that the comparison does not depend on the installed version of
greybox.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("rpy2.robjects", reason="Requires rpy2 and R")

import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError

from greybox import ALM, association, cramer, determ, determination, mcor, pcor

ro.r["library"]("greybox")
R_FOLDER = Path(__file__).resolve().parents[2] / "R"
ro.r("assocFixed <- new.env()")
for file_name in ("association.R", "pcor.R", "mcor.R", "cramer.R", "determination.R"):
    ro.r(f'sys.source("{R_FOLDER / file_name}", envir=assocFixed)')

MTCARS_DF = pd.read_csv(Path(__file__).parent / "mtcars.csv")
MTCARS = MTCARS_DF.to_numpy()
MCOR_FIELDS = {
    "value": "value",
    "statistic": "statistic",
    "df": "df",
    "df_residual": "df.residual",
    "p_value": "p.value",
}
CRAMER_FIELDS = {
    "value": "value",
    "statistic": "statistic",
    "p_value": "p.value",
    "df": "df",
}


def to_r_matrix(name, arr):
    """Assign a numpy matrix (NaN as NA) to the R global environment."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        ro.globalenv[name] = ro.FloatVector(arr.tolist())
    else:
        ro.globalenv[name] = ro.r["matrix"](
            ro.FloatVector(arr.flatten(order="F").tolist()), nrow=arr.shape[0]
        )


@pytest.fixture(scope="module")
def mtcars_na():
    data = MTCARS.copy()
    data[2, 1] = np.nan
    data[4, 6] = np.nan
    data[10, 0] = np.nan
    to_r_matrix("mtcarsNA", data)
    return data


def assert_result(py, r_code, fields):
    """Compare the fields of a result object with the elements of an R list."""
    r = ro.r(r_code)
    for py_name, r_name in fields.items():
        r_value = r.rx2(r_name)[0]
        r_value = np.nan if r_value is ro.NA_Logical else float(r_value)
        assert_allclose(
            getattr(py, py_name), r_value, rtol=1e-10, atol=1e-14, equal_nan=True
        )


def assert_association(py, r):
    """Compare the values, the p-values and the types with R's association()."""
    assert_allclose(py["value"], np.array(r.rx2("value")), rtol=1e-10)
    assert_allclose(py["p.value"], np.array(r.rx2("p.value")), rtol=1e-8, atol=1e-15)
    r_type = np.array(r.rx2("type")).reshape(py["type"].shape, order="F")
    assert (py["type"] == r_type).all()


@pytest.fixture(scope="module")
def mtcars_factors():
    """mtcars with factors (an ordered one among them) and missing values."""
    data = MTCARS_DF.copy()
    data["cyl"] = pd.Categorical(data["cyl"])
    data["am"] = pd.Categorical(data["am"].map({0: "auto", 1: "manual"}))
    data["gear"] = pd.Categorical(data["gear"], ordered=True)
    data.loc[2, "cyl"] = np.nan
    data.loc[4, "mpg"] = np.nan
    data.loc[9, "am"] = np.nan
    data.loc[6, "carb"] = np.nan
    ro.r(
        "mtF <- mtcars; mtF$cyl <- factor(mtF$cyl);"
        " mtF$am <- factor(mtF$am, labels=c('auto','manual'));"
        " mtF$gear <- factor(mtF$gear, ordered=TRUE);"
        " mtF$cyl[3] <- NA; mtF$mpg[5] <- NA; mtF$am[10] <- NA; mtF$carb[7] <- NA"
    )
    return data


@pytest.fixture(scope="module")
def continuous():
    """Continuous data without ties, n=60 (Kendall's asymptotic test)."""
    rng = np.random.default_rng(11)
    data = rng.normal(size=(60, 4))
    data[:, 1] += 0.5 * data[:, 0]
    data[:, 3] += 0.3 * data[:, 2] - 0.4 * data[:, 0]
    to_r_matrix("contData", data)
    return data


class TestAssociationCompare:
    def test_auto_mtcars(self):
        py = association(MTCARS)
        r = ro.r("assocFixed$association(as.matrix(mtcars))")
        assert_allclose(py["value"], np.array(r.rx2("value")), rtol=1e-10)
        assert_allclose(py["p.value"], np.array(r.rx2("p.value")), rtol=1e-8)
        r_type = np.array(r.rx2("type")).reshape(py["type"].shape, order="F")
        assert (py["type"] == r_type).all()

    def test_mpg_cyl(self):
        py = association(MTCARS[:, :2])
        assert py["value"][0, 1] == pytest.approx(0.8558388046971949, rel=1e-12)
        assert py["type"][0, 1] == "mcor"

    def test_with_y_shape(self):
        py = association(MTCARS[:, :3], MTCARS[:, 3:6])
        r = ro.r(
            "assocFixed$association(as.matrix(mtcars[,1:3]), as.matrix(mtcars[,4:6]))"
        )
        assert py["value"].shape == (3, 3)
        assert_allclose(py["value"], np.array(r.rx2("value")), rtol=1e-10)
        assert_allclose(py["p.value"], np.array(r.rx2("p.value")), rtol=1e-8)
        r_type = np.array(r.rx2("type")).reshape(py["type"].shape, order="F")
        assert (py["type"] == r_type).all()

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall", "cramer"])
    def test_methods_with_y(self, method):
        py = association(MTCARS[:, :3], MTCARS[:, 3:5], method=method)
        r = ro.r(
            "suppressWarnings(assocFixed$association(as.matrix(mtcars[,1:3]),"
            f' as.matrix(mtcars[,4:5]), method="{method}"))'
        )
        assert py["value"].shape == (2, 3)
        assert_allclose(py["value"], np.array(r.rx2("value")), rtol=1e-10)
        assert_allclose(py["p.value"], np.array(r.rx2("p.value")), rtol=1e-8)
        r_type = np.array(r.rx2("type")).reshape(py["type"].shape, order="F")
        assert (py["type"] == r_type).all()

    @pytest.mark.parametrize("method", ["spearman", "kendall", "cramer"])
    def test_methods(self, method):
        py = association(MTCARS, method=method)
        r = ro.r(f'suppressWarnings(assocFixed$association(mtcars, method="{method}"))')
        assert_allclose(py["value"], np.array(r.rx2("value")), rtol=1e-10)
        assert_allclose(py["p.value"], np.array(r.rx2("p.value")), rtol=1e-8)

    def test_cramer_two_by_two(self):
        # vs and am: a 2x2 table, with Yates' continuity correction in R
        py = association(MTCARS[:, [7, 8]])
        r = ro.r("assocFixed$association(as.matrix(mtcars[,c('vs','am')]))")
        assert_allclose(py["value"], np.array(r.rx2("value")), rtol=1e-10)
        assert_allclose(py["p.value"], np.array(r.rx2("p.value")), rtol=1e-10)

    def test_missing_values(self, mtcars_na):
        py = association(mtcars_na)
        r = ro.r("assocFixed$association(mtcarsNA)")
        assert np.isfinite(py["value"]).all()
        assert_allclose(py["value"], np.array(r.rx2("value")), rtol=1e-10)
        assert_allclose(py["p.value"], np.array(r.rx2("p.value")), rtol=1e-8)

    def test_single_variable(self):
        py = association(MTCARS[:, 0])
        assert py == {"value": 1.0, "p.value": 1.0, "type": "none"}


class TestPcorCompare:
    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_mtcars(self, method):
        py = pcor(MTCARS, method=method)
        r = ro.r(f'assocFixed$pcor(as.matrix(mtcars), method="{method}")')
        assert_allclose(py["value"], np.array(r.rx2("value")), rtol=1e-8)
        assert_allclose(py["p.value"], np.array(r.rx2("p.value")), rtol=1e-8)

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_continuous(self, continuous, method):
        # No ties: Spearman's Edgeworth series and Kendall's normal approximation
        py = pcor(continuous, method=method)
        r = ro.r(f'assocFixed$pcor(contData, method="{method}")')
        assert_allclose(py["value"], np.array(r.rx2("value")), rtol=1e-8)
        assert_allclose(py["p.value"], np.array(r.rx2("p.value")), rtol=1e-8)

    @pytest.mark.parametrize("method", ["spearman", "kendall"])
    def test_small_sample_exact(self, continuous, method):
        # n <= 9: Spearman's exact enumeration; n < 50: Kendall's exact test
        py = pcor(continuous[:8], method=method)
        r = ro.r(f'assocFixed$pcor(contData[1:8,], method="{method}")')
        assert_allclose(py["value"], np.array(r.rx2("value")), rtol=1e-8)
        assert_allclose(py["p.value"], np.array(r.rx2("p.value")), rtol=1e-8)

    def test_mpg_qsec(self):
        py = pcor(MTCARS)
        r = ro.r("assocFixed$pcor(as.matrix(mtcars))")
        assert py["value"][0, 6] == pytest.approx(
            np.array(r.rx2("value"))[0, 6], rel=1e-10
        )

    def test_with_y(self):
        py = pcor(MTCARS[:, :3], MTCARS[:, 3:5])
        r = ro.r("assocFixed$pcor(as.matrix(mtcars[,1:3]), as.matrix(mtcars[,4:5]))")
        assert py["value"].shape == (2, 3)
        assert_allclose(py["value"], np.array(r.rx2("value")), rtol=1e-8)
        assert_allclose(py["p.value"], np.array(r.rx2("p.value")), rtol=1e-8)

    def test_missing_values(self, mtcars_na):
        py = pcor(mtcars_na)
        r = ro.r("assocFixed$pcor(mtcarsNA)")
        assert_allclose(py["value"], np.array(r.rx2("value")), rtol=1e-8)
        assert_allclose(py["p.value"], np.array(r.rx2("p.value")), rtol=1e-8)


class TestMcorCompare:
    def test_categorical_vector(self):
        py = mcor(MTCARS[:, 1], MTCARS[:, 0])
        assert_result(py, "assocFixed$mcor(mtcars$cyl, mtcars$mpg)", MCOR_FIELDS)
        assert py.value == pytest.approx(0.8558388046971949, rel=1e-12)

    def test_matrix(self):
        py = mcor(MTCARS[:, 1:4], MTCARS[:, 0])
        r_code = "assocFixed$mcor(as.matrix(mtcars[,2:4]), mtcars$mpg)"
        assert_result(py, r_code, MCOR_FIELDS)

    def test_missing_values(self, mtcars_na):
        py = mcor(mtcars_na[:, 1], mtcars_na[:, 0])
        r_code = "assocFixed$mcor(mtcarsNA[,2], mtcarsNA[,1])"
        assert_result(py, r_code, MCOR_FIELDS)

    def test_matrix_missing_values(self, mtcars_na):
        # R's mcor() failed on a matrix with missing values before this branch
        py = mcor(mtcars_na[:, 1:4], mtcars_na[:, 0])
        r_code = "assocFixed$mcor(mtcarsNA[,2:4], mtcarsNA[,1])"
        assert_result(py, r_code, MCOR_FIELDS)

    def test_factor(self, mtcars_factors):
        py = mcor(mtcars_factors["cyl"], mtcars_factors["mpg"])
        assert_result(py, "assocFixed$mcor(mtF$cyl, mtF$mpg)", MCOR_FIELDS)

    @pytest.mark.parametrize("columns", [["cyl", "wt", "am"], ["cyl"]])
    def test_data_frame(self, mtcars_factors, columns):
        py = mcor(mtcars_factors[columns], mtcars_factors["mpg"])
        r_columns = ", ".join(f"'{name}'" for name in columns)
        r_code = f"assocFixed$mcor(mtF[,c({r_columns}),drop=FALSE], mtF$mpg)"
        assert_result(py, r_code, MCOR_FIELDS)

    def test_factor_response(self, mtcars_factors):
        with pytest.warns(UserWarning, match="should be numeric"):
            py = mcor(mtcars_factors["mpg"], mtcars_factors["gear"])
        r_code = "suppressWarnings(assocFixed$mcor(mtF$mpg, mtF$gear))"
        assert_result(py, r_code, MCOR_FIELDS)

    @pytest.mark.parametrize("use", ["everything", "complete.obs"])
    def test_use(self, mtcars_factors, use):
        py = mcor(mtcars_factors["cyl"], mtcars_factors["mpg"], use=use)
        r_code = f"assocFixed$mcor(mtF$cyl, mtF$mpg, use='{use}')"
        assert_result(py, r_code, MCOR_FIELDS)

    def test_too_few_observations(self):
        x = np.array([1.0, np.nan, 3.0])
        y = np.array([np.nan, 2.0, 1.0])
        to_r_matrix("mcorX", x)
        to_r_matrix("mcorY", y)
        py = mcor(x, y)
        assert_result(py, "assocFixed$mcor(mcorX, mcorY)", MCOR_FIELDS)
        assert np.isnan(py.value)
        with pytest.raises(ValueError, match="Missing observations"):
            mcor(x, y, use="complete.obs")
        with pytest.raises(RRuntimeError, match="Missing observations"):
            ro.r("assocFixed$mcor(mcorX, mcorY, use='complete.obs')")


class TestCramerCompare:
    @pytest.mark.parametrize("pair", [("am", "gear"), ("cyl", "gear"), ("vs", "am")])
    def test_mtcars(self, pair):
        py = cramer(MTCARS_DF[pair[0]], MTCARS_DF[pair[1]])
        r_code = f"assocFixed$cramer(mtcars${pair[0]}, mtcars${pair[1]})"
        assert_result(py, r_code, CRAMER_FIELDS)

    def test_biased(self):
        py = cramer(MTCARS_DF["cyl"], MTCARS_DF["gear"], unbiased=False)
        r_code = "assocFixed$cramer(mtcars$cyl, mtcars$gear, unbiased=FALSE)"
        assert_result(py, r_code, CRAMER_FIELDS)

    @pytest.mark.parametrize("use", ["na.or.complete", "complete.obs", "everything"])
    def test_factors_missing_values(self, mtcars_factors, use):
        py = cramer(mtcars_factors["cyl"], mtcars_factors["am"], use=use)
        r_code = f"assocFixed$cramer(mtF$cyl, mtF$am, use='{use}')"
        assert_result(py, r_code, CRAMER_FIELDS)

    def test_all_obs(self, mtcars_factors):
        with pytest.raises(ValueError, match="Missing observations in cramer"):
            cramer(mtcars_factors["cyl"], mtcars_factors["am"], use="all.obs")

    @pytest.mark.parametrize(
        "x, y", [([1, 2], [1, 2]), ([1, 2, 1], [1, 2, 2]), ([1, 2, 1, 2], [1, 2, 2, 1])]
    )
    def test_small_samples(self, x, y):
        # The bias correction divides by zero for n=2, giving NaN as in R
        py = cramer(x, y)
        r_code = f"assocFixed$cramer(c({str(x)[1:-1]}), c({str(y)[1:-1]}))"
        assert_result(py, r_code, CRAMER_FIELDS)

    def test_numeric_warning(self):
        with pytest.warns(UserWarning, match="x is numeric"):
            cramer(MTCARS_DF["mpg"], MTCARS_DF["am"])


class TestFactorsCompare:
    @pytest.mark.parametrize("use", ["na.or.complete", "complete.obs", "everything"])
    def test_association(self, mtcars_factors, use):
        py = association(mtcars_factors, use=use)
        assert_association(py, ro.r(f"assocFixed$association(mtF, use='{use}')"))

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall", "cramer"])
    def test_association_methods(self, mtcars_factors, method):
        py = association(mtcars_factors, method=method)
        r = ro.r(f"suppressWarnings(assocFixed$association(mtF, method='{method}'))")
        assert_association(py, r)

    def test_association_with_y(self, mtcars_factors):
        py = association(
            mtcars_factors[["mpg", "cyl", "am"]], mtcars_factors[["hp", "gear", "carb"]]
        )
        r = ro.r(
            "assocFixed$association(mtF[,c('mpg','cyl','am')],"
            " mtF[,c('hp','gear','carb')])"
        )
        assert_association(py, r)

    def test_association_all_obs(self, mtcars_factors):
        with pytest.raises(ValueError, match="mpg, cyl, am, carb"):
            association(mtcars_factors, use="all.obs")

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_pcor(self, mtcars_factors, method):
        with pytest.warns(UserWarning, match="categorical scales"):
            py = pcor(mtcars_factors, method=method)
        r = ro.r(f"suppressWarnings(assocFixed$pcor(mtF, method='{method}'))")
        assert_allclose(py["value"], np.array(r.rx2("value")), rtol=1e-8)
        assert_allclose(py["p.value"], np.array(r.rx2("p.value")), rtol=1e-8)

    def test_pcor_use(self, mtcars_na):
        py = pcor(mtcars_na, use="complete.obs")
        r = ro.r("assocFixed$pcor(mtcarsNA, use='complete.obs')")
        assert_allclose(py["value"], np.array(r.rx2("value")), rtol=1e-8)
        for use in ("everything", "all.obs"):
            with pytest.raises(ValueError):
                pcor(mtcars_na, use=use)
            with pytest.raises(RRuntimeError):
                ro.r(f"assocFixed$pcor(mtcarsNA, use='{use}')")

    def test_determination(self, mtcars_factors):
        py = determination(mtcars_factors)
        r = np.array(ro.r("assocFixed$determination(mtF)"))
        assert_allclose(py, r, rtol=1e-8)


class TestDeterminationCompare:
    @pytest.mark.parametrize("bruteforce", [True, False])
    def test_mtcars(self, bruteforce):
        py = determination(MTCARS, bruteforce=bruteforce)
        r = np.array(
            ro.r(
                "assocFixed$determination(mtcars,"
                f" bruteforce={str(bruteforce).upper()})"
            )
        )
        assert_allclose(py, r, rtol=1e-8)

    @pytest.mark.parametrize("bruteforce", [True, False])
    def test_missing_values(self, mtcars_na, bruteforce):
        py = determination(mtcars_na, bruteforce=bruteforce)
        r = np.array(
            ro.r(
                "assocFixed$determination(mtcarsNA,"
                f" bruteforce={str(bruteforce).upper()})"
            )
        )
        assert_allclose(py, r, rtol=1e-8)


class TestDetermCompare:
    @pytest.mark.parametrize(
        "columns, orders, r_code",
        [
            (list(MTCARS_DF.columns[1:]), (0, 0, 0), "alm(mpg ~ ., mtcars)"),
            (["wt", "hp"], (2, 0, 0), "alm(mpg ~ wt + hp, mtcars, orders=c(2,0,0))"),
        ],
    )
    def test_alm(self, columns, orders, r_code):
        X = np.column_stack([np.ones(len(MTCARS_DF)), MTCARS_DF[columns]])
        model = ALM(orders=orders).fit(X, MTCARS_DF["mpg"].to_numpy())
        assert_allclose(determ(model), np.array(ro.r(f"determ({r_code})")), rtol=1e-10)

    def test_alm_no_intercept(self):
        X = MTCARS_DF[["wt", "hp"]].to_numpy()
        model = ALM().fit(X, MTCARS_DF["mpg"].to_numpy())
        r = np.array(ro.r("suppressWarnings(determ(alm(mpg ~ wt + hp - 1, mtcars)))"))
        assert_allclose(determ(model), r, rtol=1e-10)

    def test_alm_factors(self):
        # The dummy variables of the factors, as in R's alm()
        cyl, am = MTCARS_DF["cyl"], MTCARS_DF["am"]
        X = np.column_stack(
            [np.ones(len(cyl)), cyl == 6, cyl == 8, am, MTCARS_DF[["wt", "hp"]]]
        ).astype(float)
        model = ALM().fit(X, MTCARS_DF["mpg"].to_numpy())
        ro.r("mtFactors <- mtcars; mtFactors$cyl <- factor(mtFactors$cyl)")
        ro.r("mtFactors$am <- factor(mtFactors$am)")
        r = np.array(ro.r("determ(alm(mpg ~ cyl + am + wt + hp, mtFactors))"))
        assert_allclose(determ(model), r, rtol=1e-10)

    def test_default(self):
        py = determ(MTCARS_DF.iloc[:, 1:])
        assert_allclose(py, np.array(ro.r("determ(mtcars[,-1])")), rtol=1e-10)
