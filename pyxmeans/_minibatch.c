
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "c_minibatch/minibatch.h"
#include "c_minibatch/distance.h"

/* Docstrings */
static char module_docstring[] =
    "This module provides an interface for calculating minibatches using C.";
static char minibatch_docstring[] =
    "Cluster data with minibatch.";
static char minibatch_multi_docstring[] =
    "Cluster data with minibatch.  Runs n_runs independent clustering "
    "operations using n_jobs threads and returns the model with the "
    "lowest variance.";
static char kmeanspp_docstring[] =
    "Initialize cluster locations using the kmeans++ algorithm";
static char kmeanspp_multi_docstring[] =
    "Initialize cluster locations using the kmeans++ algorithm, "
    "by running n_runs independent kmeans++ operations and picking "
    "the centroids with the lowest variance";
static char bic_docstring[] =
    "Calculate the bayesian information criterion of the clusters "
    "given the data.";
static char model_variance_docstring[] =
    "Calculate the model variance of the clusters given the data.";
static char assign_centroids_docstring[] =
    "Assigns each piece of data to a centroid";
static char set_metric_docstring[] =
    "Sets the metric to be used when calculating mini-batches.  "
    "Built-in metrics are 'euclidian' and 'cosine'.  "
    "You can also supply a callable, which should take two vectors "
    "and return a nonnegative number.";

static PyObject *python_metric = NULL;

/* common argument-checking fragments */

#define require_contiguous_ndarray(arg, dim, etype, etypename) do {     \
        if (!PyArray_Check(arg) || !PyArray_ISCONTIGUOUS(arg)) {        \
            PyErr_SetString(PyExc_TypeError,                            \
                            #arg " must be a contiguous numpy array."); \
            return NULL;                                                \
        }                                                               \
        if (PyArray_NDIM(arg) != dim) {                                 \
            PyErr_SetString(PyExc_TypeError,                            \
                            #arg " must be " #dim "-dimensional.");     \
            return NULL;                                                \
        }                                                               \
        if (PyArray_TYPE(arg) != etype) {                               \
            PyErr_SetString(PyExc_TypeError,                            \
                            #arg " element type must be " etypename);   \
            return NULL;                                                \
        }                                                               \
    } while (0)

#define require_positive(x, what) do {                                  \
        if ((x) <= 0) {                                                 \
            PyErr_SetString(PyExc_ValueError,                           \
                "not enough " what " (need at least one)");             \
            return NULL;                                                \
        }                                                               \
    } while (0)

#define require_positive_as_int(x, what) do {                           \
        require_positive(x, what);                                      \
        if ((x) > (npy_intp)INT_MAX) {                                  \
            PyErr_SetString(PyExc_ValueError,                           \
                "too many " what " (can only go up to INT_MAX)");       \
            return NULL;                                                \
        }                                                               \
    } while (0)

#define require_dimension_match(a_arr, a_dim, b_arr, b_dim, what) do {   \
        if (PyArray_DIM(a_arr, a_dim) != PyArray_DIM(b_arr, b_dim)) {    \
            PyErr_SetString(PyExc_ValueError,                            \
                #a_arr " and " #b_arr " must agree on number of " what); \
            return NULL;                                                 \
        }                                                                \
    } while (0)

/* module functions */

static PyObject *
py_assign_centroids(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyArrayObject *data;
    PyArrayObject *centroids;
    PyArrayObject *assignments;
    int n_jobs;

    if (!PyArg_ParseTuple(args, "OOOi",
                          &data, &centroids, &assignments, &n_jobs))
        return NULL;

    require_contiguous_ndarray(data,        2, NPY_DOUBLE, "double");
    require_contiguous_ndarray(centroids,   2, NPY_DOUBLE, "double");
    require_contiguous_ndarray(assignments, 1, NPY_INT,    "int");

    require_dimension_match(data, 1, centroids, 1, "features");
    require_dimension_match(data, 0, assignments, 0, "samples");

    const npy_intp N = PyArray_DIM(data, 0);
    const npy_intp D = PyArray_DIM(data, 1);
    const npy_intp k = PyArray_DIM(centroids, 0);

    require_positive_as_int(N, "samples");
    require_positive_as_int(D, "features");
    require_positive_as_int(k, "centroids");
    require_positive(n_jobs, "concurrent jobs");

    assign_centroids_multi(
        PyArray_DATA(data),
        PyArray_DATA(centroids),
        PyArray_DATA(assignments),
        n_jobs,
        (int)k, (int)N, (int)D
    );

    Py_XINCREF(assignments);
    return (PyObject *)assignments;
}

static PyObject *
py_minibatch_multi(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyArrayObject *data;
    PyArrayObject *centroids;
    int n_samples, max_iter, n_runs, n_jobs;
    double bic_ratio_termination, reassignment_ratio;

    if (!PyArg_ParseTuple(args, "OOiiiidd",
                          &data, &centroids, &n_samples, &max_iter,
                          &n_runs, &n_jobs, &bic_ratio_termination,
                          &reassignment_ratio))
        return NULL;

    require_contiguous_ndarray(data,      2, NPY_DOUBLE, "double");
    require_contiguous_ndarray(centroids, 2, NPY_DOUBLE, "double");

    require_dimension_match(data, 1, centroids, 1, "features");

    const npy_intp N = PyArray_DIM(data, 0);
    const npy_intp D = PyArray_DIM(data, 1);
    const npy_intp k = PyArray_DIM(centroids, 0);

    require_positive_as_int(N, "samples");
    require_positive_as_int(D, "features");
    require_positive_as_int(k, "centroids");
    require_positive(n_jobs, "concurrent jobs");
    require_positive(n_runs, "total runs");
    require_positive(max_iter, "allowed iterations");
    require_positive(n_samples, "requested samples");

    if (n_samples > (int)N) {
        PyErr_SetString(PyExc_RuntimeError,
                        "more samples requested than data.");
        return NULL;
    }

    minibatch_multi(
        PyArray_DATA(data),
        PyArray_DATA(centroids),
        n_samples,
        max_iter,
        n_runs,
        n_jobs,
        bic_ratio_termination,
        reassignment_ratio,
        (int)k, (int)N, (int)D
    );

    Py_XINCREF(centroids);
    return (PyObject *)centroids;
}

static PyObject *
py_minibatch(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyArrayObject *data;
    PyArrayObject *centroids;
    int n_samples, max_iter;
    double reassignment_ratio, bic_ratio_termination;

    if (!PyArg_ParseTuple(args, "OOiidd",
                          &data, &centroids, &n_samples, &max_iter,
                          &bic_ratio_termination, &reassignment_ratio)) {
        return NULL;
    }

    require_contiguous_ndarray(data,      2, NPY_DOUBLE, "double");
    require_contiguous_ndarray(centroids, 2, NPY_DOUBLE, "double");

    require_dimension_match(data, 1, centroids, 1, "features");

    const npy_intp N = PyArray_DIM(data, 0);
    const npy_intp D = PyArray_DIM(data, 1);
    const npy_intp k = PyArray_DIM(centroids, 0);

    require_positive_as_int(N, "samples");
    require_positive_as_int(D, "features");
    require_positive_as_int(k, "centroids");
    require_positive(max_iter, "allowed iterations");
    require_positive(n_samples, "requested samples");

    if (n_samples > (int)N) {
        PyErr_SetString(PyExc_RuntimeError,
                        "more samples requested than data.");
        return NULL;
    }

    minibatch(
        PyArray_DATA(data),
        PyArray_DATA(centroids),
        n_samples,
        max_iter,
        bic_ratio_termination,
        reassignment_ratio,
        (int)k, (int)N, (int)D
    );

    Py_XINCREF(centroids);
    return (PyObject *)centroids;
}

static PyObject *
py_model_variance(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyArrayObject *data;
    PyArrayObject *centroids;

    if (!PyArg_ParseTuple(args, "OO", &data, &centroids)) {
        return NULL;
    }

    require_contiguous_ndarray(data,      2, NPY_DOUBLE, "double");
    require_contiguous_ndarray(centroids, 2, NPY_DOUBLE, "double");

    require_dimension_match(data, 1, centroids, 1, "features");

    const npy_intp N = PyArray_DIM(data, 0);
    const npy_intp D = PyArray_DIM(data, 1);
    const npy_intp k = PyArray_DIM(centroids, 0);

    require_positive_as_int(N, "samples");
    require_positive_as_int(D, "features");
    require_positive_as_int(k, "centroids");

    double variance = model_variance(
        PyArray_DATA(data),
        PyArray_DATA(centroids),
        (int)k, (int)N, (int)D
    );

    return Py_BuildValue("d", variance);
}

static PyObject *
py_bic(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyArrayObject *data;
    PyArrayObject *centroids;

    if (!PyArg_ParseTuple(args, "OO", &data, &centroids)) {
        return NULL;
    }

    require_contiguous_ndarray(data,      2, NPY_DOUBLE, "double");
    require_contiguous_ndarray(centroids, 2, NPY_DOUBLE, "double");

    require_dimension_match(data, 1, centroids, 1, "features");

    const npy_intp N = PyArray_DIM(data, 0);
    const npy_intp D = PyArray_DIM(data, 1);
    const npy_intp k = PyArray_DIM(centroids, 0);

    require_positive_as_int(N, "samples");
    require_positive_as_int(D, "features");
    require_positive_as_int(k, "centroids");

    double bic = bayesian_information_criterion(
        PyArray_DATA(data),
        PyArray_DATA(centroids),
        (int)k, (int)N, (int)D
    );

    return Py_BuildValue("d", bic);
}

static PyObject *
py_kmeanspp_multi(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyArrayObject *data;
    PyArrayObject *centroids;
    int n_samples, n_runs, n_jobs;

    if (!PyArg_ParseTuple(args, "OOiii",
                          &data, &centroids, &n_samples, &n_runs, &n_jobs)) {
        return NULL;
    }

    require_contiguous_ndarray(data,      2, NPY_DOUBLE, "double");
    require_contiguous_ndarray(centroids, 2, NPY_DOUBLE, "double");

    require_dimension_match(data, 1, centroids, 1, "features");

    const npy_intp N = PyArray_DIM(data, 0);
    const npy_intp D = PyArray_DIM(data, 1);
    const npy_intp k = PyArray_DIM(centroids, 0);

    require_positive_as_int(N, "samples");
    require_positive_as_int(D, "features");
    require_positive_as_int(k, "centroids");
    require_positive(n_jobs, "concurrent jobs");
    require_positive(n_runs, "total runs");
    require_positive(n_samples, "requested samples");
    if (n_samples > (int)N) {
        PyErr_SetString(PyExc_RuntimeError,
                        "more samples requested than data.");
        return NULL;
    }

    kmeanspp_multi(
        PyArray_DATA(data),
        PyArray_DATA(centroids),
        n_samples,
        n_runs,
        n_jobs,
        k, N, D
    );

    Py_XINCREF(centroids);
    return (PyObject *)centroids;
}

static PyObject *
py_kmeanspp(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyArrayObject *data;
    PyArrayObject *centroids;
    int n_samples;

    if (!PyArg_ParseTuple(args, "OOi", &data, &centroids, &n_samples)) {
        return NULL;
    }

    require_contiguous_ndarray(data,      2, NPY_DOUBLE, "double");
    require_contiguous_ndarray(centroids, 2, NPY_DOUBLE, "double");

    require_dimension_match(data, 1, centroids, 1, "features");

    const npy_intp N = PyArray_DIM(data, 0);
    const npy_intp D = PyArray_DIM(data, 1);
    const npy_intp k = PyArray_DIM(centroids, 0);

    require_positive_as_int(N, "samples");
    require_positive_as_int(D, "features");
    require_positive_as_int(k, "centroids");
    require_positive(n_samples, "requested samples");
    if (n_samples > (int)N) {
        PyErr_SetString(PyExc_RuntimeError,
                        "more samples requested than data.");
        return NULL;
    }


    kmeanspp(
        PyArray_DATA(data),
        PyArray_DATA(centroids),
        n_samples,
        (int)k, (int)N, (int)D
    );

    Py_XINCREF(centroids);
    return (PyObject *)centroids;
}

static double
python_distance(double *A, double *B, int D)
{
    double result;
    npy_intp dims[] = {D};
    PyObject *arglist;
    PyObject *pyresult;

    PyGILState_STATE gstate = PyGILState_Ensure();
    {
        // We have the GIL locked
        PyObject *pyA = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, A);
        PyObject *pyB = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, B);

        arglist = Py_BuildValue("OO", pyA, pyB);
        pyresult = PyEval_CallObject(python_metric, arglist);

        Py_XDECREF(arglist);
        Py_XDECREF(pyA);
        Py_XDECREF(pyB);

        if (pyresult == NULL) {
            PyErr_Print();
            result = 0.0;
        }

        if (PyFloat_Check(pyresult)) {
            result = PyFloat_AsDouble(pyresult);
        } else {
            _LOG("Invalid result from python metric: not a float!\n");
            result = 0.0;
        }
        Py_XDECREF(pyresult);
    }
    PyGILState_Release(gstate);

    return result;
}

static PyObject *
py_set_metric(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *metric_object;

    if (!PyArg_ParseTuple(args, "O", &metric_object)) {
        return NULL;
    }

    if (PyUnicode_Check(metric_object)) {
        char *metric = PyUnicode_AsUTF8(metric_object);
        if (strcmp("euclidian", metric) == 0) {
            set_distance_metric(0);
        } else if (strcmp("cosine", metric) == 0) {
            set_distance_metric(1);
        } else {
            PyErr_SetString(PyExc_ValueError, "Invalid metric string value");
            return NULL;
        }
    } else if (PyCallable_Check(metric_object)) {
        Py_XDECREF(python_metric);
        Py_XINCREF(metric_object);
        python_metric = metric_object;
        distance_metric = python_distance;
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid metric type");
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

/* Module specification */
static PyMethodDef module_methods[] = {
#define M(s) { #s, py_##s, METH_VARARGS, s##_docstring }
    M(minibatch),
    M(minibatch_multi),
    M(assign_centroids),
    M(kmeanspp),
    M(kmeanspp_multi),
    M(bic),
    M(model_variance),
    M(set_metric),
    { 0, 0, 0, 0 }
};

static struct PyModuleDef minibatch_module = {
  PyModuleDef_HEAD_INIT,
  "_minibatch",
  module_docstring,
  -1,
  module_methods
};

/* Initialize the module */
PyMODINIT_FUNC PyInit__minibatch(void)
{
    PyObject *m = PyModule_Create(&minibatch_module);
    if (!m)
        return NULL;

    /* Ensure numpy is initialized. */
    import_array();

    /* Ensure c_minibatch is initialized. */
    set_distance_metric(0);

    return m;
}
