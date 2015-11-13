
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

static PyObject *
py_assign_centroids(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyArrayObject *data;
    PyArrayObject *centroids;
    PyArrayObject *assignments;
    int n_jobs;

    if (!PyArg_ParseTuple(args, "OOOi",
                          &data, &centroids, &assignments, &n_jobs)) {
        return NULL;
    }
    if (!PyArray_Check(data) || !PyArray_ISCONTIGUOUS(data)) {
        PyErr_SetString(PyExc_RuntimeError, "data not what was expected.");
        return NULL;
    }
    if (!PyArray_Check(centroids) || !PyArray_ISCONTIGUOUS(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids not what was expected.");
        return NULL;
    }
    if (!PyArray_Check(assignments) || !PyArray_ISCONTIGUOUS(assignments)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "assignments not what was expected.");
        return NULL;
    }
    if (PyArray_TYPE(data) != PyArray_TYPE(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids and data should have same type.");
        return NULL;
    }
    if (PyArray_NDIM(data) != 2) {
        PyErr_SetString(PyExc_RuntimeError,
                        "data should be two dimensional.");
        return NULL;
    }
    if (PyArray_NDIM(centroids) != 2) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids should be two dimensional.");
        return NULL;
    }
    if (PyArray_NDIM(assignments) != 1) {
        PyErr_SetString(PyExc_RuntimeError,
                        "assignments should be one dimensional.");
        return NULL;
    }

    const int N = (int) PyArray_DIM(data, 0);
    const int D = (int) PyArray_DIM(data, 1);
    const int k = (int) PyArray_DIM(centroids, 0);

    if (PyArray_DIM(centroids, 1) != D) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids has wrong number of features.");
        return NULL;
    }
    if (PyArray_DIM(assignments, 0) != N) {
        PyErr_SetString(PyExc_RuntimeError,
                        "assignments has wrong number of samples.");
        return NULL;
    }

    assign_centroids_multi(
        PyArray_DATA(data),
        PyArray_DATA(centroids),
        PyArray_DATA(assignments),
        n_jobs,
        k, N, D
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
                          &reassignment_ratio)) {
        return NULL;
    }
    if (!PyArray_Check(data) || !PyArray_ISCONTIGUOUS(data)) {
        PyErr_SetString(PyExc_RuntimeError, "data not what was expected.");
        return NULL;
    }
    if (!PyArray_Check(centroids) || !PyArray_ISCONTIGUOUS(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids not what was expected.");
        return NULL;
    }
    if (PyArray_TYPE(data) != PyArray_TYPE(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids and data should have same type.");
        return NULL;
    }
    if (PyArray_NDIM(data) != 2) {
        PyErr_SetString(PyExc_RuntimeError,
                        "data should be two dimensional");
        return NULL;
    }
    if (PyArray_NDIM(centroids) != 2) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids should be two dimensional");
        return NULL;
    }

    const int N = (int) PyArray_DIM(data, 0);
    const int D = (int) PyArray_DIM(data, 1);
    const int k = (int) PyArray_DIM(centroids, 0);

    if (PyArray_DIM(centroids, 1) != D) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids has wrong number of features.");
        return NULL;
    }
    if (n_samples > N) {
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
        k, N, D
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
    if (!PyArray_Check(data) || !PyArray_ISCONTIGUOUS(data)) {
        PyErr_SetString(PyExc_RuntimeError, "data not what was expected.");
        return NULL;
    }
    if (!PyArray_Check(centroids) || !PyArray_ISCONTIGUOUS(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids not what was expected.");
        return NULL;
    }
    if (PyArray_TYPE(data) != PyArray_TYPE(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids and data should have same type.");
        return NULL;
    }
    if (PyArray_NDIM(data) != 2) {
        PyErr_SetString(PyExc_RuntimeError,
                        "data should be two dimensional");
        return NULL;
    }
    if (PyArray_NDIM(centroids) != 2) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids should be two dimensional");
        return NULL;
    }

    const int N = (int) PyArray_DIM(data, 0);
    const int D = (int) PyArray_DIM(data, 1);
    const int k = (int) PyArray_DIM(centroids, 0);

    if (PyArray_DIM(centroids, 1) != D) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids has wrong number of features.");
        return NULL;
    }
    if (n_samples > N) {
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
        k, N, D
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
    if (!PyArray_Check(data) || !PyArray_ISCONTIGUOUS(data)) {
        PyErr_SetString(PyExc_RuntimeError, "data not what was expected.");
        return NULL;
    }
    if (!PyArray_Check(centroids) || !PyArray_ISCONTIGUOUS(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids not what was expected.");
        return NULL;
    }
    if (PyArray_TYPE(data) != PyArray_TYPE(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids and data should have same type.");
        return NULL;
    }
    if (PyArray_NDIM(data) != 2) {
        PyErr_SetString(PyExc_RuntimeError,
                        "data should be two dimensional");
        return NULL;
    }
    if (PyArray_NDIM(centroids) != 2) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids should be two dimensional");
        return NULL;
    }

    const int N = (int) PyArray_DIM(data, 0);
    const int D = (int) PyArray_DIM(data, 1);
    const int k = (int) PyArray_DIM(centroids, 0);

    if (PyArray_DIM(centroids, 1) != D) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids has wrong number of features.");
        return NULL;
    }

    double variance = model_variance(
        PyArray_DATA(data),
        PyArray_DATA(centroids),
        k, N, D
    );

    PyObject *ret = Py_BuildValue("d", variance);
    return ret;
}

static PyObject *
py_bic(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyArrayObject *data;
    PyArrayObject *centroids;

    if (!PyArg_ParseTuple(args, "OO", &data, &centroids)) {
        return NULL;
    }
    if (!PyArray_Check(data) || !PyArray_ISCONTIGUOUS(data)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "data not what was expected.");
        return NULL;
    }
    if (!PyArray_Check(centroids) || !PyArray_ISCONTIGUOUS(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids not what was expected.");
        return NULL;
    }
    if (PyArray_TYPE(data) != PyArray_TYPE(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids and data should have same type.");
        return NULL;
    }
    if (PyArray_NDIM(data) != 2) {
        PyErr_SetString(PyExc_RuntimeError,
                        "data should be two dimensional");
        return NULL;
    }
    if (PyArray_NDIM(centroids) != 2) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids should be two dimensional");
        return NULL;
    }

    const int N = (int) PyArray_DIM(data, 0);
    const int D = (int) PyArray_DIM(data, 1);
    const int k = (int) PyArray_DIM(centroids, 0);

    if (PyArray_DIM(centroids, 1) != D) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids has wrong number of features.");
        return NULL;
    }

    double bic = bayesian_information_criterion(
        PyArray_DATA(data),
        PyArray_DATA(centroids),
        k, N, D
    );

    PyObject *ret = Py_BuildValue("d", bic);
    return ret;
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
    if (!PyArray_Check(data) || !PyArray_ISCONTIGUOUS(data)) {
        PyErr_SetString(PyExc_RuntimeError, "data not what was expected.");
        return NULL;
    }
    if (!PyArray_Check(centroids) || !PyArray_ISCONTIGUOUS(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids not what was expected.");
        return NULL;
    }
    if (PyArray_TYPE(data) != PyArray_TYPE(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids and data should have same type.");
        return NULL;
    }
    if (PyArray_NDIM(data) != 2) {
        PyErr_SetString(PyExc_RuntimeError,
                        "data should be two dimensional");
        return NULL;
    }
    if (PyArray_NDIM(centroids) != 2) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids should be two dimensional");
        return NULL;
    }

    const int N = (int) PyArray_DIM(data, 0);
    const int D = (int) PyArray_DIM(data, 1);
    const int k = (int) PyArray_DIM(centroids, 0);

    if (n_samples >= N) {
        PyErr_SetString(PyExc_RuntimeError,
                        "n_samples must be smaller than the number of samples");
        return NULL;
    }
    if (PyArray_DIM(centroids, 1) != D) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids has wrong number of features.");
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
    if (!PyArray_Check(data) || !PyArray_ISCONTIGUOUS(data)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "data not what was expected.");
        return NULL;
    }
    if (!PyArray_Check(centroids) || !PyArray_ISCONTIGUOUS(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids not what was expected.");
        return NULL;
    }
    if (PyArray_TYPE(data) != PyArray_TYPE(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids and data should have same type.");
        return NULL;
    }
    if (PyArray_NDIM(data) != 2) {
        PyErr_SetString(PyExc_RuntimeError,
                        "data should be two dimensional");
        return NULL;
    }
    if (PyArray_NDIM(centroids) != 2) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids should be two dimensional");
        return NULL;
    }

    const int N = (int) PyArray_DIM(data, 0);
    const int D = (int) PyArray_DIM(data, 1);
    const int k = (int) PyArray_DIM(centroids, 0);

    if (n_samples >= N) {
        PyErr_SetString(PyExc_RuntimeError,
                        "n_samples must be smaller than the number of samples");
        return NULL;
    }
    if (PyArray_DIM(centroids, 1) != D) {
        PyErr_SetString(PyExc_RuntimeError,
                        "centroids has wrong number of features.");
        return NULL;
    }

    kmeanspp(
        PyArray_DATA(data),
        PyArray_DATA(centroids),
        n_samples,
        k, N, D
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
