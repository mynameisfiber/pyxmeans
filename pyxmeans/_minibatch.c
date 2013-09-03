#include <Python.h>
#include <numpy/arrayobject.h>
#include "c_minibatch/minibatch.h"
 
/* Docstrings */
static char module_docstring[] =
    "This module provides an interface for calculating minibatches using C.";
static char minibatch_docstring[] =
    "Cluster data with minibatch";
static char kmeanspp_docstring[] =
    "Initialize cluster locations using the kmeans++ algorithm";
static char bic_docstring[] =
    "Calculate the bayesian information criterion of the clusters given the data";

PyArrayObject* py_minibatch(PyObject* self, PyObject* args) {
    PyArrayObject* data;
    PyArrayObject* centroids;
    int n_samples, max_iter;

    if (!PyArg_ParseTuple(args, "OOii", &data, &centroids, &n_samples, &max_iter)) { 
        PyErr_SetString(PyExc_RuntimeError, "Invalid arguments");
        return NULL;
    }
    if (!PyArray_Check(data) || !PyArray_ISCONTIGUOUS(data)) {
        PyErr_SetString(PyExc_RuntimeError,"data not what was expected.");
        return NULL;
    }
    if (!PyArray_Check(centroids) || !PyArray_ISCONTIGUOUS(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,"centroids not what was expected.");
        return NULL;
    }
    if (PyArray_TYPE(data) != PyArray_TYPE(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,"centroids and data should have same type.");
        return NULL;
    }
    if (PyArray_NDIM(data) != 2) {
        PyErr_SetString(PyExc_RuntimeError,"data should be two dimensional");
        return NULL;
    }
    if (PyArray_NDIM(centroids) != 2) {
        PyErr_SetString(PyExc_RuntimeError,"centroids should be two dimensional");
        return NULL;
    }
    
    const int N = (int) PyArray_DIM(data, 0);
    const int D = (int) PyArray_DIM(data, 1);
    const int k = (int) PyArray_DIM(centroids, 0);

    if (PyArray_DIM(centroids, 1) != D) {
        PyErr_SetString(PyExc_RuntimeError,"centroids has wrong number of features.");
        return NULL;
    }
    if (n_samples > N) {
        PyErr_SetString(PyExc_RuntimeError,"more samples requested than data.");
        return NULL;
    }

    minibatch(
        PyArray_DATA(data), 
        PyArray_DATA(centroids), 
        n_samples,
        max_iter,
        k, N, D
    );

    return centroids;
}


PyObject* py_bic(PyObject* self, PyObject* args) {
    PyArrayObject* data;
    PyArrayObject* centroids;

    if (!PyArg_ParseTuple(args, "OO", &data, &centroids)) { 
        PyErr_SetString(PyExc_RuntimeError, "Invalid arguments");
        return NULL;
    }
    if (!PyArray_Check(data) || !PyArray_ISCONTIGUOUS(data)) {
        PyErr_SetString(PyExc_RuntimeError,"data not what was expected.");
        return NULL;
    }
    if (!PyArray_Check(centroids) || !PyArray_ISCONTIGUOUS(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,"centroids not what was expected.");
        return NULL;
    }
    if (PyArray_TYPE(data) != PyArray_TYPE(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,"centroids and data should have same type.");
        return NULL;
    }
    if (PyArray_NDIM(data) != 2) {
        PyErr_SetString(PyExc_RuntimeError,"data should be two dimensional");
        return NULL;
    }
    if (PyArray_NDIM(centroids) != 2) {
        PyErr_SetString(PyExc_RuntimeError,"centroids should be two dimensional");
        return NULL;
    }
    
    const int N = (int) PyArray_DIM(data, 0);
    const int D = (int) PyArray_DIM(data, 1);
    const int k = (int) PyArray_DIM(centroids, 0);

    if (PyArray_DIM(centroids, 1) != D) {
        PyErr_SetString(PyExc_RuntimeError,"centroids has wrong number of features.");
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


PyArrayObject* py_kmeanspp(PyObject* self, PyObject* args) {
    PyArrayObject* data;
    PyArrayObject* centroids;

    if (!PyArg_ParseTuple(args, "OO", &data, &centroids)) { 
        PyErr_SetString(PyExc_RuntimeError, "Invalid arguments");
        return NULL;
    }
    if (!PyArray_Check(data) || !PyArray_ISCONTIGUOUS(data)) {
        PyErr_SetString(PyExc_RuntimeError,"data not what was expected.");
        return NULL;
    }
    if (!PyArray_Check(centroids) || !PyArray_ISCONTIGUOUS(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,"centroids not what was expected.");
        return NULL;
    }
    if (PyArray_TYPE(data) != PyArray_TYPE(centroids)) {
        PyErr_SetString(PyExc_RuntimeError,"centroids and data should have same type.");
        return NULL;
    }
    if (PyArray_NDIM(data) != 2) {
        PyErr_SetString(PyExc_RuntimeError,"data should be two dimensional");
        return NULL;
    }
    if (PyArray_NDIM(centroids) != 2) {
        PyErr_SetString(PyExc_RuntimeError,"centroids should be two dimensional");
        return NULL;
    }
    
    const int N = (int) PyArray_DIM(data, 0);
    const int D = (int) PyArray_DIM(data, 1);
    const int k = (int) PyArray_DIM(centroids, 0);

    if (PyArray_DIM(centroids, 1) != D) {
        PyErr_SetString(PyExc_RuntimeError,"centroids has wrong number of features.");
        return NULL;
    }

    kmeanspp(
        PyArray_DATA(data), 
        PyArray_DATA(centroids), 
        k, N, D
    );

    return centroids;
}

/* Module specification */
static PyMethodDef module_methods[] = {
    {"minibatch" , py_minibatch , METH_VARARGS , minibatch_docstring} , 
    {"kmeanspp"  , py_kmeanspp  , METH_VARARGS , kmeanspp_docstring}  , 
    {"bic"       , py_bic       , METH_VARARGS , bic_docstring}       , 
    {NULL        , NULL         , 0            , NULL}
};
 
/* Initialize the module */
PyMODINIT_FUNC init_minibatch(void)
{
    PyObject *m = Py_InitModule3("_minibatch", module_methods, module_docstring);
    if (m == NULL)
        return;
 
    /* Load `numpy` functionality. */
    import_array();
}

