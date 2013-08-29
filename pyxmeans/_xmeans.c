#include <Python.h>
#include <numpy/arrayobject.h>
#include "chi2.h"
 
/* Docstrings */
static char module_docstring[] =
    "This module provides an interface for calculating xmeans using C.";
static char minibatch_docstring[] =
    "Cluster data with minibatch and return the model with the best BIC score.";
 
bool assert_type_contiguous(PyArrayObject* array,int type) { 
    if (!PyArray_Check(array) ||
        PyArray_TYPE(array) != type ||
        !PyArray_ISCONTIGUOUS(array)) {
        return false;
    }
    return true;
}

int computecentroids(double* data, double* centroids, int* a_assignments, const int N, const int Nf, const int k) {
    return 1;
}

PyObject* py_minibatch(PyObject* self, PyObject* args) {
    PyArrayObject* data;
    PyArrayObject* centroids;
    PyArrayObject* assignments;
    if (!PyArg_ParseTuple(args, "OOO", &data, &centroids, &assignments)) { throw Kmeans_Exception("Wrong number of arguments for computecentroids."); }
    if (!PyArray_Check(data) || !PyArray_ISCONTIGUOUS(data)) throw Kmeans_Exception("data not what was expected.");
    if (!PyArray_Check(centroids) || !PyArray_ISCONTIGUOUS(centroids)) throw Kmeans_Exception("centroids not what was expected.");
    if (!PyArray_Check(assignments) || !PyArray_ISCONTIGUOUS(assignments)) throw Kmeans_Exception("assignments not what was expected.");
    if (PyArray_TYPE(data) != PyArray_TYPE(centroids)) throw Kmeans_Exception("centroids and data should have same type.");
    if (PyArray_NDIM(data) != 2) throw Kmeans_Exception("data should be two dimensional");
    if (PyArray_NDIM(centroids) != 2) throw Kmeans_Exception("centroids should be two dimensional");
    if (PyArray_NDIM(assignments) != 1) throw Kmeans_Exception("assignments should be two dimensional");
    
    const int N = PyArray_DIM(data, 0);
    const int Nf = PyArray_DIM(data, 1);
    const int k = PyArray_DIM(centroids, 0);
    if (PyArray_DIM(centroids, 1) != Nf) throw Kmeans_Exception("centroids has wrong number of features.");
    if (PyArray_DIM(assignments, 0) != N) throw Kmeans_Exception("assignments has wrong size.");
    if (minibatch(
        PyArray_DATA(data), 
        PyArray_DATA(centroids), 
        PyArray_DATA(assignments), 
        N, Nf, k)) { 
        Py_RETURN_TRUE; 
    } 
    Py_RETURN_FALSE;
}

/* Module specification */
static PyMethodDef module_methods[] = {
    {"minibatch", py_minibatch, METH_VARARGS, minibatch_docstring},
    {NULL, NULL, 0, NULL}
};
 
/* Initialize the module */
PyMODINIT_FUNC init_xmeans(void)
{
    PyObject *m = Py_InitModule3("_xmeans", module_methods, module_docstring);
    if (m == NULL)
        return;
 
    /* Load `numpy` functionality. */
    import_array();
}

