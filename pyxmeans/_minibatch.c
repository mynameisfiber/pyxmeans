#include <Python.h>
#include <numpy/arrayobject.h>
#include "c_minibatch/minibatch.h"
#include "c_minibatch/distance.h"
 
/* Docstrings */
static char module_docstring[] =
    "This module provides an interface for calculating minibatches using C.";
static char minibatch_docstring[] =
    "Cluster data with minibatch";
static char minibatch_multi_docstring[] =
    "Cluster data with minibatch.  Runs n_runs independent clustering operations using n_jobs threads and returns the model with the lowest variance";
static char kmeanspp_docstring[] =
    "Initialize cluster locations using the kmeans++ algorithm";
static char kmeanspp_multi_docstring[] =
    "Initialize cluster locations using the kmeans++ algorithm by running n_runs independant kmeans++ and picking the centroids with the lowest variance";
static char bic_docstring[] =
    "Calculate the bayesian information criterion of the clusters given the data";
static char model_variance_docstring[] =
    "Calculate the model variance of the clusters given the data";
static char assign_centroids_docstring[] =
    "Assigns each piece of data to a centroid";
static char set_metric_docstring[] =
    "Sets the metric to be used when calculating mini-batches. Input value can be 'euclidian' or 'cosine'";

PyObject *python_metric = NULL;

PyArrayObject* py_assign_centroids(PyObject* self, PyObject* args) {
    PyArrayObject* data;
    PyArrayObject* centroids;
    PyArrayObject* assignments;
    int n_jobs;

    if (!PyArg_ParseTuple(args, "OOOi", &data, &centroids, &assignments, &n_jobs)) { 
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
    if (!PyArray_Check(assignments) || !PyArray_ISCONTIGUOUS(assignments)) {
        PyErr_SetString(PyExc_RuntimeError,"assignments not what was expected.");
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
    if (PyArray_NDIM(assignments) != 1) {
        PyErr_SetString(PyExc_RuntimeError,"assignments should be one dimensional");
        return NULL;
    }
    
    const int N = (int) PyArray_DIM(data, 0);
    const int D = (int) PyArray_DIM(data, 1);
    const int k = (int) PyArray_DIM(centroids, 0);

    if (PyArray_DIM(centroids, 1) != D) {
        PyErr_SetString(PyExc_RuntimeError,"centroids has wrong number of features.");
        return NULL;
    }
    if (PyArray_DIM(assignments, 0) != N) {
        PyErr_SetString(PyExc_RuntimeError,"assignments has wrong number of samples.");
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
    return assignments;
}

PyArrayObject* py_minibatch_multi(PyObject* self, PyObject* args) {
    PyArrayObject* data;
    PyArrayObject* centroids;
    int n_samples, max_iter, n_runs, n_jobs;
    double bic_ratio_termination, reassignment_ratio;

    if (!PyArg_ParseTuple(args, "OOiiiidd", &data, &centroids, &n_samples, &max_iter, &n_runs, &n_jobs, &bic_ratio_termination, &reassignment_ratio)) { 
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
    return centroids;
}

PyArrayObject* py_minibatch(PyObject* self, PyObject* args) {
    PyArrayObject* data;
    PyArrayObject* centroids;
    int n_samples, max_iter;
    double reassignment_ratio, bic_ratio_termination;

    if (!PyArg_ParseTuple(args, "OOiidd", &data, &centroids, &n_samples, &max_iter, &bic_ratio_termination, &reassignment_ratio)) { 
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
        bic_ratio_termination,
        reassignment_ratio,
        k, N, D
    );

    Py_XINCREF(centroids);
    return centroids;
}


PyObject* py_model_variance(PyObject* self, PyObject* args) {
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

    double variance = model_variance(
        PyArray_DATA(data), 
        PyArray_DATA(centroids), 
        k, N, D
    );

    PyObject *ret = Py_BuildValue("d", variance);
    return ret;
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

PyArrayObject* py_kmeanspp_multi(PyObject* self, PyObject* args) {
    PyArrayObject* data;
    PyArrayObject* centroids;
    int n_samples, n_runs, n_jobs;

    if (!PyArg_ParseTuple(args, "OOiii", &data, &centroids, &n_samples, &n_runs, &n_jobs)) { 
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

    if (n_samples >= N) {
        PyErr_SetString(PyExc_RuntimeError,"n_samples must be smaller than the number of samples");
        return NULL;
    }
    if (PyArray_DIM(centroids, 1) != D) {
        PyErr_SetString(PyExc_RuntimeError,"centroids has wrong number of features.");
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
    return centroids;
}


PyArrayObject* py_kmeanspp(PyObject* self, PyObject* args) {
    PyArrayObject* data;
    PyArrayObject* centroids;
    int n_samples;

    if (!PyArg_ParseTuple(args, "OOi", &data, &centroids, &n_samples)) { 
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

    if (n_samples >= N) {
        PyErr_SetString(PyExc_RuntimeError,"n_samples must be smaller than the number of samples");
        return NULL;
    }
    if (PyArray_DIM(centroids, 1) != D) {
        PyErr_SetString(PyExc_RuntimeError,"centroids has wrong number of features.");
        return NULL;
    }

    kmeanspp(
        PyArray_DATA(data), 
        PyArray_DATA(centroids), 
        n_samples,
        k, N, D
    );

    Py_XINCREF(centroids);
    return centroids;
}

double python_distance(double *A, double *B, int D) {
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

PyObject* py_set_metric(PyObject* self, PyObject* args) {
    PyObject *metric_object;

    if (!PyArg_ParseTuple(args, "O", &metric_object)) { 
        PyErr_SetString(PyExc_RuntimeError, "Invalid arguments");
        return NULL;
    }

    if (PyString_Check(metric_object)) {
        char *metric = PyString_AsString(metric_object);
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
    {"minibatch"        , py_minibatch        , METH_VARARGS , minibatch_docstring        }  , 
    {"minibatch_multi"  , py_minibatch_multi  , METH_VARARGS , minibatch_multi_docstring  }  , 
    {"assign_centroids" , py_assign_centroids , METH_VARARGS , assign_centroids_docstring }  , 
    {"kmeanspp"         , py_kmeanspp         , METH_VARARGS , kmeanspp_docstring         }  , 
    {"kmeanspp_multi"   , py_kmeanspp_multi   , METH_VARARGS , kmeanspp_multi_docstring   }  , 
    {"bic"              , py_bic              , METH_VARARGS , bic_docstring              }  , 
    {"model_variance"   , py_model_variance   , METH_VARARGS , model_variance_docstring   }  , 
    {"set_metric"       , py_set_metric       , METH_VARARGS , set_metric_docstring       }  , 
    {NULL               , NULL                , 0            , NULL                       } 
};
 
/* Initialize the module */
PyMODINIT_FUNC init_minibatch(void)
{
    PyObject *m = Py_InitModule3("_minibatch", module_methods, module_docstring);
    if (m == NULL)
        return;

    set_distance_metric(0);
 
    /* Load `numpy` functionality. */
    import_array();
}

