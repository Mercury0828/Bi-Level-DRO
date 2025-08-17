#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>

typedef struct {
    double* data;
    int rows;
    int cols;
} Matrix;

typedef struct {
    int n_locations;
    int n_routes;
    double* distance_matrix;
    double* storage_capacity;
    int* transport_capacity;
} NetworkTopology;

typedef struct {
    double inventory_cost;
    double transport_cost;
    double penalty_cost;
} CostParams;


static Matrix* create_matrix(int rows, int cols) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double*)calloc(rows * cols, sizeof(double));
    return mat;
}

static void free_matrix(Matrix* mat) {
    if (mat) {
        free(mat->data);
        free(mat);
    }
}

static double get_element(Matrix* mat, int i, int j) {
    return mat->data[i * mat->cols + j];
}

static void set_element(Matrix* mat, int i, int j, double value) {
    mat->data[i * mat->cols + j] = value;
}

static double matrix_multiply_sum(double* mat1, double* mat2, int size) {
    double sum = 0.0;
    for (int i = 0; i < size * size; i++) {
        sum += mat1[i] * mat2[i];
    }
    return sum;
}

static void solve_transportation_problem(
    int n_loc,
    double* x_inventory,
    double* demand,
    double* distance_matrix,
    int* transport_capacity,
    double* y_transport_out 
) {

    memset(y_transport_out, 0, n_loc * n_loc * sizeof(double));
    
    for (int j = 0; j < n_loc; j++) {
        double remaining_demand = demand[j];
        
        for (int i = 0; i < n_loc; i++) {
            if (i == j) continue;
            
            double available = fmin(x_inventory[i], remaining_demand);
            available = fmin(available, transport_capacity[i * n_loc + j]);
            
            if (available > 0) {
                y_transport_out[i * n_loc + j] = available;
                remaining_demand -= available;
                
                if (remaining_demand <= 0) break;
            }
        }
    }
}


static void solve_upper_level_dro(
    int n_loc,
    int n_scenarios,
    double* demand_scenarios,
    double* storage_capacity,
    double epsilon,
    CostParams* costs,
    double* x_out 
) {
    double* avg_demand = (double*)calloc(n_loc, sizeof(double));
    for (int s = 0; s < n_scenarios; s++) {
        for (int i = 0; i < n_loc; i++) {
            avg_demand[i] += demand_scenarios[s * n_loc + i] / n_scenarios;
        }
    }
    
    double* std_demand = (double*)calloc(n_loc, sizeof(double));
    for (int s = 0; s < n_scenarios; s++) {
        for (int i = 0; i < n_loc; i++) {
            double diff = demand_scenarios[s * n_loc + i] - avg_demand[i];
            std_demand[i] += diff * diff / n_scenarios;
        }
    }
    for (int i = 0; i < n_loc; i++) {
        std_demand[i] = sqrt(std_demand[i]);
    }
    
    for (int i = 0; i < n_loc; i++) {
        x_out[i] = avg_demand[i] + epsilon * std_demand[i];
        x_out[i] = fmin(x_out[i], storage_capacity[i]);
        x_out[i] = fmax(x_out[i], 0.0);
    }
    
    free(avg_demand);
    free(std_demand);
}

static double compute_total_cost(
    double* x, int n_loc,
    double* y, 
    double* distance_matrix,
    double* demand_scenarios, int n_scenarios,
    CostParams* costs
) {
    double total_cost = 0.0;
    
    for (int s = 0; s < n_scenarios; s++) {
        double scenario_cost = 0.0;
        double* demand = &demand_scenarios[s * n_loc];
        
        for (int i = 0; i < n_loc; i++) {
            scenario_cost += x[i] * costs->inventory_cost;
        }
        
        for (int i = 0; i < n_loc; i++) {
            for (int j = 0; j < n_loc; j++) {
                scenario_cost += y[i * n_loc + j] * distance_matrix[i * n_loc + j] * 
                                costs->transport_cost / 100.0;
            }
        }
        
        double* total_supply = (double*)calloc(n_loc, sizeof(double));
        for (int i = 0; i < n_loc; i++) {
            total_supply[i] = x[i];
            for (int j = 0; j < n_loc; j++) {
                total_supply[i] += y[j * n_loc + i];
            }
            for (int j = 0; j < n_loc; j++) {
                total_supply[i] -= y[i * n_loc + j];
            }
            
            double unmet = fmax(0.0, demand[i] - total_supply[i]);
            scenario_cost += unmet * costs->penalty_cost;
        }
        free(total_supply);
        
        total_cost += scenario_cost;
    }
    
    return total_cost / n_scenarios;
}

static double compute_service_level(
    double* x, int n_loc,
    double* y,
    double* demand_scenarios, int n_scenarios
) {
    double total_service = 0.0;
    
    for (int s = 0; s < n_scenarios; s++) {
        double* demand = &demand_scenarios[s * n_loc];
        double* total_supply = (double*)calloc(n_loc, sizeof(double));
        
        double total_demand = 0.0;
        double met_demand = 0.0;
        
        for (int i = 0; i < n_loc; i++) {
            total_supply[i] = x[i];
            for (int j = 0; j < n_loc; j++) {
                total_supply[i] += y[j * n_loc + i] - y[i * n_loc + j];
            }
            
            total_demand += demand[i];
            met_demand += fmin(demand[i], total_supply[i]);
        }
        
        if (total_demand > 0) {
            total_service += met_demand / total_demand;
        } else {
            total_service += 1.0;
        }
        
        free(total_supply);
    }
    
    return total_service / n_scenarios;
}

static PyObject* compute_cost_distribution(
    double* x, int n_loc,
    double* y,
    double* distance_matrix,
    double* demand_scenarios, int n_scenarios,
    CostParams* costs
) {
    npy_intp dims[1] = {n_scenarios};
    PyObject* array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double* data = (double*)PyArray_DATA((PyArrayObject*)array);
    
    for (int s = 0; s < n_scenarios; s++) {
        double scenario_cost = 0.0;
        double* demand = &demand_scenarios[s * n_loc];
        
        for (int i = 0; i < n_loc; i++) {
            scenario_cost += x[i] * costs->inventory_cost;
        }
        
        for (int i = 0; i < n_loc; i++) {
            for (int j = 0; j < n_loc; j++) {
                scenario_cost += y[i * n_loc + j] * distance_matrix[i * n_loc + j] * 
                                costs->transport_cost / 100.0;
            }
        }
        
        double* total_supply = (double*)calloc(n_loc, sizeof(double));
        for (int i = 0; i < n_loc; i++) {
            total_supply[i] = x[i];
            for (int j = 0; j < n_loc; j++) {
                total_supply[i] += y[j * n_loc + i] - y[i * n_loc + j];
            }
            double unmet = fmax(0.0, demand[i] - total_supply[i]);
            scenario_cost += unmet * costs->penalty_cost;
        }
        free(total_supply);
        
        data[s] = scenario_cost;
    }
    
    return array;
}

static PyObject* solve_dro(PyObject* self, PyObject* args) {
    PyArrayObject *demand_scenarios_array, *distance_matrix_array;
    PyArrayObject *storage_capacity_array, *transport_capacity_array;
    double epsilon;
    double inv_cost, trans_cost, penalty_cost;
    int max_iterations = 20;
    double tolerance = 1e-4;
    
    if (!PyArg_ParseTuple(args, "O!O!O!O!dddd|id",
                         &PyArray_Type, &demand_scenarios_array,
                         &PyArray_Type, &distance_matrix_array,
                         &PyArray_Type, &storage_capacity_array,
                         &PyArray_Type, &transport_capacity_array,
                         &epsilon,
                         &inv_cost, &trans_cost, &penalty_cost,
                         &max_iterations, &tolerance)) {
        return NULL;
    }
    
    int n_scenarios = (int)PyArray_DIM(demand_scenarios_array, 0);
    int n_locations = (int)PyArray_DIM(demand_scenarios_array, 1);
    
    double* demand_scenarios = (double*)PyArray_DATA(demand_scenarios_array);
    double* distance_matrix = (double*)PyArray_DATA(distance_matrix_array);
    double* storage_capacity = (double*)PyArray_DATA(storage_capacity_array);
    double* transport_capacity = (double*)PyArray_DATA(transport_capacity_array);
    
    CostParams costs = {inv_cost, trans_cost, penalty_cost};
    
    clock_t start = clock();
    
    double* x_current = (double*)malloc(n_locations * sizeof(double));
    double* y_current = (double*)calloc(n_locations * n_locations, sizeof(double));
    
    for (int i = 0; i < n_locations; i++) {
        x_current[i] = 50.0;
    }
    
    PyObject* convergence_history = PyDict_New();
    PyObject* objective_list = PyList_New(0);
    PyObject* primal_list = PyList_New(0);
    
    double obj_val = 0.0;
    int actual_iterations = 0;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        double* x_new = (double*)malloc(n_locations * sizeof(double));
        double* y_new = (double*)calloc(n_locations * n_locations, sizeof(double));
        
        solve_upper_level_dro(n_locations, n_scenarios, demand_scenarios,
                            storage_capacity, epsilon, &costs, x_new);
        
        double* avg_demand = (double*)calloc(n_locations, sizeof(double));
        for (int s = 0; s < n_scenarios; s++) {
            for (int i = 0; i < n_locations; i++) {
                avg_demand[i] += demand_scenarios[s * n_locations + i] / n_scenarios;
            }
        }
        
        solve_transportation_problem(n_locations, x_new, avg_demand,
                                    distance_matrix, transport_capacity, y_new);
        
        double primal_residual = 0.0;
        for (int i = 0; i < n_locations; i++) {
            double diff = x_new[i] - x_current[i];
            primal_residual += diff * diff;
        }
        primal_residual = sqrt(primal_residual);
        
        obj_val = compute_total_cost(x_new, n_locations, y_new,
                                    distance_matrix, demand_scenarios,
                                    n_scenarios, &costs);
        
        PyList_Append(objective_list, PyFloat_FromDouble(obj_val));
        PyList_Append(primal_list, PyFloat_FromDouble(primal_residual));
        
        if (primal_residual < tolerance && iter > 0) {
            free(x_current);
            free(y_current);
            x_current = x_new;
            y_current = y_new;
            actual_iterations = iter + 1;
            break;
        }
        
        free(x_current);
        free(y_current);
        x_current = x_new;
        y_current = y_new;
        actual_iterations = iter + 1;
        
        free(avg_demand);
    }
    
    PyDict_SetItemString(convergence_history, "objective", objective_list);
    PyDict_SetItemString(convergence_history, "primal_residual", primal_list);
    
    double total_cost = compute_total_cost(x_current, n_locations, y_current,
                                          distance_matrix, demand_scenarios,
                                          n_scenarios, &costs);
    double service_level = compute_service_level(x_current, n_locations,
                                                y_current, demand_scenarios,
                                                n_scenarios);
    PyObject* cost_per_scenario = compute_cost_distribution(x_current, n_locations,
                                                           y_current, distance_matrix,
                                                           demand_scenarios, n_scenarios,
                                                           &costs);
    
    double* cost_data = (double*)PyArray_DATA((PyArrayObject*)cost_per_scenario);
    double mean_cost = 0.0, variance = 0.0;
    for (int i = 0; i < n_scenarios; i++) {
        mean_cost += cost_data[i] / n_scenarios;
    }
    for (int i = 0; i < n_scenarios; i++) {
        double diff = cost_data[i] - mean_cost;
        variance += diff * diff / n_scenarios;
    }
    
    double unmet_demand = 0.0;
    for (int s = 0; s < n_scenarios; s++) {
        for (int i = 0; i < n_locations; i++) {
            double supply = x_current[i];
            for (int j = 0; j < n_locations; j++) {
                supply += y_current[j * n_locations + i] - y_current[i * n_locations + j];
            }
            unmet_demand += fmax(0.0, demand_scenarios[s * n_locations + i] - supply);
        }
    }
    
    double computation_time = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    
    npy_intp x_dims[1] = {n_locations};
    PyObject* x_optimal = PyArray_SimpleNewFromData(1, x_dims, NPY_DOUBLE, x_current);
    PyArray_ENABLEFLAGS((PyArrayObject*)x_optimal, NPY_ARRAY_OWNDATA);
    
    npy_intp y_dims[2] = {n_locations, n_locations};
    PyObject* y_optimal = PyArray_SimpleNewFromData(2, y_dims, NPY_DOUBLE, y_current);
    PyArray_ENABLEFLAGS((PyArrayObject*)y_optimal, NPY_ARRAY_OWNDATA);
    
    PyObject* result = PyDict_New();
    PyDict_SetItemString(result, "method", PyUnicode_FromString("DRO"));
    
    char uncertainty_str[50];
    sprintf(uncertainty_str, "ε=%.2f", epsilon);
    PyDict_SetItemString(result, "uncertainty_level", PyUnicode_FromString(uncertainty_str));
    
    PyDict_SetItemString(result, "x_optimal", x_optimal);
    PyDict_SetItemString(result, "y_optimal", y_optimal);
    PyDict_SetItemString(result, "objective_value", PyFloat_FromDouble(obj_val));
    PyDict_SetItemString(result, "total_cost", PyFloat_FromDouble(total_cost));
    PyDict_SetItemString(result, "service_level", PyFloat_FromDouble(service_level));
    PyDict_SetItemString(result, "computation_time", PyFloat_FromDouble(computation_time));
    PyDict_SetItemString(result, "iterations", PyLong_FromLong(actual_iterations));
    PyDict_SetItemString(result, "convergence_history", convergence_history);
    PyDict_SetItemString(result, "cost_per_scenario", cost_per_scenario);
    PyDict_SetItemString(result, "variance", PyFloat_FromDouble(variance));
    PyDict_SetItemString(result, "unmet_demand", PyFloat_FromDouble(unmet_demand));
    
    return result;
}

static PyObject* solve_ro(PyObject* self, PyObject* args) {
    PyArrayObject *demand_scenarios_array, *distance_matrix_array;
    PyArrayObject *storage_capacity_array, *transport_capacity_array;
    double epsilon;
    double inv_cost, trans_cost, penalty_cost;
    
    if (!PyArg_ParseTuple(args, "O!O!O!O!dddd",
                         &PyArray_Type, &demand_scenarios_array,
                         &PyArray_Type, &distance_matrix_array,
                         &PyArray_Type, &storage_capacity_array,
                         &PyArray_Type, &transport_capacity_array,
                         &epsilon,
                         &inv_cost, &trans_cost, &penalty_cost)) {
        return NULL;
    }
    
    int n_scenarios = (int)PyArray_DIM(demand_scenarios_array, 0);
    int n_locations = (int)PyArray_DIM(demand_scenarios_array, 1);
    
    double* demand_scenarios = (double*)PyArray_DATA(demand_scenarios_array);
    double* distance_matrix = (double*)PyArray_DATA(distance_matrix_array);
    double* storage_capacity = (double*)PyArray_DATA(storage_capacity_array);
    double* transport_capacity = (double*)PyArray_DATA(transport_capacity_array);
    
    CostParams costs = {inv_cost, trans_cost, penalty_cost};
    
    clock_t start = clock();
    
    double* worst_case_demand = (double*)calloc(n_locations, sizeof(double));
    for (int i = 0; i < n_locations; i++) {
        double max_demand = 0.0;
        for (int s = 0; s < n_scenarios; s++) {
            double demand = demand_scenarios[s * n_locations + i];
            if (demand > max_demand) {
                max_demand = demand;
            }
        }
        worst_case_demand[i] = max_demand * (1 + epsilon);
    }
    
    double* x_optimal = (double*)malloc(n_locations * sizeof(double));
    for (int i = 0; i < n_locations; i++) {
        x_optimal[i] = worst_case_demand[i] * 0.8;
        x_optimal[i] = fmin(x_optimal[i], storage_capacity[i] * 0.9);
        x_optimal[i] = fmax(x_optimal[i], worst_case_demand[i] * 0.3);
    }
    
    double* y_optimal = (double*)calloc(n_locations * n_locations, sizeof(double));
    solve_transportation_problem(n_locations, x_optimal, worst_case_demand,
                                distance_matrix, transport_capacity, y_optimal);
    
    double total_cost = compute_total_cost(x_optimal, n_locations, y_optimal,
                                          distance_matrix, demand_scenarios,
                                          n_scenarios, &costs);
    double service_level = compute_service_level(x_optimal, n_locations,
                                                y_optimal, demand_scenarios,
                                                n_scenarios);
    PyObject* cost_per_scenario = compute_cost_distribution(x_optimal, n_locations,
                                                           y_optimal, distance_matrix,
                                                           demand_scenarios, n_scenarios,
                                                           &costs);
    
    double* cost_data = (double*)PyArray_DATA((PyArrayObject*)cost_per_scenario);
    double mean_cost = 0.0, variance = 0.0;
    for (int i = 0; i < n_scenarios; i++) {
        mean_cost += cost_data[i] / n_scenarios;
    }
    for (int i = 0; i < n_scenarios; i++) {
        double diff = cost_data[i] - mean_cost;
        variance += diff * diff / n_scenarios;
    }
    
    double unmet_demand = 0.0;
    for (int s = 0; s < n_scenarios; s++) {
        for (int i = 0; i < n_locations; i++) {
            double supply = x_optimal[i];
            for (int j = 0; j < n_locations; j++) {
                supply += y_optimal[j * n_locations + i] - y_optimal[i * n_locations + j];
            }
            unmet_demand += fmax(0.0, demand_scenarios[s * n_locations + i] - supply);
        }
    }
    
    double computation_time = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    
    npy_intp x_dims[1] = {n_locations};
    PyObject* x_array = PyArray_SimpleNewFromData(1, x_dims, NPY_DOUBLE, x_optimal);
    PyArray_ENABLEFLAGS((PyArrayObject*)x_array, NPY_ARRAY_OWNDATA);
    
    npy_intp y_dims[2] = {n_locations, n_locations};
    PyObject* y_array = PyArray_SimpleNewFromData(2, y_dims, NPY_DOUBLE, y_optimal);
    PyArray_ENABLEFLAGS((PyArrayObject*)y_array, NPY_ARRAY_OWNDATA);
    
    PyObject* result = PyDict_New();
    PyDict_SetItemString(result, "method", PyUnicode_FromString("RO"));
    
    char uncertainty_str[50];
    sprintf(uncertainty_str, "ε=%.2f", epsilon);
    PyDict_SetItemString(result, "uncertainty_level", PyUnicode_FromString(uncertainty_str));
    
    PyDict_SetItemString(result, "x_optimal", x_array);
    PyDict_SetItemString(result, "y_optimal", y_array);
    PyDict_SetItemString(result, "objective_value", PyFloat_FromDouble(total_cost));
    PyDict_SetItemString(result, "total_cost", PyFloat_FromDouble(total_cost));
    PyDict_SetItemString(result, "service_level", PyFloat_FromDouble(service_level));
    PyDict_SetItemString(result, "computation_time", PyFloat_FromDouble(computation_time));
    PyDict_SetItemString(result, "iterations", PyLong_FromLong(1));
    PyDict_SetItemString(result, "convergence_history", PyDict_New());
    PyDict_SetItemString(result, "cost_per_scenario", cost_per_scenario);
    PyDict_SetItemString(result, "variance", PyFloat_FromDouble(variance));
    PyDict_SetItemString(result, "unmet_demand", PyFloat_FromDouble(unmet_demand));
    
    free(worst_case_demand);
    
    return result;
}

static PyObject* solve_sp(PyObject* self, PyObject* args) {
    PyArrayObject *demand_scenarios_array, *distance_matrix_array;
    PyArrayObject *storage_capacity_array, *transport_capacity_array;
    double epsilon;
    double inv_cost, trans_cost, penalty_cost;
    
    if (!PyArg_ParseTuple(args, "O!O!O!O!dddd",
                         &PyArray_Type, &demand_scenarios_array,
                         &PyArray_Type, &distance_matrix_array,
                         &PyArray_Type, &storage_capacity_array,
                         &PyArray_Type, &transport_capacity_array,
                         &epsilon,
                         &inv_cost, &trans_cost, &penalty_cost)) {
        return NULL;
    }
    
    int n_scenarios = (int)PyArray_DIM(demand_scenarios_array, 0);
    int n_locations = (int)PyArray_DIM(demand_scenarios_array, 1);
    
    double* demand_scenarios = (double*)PyArray_DATA(demand_scenarios_array);
    double* distance_matrix = (double*)PyArray_DATA(distance_matrix_array);
    double* storage_capacity = (double*)PyArray_DATA(storage_capacity_array);
    double* transport_capacity = (double*)PyArray_DATA(transport_capacity_array);
    
    CostParams costs = {inv_cost, trans_cost, penalty_cost};
    
    clock_t start = clock();
    
    double* avg_demand = (double*)calloc(n_locations, sizeof(double));
    for (int s = 0; s < n_scenarios; s++) {
        for (int i = 0; i < n_locations; i++) {
            avg_demand[i] += demand_scenarios[s * n_locations + i] / n_scenarios;
        }
    }
    
    double* x_optimal = (double*)malloc(n_locations * sizeof(double));
    for (int i = 0; i < n_locations; i++) {
        x_optimal[i] = avg_demand[i] * (1 + epsilon * 0.5);
        x_optimal[i] = fmin(x_optimal[i], storage_capacity[i]);
        x_optimal[i] = fmax(x_optimal[i], 0.0);
    }
    
    double* y_avg = (double*)calloc(n_locations * n_locations, sizeof(double));
    for (int s = 0; s < n_scenarios; s++) {
        double* y_scenario = (double*)calloc(n_locations * n_locations, sizeof(double));
        double* demand = &demand_scenarios[s * n_locations];
        
        solve_transportation_problem(n_locations, x_optimal, demand,
                                    distance_matrix, transport_capacity, y_scenario);
        
        for (int i = 0; i < n_locations * n_locations; i++) {
            y_avg[i] += y_scenario[i] / n_scenarios;
        }
        
        free(y_scenario);
    }
    
    double total_cost = compute_total_cost(x_optimal, n_locations, y_avg,
                                          distance_matrix, demand_scenarios,
                                          n_scenarios, &costs);
    double service_level = compute_service_level(x_optimal, n_locations,
                                                y_avg, demand_scenarios,
                                                n_scenarios);
    PyObject* cost_per_scenario = compute_cost_distribution(x_optimal, n_locations,
                                                           y_avg, distance_matrix,
                                                           demand_scenarios, n_scenarios,
                                                           &costs);
    
    double* cost_data = (double*)PyArray_DATA((PyArrayObject*)cost_per_scenario);
    double mean_cost = 0.0, variance = 0.0;
    for (int i = 0; i < n_scenarios; i++) {
        mean_cost += cost_data[i] / n_scenarios;
    }
    for (int i = 0; i < n_scenarios; i++) {
        double diff = cost_data[i] - mean_cost;
        variance += diff * diff / n_scenarios;
    }
    
    double unmet_demand = 0.0;
    for (int s = 0; s < n_scenarios; s++) {
        for (int i = 0; i < n_locations; i++) {
            double supply = x_optimal[i];
            for (int j = 0; j < n_locations; j++) {
                supply += y_avg[j * n_locations + i] - y_avg[i * n_locations + j];
            }
            unmet_demand += fmax(0.0, demand_scenarios[s * n_locations + i] - supply);
        }
    }
    
    double computation_time = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    
    npy_intp x_dims[1] = {n_locations};
    PyObject* x_array = PyArray_SimpleNewFromData(1, x_dims, NPY_DOUBLE, x_optimal);
    PyArray_ENABLEFLAGS((PyArrayObject*)x_array, NPY_ARRAY_OWNDATA);
    
    npy_intp y_dims[2] = {n_locations, n_locations};
    PyObject* y_array = PyArray_SimpleNewFromData(2, y_dims, NPY_DOUBLE, y_avg);
    PyArray_ENABLEFLAGS((PyArrayObject*)y_array, NPY_ARRAY_OWNDATA);
    
    PyObject* result = PyDict_New();
    PyDict_SetItemString(result, "method", PyUnicode_FromString("SP"));
    
    char uncertainty_str[50];
    sprintf(uncertainty_str, "ε=%.2f", epsilon);
    PyDict_SetItemString(result, "uncertainty_level", PyUnicode_FromString(uncertainty_str));
    
    PyDict_SetItemString(result, "x_optimal", x_array);
    PyDict_SetItemString(result, "y_optimal", y_array);
    PyDict_SetItemString(result, "objective_value", PyFloat_FromDouble(total_cost));
    PyDict_SetItemString(result, "total_cost", PyFloat_FromDouble(total_cost));
    PyDict_SetItemString(result, "service_level", PyFloat_FromDouble(service_level));
    PyDict_SetItemString(result, "computation_time", PyFloat_FromDouble(computation_time));
    PyDict_SetItemString(result, "iterations", PyLong_FromLong(1));
    PyDict_SetItemString(result, "convergence_history", PyDict_New());
    PyDict_SetItemString(result, "cost_per_scenario", cost_per_scenario);
    PyDict_SetItemString(result, "variance", PyFloat_FromDouble(variance));
    PyDict_SetItemString(result, "unmet_demand", PyFloat_FromDouble(unmet_demand));
    
    free(avg_demand);
    
    return result;
}

static PyMethodDef CoreAlgorithmMethods[] = {
    {"solve_dro", solve_dro, METH_VARARGS,
     "Solve using Bi-Level DRO algorithm with real optimization"},
    {"solve_ro", solve_ro, METH_VARARGS,
     "Solve using Robust Optimization with real calculations"},
    {"solve_sp", solve_sp, METH_VARARGS,
     "Solve using Stochastic Programming with real calculations"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef corealgorithmmodule = {
    PyModuleDef_HEAD_INIT,
    "core_algorithm",
    "Real optimization algorithms implementation in C",
    -1,
    CoreAlgorithmMethods
};

PyMODINIT_FUNC PyInit_core_algorithm(void) {
    import_array();
    return PyModule_Create(&corealgorithmmodule);
}