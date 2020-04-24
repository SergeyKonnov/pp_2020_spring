// Copyright 2020 Koltyushkina Yanina

#include "../../../modules/task_2/koltyushkina_ya_radix_sort_for_double/radix_sort.h"
#include <omp.h>
#include <vector>
#include <ctime>
#include <random>
#include <iostream>
#include <utility>
#include <queue>
#include <limits>

sortTask::sortTask(double* array, int index, int size):
                                array(array), index(index), size(size) {
    num_of_depends = 0;
}

void sortTask::do_task() {
    double * arr = array + index;
    RadixSortAll(&arr, size);
}

mergeTask::mergeTask(double* inmas, int index1, int size1, int index2, int size2):
            inmas(inmas), index1(index1), size1(size1), index2(index2), size2(size2) {
    num_of_depends = 2;
}

void mergeTask::do_task() {
    std::vector<double>tmp(size1+size2);
    int ind1 = index1, ind2 = index2, t = 0;
    while (ind1 < size1 + index1 && ind2 < size2 + index2) {
        if (inmas[ind1] < inmas[ind2]) {
            tmp[t++] = inmas[ind1++];
        } else {
            tmp[t++] = inmas[ind2++];
        }
    }
    while (ind1 < size1 + index1)
        tmp[t++] = inmas[ind1++];
    while (ind2 < size2 + index2)
        tmp[t++] = inmas[ind2++];
    for (int i = 0; i < t; i++) {
        inmas[index1+i] = tmp[i];
    }
}

void RadixSortPart(double *inmas, double **outmas, int len, int byteN) {
    if (len == 0)
        return;
    unsigned char *bymas = (unsigned char*)inmas;
    int schet[256];
    int smesh;
    // memset(schet, 0, sizeof(int) * 256);
    for (int i = 0; i < 256; i++) {
        schet[i] = 0;
    }

    for (int i = 0; i < len; i++) {
        schet[bymas[8 * i + byteN]]++;
    }

    int s = 0;
    for (s = 0; s < 256; s++) {
        if (schet[s] != 0)
        break;
    }

    smesh = schet[s];
    schet[s] = 0;
    s++;

    int a;
    for (; s < 256; s++) {
        a = schet[s];
        schet[s] = smesh;
        smesh += a;
    }

    for (int i = 0; i < len; i++) {
        (*outmas)[schet[bymas[8 * i + byteN]]] = inmas[i];
        schet[bymas[8 * i + byteN]]++;
    }
}

void RadixSortAll(double**inmas, int len) {
    double* outmas = new double[len];
    RadixSortPart(*inmas, &outmas, len, 0);
    RadixSortPart(outmas, inmas, len, 1);
    RadixSortPart(*inmas, &outmas, len, 2);
    RadixSortPart(outmas, inmas, len, 3);
    RadixSortPart(*inmas, &outmas, len, 4);
    RadixSortPart(outmas, inmas, len, 5);
    RadixSortPart(*inmas, &outmas, len, 6);
    RadixSortPart(outmas, inmas, len, 7);
}

void oddEvenMerge(double* array, int index, int size) {
    if (size == 2) {
        if (array[index] > array[index + 1])
            std::swap(array[index], array[index+1]);
        return;
    }
    for (int i = index + 1; i < index + size-1; i+=2) {
        if (array[i] > array[i+1])
            std::swap(array[i], array[i+1]);
    }
}

double* RandMas(int len, double low, double high) {
  if (len <= 0) {
    throw std::exception();
  }
  std::random_device ran;
  std::mt19937 g(ran());
  if (low >= high) {
    throw std::exception();
  }
  std::uniform_real_distribution<> range(low, high);
  double *mas = new double[len];
  for (int i = 0; i < len; i++) {
    mas[i] = range(g);
  }
  return mas;
}

void get_tree_task(double* inmas, int left, int right, task* prev_task,
                                const std::vector<int>& portion, task* task_array[], bool is_begin) {
    if (right - left == 0) {
        int begin = ((left == 0)?0:portion[left-1]);
        int len = portion[left];
        if (left)
            len -= portion[left-1];
        sortTask* stask = new sortTask(inmas, begin, len);
        stask->ref = prev_task;
        task_array[right] = stask;
    } else {
        int mid = (left+right)/2+1;
        int indleft = 0;
        if (left)
            indleft = portion[left-1];
        int lenleft = portion[mid-1]-indleft;
        int indright = indleft+lenleft;
        int lenright = portion[right]-portion[mid-1];
        mergeTask* mtask = new mergeTask(inmas, indleft, lenleft, indright, lenright);
        mtask->ref = prev_task;
        get_tree_task(inmas, mid, right, mtask, portion, task_array);
        get_tree_task(inmas, left, mid-1, mtask, portion, task_array);
    }
}

void RadixSortAllParallel(double ** inmas, int len) {
    int num_threads = omp_get_max_threads();
    std::vector<task*>task_array(num_threads);
    std::vector<task*> tasks;
    std::vector<int>portion(num_threads);
    for (int i = 0; i < num_threads; i++) {
        portion[i] = len/num_threads +
                    (i < (len%num_threads)?1:0) + ((i == 0)?0:portion[i-1]);
    }
    task* prev_task = nullptr;
    get_tree_task(*inmas, 0, num_threads-1, prev_task, portion, task_array.data(), 1);
    while (task_array.size()) {
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(task_array.size()); i++) {
            task_array[i]->do_task();
        }
        std::vector<task*>tmp_task_array;
        for (int i = 0; i < static_cast<int>(task_array.size()); i++) {
            if (task_array[i]->ref == nullptr)
                continue;
            task_array[i]->ref->num_of_depends--;
            if (task_array[i]->ref->num_of_depends == 0)
                tmp_task_array.push_back(task_array[i]->ref);
            delete task_array[i];
        }
        std::swap(tmp_task_array, task_array);
    }
}
