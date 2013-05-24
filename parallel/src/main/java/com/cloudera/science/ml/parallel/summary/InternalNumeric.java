/**
 * Copyright (c) 2013, Cloudera, Inc. All Rights Reserved.
 *
 * Cloudera, Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"). You may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for
 * the specific language governing permissions and limitations under the
 * License.
 */
package com.cloudera.science.ml.parallel.summary;

import com.cloudera.science.ml.core.summary.Numeric;
import org.apache.crunch.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

class InternalNumeric {
  private double min = Double.POSITIVE_INFINITY;
  private double max = Double.NEGATIVE_INFINITY;
  private double sum;
  private double sumSq;
  private long missing;

  private int b;
  private double[][] medianArray;
  private int[] cursor;
  private int medianK;

  InternalNumeric() {
    this(11, 10);
  }

  private InternalNumeric(int b, int k) {
    this.b = b;
    medianArray = new double[k][b];
    cursor = new int[k];
    medianK = 1;

  }

  public Numeric toNumeric(long recordCount) {
    if (missing == recordCount) {
      return new Numeric(0.0, 0.0, 0.0, 0.0, missing);
    }
    long n = recordCount - missing;
    double mean = sum / n;
    double stdDev = Math.sqrt((sumSq / n) - mean * mean);
    return new Numeric(min, max, mean, stdDev, currentMedian(), missing);
  }

  public void update(double d) {
    if (Double.isNaN(d)) {
      missing++;
    } else {
      sum += d;
      sumSq += d * d;
      if (d < min) {
        min = d;
      }
      if (d > max) {
        max = d;
      }
      updateMedian(d);
    }
  }

  private void updateMedian(double d) {
    medianArray[0][cursor[0]++] = d;
    int i = 1;
    while (i < medianArray.length - 1) {
      if (cursor[i - 1] == b) {
        medianArray[i][cursor[i]] = computeFullArrayMedian(medianArray[i - 1]);
        cursor[i - 1] = 0;
        cursor[i]++;
        if (medianK < i) {
          medianK = i;
        }
      } else break;
      i++;
    }
  }

  private double computeFullArrayMedian(double[] values) {
    double[] cloned = values.clone();
    Arrays.sort(cloned);
    int index = cloned.length % 2 == 0 ? cloned.length / 2 - 1 : cloned.length / 2;
    return cloned[index];
  }

  private double weightedMedian() {
    List<Pair<Double, Double>> valueList = weightedList();
    double sumWeights = sumOfSeconds(valueList);
    double s = sumWeights;
    int j = 0;
    System.out.println(valueList);

    while (s > sumWeights / 2) {
      s -= valueList.get(j++).second();
    }
    return valueList.get(j-1).first();
  }

  private List<Pair<Double, Double>> weightedList() {
    List<Pair<Double, Double>> valueList = new ArrayList<Pair<Double, Double>>();

    for (int i = 0; i <= medianK; i++) {
      double weight = Math.pow(b, i);
      for (int j = 0; j < cursor[i]; j++) {
        valueList.add(new Pair<Double, Double>(medianArray[i][j], weight));
      }
    }

    Collections.sort(valueList, new Comparator<Pair<Double, Double>>() {
      @Override
      public int compare(Pair<Double, Double> doubleDoublePair, Pair<Double, Double> doubleDoublePair2) {
        int c = doubleDoublePair.first().compareTo(doubleDoublePair2.first());
        if (c == 0) {
          return doubleDoublePair.second().compareTo(doubleDoublePair2.second());
        }
        return c;
      }
    });
    return valueList;
  }

  private double sumOfSeconds(List<Pair<Double, Double>> list) {
    double sum = 0;
    for (Pair<Double, Double> pair : list) {
      sum += pair.second();
    }
    return sum;
  }


  private double currentMedian() {
    if (cursor[medianK] == 0) {
      return computeFullArrayMedian(medianArray[medianK - 1]);
    } else if (cursor[medianK] == b - 1) {
      return computeFullArrayMedian(medianArray[medianK]);
    } else {
      return weightedMedian();
    }
  }

  public void merge(InternalNumeric other) {
    sum += other.sum;
    sumSq += other.sumSq;
    missing += other.missing;
    if (other.min < min) {
      min = other.min;
    }
    if (other.max > max) {
      max = other.max;
    }
    List<Pair<Double, Double>> valueList = other.weightedList();
    for (Pair<Double, Double> weightedValue : valueList) {
      double value = weightedValue.first();
      double weight = weightedValue.second();
      for (int i = 0; i < weight; i++) {
        this.updateMedian(value);
      }
    }
  }
}