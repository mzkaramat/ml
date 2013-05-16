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

import java.util.Arrays;

class InternalNumeric {
  private double min = Double.POSITIVE_INFINITY;
  private double max = Double.NEGATIVE_INFINITY;
  private double sum;
  private double sumSq;
  private long missing;
  private double median;

  private int b;
  private double[][] medianArray;
  private int dataCursor = 0;
  private int medianK = 1;
  private int medianCursor = 0;

  InternalNumeric() {
  }

  InternalNumeric(int b, int k) {
    if (b % 2 == 0) {
      b += 1;
    }
    this.b = b;
    medianArray = new double[k - 1][b];

  }

  public Numeric toNumeric(long recordCount) {
    if (missing == recordCount) {
      return new Numeric(0.0, 0.0, 0.0, 0.0, missing);
    }
    long n = recordCount - missing;
    double mean = sum / n;
    double stdDev = Math.sqrt((sumSq / n) - mean * mean);
    return new Numeric(min, max, mean, stdDev, missing, median);
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
      medianArray[0][dataCursor++] = (long) Math.floor(d);
      if (dataCursor == b) {
        dataCursor = 0;
        medianArray[medianK][medianCursor++] = computeFullArrayMedian(medianArray[medianK - 1]);
        if (medianCursor == b) {
          medianK++;
          medianArray[medianK][0] = computeFullArrayMedian(medianArray[medianK - 1]);
          medianCursor = 1;
        }
      }
      median = currentMedian();
    }
  }

  private double computeFullArrayMedian(double[] values) {
    double[] cloned = values.clone();
    Arrays.sort(cloned);
    return cloned[cloned.length / 2 + 1];
  }

  private double computePartialArrayMedian(double[] values, int size) {
    double[] cloned = Arrays.copyOf(values, size).clone();
    Arrays.sort(cloned);
    return cloned[cloned.length / 2 + 1];
  }

  private double currentMedian() {
    if (medianCursor == 1) {
      return medianArray[medianK][0];
    } else {
      return computePartialArrayMedian(medianArray[medianK], medianK);
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
  }
}