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
package com.cloudera.science.ml.parallel.pivot;

class Stat {
  private long[] counts;
  private double[] sums;
  
  // For Avro serialization
  private Stat() { }
  
  Stat(int n) {
    this.counts = new long[n];
    this.sums = new double[n];
  }
  
  Stat(long[] counts, double[] sums) {
    this.counts = new long[counts.length];
    this.sums = new double[sums.length];
    System.arraycopy(counts, 0, this.counts, 0, counts.length);
    System.arraycopy(sums, 0, this.sums, 0, sums.length);
  }
  
  public Stat copy() {
    return new Stat(counts, sums);
  }
  
  int size() {
    return counts.length;
  }
  
  long getCount(int i) {
    return counts[i];
  }
  
  double getSum(int i) {
    return sums[i];
  }
  
  public void inc(int i, double value) {
    if (!Double.isNaN(value)) {
      counts[i]++;
      sums[i] += value;
      System.out.println(String.format("Inc %d by %f to get %f", i, value, sums[i]));
    }
  }
  
  public Stat merge(Stat other) {
    long[] c = new long[counts.length];
    double[] s = new double[sums.length];
    for (int i = 0; i < c.length; i++) {
      c[i] = counts[i] + other.counts[i];
      s[i] = sums[i] + other.sums[i];
    }
    return new Stat(c, s);
  }
}
