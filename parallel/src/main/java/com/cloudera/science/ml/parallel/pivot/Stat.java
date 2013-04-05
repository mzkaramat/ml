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
  private long count = 0L;
  private double sum = 0.0;
  
  Stat() { this(0L, 0.0); }
  
  Stat(long count, double sum) {
    this.count = count;
    this.sum = sum;
  }
  
  long getCount() {
    return count;
  }
  
  double getSum() {
    return sum;
  }
  
  public void inc(double value) {
    this.count++;
    this.sum += value;
  }
  
  public Stat merge(Stat other) {
    return new Stat(count + other.count, sum + other.sum);
  }
}
