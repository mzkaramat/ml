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
package com.cloudera.science.ml.core.summary;

import java.io.Serializable;

public class Numeric implements Serializable {
  private double min;
  private double max;
  private double mean;
  private double stdDev;
  private double firstQuartile;
  private double remedian;
  private double thirdQuartile;

  private Long missing;
  private String transform;

  // For serialization
  private Numeric() {
  }

  //  public Numeric(double min, double max, double mean, double stdDev, long missing) {
//    this(min, max, mean, stdDev);
//    if (missing > 0) {
//      this.missing = missing;
//    }
//  }
  public Numeric(double min, double max, double mean, double stdDev, double remedian, double firstQuartile,
                 double thirdQuartile, long missing) {
    this(min, max, mean, stdDev, remedian, firstQuartile, thirdQuartile);
    if (missing > 0) {
      this.missing = missing;
    }
  }

  public Numeric(double min, double max, double mean, double stdDev, double remedian, double firstQuartile,
                 double thirdQuartile) {
    this.min = min;
    this.max = max;
    this.mean = mean;
    this.stdDev = stdDev;
    this.firstQuartile = firstQuartile;
    this.remedian = remedian;
    this.thirdQuartile = thirdQuartile;
  }

  public double min() {
    return min;
  }

  public double max() {
    return max;
  }

  public double mean() {
    return mean;
  }

  public double stdDev() {
    return stdDev;
  }

  public double range() {
    return max - min;
  }

  public double firstQuartile() {
    return firstQuartile;
  }

  public double remedian() {
    return remedian;
  }

  double thirdQuartile() {
    return thirdQuartile;
  }

  public String getTransform() {
    return transform;
  }

  public long getMissing() {
    return missing == null ? 0L : missing;
  }
}
