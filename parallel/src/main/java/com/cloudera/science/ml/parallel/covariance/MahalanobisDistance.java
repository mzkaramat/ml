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
package com.cloudera.science.ml.parallel.covariance;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import java.io.Serializable;

public class MahalanobisDistance implements Serializable {
  private final double[] means;
  private final double[][] covInv;
  private final long n;
  private transient Vector m;
  private transient Matrix ci;

  public MahalanobisDistance(double[] means, double[][] covInv, long n) {
    this.means = means;
    this.covInv = covInv;
    this.n = n;
  }

  public void initialize() {
    if (m == null) {
      this.m = new DenseVector(means);
      this.ci = new DenseMatrix(covInv);
    }
  }

  public double distance(Vector v) {
    Vector d = v.minus(m);
    return d.dot(ci.times(d));
  }
}
