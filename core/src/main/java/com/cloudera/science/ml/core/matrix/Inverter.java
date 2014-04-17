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
package com.cloudera.science.ml.core.matrix;


import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import java.io.Serializable;

/**
 * A collection of strategies for inverting a {@code RealMatrix}.
 */
public enum Inverter implements Serializable {

  LU {
    @Override
    public RealMatrix apply(RealMatrix matrix) {
      return new LUDecomposition(matrix).getSolver().getInverse();
    }
  },

  SVD {
    @Override
    public RealMatrix apply(RealMatrix matrix) {
      return new SingularValueDecomposition(matrix).getSolver().getInverse();
    }
  },

  QR {
    @Override
    public RealMatrix apply(RealMatrix matrix) {
      return new QRDecomposition(matrix).getSolver().getInverse();
    }
  };

  public abstract RealMatrix apply(RealMatrix matrix);
}
