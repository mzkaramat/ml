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

import com.cloudera.science.ml.avro.MLMatrixEntry;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

public class MatrixUtils {
  public static RealMatrix toRealMatrix(int rows,
                                        int columns,
                                        Iterable<MLMatrixEntry> entries,
                                        boolean symmetric) {
    double[][] data = new double[rows][columns];
    for (MLMatrixEntry e : entries) {
      data[e.getRow()][e.getColumn()] = e.getValue();
      if (symmetric) {
        data[e.getColumn()][e.getRow()] = e.getValue();
      }
    }
    return new Array2DRowRealMatrix(data);
  }
}
